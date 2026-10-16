"""Local web page to review recorded evals, edit scores/feedback, then Submit or Discard each one.

  .venv/bin/python scripts/review_evals.py --config configs/robot.yam-eef-llm.yaml [--port 8765]
  open http://127.0.0.1:8765

Viewing and "Save locally" only touch local files. Only the Submit and Discard buttons contact the Router:
  Submit  = colosseum-robot <config> --resume ev_<id>  (sends scores/feedback, uploads videos + dataset; public)
  Discard = colosseum-robot <config> --abort ev_<id>   (closes the eval; nothing published)
Preview videos are encoded into eval_review_cache/ next to the eval folder, which stays untouched.
"""
import argparse
import html
import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml
from urllib.parse import parse_qs, unquote, urlparse

CAMS = ('head_image', 'left_image', 'right_image')
HERE = Path(__file__).parent
LOCK = threading.Lock()
CLIENT = HERE.parent
JOBS = {}  # ev -> Popen of a running submit/discard

def load(ev):
    return json.loads((ROOT / ev / 'manifest.json').read_text())


def runs(manifest):
    out = []
    for run in manifest['assignment'].get('runs', []):
        path = ROOT / manifest['assignment']['id'] / run['id']
        status = path / 'recording-status.json'
        done = (path / 'frames.jsonl').exists() and (not status.exists() or json.loads(status.read_text()).get('complete'))
        out.append(dict(run, path=path, recorded=bool(done)))
    return out


def preview(ev, run_id, cam):
    """Encode PNG frames to a cached mp4 (once). Returns the path or None."""
    src = ROOT / ev / run_id
    dst = CACHE / ev / run_id / f'{cam}.mp4'
    if dst.exists():
        return dst
    if (src / cam / 'video.mp4').exists():
        return src / cam / 'video.mp4'
    frames = src / 'frames.jsonl'
    if not frames.exists() or not (src / cam).is_dir():
        return None
    with LOCK:
        if dst.exists():
            return dst
        rows = [json.loads(line) for line in frames.read_text().splitlines() if line.strip()]
        if not rows:
            return None
        lines = []
        for i, row in enumerate(rows):
            d = rows[i + 1]['seconds'] - row['seconds'] if i + 1 < len(rows) else 1 / 15
            lines += [f"file '{(src / cam / f'{row['frame']:06d}.png').resolve()}'", f'duration {max(d, .001):.6f}']
        lines.append(f"file '{(src / cam / f'{rows[-1]['frame']:06d}.png').resolve()}'")
        dst.parent.mkdir(parents=True, exist_ok=True)
        listing = dst.with_suffix('.txt')
        listing.write_text('\n'.join(lines) + '\n')
        tmp = dst.with_suffix('.pending.mp4')
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0',
                        '-i', str(listing), '-vsync', 'vfr', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                        '-c:v', 'libx264', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                        str(tmp)], check=True)
        tmp.replace(dst)
    return dst


CSS = """
:root{--bg:#f6f6f4;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#deded8;--acc:#2f6fdd;--warn:#b54708;--ok:#1b7f3b}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ececea;--mut:#9a9a94;--card:#1f1f1d;--line:#34342f;--acc:#6b9cff;--warn:#f0a35c;--ok:#5cc785}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}
main{max-width:1400px;margin:0 auto;padding:16px}a{color:var(--acc)}h1{font-size:20px}h2{font-size:17px;margin:0 0 8px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:12px 0}
.mut{color:var(--mut)}.warn{color:var(--warn)}.ok{color:var(--ok)}
table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid var(--line);text-align:left}
.vids{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.vids video{width:100%;border-radius:6px;background:#000}
@media (max-width:800px){.vids{grid-template-columns:1fr}}
label{display:block;margin:8px 0 4px;font-weight:600}input,textarea,select{font:inherit;padding:6px;border:1px solid var(--line);
border-radius:6px;background:var(--bg);color:var(--fg)}textarea{width:100%;min-height:110px}
button{font:inherit;padding:8px 16px;border:0;border-radius:6px;background:var(--acc);color:#fff;cursor:pointer}
.ctl{display:flex;gap:8px;align-items:center;margin:8px 0;flex-wrap:wrap}code{background:var(--bg);padding:2px 5px;border-radius:4px}
"""


def page(title, body):
    return (f'<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(title)}</title><style>{CSS}</style><main>{body}</main>').encode()


def index():
    rows = []
    for m in sorted(ROOT.glob('*/manifest.json'), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            d = json.loads(m.read_text())
        except Exception:
            continue
        a = d['assignment']
        state = ('<span class=ok>submitted</span>' if d.get('submitted')
                 else '<span class=mut>discarded</span>' if d.get('discarded')
                 else '<span class=mut>superseded by a new eval</span>' if d.get('review') == 'superseded'
                 else '<span class=warn>scored, not submitted</span>' if 'result' in d else 'needs review')
        rec = sum(r['recorded'] for r in runs(d))
        rows.append(f"<tr><td><a href='/ev/{a['id']}'>{a['id']}</a></td>"
                    f"<td>{html.escape(a.get('task', {}).get('instruction', ''))}</td>"
                    f"<td>{rec}/{len(a.get('runs', []))}</td><td>{state}</td></tr>")
    body = ''.join(rows) or "<tr><td colspan=4 class=mut>No evals in this folder yet.</td></tr>"
    return page('Eval review', f"<h1>Eval review</h1><p class=mut>{html.escape(str(ROOT))}</p>"
                f"<div class=card><table><tr><th>Eval</th><th>Instruction</th><th>Runs recorded</th><th>State</th></tr>"
                f"{body}</table></div>")


def eval_page(ev, msg=''):
    d = load(ev)
    a = d['assignment']
    locked = d.get('submitted') or d.get('discarded') or d.get('review') == 'superseded' or busy(ev)
    res = d.get('result', {})
    outcomes = res.get('outcomes') or d.get('outcomes', {})
    dis = ' disabled' if locked else ''
    parts = [f"<p><a href='/'>← all evals</a></p><h1>{ev}</h1>"]
    if msg:
        parts.append(f"<div class=card>{msg}</div>")
    parts.append(f"<div class=card><b>Instruction:</b> {html.escape(a.get('task', {}).get('instruction', ''))}<br>"
                 f"<b>Scene:</b> {html.escape(str(a.get('scene', '')))} <span class=mut>(instruction and scene are fixed "
                 f"on the Router; not editable)</span></div>")
    if d.get('submitted'):
        parts.append("<div class='card warn'>Submitted. Ask the Router administrator to change it.</div>")
    elif d.get('discarded'):
        parts.append("<div class='card mut'>Discarded on the Router. Nothing was published.</div>")
    elif d.get('review') == 'superseded':
        parts.append("<div class='card mut'>A newer eval replaced this one on the Router. Local recordings remain available, but this eval can no longer be submitted.</div>")
    elif busy(ev):
        parts.append("<div class='card warn'>Talking to the Router… <a href=''>refresh</a></div>")
    parts.append("<form method=post>")
    for r in runs(d):
        side = r['side']
        o = outcomes.get(side, {})
        val = '' if o.get('partial_success') is None else round(o['partial_success'] * 100)
        if r['recorded']:
            vids = ''.join(f"<div><div class=mut>{c}</div><video data-side={side} src='/video/{ev}/{r['id']}/{c}' "
                           f"controls muted preload=metadata></video></div>" for c in CAMS
                           if (r['path'] / c).is_dir())
            ctl = (f"<div class=ctl><button type=button onclick=\"sync('{side}','play')\">▶ play all</button>"
                   f"<button type=button onclick=\"sync('{side}','pause')\">❚❚ pause</button>"
                   f"<button type=button onclick=\"sync('{side}','restart')\">⟲ restart</button>"
                   f"<label style='margin:0'>speed <select onchange=\"rate('{side}',this.value)\">"
                   f"<option>0.5<option selected>1<option>2<option>4</select></label></div>")
        else:
            vids, ctl = "<p class=mut>Not recorded yet (run it with --resume first).</p>", ''
        parts.append(f"<div class=card><h2>Policy {side} <span class=mut>{html.escape(r['id'])}</span></h2>{ctl}"
                     f"<div class=vids>{vids}</div><label>{side} partial success (0–100)</label>"
                     f"<input type=number min=0 max=100 step=1 name=score_{side} value='{val}'{dis}></div>")
    pref = res.get('preference') or d.get('draft', {}).get('preference')
    opts = ''.join(f"<option value='{v}'{' selected' if pref == v else ''}>{v or '—'}" for v in ('', 'A', 'B', 'tie'))
    parts.append(f"<div class=card><label>Preference</label><select name=preference{dis}>{opts}</select>"
                 f"<label>Feedback</label><textarea name=feedback{dis}>{html.escape(res.get('feedback', d.get('draft', {}).get('feedback', '')))}</textarea>"
                 + ('' if locked else "<div class=ctl><button name=action value=save>Save locally</button>"
                    "<button name=action value=submit style='background:#c4320a' onclick=\"return confirm("
                    "'Submit to the Router? Scores, feedback, videos and dataset become public and cannot be edited.')\">"
                    "Save + Submit to Router</button></div>") + "</div></form>")
    if not locked:
        parts.append("<form method=post class=card><b>Discard</b> <span class=mut>closes this eval on the Router; "
                     "nothing is published.</span><div class=ctl><input name=reason placeholder='reason (required)' "
                     "required style='flex:1'><button name=action value=discard onclick=\"return confirm("
                     "'Discard this eval? It cannot be submitted afterwards.')\">Discard</button></div></form>")
    log = ROOT / ev / 'review-router.log'
    if log.exists():
        parts.append(f"<div class=card><b>Router log</b><pre style='white-space:pre-wrap'>"
                     f"{html.escape(log.read_text()[-6000:])}</pre></div>")
    parts.append("""<script>
const V=s=>document.querySelectorAll(`video[data-side="${s}"]`);
function sync(s,a){V(s).forEach(v=>{if(a=='restart'){v.currentTime=0;v.play()}else v[a]()})}
function rate(s,r){V(s).forEach(v=>v.playbackRate=+r)}
</script>""")
    return page(ev, ''.join(parts))


def save(ev, form):
    path = ROOT / ev / 'manifest.json'
    d = json.loads(path.read_text())
    if d.get('submitted') or d.get('review') == 'superseded':
        return '<span class=warn>This eval is closed; nothing changed.</span>'
    outcomes = dict(d.get('outcomes', {}))
    for r in runs(d):
        raw = form.get(f"score_{r['side']}", [''])[0].strip()
        if raw:
            p = min(max(float(raw), 0), 100) / 100
            outcomes[r['side']] = {'success': p == 1.0, 'partial_success': p}
    d['outcomes'] = outcomes
    pref = form.get('preference', [''])[0]
    feedback = form.get('feedback', [''])[0].strip()
    sides = [r['side'] for r in runs(d)]
    if all(s in outcomes for s in sides) and pref in ('A', 'B', 'tie'):
        d['result'] = {'outcomes': outcomes, 'preference': pref, 'feedback': feedback}
        msg = '<span class=ok>Saved. Complete: the next --resume will submit this.</span>'
    else:
        d.pop('result', None)
        d['draft'] = {'preference': pref, 'feedback': feedback}
        msg = ('<span class=warn>Scores saved. Preference/feedback kept as a draft only (needs every score and a '
               'preference); --resume will prompt for them.</span>')
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(d, indent=2))
    tmp.replace(path)
    return msg


def busy(ev):
    job = JOBS.get(ev)
    return job is not None and job.poll() is None


def router(ev, action, reason=''):
    """Run the real client for this one eval; mark the manifest when it succeeds."""
    if load(ev).get('review') == 'superseded':
        return '<span class=warn>This eval was replaced by a newer one.</span>'
    if any(busy(e) for e in JOBS):
        return '<span class=warn>Another Router action is still running.</span>'
    flag = '--resume' if action == 'submit' else '--abort'
    stdin = 'n\n' if action == 'submit' else reason.replace('\n', ' ') + '\n'  # n = don't start a next eval
    log = (ROOT / ev / 'review-router.log').open('a')
    log.write(f'\n$ colosseum-robot {CONFIG} {flag} {ev}\n'); log.flush()
    job = subprocess.Popen([str(CLIENT / '.venv/bin/colosseum-robot'), str(CONFIG), flag, ev], cwd=CLIENT,
                           stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, text=True)
    job.stdin.write(stdin); job.stdin.close()
    JOBS[ev] = job

    def finish():
        code = job.wait()
        path = ROOT / ev / 'manifest.json'
        with LOCK:
            d = json.loads(path.read_text())
            if action == 'discard' and code == 0:
                d['discarded'] = True
            d.pop('review', None) if (d.get('submitted') or d.get('discarded')) else None
            tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(d, indent=2)); tmp.replace(path)
        log.write(f'[exit {code}]\n'); log.close()
    threading.Thread(target=finish, daemon=True).start()
    return '<span class=warn>Sent to the Router. Refresh in a few seconds to see the log.</span>'


class H(BaseHTTPRequestHandler):
    def send(self, code, body, ctype='text/html; charset=utf-8'):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def ev_ok(self, ev):
        return ev and '/' not in ev and '..' not in ev and (ROOT / ev / 'manifest.json').exists()

    def do_GET(self):
        p = [unquote(x) for x in urlparse(self.path).path.strip('/').split('/') if x]
        try:
            if not p:
                return self.send(200, index())
            if p[0] == 'ev' and len(p) == 2 and self.ev_ok(p[1]):
                return self.send(200, eval_page(p[1]))
            if p[0] == 'video' and len(p) == 4 and self.ev_ok(p[1]) and p[3] in CAMS and p[2] in {
                    r['id'] for r in load(p[1])['assignment'].get('runs', [])}:
                f = preview(*p[1:])
                if f:
                    return self.file(f)
            self.send(404, b'not found', 'text/plain')
        except Exception as e:
            self.send(500, page('Error', f'<pre>{html.escape(repr(e))}</pre>'))

    def file(self, f):  # with Range support so the browser can seek
        size = f.stat().st_size
        rng = self.headers.get('Range', '')
        start, end = 0, size - 1
        if rng.startswith('bytes='):
            a, _, b = rng[6:].partition('-')
            start = int(a) if a else size - int(b)
            end = int(b) if a and b else size - 1
        self.send_response(206 if rng else 200)
        self.send_header('Content-Type', 'video/mp4')
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(end - start + 1))
        if rng:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        with open(f, 'rb') as fh:
            fh.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = fh.read(min(1 << 20, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)

    def do_POST(self):
        p = [unquote(x) for x in urlparse(self.path).path.strip('/').split('/') if x]
        if len(p) != 2 or p[0] != 'ev' or not self.ev_ok(p[1]):
            return self.send(404, b'not found', 'text/plain')
        form = parse_qs(self.rfile.read(int(self.headers.get('Content-Length', 0))).decode(), keep_blank_values=True)
        action = form.get('action', ['save'])[0]
        try:
            if busy(p[1]):
                msg = '<span class=warn>Busy; wait for the Router action to finish.</span>'
            elif action == 'discard':
                reason = form.get('reason', [''])[0].strip()
                msg = router(p[1], 'discard', reason) if reason else '<span class=warn>A reason is required.</span>'
            else:
                msg = save(p[1], form)
                if action == 'submit':
                    msg = (router(p[1], 'submit') if 'result' in load(p[1])
                           else '<span class=warn>Not submitted: every score and a preference are required.</span>')
        except ValueError as e:
            msg = f'<span class=warn>Not saved: {html.escape(str(e))}</span>'
        self.send(200, eval_page(p[1], msg))

    def log_message(self, *a):
        pass


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, help='robot config used for Submit/Discard')
    ap.add_argument('--root', help='eval folder (default: evaluation_dir from the config)')
    ap.add_argument('--port', type=int, default=8765)
    args = ap.parse_args()
    CONFIG = Path(args.config).resolve()
    ROOT = Path(args.root or (CLIENT / yaml.safe_load(CONFIG.read_text()).get('evaluation_dir', 'eval_runs'))).resolve()
    CACHE = ROOT.parent / 'eval_review_cache'
    print(f'Reviewing {ROOT}\nOpen http://127.0.0.1:{args.port}  (local only; Ctrl-C to stop)')
    ThreadingHTTPServer(('127.0.0.1', args.port), H).serve_forever()
