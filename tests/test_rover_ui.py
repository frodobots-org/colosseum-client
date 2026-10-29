"""The local operator page runs the terminal flow behind forms."""
import json
import threading
import time
import urllib.request

import pytest

from colosseum_client import evaluation as e
from colosseum_client import rover_ui
from colosseum_client.navigation import ROVER_TYPE
from colosseum_client.robot_config import RobotClientConfig


def config(tmp_path, **settings):
    return RobotClientConfig(url='ws://localhost', token='t', cameras={'front_image': 'front'}, robot_type=ROVER_TYPE,
                             control_hz=10, evaluation_dir=str(tmp_path), adapter_config={'link': 'simulated', **settings})


@pytest.fixture
def page(tmp_path):
    server, operator = rover_ui.serve(config(tmp_path), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_address[1]}'
    def call(path, body=None):
        request = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'}, method='POST' if body is not None else 'GET')
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()
    yield operator, call
    operator.release_manual()
    server.shutdown()
    server.server_close()
    e.ask = lambda prompt, options=None: input(prompt)
    e.INTERRUPT.clear()
    from colosseum_client.robots.earth_rover import use_shared_link
    use_shared_link(None)


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def state(call):
    return json.loads(call('/api/state')[1])


def connect(call, bot='frodobot_x'):
    assert json.loads(call('/api/connect', {'bot': bot})[1])['ok']
    assert wait_until(lambda: state(call)['link_phase'] == 'connected')


def test_prompts_become_forms_and_answers_flow_back(page, monkeypatch):
    operator, call = page
    monkeypatch.setattr(operator, 'router_ok', lambda: True)
    seen = {}
    def fake_run(cfg, **kwargs):
        seen['bot'] = cfg.adapter_config.get('bot_slug')
        seen['mission'] = cfg.adapter_config.get('mission_slug')
        seen['track'] = kwargs.get('track')
        print('Missions on the bot:')
        seen['choice'] = e.ask('Select mission number: ', options=[('1', 'courtyard-loop'), ('2', 'plaza')])
        seen['free'] = e.ask('Feedback (optional): ')
        print('Evaluation result submitted.')
    monkeypatch.setattr(e, 'run_evaluation', fake_run)
    status, body = call('/api/start', {'bot': 'frodobot_x', 'mission': 'plaza', 'checkpoint': '', 'track': 'open', 'execute': False})
    assert status == 200 and json.loads(body)['ok']
    assert wait_until(lambda: state(call)['pending'] is not None)
    current = state(call)
    assert current['running'] and current['pending']['prompt'] == 'Select mission number:'
    assert current['pending']['options'] == [{'value': '1', 'label': 'courtyard-loop'}, {'value': '2', 'label': 'plaza'}]
    assert 'Missions on the bot:' in current['log']
    assert call('/api/start', {'bot': 'x'})[0] == 409  # one session at a time
    assert json.loads(call('/api/answer', {'value': '2'})[1])['ok']
    assert wait_until(lambda: (state(call)['pending'] or {}).get('prompt') == 'Feedback (optional):')
    assert state(call)['pending']['options'] is None
    call('/api/answer', {'value': 'smooth run'})
    assert wait_until(lambda: state(call)['phase'] == 'finished')
    assert seen == {'bot': 'frodobot_x', 'mission': 'plaza', 'track': 'open', 'choice': '2', 'free': 'smooth run'}
    assert json.loads(call('/api/answer', {'value': 'late'})[1])['ok'] is False


def test_stop_abort_frame_and_results(page, monkeypatch, tmp_path):
    operator, call = page
    monkeypatch.setattr(operator, 'router_ok', lambda: True)
    def fake_run(cfg, **kwargs):
        e.ask('press Enter to start', options=[('', 'Start trial')])
    monkeypatch.setattr(e, 'run_evaluation', fake_run)
    call('/api/start', {'bot': 'frodobot_x'})
    assert wait_until(lambda: state(call)['pending'] is not None)
    assert json.loads(call('/api/abort', {})[1])['ok'] and e.INTERRUPT.is_set()
    e.INTERRUPT.clear()
    call('/api/stop', {})
    assert wait_until(lambda: state(call)['phase'] == 'stopped')
    assert call('/frame.jpg')[0] == 404
    status, body = call('/')
    assert status == 200 and b'Earth Rover operator' in body
    manifest = tmp_path / 'FT-1' / 'manifest.json'
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({'assignment': {'id': 'FT-1', 'track': 'fine-tuning', 'task': {'instruction': 'Go', 'goal': {'mission_slug': 'plaza'}}},
                                    'outcomes': {'trial': {'success': True, 'partial_success': 1.0}}, 'submitted': True,
                                    'datasets_uploaded': {'run_1': True}}))
    results = state(call)['results']
    assert results[0]['id'] == 'FT-1' and results[0]['submitted'] and results[0]['uploaded'] == ['run_1']


def test_missions_endpoint_uses_the_backend_session(page, monkeypatch):
    operator, call = page
    class Session:
        def missions(self): return [{'slug': 'loop', 'type': 'checkpoints'}]
        def checkpoints(self, slug): return [{'sequence': 1, 'latitude': '1', 'longitude': '2'}]
        def close(self): pass
    monkeypatch.setattr(e, 'rover_session', lambda cfg: Session() if cfg.adapter_config['bot_slug'] == 'frodobot_x' else None)
    assert json.loads(call('/api/missions?bot=frodobot_x')[1])['missions'][0]['checkpoints'][0]['sequence'] == 1
    assert 'note' in json.loads(call('/api/missions?bot=other')[1])
    assert 'error' in json.loads(call('/api/missions')[1])


def test_terminal_prompts_still_read_stdin(monkeypatch):
    monkeypatch.setattr('builtins.input', lambda prompt: 'y')
    assert e.yes_no('Continue?') is True
    monkeypatch.setattr('builtins.input', lambda prompt: '3')
    assert e.rover_score('trial') == {'success': False, 'partial_success': 0.75}


def test_connect_first_then_the_trial_rides_on_the_shared_link(page, monkeypatch):
    operator, call = page
    monkeypatch.setattr(operator, 'router_ok', lambda: True)
    assert call('/api/connect', {'bot': ''})[0] == 409
    connect(call)
    current = state(call)
    assert current['connected'] and current['mission'] is None and current['telemetry']['battery'] == 100
    status, body = call('/frame.jpg')
    assert status == 200 and body[:2] == b'\xff\xd8'
    assert call('/api/connect', {'bot': 'frodobot_x'})[0] == 409  # already connected
    assert call('/api/start', {'bot': 'other'})[0] == 409  # another rover needs a disconnect first
    preview = operator.link
    seen = {}
    def fake_run(cfg, **kwargs):
        from colosseum_client.robots import earth_rover_mini_plus as rover
        robot = rover.EarthRoverMiniPlusRobot(cfg)
        robot.prepare_for_task({'goal': {'type': 'gps', 'latitude': 1.0, 'longitude': 2.0, 'mission_slug': 'plaza'}})
        seen['same_link'] = robot.link is preview
        seen['mission'] = preview.mission_slug
        robot.get_observation()
        robot.close()
        seen['after'] = (preview.started, preview.mission_slug, preview.stops)
    monkeypatch.setattr(e, 'run_evaluation', fake_run)
    assert json.loads(call('/api/start', {'bot': 'frodobot_x', 'mission': 'plaza'})[1])['ok']
    assert wait_until(lambda: state(call)['phase'] == 'finished')
    assert seen['same_link'] and seen['mission'] == 'plaza'
    assert seen['after'][0] is True and seen['after'][1] is None and seen['after'][2] >= 1
    assert state(call)['connected']  # the preview survives the trial
    assert json.loads(call('/api/disconnect', {})[1])['ok']
    assert preview.closed and state(call)['link_phase'] == 'disconnected'


def test_start_explains_an_unreachable_router(page, monkeypatch):
    operator, call = page
    monkeypatch.setattr(e, 'run_evaluation', lambda cfg, **kwargs: None)
    status, body = call('/api/start', {'bot': 'frodobot_x'})
    assert status == 409 and 'not reachable' in json.loads(body)['error'] and 'start-router.sh' in json.loads(body)['error']


def test_manual_drive_streams_while_held_and_sends_one_zero_on_release(page):
    operator, call = page
    assert call('/api/drive', {'linear': 0.5, 'angular': 0})[0] == 409  # not connected
    assert call('/api/halt', {})[0] == 409
    connect(call)
    link = operator.link
    current = state(call)
    assert current['manual'] and not current['trial'] and current['lamp'] == 0 and current['drive'] == [0, 0]
    assert call('/api/drive', {'linear': 2, 'angular': 0})[0] == 409
    assert call('/api/drive', {'linear': 'fast', 'angular': 0})[0] == 409
    for _ in range(3):  # the page re-sends the held command at 10 Hz
        assert json.loads(call('/api/drive', {'linear': 0.5, 'angular': -0.2})[1])['ok']
        time.sleep(0.1)
    assert wait_until(lambda: sum(1 for sent in link.sent if sent[:2] == (0.5, -0.2)) >= 3)
    assert {sent[2] for sent in link.sent} == {operator.manual.ident}  # a single sender thread
    assert state(call)['drive'] == [0.5, -0.2]
    assert json.loads(call('/api/drive', {'linear': 0, 'angular': 0})[1])['ok']
    assert wait_until(lambda: link.sent and link.sent[-1][:2] == (0.0, 0.0))
    count = len(link.sent)
    time.sleep(0.35)
    assert len(link.sent) == count  # exactly one zero, then silence
    data = json.loads(call('/api/lamp', {'on': True})[1])
    assert data['lamp'] == 1 and link.lamp == 1 and state(call)['lamp'] == 1
    assert wait_until(lambda: len(link.sent) == count + 1)  # one command carries the lamp bit
    assert json.loads(call('/api/halt', {})[1]) == {'ok': True, 'aborting': False}
    assert link.stops == 1 and link.sent[-1][:2] == (0.0, 0.0)


def test_manual_drive_watchdog_zeros_when_the_page_goes_silent(page, monkeypatch):
    operator, call = page
    monkeypatch.setattr(rover_ui, 'MANUAL_TTL_S', 0.2)
    connect(call)
    link = operator.link
    assert json.loads(call('/api/drive', {'linear': 0.4, 'angular': 0})[1])['ok']
    assert wait_until(lambda: link.sent and link.sent[-1][:2] == (0.4, 0.0))
    assert wait_until(lambda: link.sent[-1][:2] == (0.0, 0.0), timeout=1.5)
    assert operator.manual.expiries == 1
    count = len(link.sent)
    time.sleep(0.3)
    assert len(link.sent) == count


def test_a_trial_claims_the_link_and_manual_drive_resumes_after_it(page):
    operator, call = page
    connect(call)
    link = operator.link
    assert json.loads(call('/api/drive', {'linear': 0.3, 'angular': 0.1})[1])['ok']
    assert wait_until(lambda: link.sent and link.sent[-1][:2] == (0.3, 0.1))
    manual = operator.manual
    from colosseum_client.robots import earth_rover_mini_plus as rover
    robot = rover.EarthRoverMiniPlusRobot(operator.config)  # claims the shared link: manual driving ends with a zero
    assert not manual.is_alive() and link.sent[-1][:2] == (0.0, 0.0)
    assert call('/api/drive', {'linear': 0.3, 'angular': 0})[0] == 409
    assert call('/api/lamp', {'on': True})[0] == 409
    current = state(call)
    assert current['trial'] and not current['manual'] and current['goal'] is None
    robot.prepare_for_task({'goal': {'type': 'gps', 'latitude': -34.9, 'longitude': -57.9}})
    assert state(call)['goal'] == {'latitude': -34.9, 'longitude': -57.9}
    assert json.loads(call('/api/halt', {})[1]) == {'ok': True, 'aborting': True} and e.INTERRUPT.is_set()
    robot.close()
    current = state(call)
    assert current['manual'] and not current['trial'] and current['goal'] is None
    assert json.loads(call('/api/drive', {'linear': 0.2, 'angular': 0})[1])['ok']
    assert wait_until(lambda: link.sent[-1][:2] == (0.2, 0.0))
    assert json.loads(call('/api/disconnect', {})[1])['ok']
    assert link.sent[-1][:2] == (0.0, 0.0) and not operator.manual


def test_static_files_are_served_from_the_package(page):
    _, call = page
    status, body = call('/static/leaflet/leaflet.js')
    assert status == 200 and b'Leaflet 1.9.4' in body
    assert call('/static/leaflet/leaflet.css')[0] == 200
    assert call('/static/leaflet/images/marker-icon.png')[0] == 200
    assert call('/static/operator.js')[0] == 200
    assert call('/static/vendor/AgoraRTC_N-4.24.2.js')[0] == 200  # the SDK's Agora web client, shared with the link
    assert call('/static/vendor/rover.html')[0] == 200
    assert call('/static/vendor/../browser.py')[0] == 404
    assert call('/static/../rover_ui.py')[0] == 404
    assert call('/static/')[0] == 404
    status, body = call('/')
    assert b'/static/operator.js' in body and b'id="map"' in body and b'data-dir="forward"' in body
    assert b'id="front-player"' in body and b'AgoraRTC_N-4.24.2.js' in body


def test_state_carries_the_spectator_identity_for_the_page_to_play_video(page):
    operator, call = page
    connect(call)
    assert state(call)['video'] is None  # the simulator issues no spectator token: the page falls back to snapshots
    operator.link.spectator = lambda: {'appid': 'app', 'channel': 'ride_bot_x', 'uid': 421, 'rtcToken': 'rtc-spectator'}
    assert state(call)['video'] == {'appid': 'app', 'channel': 'ride_bot_x', 'uid': 421, 'rtcToken': 'rtc-spectator'}


def test_diagnostics_and_events_reach_the_page(page):
    operator, call = page
    assert state(call)['diagnostics'] is None
    connect(call)
    s = state(call)
    assert s['link_live'] and s['diagnostics']['transport'] == 'simulated'
    assert s['diagnostics']['rtm']['publishers'] == [1000, 1001] and 'front_image' in s['diagnostics']['frames']
    assert any('Connected to frodobot_x' in line for line in s['events'])
    status, body = call('/api/diagnostics')
    dump = json.loads(body)
    assert status == 200 and dump['bot'] == 'frodobot_x' and dump['diagnostics']['started'] and dump['events']
    call('/api/disconnect', {})
    assert any(line.endswith('Disconnected from the rover') for line in state(call)['events'])
    assert state(call)['diagnostics'] is None
