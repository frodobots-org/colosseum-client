"""Phase 0 link check: exercise the production rover link (FrodoBots ride + the embedded SDK browser page) without the Router.

    FRODOBOTS_SDK_TOKEN=... python scripts/earth_rover_link_check.py --bot-slug BOT [--mission-slug MISSION]
        [--transport browser|native] [--drive] [--seconds 20]

Without --mission-slug it uses the SDK's token-only session (no ride, no platform recording).
Without --drive it only observes (frames, telemetry) and sends zeros plus a confirmed stop.
With --drive it sends a slow forward command for one second, then a slow left turn, so the
angular sign and the command scale can be calibrated by eye (operator present).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

from colosseum_client.navigation import FRONT_CAMERA, REAR_CAMERA
from colosseum_client.robots.earth_rover.frodobots_link import FrodobotsRoverLink
from colosseum_client.robots.earth_rover.session import DEFAULT_API_URL


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bot-slug', required=True)
    parser.add_argument('--mission-slug', default='', help='omit for a token-only session (no ride, no platform recording)')
    parser.add_argument('--api-url', default=DEFAULT_API_URL)
    parser.add_argument('--transport', choices=['browser', 'native'], default='browser')
    parser.add_argument('--seconds', type=float, default=20, help='observation window')
    parser.add_argument('--drive', action='store_true', help='send a short forward then left command (operator present)')
    parser.add_argument('--report', default='earth-rover-link-check.json')
    args = parser.parse_args()
    token = os.environ.get('FRODOBOTS_SDK_TOKEN')
    if not token:
        sys.exit('Set FRODOBOTS_SDK_TOKEN to the organizer sdk_access_token')
    report = {'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'transport': args.transport,
              'bot_slug': args.bot_slug, 'mission_slug': args.mission_slug}
    settings = {'sdk_token': token, 'bot_slug': args.bot_slug, 'mission_slug': args.mission_slug, 'frodobots_api_url': args.api_url,
                'transport': args.transport, 'first_frame_timeout_s': 20}
    link = FrodobotsRoverLink(settings, [FRONT_CAMERA, REAR_CAMERA], (1024, 576))
    try:
        started = time.monotonic()
        try:
            link.start(args.mission_slug or None)
            report['connect_s'] = round(time.monotonic() - started, 2)
        except Exception as exc:
            report['connect_error'] = f'{type(exc).__name__}: {exc}'[:400]
            report['health'] = link.health()
            report['errors'] = list(getattr(link.browser, 'errors', []))[-10:] if link.browser else []
            print(json.dumps(report, indent=2, default=str), flush=True)
            raise
        credentials = link.credentials
        report['ride'] = {'channel': credentials.channel, 'user_id': credentials.user_id, 'bot_uid': credentials.bot_uid,
                          'bot_type': credentials.bot_type, 'browser': getattr(link.browser, 'browser_name', '')}
        print(f"Connected in {report['connect_s']}s: channel={credentials.channel} user={credentials.user_id} bot={credentials.bot_uid}", flush=True)

        ages = {FRONT_CAMERA: [], REAR_CAMERA: []}
        stamps = []
        last = None
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            now = time.monotonic()
            for camera, (_, received) in link.latest_frames().items():
                ages[camera].append(now - received)
            latest = link.latest_telemetry()
            if latest is not None and latest[1] != last:
                last = latest[1]
                stamps.append(last)
            time.sleep(0.2)
        counts = link.browser.frames.counts if link.browser else (link.video.store.counts if link.video else {})
        report['video'] = {camera: {'frames': counts.get(camera, 0), 'median_age_s': statistics.median(values) if values else None,
                                    'max_age_s': max(values) if values else None} for camera, values in ages.items()}
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        latest = link.latest_telemetry()
        report['telemetry'] = {'reports': len(stamps), 'median_interval_s': statistics.median(gaps) if gaps else None,
                               'last_values': None if latest is None else latest[0]}
        report['health'] = link.health()
        print(json.dumps({'video': report['video'], 'telemetry': report['telemetry'], 'health': report['health']}, indent=2, default=str), flush=True)

        commands = []
        if args.drive:
            commands += [(0.2, 0.0)] * 10 + [(0.0, 0.0)] * 5 + [(0.0, 0.3)] * 10
            print('Driving: forward for 1 s, pause, then a left turn for 1 s. Watch the rover.', flush=True)
        commands += [(0.0, 0.0)] * 10
        latencies = []
        for linear, angular in commands:
            tick = time.monotonic()
            try:
                link.send(linear, angular)
                latencies.append(time.monotonic() - tick)
            except Exception as exc:
                latencies.append(None)
                print(f'send failed: {type(exc).__name__}', flush=True)
            time.sleep(max(0.0, 0.1 - (time.monotonic() - tick)))
        measured = [value for value in latencies if value is not None]
        tick = time.monotonic()
        confirmed = link.stop(confirm=True)
        report['commands'] = {'sent': len(commands), 'failed': latencies.count(None),
                              'median_dispatch_s': statistics.median(measured) if measured else None,
                              'max_dispatch_s': max(measured) if measured else None,
                              'stop_confirmed': confirmed, 'stop_receipt_s': round(time.monotonic() - tick, 3),
                              'rtm': link.health()}
        print(json.dumps(report['commands'], indent=2, default=str), flush=True)
        report['errors'] = list(getattr(link.browser, 'errors', []))[-10:] if link.browser else []
    finally:
        try:
            link.close()
            report['closed'] = True
        except Exception as exc:
            report['closed'] = False
            report['close_error'] = f'{type(exc).__name__}: {exc}'[:300]
            print(report['close_error'], flush=True)
        with open(args.report, 'w', encoding='utf-8') as output:
            json.dump(report, output, indent=2, default=str)
        print(f'Report written to {args.report}', flush=True)


if __name__ == '__main__':
    main()
