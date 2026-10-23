"""Drive commands: RTM 1.x peer messages to the rover through Agora's Signaling REST API, sent as the ride user.

The rover accepts ``{"linear", "angular", "lamp"}`` JSON peer messages from the
ride's user id and keeps executing the last one until a new one arrives. The
REST path needs no RTM session: the ride's RTM token authenticates the sender,
and ``wait_for_ack`` reports the rover's receipt, which is how a stop is confirmed.
"""
from __future__ import annotations

import json
import time

import httpx

REST_BASE = 'https://api.agora.io'
SEND_TIMEOUT_S = 1.0  # Measured round trips: ~160 ms median, 700 ms worst case.
ACK_TIMEOUT_S = 4.0
STOP_BUDGET_S = 10.0


def command_payload(linear, angular, lamp=0):
    clip = lambda value: round(max(-1.0, min(1.0, float(value))), 3)
    return json.dumps({'linear': clip(linear), 'angular': clip(angular), 'lamp': int(lamp)}, separators=(',', ':'))


class RtmRestSender:
    def __init__(self, credentials, *, base_url=REST_BASE):
        self.destination = credentials.bot_uid
        self.path = f'/dev/v2/project/{credentials.app_id}/rtm/users/{credentials.user_id}/peer_messages'
        self.http = httpx.Client(base_url=base_url, trust_env=False, timeout=SEND_TIMEOUT_S, headers={
            'x-agora-uid': str(credentials.user_id), 'x-agora-token': credentials.rtm_token,
            'Content-Type': 'application/json;charset=utf-8'})
        self.lamp = 0
        self.sent = 0
        self.last_error = ''

    def _post(self, payload, *, wait_for_ack, timeout):
        body = {'destination': self.destination, 'enable_offline_messaging': False,
                'enable_historical_messaging': False, 'payload': payload}
        response = self.http.post(self.path, params={'wait_for_ack': 'true' if wait_for_ack else 'false'},
                                  json=body, timeout=timeout)
        if response.is_error:
            # Never echo the response body: it can describe the token.
            raise RuntimeError(f'Agora RTM REST send failed ({response.status_code})')
        return response.json()

    def send(self, linear, angular):
        self._post(command_payload(linear, angular, self.lamp), wait_for_ack=False, timeout=SEND_TIMEOUT_S)
        self.sent += 1

    def stop(self, confirm=True):
        """Send zeros; with ``confirm`` retry until the rover acknowledges or the budget runs out."""
        deadline = time.monotonic() + STOP_BUDGET_S
        delay = 1.0
        while True:
            try:
                result = self._post(command_payload(0.0, 0.0, self.lamp), wait_for_ack=confirm,
                                    timeout=ACK_TIMEOUT_S if confirm else SEND_TIMEOUT_S)
                self.sent += 1
                if not confirm or result.get('code') == 'message_delivered':
                    return True
                self.last_error = str(result.get('code', 'no acknowledgement'))
            except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                self.last_error = type(exc).__name__
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(delay, remaining))
            delay = min(delay * 1.5, 5.0)

    def close(self):
        self.http.close()
