"""FrodoBots ride session: the backend locks the rover and issues the Agora credentials.

These are frodobots_web's existing programmatic endpoints, called directly with an
organizer ``sdk_access_token``, exactly as the Earth Rovers SDK does: with a mission,
``/sdk/start_ride`` opens a recorded ride; without one, ``/sdk/token`` grants the
bot and its credentials (no ride row, no platform recording). ``/sdk/end_ride``
releases a ride.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import httpx

DEFAULT_API_URL = 'https://frodobots-web-api.onrender.com/api/v1'
TOKEN_LIFETIME_S = 6 * 3600  # Backend privilege expiry; re-run start_ride before it passes.
REQUIRED_KEYS = ('APP_ID', 'CHANNEL_NAME', 'USERID', 'RTC_TOKEN', 'RTM_TOKEN', 'BOT_UID')


@dataclass(frozen=True)
class RideCredentials:
    app_id: str
    channel: str
    user_id: int
    rtc_token: str
    rtm_token: str
    bot_uid: str      # The rover's device reference; RTM peer for commands and telemetry.
    bot_type: str
    obtained_at: float
    # A second RTC identity for watching the cameras alongside the link, the way the
    # SDK dashboard does (its page joins as the spectator while the headless page keeps USERID).
    spectator_user_id: int | None = None
    spectator_rtc_token: str = ''

    def expires_in(self):
        return TOKEN_LIFETIME_S - (time.monotonic() - self.obtained_at)

    def spectator(self):
        """Agora RTC join parameters for a viewer, or None when the backend issued no spectator token."""
        if self.spectator_user_id is None or not self.spectator_rtc_token:
            return None
        return {'appid': self.app_id, 'channel': self.channel, 'uid': self.spectator_user_id, 'rtcToken': self.spectator_rtc_token}


class FrodobotsSession:
    def __init__(self, token, bot_slug, mission_slug='', *, api_url=DEFAULT_API_URL, timeout=30, transport=None):
        if not token or not bot_slug:
            raise ValueError('sdk_token and bot_slug are required')
        self.bot_slug = bot_slug
        self.mission_slug = mission_slug or ''
        self.http = httpx.Client(base_url=api_url.rstrip('/'), timeout=timeout, trust_env=False,
                                 headers={'Authorization': f'Bearer {token}'}, transport=transport)
        self.credentials = None

    def _request(self, method, path, **kwargs):
        response = self.http.request(method, path, **kwargs)
        if response.is_error:
            detail = ''
            try:
                body = response.json()
                detail = body.get('error') or body.get('message') or '' if isinstance(body, dict) else ''
            except ValueError:
                pass
            raise RuntimeError(f'FrodoBots {path} failed ({response.status_code})' + (f': {detail}' if detail else ''))
        return response.json()

    def missions(self):
        """The bot's SDK-enabled active missions, as the SDK's /missions lists them."""
        data = self._request('GET', '/sdk/missions', params={'bot_slug': self.bot_slug})
        return list(data.get('missions') or [])

    def checkpoints(self, mission_slug):
        """Ordered checkpoints of an SDK-enabled mission: id, sequence, latitude, longitude."""
        data = self._request('POST', '/sdk/checkpoints_list', json={'bot_slug': self.bot_slug, 'mission_slug': mission_slug})
        return sorted((dict(item) for item in data.get('checkpoints_list') or []), key=lambda item: int(item.get('sequence', 0)))

    def start_ride(self, mission_slug=None):
        """Acquire the rover for this account; a repeated call rejoins the same active ride.

        Without a mission this is the SDK's token-only session: the backend still
        checks that nobody else holds the bot and points the rover at our channel.
        """
        if mission_slug is not None:
            self.mission_slug = mission_slug or ''
        if self.mission_slug:
            data = self._request('POST', '/sdk/start_ride', json={'bot_slug': self.bot_slug, 'mission_slug': self.mission_slug})
        else:
            data = self._request('POST', '/sdk/token', json={'bot_slug': self.bot_slug})
        missing = [key for key in REQUIRED_KEYS if not data.get(key)]
        if missing:
            raise RuntimeError('FrodoBots start_ride response lacks ' + ', '.join(missing))
        spectator = data.get('SPECTATOR_USERID')
        self.credentials = RideCredentials(
            app_id=str(data['APP_ID']), channel=str(data['CHANNEL_NAME']), user_id=int(data['USERID']),
            rtc_token=str(data['RTC_TOKEN']), rtm_token=str(data['RTM_TOKEN']), bot_uid=str(data['BOT_UID']),
            bot_type=str(data.get('BOT_TYPE', '')), obtained_at=time.monotonic(),
            spectator_user_id=int(spectator) if spectator not in (None, '') else None,
            spectator_rtc_token=str(data.get('SPECTATOR_RTC_TOKEN') or ''))
        return self.credentials

    def end_ride(self):
        """Release the rover. A missing active ride (404) counts as already ended; token-only sessions have no ride."""
        if not self.mission_slug:
            self.credentials = None
            return
        try:
            response = self.http.post('/sdk/end_ride', json={'bot_slug': self.bot_slug, 'mission_slug': self.mission_slug})
            if response.is_error and response.status_code != 404:
                raise RuntimeError(f'FrodoBots /sdk/end_ride failed ({response.status_code})')
        finally:
            self.credentials = None

    def shadow_gps(self):
        """The rover's last reported position from the backend's device shadow (low rate, no heading)."""
        data = self._request('GET', '/sdk/data', params={'bot_slug': self.bot_slug})
        return {'latitude': float(data['latitude']), 'longitude': float(data['longitude'])}

    def close(self):
        self.http.close()
