"""Rover cameras: subscribe to the ride's Agora RTC channel with the Agora Python Server SDK.

The rover publishes each camera as its own RTC user: uid 1000 is the front
camera, 1001 the rear. Decoded I420 frames arrive on the SDK's callback thread
and are converted to RGB here; the adapter reads the latest frame per camera.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from ...navigation import FRONT_CAMERA, REAR_CAMERA

CAMERA_UIDS = {'1000': FRONT_CAMERA, '1001': REAR_CAMERA}


def yuv420_to_rgb(frame) -> np.ndarray:
    """I420 planes with strides → RGB uint8 (height, width, 3), BT.601 limited range."""
    height, width = int(frame.height), int(frame.width)
    half_height, half_width = (height + 1) // 2, (width + 1) // 2

    def plane(buffer, stride, rows, columns):
        data = np.frombuffer(bytes(buffer), dtype=np.uint8)[: stride * rows]
        return data.reshape(rows, stride)[:, :columns].astype(np.float32)

    luma = (plane(frame.y_buffer, int(frame.y_stride), height, width) - 16.0) * 1.164
    chroma_u = plane(frame.u_buffer, int(frame.u_stride), half_height, half_width) - 128.0
    chroma_v = plane(frame.v_buffer, int(frame.v_stride), half_height, half_width) - 128.0
    chroma_u = np.repeat(np.repeat(chroma_u, 2, axis=0), 2, axis=1)[:height, :width]
    chroma_v = np.repeat(np.repeat(chroma_v, 2, axis=0), 2, axis=1)[:height, :width]
    rgb = np.stack([luma + 1.596 * chroma_v, luma - 0.392 * chroma_u - 0.813 * chroma_v, luma + 2.017 * chroma_u], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


class FrameStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._frames = {}
        self.counts = {}

    def put(self, camera, rgb):
        with self._lock:
            self._frames[camera] = (rgb, time.monotonic())
            self.counts[camera] = self.counts.get(camera, 0) + 1

    def latest(self):
        with self._lock:
            return dict(self._frames)

    def clear(self):
        """A new channel starts empty: the previous page's last frame must not pass as a fresh one."""
        with self._lock:
            self._frames.clear()
            self.counts.clear()


class AgoraVideo:
    def __init__(self, credentials, cameras, *, log_path='agora_rtc.log'):
        self.credentials = credentials
        self.cameras = set(cameras)
        self.store = FrameStore()
        self.log_path = log_path
        self.service = None
        self.connection = None
        self.decode_errors = 0

    def start(self):
        try:
            from agora.rtc.agora_base import ChannelProfileType, ClientRoleType
            from agora.rtc.agora_service import AgoraService, AgoraServiceConfig, RTCConnConfig
            from agora.rtc.video_frame_observer import IVideoFrameObserver
        except ImportError as exc:
            raise RuntimeError("The Earth Rover link needs the 'earth-rover' extra: uv sync --extra earth-rover") from exc
        video = self

        class Observer(IVideoFrameObserver):
            def on_frame(self, channel_id, remote_uid, frame):
                camera = CAMERA_UIDS.get(str(remote_uid))
                if camera in video.cameras:
                    try:
                        video.store.put(camera, yuv420_to_rgb(frame))
                    except Exception:
                        video.decode_errors += 1  # One bad frame must not take down the SDK callback thread.
                return 1

        service_config = AgoraServiceConfig()
        service_config.appid = self.credentials.app_id
        service_config.log_path = self.log_path
        if hasattr(service_config, 'enable_video'):
            service_config.enable_video = 1
        self.service = AgoraService()
        self.service.initialize(service_config)
        connection_config = RTCConnConfig(
            auto_subscribe_audio=0, auto_subscribe_video=1,
            client_role_type=ClientRoleType.CLIENT_ROLE_AUDIENCE,
            channel_profile=ChannelProfileType.CHANNEL_PROFILE_LIVE_BROADCASTING)
        publish_config = None
        try:  # Subscriber only; the SDK's publish settings vary by version.
            from agora.rtc.agora_base import RtcConnectionPublishConfig
            publish_config = RtcConnectionPublishConfig(is_publish_audio=False, is_publish_video=False)
        except (ImportError, TypeError):
            pass
        self.connection = self.service.create_rtc_connection(connection_config, publish_config)
        self._observer = Observer()
        self.connection.register_video_frame_observer(self._observer)
        self.connection.connect(self.credentials.rtc_token, self.credentials.channel, str(self.credentials.user_id))

    def latest(self):
        return self.store.latest()

    def wait_for(self, camera, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if camera in self.store.latest():
                return True
            time.sleep(0.1)
        return camera in self.store.latest()

    def close(self):
        connection, self.connection = self.connection, None
        service, self.service = self.service, None
        for call in ((connection.disconnect, connection.release) if connection else ()) + ((service.release,) if service else ()):
            try:
                call()
            except Exception:
                pass
