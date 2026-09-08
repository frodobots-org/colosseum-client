from __future__ import annotations

import asyncio
import ssl
import time

from websockets.asyncio.client import ClientConnection, connect

from . import colosseum_pb2 as pb

PROTOCOL_VERSION = 1


class ProtocolError(RuntimeError):
    pass


class ColosseumClient:
    def __init__(
        self,
        router_url: str,
        token: str,
        client_id: str,
        *,
        robot_type: str = "DROID",
        joint_count: int = 7,
        has_gripper: bool = True,
        control_hz: int = 15,
        action_spaces: dict[str, int] | None = None,
    ) -> None:
        if not router_url.startswith(("wss://", "ws://")):
            raise ValueError("router_url must use wss:// or ws://")
        self.router_url = router_url
        self.token = token
        self.client_id = client_id
        if joint_count <= 0 or control_hz <= 0:
            raise ValueError("joint_count and control_hz must be positive")
        if action_spaces is None:
            action_spaces = {"joint_position": joint_count + int(has_gripper)}
        if not action_spaces or any(not name or dimension <= 0 for name, dimension in action_spaces.items()):
            raise ValueError("action_spaces must map names to positive action dimensions")
        self.robot_spec = pb.RobotSpec(
            robot_type=robot_type,
            joint_count=joint_count,
            has_gripper=has_gripper,
            control_hz=control_hz,
            action_spaces=[
                pb.ActionSpaceSpec(name=name, action_dim=dimension)
                for name, dimension in action_spaces.items()
            ],
        )
        self.connection: ClientConnection | None = None
        self.session_id = ""
        self.policy_server_id = ""
        self.policy_id = ""
        self._sequence = 0

    @staticmethod
    def _frame(message_type: int, payload=None, *, session_id: str = "", sequence: int = 0, deadline_ms: int = 0) -> pb.RelayFrame:
        return pb.RelayFrame(
            protocol_version=PROTOCOL_VERSION,
            type=message_type,
            session_id=session_id,
            sequence=sequence,
            sent_at_ns=time.time_ns(),
            deadline_ms=deadline_ms,
            payload=payload.SerializeToString() if payload is not None else b"",
        )

    async def connect(self, *, requested_policy_id: str = "", ssl_context: ssl.SSLContext | None = None, evaluation_run: str = "") -> pb.SessionReady:
        options = {
            "additional_headers": {"Authorization": f"Bearer {self.token}"},
            "compression": None,
            "max_size": 32 * 1024 * 1024,
            "ping_interval": 10,
            "ping_timeout": 10,
        }
        if evaluation_run:
            options["additional_headers"]["X-Colosseum-Run"] = evaluation_run
        if ssl_context is not None:
            options["ssl"] = ssl_context
        self.connection = await connect(self.router_url, **options)
        hello = pb.Hello(role=pb.CLIENT, peer_id=self.client_id)
        hello.robot.CopyFrom(self.robot_spec)
        await self.connection.send(self._frame(pb.HELLO, hello).SerializeToString())
        registered = pb.Registered.FromString((await self._expect(pb.REGISTERED)).payload)
        self.client_id = registered.peer_id
        await self.connection.send(
            self._frame(pb.MATCH_REQUEST, pb.MatchRequest(requested_policy_id=requested_policy_id)).SerializeToString()
        )
        matched = await self._expect(pb.MATCHED)
        result = pb.MatchResult.FromString(matched.payload)
        self.session_id = result.session_id
        self.policy_server_id = result.policy_server_id
        self.policy_id = result.policy_id
        ready = await self._expect(pb.SESSION_READY)
        return pb.SessionReady.FromString(ready.payload)

    async def infer(self, observation: pb.Observation, *, deadline_ms: int = 1000) -> pb.ActionPlan:
        if self.connection is None or not self.session_id:
            raise RuntimeError("client is not connected to a policy session")
        self._sequence += 1
        sequence = self._sequence
        frame = self._frame(
            pb.OBSERVATION,
            observation,
            session_id=self.session_id,
            sequence=sequence,
            deadline_ms=deadline_ms,
        )
        await self.connection.send(frame.SerializeToString())
        try:
            response = await asyncio.wait_for(self._expect(pb.ACTION_PLAN), timeout=deadline_ms / 1000)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"inference deadline exceeded ({deadline_ms} ms)") from exc
        if response.sequence != sequence:
            raise ProtocolError(f"action sequence mismatch: expected {sequence}, got {response.sequence}")
        plan = pb.ActionPlan.FromString(response.payload)
        if plan.request_sequence != sequence:
            raise ProtocolError("action plan request_sequence mismatch")
        return plan

    async def reset(self, reason: str = "") -> None:
        if self.connection is None or not self.session_id:
            raise RuntimeError("client is not connected to a policy session")
        await self.connection.send(
            self._frame(pb.RESET, pb.ResetRequest(reason=reason), session_id=self.session_id).SerializeToString()
        )

    async def close(self) -> None:
        connection = self.connection
        try:
            if connection is not None:
                try:
                    if self.session_id:
                        await connection.send(
                            self._frame(pb.SESSION_CLOSE, session_id=self.session_id).SerializeToString()
                        )
                finally:
                    await connection.close()
        finally:
            self.connection = None
            self.session_id = ""

    async def _expect(self, expected_type: int) -> pb.RelayFrame:
        if self.connection is None:
            raise RuntimeError("not connected")
        raw = await self.connection.recv()
        if isinstance(raw, str):
            raise ProtocolError("router returned a text frame; protobuf binary required")
        frame = pb.RelayFrame.FromString(raw)
        if frame.protocol_version != PROTOCOL_VERSION:
            raise ProtocolError("protocol version mismatch")
        if frame.type == pb.ERROR:
            error = pb.Error.FromString(frame.payload)
            raise ProtocolError(f"{error.code}: {error.message}")
        if frame.type == pb.SESSION_CLOSE:
            self.session_id = ""
            raise ProtocolError("policy session was closed")
        if frame.type != expected_type:
            raise ProtocolError(f"expected message type {expected_type}, got {frame.type}")
        return frame

    async def __aenter__(self) -> "ColosseumClient":
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()
