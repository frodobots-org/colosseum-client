"""Local policy WS protocol: Protobuf control and inference frames."""
import asyncio
import contextlib
import hashlib
import struct
import math
from urllib.parse import urlsplit, urlunsplit, quote

from websockets.asyncio.client import connect

from .local_protocol import encode_control, decode_control

from .client import ColosseumClient, ProtocolError
from . import colosseum_pb2 as pb


async def receive_control(connection, timeout=30):
    raw = await asyncio.wait_for(connection.recv(), timeout)
    value = decode_control(raw)
    if not isinstance(value, dict) or value.get('type') == 'error':
        raise ProtocolError('Local policy/control request failed')
    return value


async def capabilities(url, *, details=False):
    async with connect(url, proxy=None, open_timeout=10, max_size=65536) as connection:
        await connection.send(encode_control(dict(type='capabilities', protocol_version=1)))
        reply = await receive_control(connection)
        profiles = reply.get('runtime_profiles')
        if (reply.get('type') != 'capabilities' or reply.get('protocol_version') != 1
                or not isinstance(profiles, list) or not profiles or len(profiles) > 100
                or any(not isinstance(p, str) or not p or len(p) > 100 for p in profiles)):
            raise ProtocolError('Invalid local runtime capabilities')
        return reply if details else profiles


class LocalPolicyClient(ColosseumClient):
    def __init__(self, config, api_base, **kwargs):
        super().__init__(config.policy_server_url, '', '', **kwargs)
        parts = urlsplit(api_base)
        self.control_base = urlunsplit(('wss' if parts.scheme == 'https' else 'ws', parts.netloc,
                                       parts.path.rstrip('/'), '', ''))
        self.router_token = config.token
        self.institution = config.institution.strip()
        self.test_robot = config.test or config.robot_type == "test"
        self.inference_timeout = config.deadline_ms / 1000
        self.prepare_timeout = config.prepare_timeout
        self.control = None
        self.control_lock = asyncio.Lock()
        self.heartbeat_task = None
        self.failure = None
        self.incoming = asyncio.Queue(maxsize=4)
        self.receiver_task = None
        self.reporting_task = None
        self.reports = asyncio.Queue(maxsize=1024)

    async def connect(self, *, evaluation_run='', verification_only=False, send_dummy=False, **kwargs):
        if send_dummy and not self.test_robot:
            raise ProtocolError("Synthetic inference requires robot_type: test")
        self.run_id = evaluation_run
        self.control = await connect(self.control_base + f'/api/eval/runs/{evaluation_run}/local',
            additional_headers={'Authorization': f'Bearer {self.router_token}', 'X-Colosseum-Institution': quote(self.institution,safe='')},
            proxy=None, open_timeout=10, max_size=65536)
        preparation = await receive_control(self.control)
        if preparation.get('type') != 'prepare' or preparation.get('run_id') != evaluation_run or preparation.get('protocol_version') != 1:
            raise ProtocolError('Invalid Router preparation')
        if not verification_only and bool(preparation.get('test', False)) != self.test_robot:
            raise ProtocolError('Router test flag differs from client configuration')
        self.model = preparation['model']
        self.task = preparation['task']
        dimensions = {a.name: a.action_dim for a in self.robot_spec.action_spaces}
        if not verification_only and not self.test_robot and (dimensions.get(self.model['action_space']) != self.model['action_dim'] or self.model['control_hz'] != self.robot_spec.control_hz):
            raise ProtocolError('Assigned model does not support this robot action contract')
        simulate = False
        # Router credentials never travel to the local service.
        self.connection = await connect(self.router_url, proxy=None, open_timeout=10,
            compression=None, max_size=32 * 1024 * 1024, ping_interval=10, ping_timeout=10)
        if send_dummy or simulate:
            preparation = {**preparation, "state":"simulate", "verification_only":True}
        await self.connection.send(encode_control(preparation))
        async with asyncio.timeout(self.prepare_timeout):
            while True:
                ready = await receive_control(self.connection, self.prepare_timeout)
                if ready.get('run_id') != evaluation_run or ready.get('preparation_id') != preparation['preparation_id']:
                    raise ProtocolError('Local preparation correlation mismatch')
                if not verification_only and ready.get('type') == 'received':
                    simulate = ready.get('verification_only') is True
                    if simulate and not self.test_robot:
                        raise ProtocolError('Simulation server requires test: true')
                    if ready.get('model') != self.model:
                        raise ProtocolError('Invalid model receipt')
                    continue
                if ready.get('type') == 'progress':
                    state = ready.get('state')
                    if state not in {'downloading', 'loading', 'warming_up'}:
                        raise ProtocolError('Invalid preparation progress')
                    print(f"Policy: {state} {ready.get('message', '')}", flush=True)
                    continue
                if verification_only:
                    if (ready.get('type') != 'received' or ready.get('model') != self.model
                            or ready.get('verification_only') is not True or ready.get('loaded') is not False
                            or ready.get('transport') != urlsplit(self.router_url).scheme):
                        raise ProtocolError('Invalid WebSocket verification receipt')
                    break
                if ready.get('type') == 'simulation_ready' or ready.get('verification_only') is True:
                    if not self.test_robot:
                        raise ProtocolError('Simulation server requires test: true')
                if simulate and ready.get('type') == 'simulation_ready':
                    if ready.get('verification_only') is not True or ready.get('loaded') is not False:
                        raise ProtocolError('Invalid simulation readiness')
                    ready = {**ready, 'type':'ready', 'state':'simulation'}
                if ready.get('type') != 'ready' or ready.get('model') != self.model:
                    raise ProtocolError('Local model revision/profile mismatch')
                break
        if verification_only:
            if send_dummy:
                ready['simulation'] = await self._simulate(preparation)
            return ready
        self.session_id = evaluation_run
        self.receiver_task = asyncio.create_task(self._receive())
        # Readiness comes from Policy Server. Router acknowledgments are consumed
        # in order in the background, never on the inference/action start path.
        self.reporting_task = asyncio.create_task(self._report_lifecycle(ready))
        self.heartbeat_task = asyncio.create_task(self._heartbeat())
        return pb.SessionReady(policy_id='assigned-policy', action_spaces=[self.model['action_space']],
            control_hz=self.model['control_hz'], max_horizon=self.model['max_horizon'])

    async def _simulate(self, preparation):
        print(f'SIMULATION: preparing model (timeout {self.prepare_timeout}s)', flush=True)
        started = asyncio.get_running_loop().time()
        try:
            async with asyncio.timeout(self.prepare_timeout):
                while True:
                    reply = await receive_control(self.connection, self.prepare_timeout)
                    if (reply.get('run_id') != self.run_id or
                            reply.get('preparation_id') != preparation['preparation_id'] or
                            reply.get('verification_only') is not True or reply.get('loaded') is not False):
                        raise ProtocolError('Invalid simulation preparation identity')
                    if reply.get('type') == 'simulation_ready' and reply.get('model') == self.model:
                        break
                    if reply.get('type') != 'progress' or reply.get('state') not in {'downloading','loading','warming_up'}:
                        raise ProtocolError('Expected simulation progress or readiness')
                    print(f"Policy: {reply['state']} — {reply.get('message','')} "
                          f"[elapsed {asyncio.get_running_loop().time()-started:.1f}s]", flush=True)
        except TimeoutError as exc:
            raise ProtocolError(f'Model preparation timed out after {self.prepare_timeout}s') from exc
        print('SIMULATION ready; sending 3 dummy observations; returned actions will only be validated.', flush=True)
        results = []
        for sequence in range(1, 4):
            obs = pb.Observation(instruction=self.task['instruction'], control_step=sequence-1)
            for name, size in [('joint_position',7), ('gripper_position',1), ('cartesian_position',6)]:
                obs.state[name].CopyFrom(pb.Tensor(shape=[size], dtype=pb.FLOAT32,
                    data=struct.pack(f'<{size}f', *([0.0]*size))))
            obs.sensors.add(sensor_id='head_image', encoding=pb.RAW_RGB, width=32, height=24,
                            data=bytes([32,128,224])*(32*24))
            frame = self._frame(pb.OBSERVATION, obs, session_id=self.run_id, sequence=sequence)
            frame.deadline_ms = int(self.inference_timeout * 1000)
            started = asyncio.get_running_loop().time()
            print(f'Inference {sequence}/3: waiting (timeout {self.inference_timeout:g}s)...', flush=True)
            try:
                async with asyncio.timeout(self.inference_timeout):
                    await self.connection.send(frame.SerializeToString())
                    raw = await self.connection.recv()
            except TimeoutError as exc:
                raise ProtocolError(f'Inference {sequence} timed out after {self.inference_timeout:g}s') from exc
            if not isinstance(raw, bytes):
                raise ProtocolError('Expected binary synthetic ActionPlan')
            reply = pb.RelayFrame.FromString(raw)
            if reply.type == pb.LOCAL_CONTROL:
                error = decode_control(raw)
                raise ProtocolError(f"Simulation failed: {error.get('code')} {error.get('message','')}")
            if reply.protocol_version != 1 or reply.type != pb.ACTION_PLAN or reply.session_id != self.run_id:
                raise ProtocolError('Invalid synthetic ActionPlan frame')
            plan = pb.ActionPlan.FromString(reply.payload)
            if (reply.sequence != sequence or plan.request_sequence != sequence or plan.plan_id != sequence
                    or plan.start_step != sequence-1 or plan.valid_until_step != sequence-1
                    or plan.control_hz != self.model['control_hz']
                    or list(plan.actions.shape) != [1, self.model['action_dim']]
                    or plan.actions.dtype != pb.FLOAT32 or len(plan.actions.data) != self.model['action_dim']*4
                    or any(not math.isfinite(x[0]) for x in struct.iter_unpack('<f',plan.actions.data))):
                raise ProtocolError('Invalid synthetic ActionPlan contract')
            elapsed = asyncio.get_running_loop().time()-started
            results.append(dict(sequence=sequence, elapsed_seconds=round(elapsed,3),
                action_shape=list(plan.actions.shape), data_sha256=hashlib.sha256(frame.payload).hexdigest()))
            print(f'Inference {sequence}/3: received fake action {list(plan.actions.shape)} in {elapsed:.2f}s', flush=True)
        await self.connection.send(pb.RelayFrame(protocol_version=1,type=pb.SESSION_CLOSE,
                                                session_id=self.run_id).SerializeToString())
        return dict(loaded=False, synthetic=True, inferences=results)

    async def _receive(self):
        try:
            async for raw in self.connection:
                if isinstance(raw, str):
                    raise ProtocolError('Expected binary inference frame')
                frame = pb.RelayFrame.FromString(raw)
                if frame.protocol_version != 1 or frame.session_id != self.run_id or frame.type != pb.ACTION_PLAN:
                    raise ProtocolError('Invalid local inference frame')
                self.incoming.put_nowait(frame)
            raise ProtocolError('Local policy disconnected')
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.failure = exc
            if not self.incoming.full():
                self.incoming.put_nowait(exc)

    async def _expect(self, expected_type):
        self.ensure_active()
        frame = await self.incoming.get()
        if isinstance(frame, Exception):
            raise ProtocolError('Local inference interrupted') from frame
        self.ensure_active()
        if frame.type != expected_type:
            raise ProtocolError('Unexpected local message')
        return frame

    def ensure_active(self):
        if self.failure or not self.session_id:
            raise ProtocolError('Local Trial connection is not active') from self.failure

    async def _control(self, kind, expected, **fields):
        self.ensure_active()
        try:
            async with self.control_lock:
                await self.control.send(encode_control(dict(type=kind, run_id=self.run_id, **fields)))
                reply = await receive_control(self.control, timeout=10)
                if reply != dict(type=expected, run_id=self.run_id, **fields):
                    raise ProtocolError('Invalid Router lifecycle acknowledgment')
        except Exception as exc:
            self.failure = exc
            raise

    async def _report_lifecycle(self, ready):
        try:
            async with asyncio.timeout(10):
                await self.control.send(encode_control(ready))
                started = await receive_control(self.control, timeout=10)
                if started != dict(type='started', run_id=self.run_id):
                    raise ProtocolError('Router rejected local Trial start report')
            while True:
                kind, expected, fields = await self.reports.get()
                await self._control(kind, expected, **fields)
                if kind == 'finish':
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.failure = exc
            raise

    def _queue_report(self, kind, expected, **fields):
        self.ensure_active()
        try:
            self.reports.put_nowait((kind, expected, fields))
        except asyncio.QueueFull as exc:
            self.failure = exc
            raise ProtocolError('Router reporting backlog exceeded limit') from exc

    async def _heartbeat(self):
        try:
            while True:
                await asyncio.sleep(10)
                self._queue_report('heartbeat', 'heartbeat')
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.failure = exc

    async def before_action(self, step):
        self._queue_report('step', 'step', step=step)

    async def infer(self, observation, *, deadline_ms=1000):
        self.ensure_active()
        observation.instruction = self.task['instruction']
        plan = await super().infer(observation, deadline_ms=deadline_ms)
        if plan.control_hz != self.model['control_hz']:
            raise ProtocolError('Local policy action control rate mismatch')
        if not plan.actions.shape or plan.actions.shape[0] > self.model['max_horizon']:
            raise ProtocolError('Local policy action horizon exceeds the assigned limit')
        return plan

    async def finish(self):
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.heartbeat_task
            self.heartbeat_task = None
        self._queue_report('finish', 'finished')
        # Before saving scores or starting B, confirm all A events were persisted.
        async with asyncio.timeout(30):
            await self.reporting_task
        self.ensure_active()

    async def close(self):
        for task in (self.heartbeat_task, self.receiver_task, self.reporting_task):
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        try:
            await super().close()
        finally:
            if self.control:
                await self.control.close()
