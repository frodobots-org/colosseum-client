import asyncio
import json
import pytest
from colosseum_client.local_policy import LocalPolicyClient
from colosseum_client.robot_config import RobotClientConfig
from colosseum_client.client import ProtocolError
from colosseum_client import colosseum_pb2 as pb


class Stream:
    def __init__(self, messages): self.messages = iter(messages)
    def __aiter__(self): return self
    async def __anext__(self):
        try: return next(self.messages)
        except StopIteration: raise StopAsyncIteration


def client():
    c = LocalPolicyClient(RobotClientConfig(url='ws://router',token='secret',cameras={},
        policy_server_url='ws://local'), 'http://router')
    c.run_id = c.session_id = 'run-current'
    return c


@pytest.mark.parametrize('frame', [None, pb.RelayFrame(protocol_version=1, session_id='run-old',type=pb.ACTION_PLAN),
                                   pb.RelayFrame(protocol_version=1, session_id='run-current',type=pb.SESSION_CLOSE)])
async def test_local_disconnect_or_stale_session_blocks_cached_actions(frame):
    c = client()
    c.connection = Stream([] if frame is None else [frame.SerializeToString()])
    await c._receive()
    with pytest.raises(ProtocolError, match='not active'):
        c.ensure_active()


async def test_router_control_failure_blocks_actions():
    class Control:
        async def send(self, value): raise ConnectionError('disconnected')
    c = client()
    c.control = Control()
    with pytest.raises(ConnectionError):
        await c._control('step','step',step=0)
    with pytest.raises(ProtocolError):
        c.ensure_active()


async def test_simulation_preparation_has_total_deadline_despite_progress():
    from colosseum_client.local_protocol import encode_control
    c = client()
    c.prepare_timeout = .03
    class Progress:
        async def recv(self):
            await asyncio.sleep(.005)
            return encode_control(dict(type='progress', run_id=c.run_id, preparation_id='p',
                verification_only=True, loaded=False, state='downloading'))
    c.connection = Progress()
    with pytest.raises(ProtocolError, match='preparation timed out'):
        await c._simulate({'preparation_id':'p'})


async def test_simulation_inference_deadline():
    from colosseum_client.local_protocol import encode_control
    c = client()
    c.model = {'url':'https://huggingface.co/example/model', 'subfolder':'', 'runtime_profile':''}
    c.task = {'instruction':'Close laptop'}
    c.inference_timeout = .01
    class SlowInference:
        calls = 0
        async def send(self, value): pass
        async def recv(self):
            self.calls += 1
            if self.calls == 1:
                return encode_control(dict(type='simulation_ready', run_id=c.run_id, preparation_id='p',
                    model=c.model, verification_only=True, loaded=False))
            await asyncio.sleep(1)
    c.connection = SlowInference()
    with pytest.raises(ProtocolError, match='Inference 1 timed out'):
        await c._simulate({'preparation_id':'p'})


@pytest.mark.parametrize('subfolder', ['', 'g05-so101'])
@pytest.mark.parametrize('ping', [(20, 60), (7, 15)])
@pytest.mark.parametrize('receipt', [False, True])
async def test_ready_starts_inference_without_router_ack(monkeypatch, receipt, ping, subfolder):
    from colosseum_client.local_protocol import encode_control, decode_control
    from colosseum_client import local_policy
    c = client()
    c.session_id = ''
    model = dict(url='https://huggingface.co/example/model',revision='a'*40,
        subfolder=subfolder,action_space='joint_position',action_dim=8,control_hz=15,max_horizon=1)
    preparation = dict(type='prepare', protocol_version=1, run_id=c.run_id,
        preparation_id='p',model=model,test=False,task={'instruction':'Move cup'})
    ack = asyncio.Event()
    class Router:
        calls = 0
        async def recv(self):
            self.calls += 1
            if self.calls == 1: return encode_control(preparation)
            await ack.wait()
            return encode_control(dict(type='started',run_id=c.run_id))
        async def send(self,raw): pass
    class Policy:
        calls = 0
        async def recv(self):
            self.calls += 1
            kind = 'received' if receipt and self.calls == 1 else 'ready'
            return encode_control({**preparation, 'type': kind, 'verification_only': False})
        async def send(self,raw):
            message = decode_control(raw)
            assert message['type'] == 'prepare' and message['test'] is False
            assert message['robot_type'] == c.hardware_robot_type
            assert 'robot_type' not in preparation  # Router preparation is unchanged.
            assert message.get('state') != 'simulate'
        def __aiter__(self): return self
        async def __anext__(self):
            await asyncio.Event().wait()
    connections = iter([Router(),Policy()])
    c.policy_ping_interval, c.policy_ping_timeout = ping
    async def connect(*args, **kwargs):
        if args[0] == c.router_url:
            assert (kwargs['ping_interval'], kwargs['ping_timeout']) == ping
        return next(connections)
    monkeypatch.setattr(local_policy,'connect',connect)
    async def capabilities(*args,**kwargs):
        raise AssertionError('Preparation must not query policy capabilities')
    monkeypatch.setattr(local_policy,'capabilities',capabilities)
    try:
        await asyncio.wait_for(c.connect(evaluation_run=c.run_id),.2)
        assert not ack.is_set()
        await asyncio.wait_for(c.before_action(0),.1)
        assert c.reports.empty()
        c.ensure_active()
    finally:
        for task in [c.receiver_task,c.reporting_task,c.heartbeat_task]:
            task.cancel()
        await asyncio.gather(c.receiver_task,c.reporting_task,c.heartbeat_task,return_exceptions=True)


@pytest.mark.parametrize('mode,accepted', [('base_velocity', True), ('joint_pos', False)])
async def test_rover_models_run_in_base_velocity_mode(monkeypatch, mode, accepted):
    from colosseum_client.local_protocol import encode_control, decode_control
    from colosseum_client import local_policy
    cfg = RobotClientConfig(url='ws://router', token='secret', cameras={'front_image': 'front'},
        robot_type='earth_rover_mini_plus', control_hz=10,
        llm_api_keys={'anthropic': 'key'}, policy_server_url='ws://127.0.0.1:8000')
    c = LocalPolicyClient(cfg, 'http://router', robot_type='earth_rover_mini_plus', joint_count=0,
                          has_gripper=False, control_hz=10, action_spaces={'base_velocity': 2},
                          navigation={'timeout_s': 600, 'stopped_s': 2})
    model = dict(model_type='llm', name='claude-fable-5-2', url='https://api.anthropic.com/v1',
                 revision='rover-v1', action_space='base_velocity', action_dim=2, control_hz=10, max_horizon=30)
    preparation = dict(type='prepare', protocol_version=1, run_id='run', preparation_id='p', model=model,
                       test=False, task={'instruction': 'Drive to the second checkpoint', 'max_steps': 6000})
    sent = []

    class Router:
        first = True
        async def recv(self):
            if self.first:
                self.first = False
                return encode_control(preparation)
            await asyncio.Event().wait()
        async def send(self, raw):
            pass

    class Policy:
        async def recv(self):
            return encode_control({**preparation, 'type': 'ready', 'control_mode': mode, 'internal_action_dim': 2})
        async def send(self, raw):
            sent.append(decode_control(raw))
        def __aiter__(self): return self
        async def __anext__(self):
            await asyncio.Event().wait()

    connections = iter([Router(), Policy()])
    async def connect(*args, **kwargs):
        return next(connections)
    monkeypatch.setattr(local_policy, 'connect', connect)
    try:
        if not accepted:
            with pytest.raises(ProtocolError, match='control mode'):
                await asyncio.wait_for(c.connect(evaluation_run='run'), .2)
            return
        ready = await asyncio.wait_for(c.connect(evaluation_run='run'), .2)
        assert ready.action_spaces == ['base_velocity'] and c.native_control_mode == 'base_velocity'
        # The Policy Server gets the key for this model and the trial's time limit.
        assert sent[0]['api_key'] == 'key' and sent[0]['robot_type'] == 'earth_rover_mini_plus'
        assert json.loads(sent[0]['message'])['client_options']['navigation']['timeout_s'] == 600
    finally:
        tasks = [task for task in (c.receiver_task, c.reporting_task, c.heartbeat_task) if task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_so101_llm_uses_router_frequency_and_internal_eef_contract(monkeypatch):
    from colosseum_client.local_protocol import encode_control, decode_control
    from colosseum_client import local_policy
    cfg = RobotClientConfig(url='ws://router', token='secret', cameras={}, robot_type='so101',
        control_hz=30, llm_control_mode='eef_delta_pose',
        llm_account_logins={'anthropic': 'claude-code'},
        policy_server_url='ws://127.0.0.1:8000')
    c = LocalPolicyClient(cfg, 'http://router', robot_type='so101', joint_count=5,
                          control_hz=cfg.control_hz, action_spaces={'joint_position': 6})
    model = dict(model_type='llm', name='claude-opus-5-5',
        url='https://api.anthropic.com/v1', revision='rig-v1',
        action_space='joint_position', action_dim=6, control_hz=10, max_horizon=4)
    preparation = dict(type='prepare', protocol_version=1, run_id='run',
        preparation_id='p', model=model, test=False, task={'instruction': 'Move cup'})

    class Router:
        first = True
        async def recv(self):
            if self.first:
                self.first = False
                return encode_control(preparation)
            await asyncio.Event().wait()
        async def send(self, raw):
            pass

    class Policy:
        async def recv(self):
            return encode_control({**preparation, 'type': 'ready',
                'control_mode': 'eef_delta_pose', 'internal_action_dim': 7})
        async def send(self, raw):
            sent = decode_control(raw)
            assert json.loads(sent['message'])['client_options']['auth_method'] == 'claude-code'
            assert 'api_key' not in sent
        def __aiter__(self): return self
        async def __anext__(self):
            await asyncio.Event().wait()

    connections = iter([Router(), Policy()])
    async def connect(*args, **kwargs):
        return next(connections)
    monkeypatch.setattr(local_policy, 'connect', connect)
    try:
        ready = await asyncio.wait_for(c.connect(evaluation_run='run'), .2)
        assert ready.control_hz == 10
        assert c.native_control_mode == 'eef_delta_pose'
        assert c.native_action_dim == 7
    finally:
        for task in (c.receiver_task, c.reporting_task, c.heartbeat_task):
            if task is not None:
                task.cancel()
        await asyncio.gather(*(task for task in (c.receiver_task, c.reporting_task, c.heartbeat_task)
                               if task is not None), return_exceptions=True)


@pytest.mark.asyncio
async def test_eef_effective_contract_keeps_router_joint_identity(monkeypatch):
    from colosseum_client.local_protocol import encode_control, decode_control
    from colosseum_client import local_policy

    cfg = RobotClientConfig(url='ws://router', token='token', cameras={},
        robot_type='yam', policy_server_url='ws://127.0.0.1:8000',
        control_hz=30, use_local_action_contract=True,
        llm_api_keys={'openai': 'secret'})
    c = LocalPolicyClient(cfg, 'http://router', robot_type='yam', joint_count=12,
        has_gripper=True, control_hz=30,
        action_spaces={'joint_position': 14, 'eef_abs_pose': 14})
    model = dict(model_type='llm', name='gpt-6-astra', url='https://relay.example/v1',
        revision='router-revision', action_space='joint_position', action_dim=14,
        control_hz=30, max_horizon=1)
    preparation = dict(type='prepare', protocol_version=1, run_id='run-1',
        preparation_id='prep', model=model, task={'instruction': 'Pick up the block'},
        test=False)
    effective = {**model, 'action_space': 'eef_abs_pose'}
    stop = asyncio.Event()

    class Router:
        async def recv(self):
            if not hasattr(self, 'sent_prepare'):
                self.sent_prepare = True
                return encode_control(preparation)
            await stop.wait()

        async def send(self, value):
            ready = decode_control(value)
            assert all(ready['model'][key] == value for key, value in model.items())
            assert 'message' not in ready

    class Policy:
        async def recv(self):
            if not hasattr(self, 'ready_sent'):
                self.ready_sent = True
                return encode_control({**preparation, 'type': 'ready', 'loaded': True,
                    'message': json.dumps({'effective_model': effective})})
            await stop.wait()

        async def send(self, value):
            actual = decode_control(value)
            assert all(actual['model'][key] == value for key, value in model.items())
            assert json.loads(actual['message'])['client_options']['use_local_action_contract'] is True
            assert actual['api_key'] == 'secret'

    connections = iter([Router(), Policy()])

    async def connect(*args, **kwargs):
        return next(connections)

    monkeypatch.setattr(local_policy, 'connect', connect)
    try:
        metadata = await c.connect(evaluation_run='run-1')
        assert c.model['action_space'] == 'eef_abs_pose'
        assert list(metadata.action_spaces) == ['eef_abs_pose']
    finally:
        tasks = [task for task in (c.receiver_task, c.reporting_task, c.heartbeat_task) if task]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks,
                             return_exceptions=True)


async def test_background_router_rejection_stops_further_actions():
    from colosseum_client.local_protocol import encode_control
    c=client()
    class Rejected:
        async def send(self,raw): pass
        async def recv(self):return encode_control(dict(type='error',message='Assignment aborted'))
    c.control=Rejected()
    with pytest.raises(ProtocolError):
        await c._report_lifecycle(dict(type='ready'))
    with pytest.raises(ProtocolError,match='not active'):
        await c.before_action(1)


async def test_control_error_keeps_code_and_message():
    from colosseum_client.local_protocol import decode_control
    from colosseum_client import local_policy
    frame = pb.RelayFrame(
        protocol_version=1, type=pb.ERROR, session_id='run-test',
        payload=pb.Error(code='MODEL_START_FAILED', message='launcher failed', retryable=False).SerializeToString(),
    )
    assert decode_control(frame.SerializeToString()) == {
        'type': 'error', 'code': 'MODEL_START_FAILED', 'message': 'launcher failed', 'retryable': False,
    }
    class Connection:
        async def recv(self):
            return frame.SerializeToString()
    with pytest.raises(ProtocolError, match='MODEL_START_FAILED: launcher failed'):
        await local_policy.receive_control(Connection())


async def test_protobuf_error_frame_preserves_code_and_message():
    c = client()
    error = pb.Error(code='INFERENCE_FAILED', message='sanitized backend failure')
    class Connection:
        sent = False
        def __aiter__(self): return self
        async def __anext__(self):
            if self.sent: raise StopAsyncIteration
            self.sent = True
            return pb.RelayFrame(protocol_version=1, type=pb.ERROR, session_id=c.run_id,
                payload=error.SerializeToString()).SerializeToString()
    c.connection = Connection()
    await c._receive()
    with pytest.raises(ProtocolError, match='INFERENCE_FAILED: sanitized backend failure'):
        await c._expect(pb.ACTION_PLAN)


async def test_error_blocks_queued_and_cached_actions():
    c = client()
    c.failure = ProtocolError('INFERENCE_FAILED: backend failure')
    c.incoming.put_nowait(pb.RelayFrame(protocol_version=1, session_id=c.run_id, type=pb.ACTION_PLAN))
    with pytest.raises(ProtocolError, match='INFERENCE_FAILED: backend failure'):
        await c._expect(pb.ACTION_PLAN)
    assert c.incoming.qsize() == 1
    with pytest.raises(ProtocolError, match='INFERENCE_FAILED: backend failure'):
        await c.before_action(1)
    assert c.reports.empty()


async def test_local_close_attempts_both_connections():
    c = client()
    events = []
    class Policy:
        async def send(self, raw): pass
        async def close(self):
            events.append('policy')
            raise RuntimeError('policy close failed')
    class Router:
        async def close(self):
            events.append('router')
            raise ValueError('router close failed')
    c.connection, c.control = Policy(), Router()
    with pytest.raises(RuntimeError, match='policy close failed'):
        await c.close()
    assert events == ['policy', 'router']
    assert not c.session_id and c.connection is None
