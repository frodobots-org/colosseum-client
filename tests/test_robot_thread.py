import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from colosseum_client import evaluation as e
from colosseum_client.robot_config import RobotClientConfig


@pytest.mark.parametrize('recording', [True, False])
@pytest.mark.parametrize('execute_action', [None, False])
@pytest.mark.parametrize('fail_read', [False, True])
def test_robot_lifecycle_stays_on_one_worker_thread(tmp_path, monkeypatch, fail_read, capsys, execute_action, recording):
    events=[]
    inferences=[]
    recordings=[]
    main_thread=threading.get_ident()
    def record(name): events.append((name,threading.get_ident()))
    class Robot:
        joint_count=7; has_gripper=True; action_dim=8; action_space_name='joint_position'
        def __init__(self,config): record('create')
        def get_observation(self):
            record('read')
            if fail_read: raise RuntimeError('robot read failed')
            return object()
        def execute(self,action): record('execute')
        def close(self): record('close')
    class Client:
        def __init__(self,*args,**kwargs): pass
        async def connect(self,**kwargs): return SimpleNamespace(action_spaces=['joint_position'])
        async def infer(self,*args,**kwargs):
            inferences.append(args)
            return object()
        async def close(self): pass
    class Recorder:
        def __init__(self,*args):
            assert recording
        def add(self,*args,**kwargs): recordings.append(args)
        def close(self): pass
    finished=[]
    api=SimpleNamespace(request=lambda *args:finished.append(args))
    monkeypatch.setattr(e,'ColosseumClient',Client)
    monkeypatch.setattr(e,'TrialRecorder',Recorder)
    monkeypatch.setattr(e,'protobuf_observation',lambda *a,**kw:object())
    monkeypatch.setattr(e,'action_chunk',lambda *a,**kw:np.tile(np.arange(8, dtype=float), (3, 1)))
    config=RobotClientConfig(url='ws://test',token='test',cameras={},control_hz=1000,recording=recording)
    assignment={'task':{'instruction':'Close laptop','cameras':['head_image'],'max_steps':5}}
    options={} if execute_action is None else {'execute_action':execute_action}
    coro=e.run_trial(config,assignment,{'id':'run-test'},tmp_path,api,robot_factory=Robot,**options)
    if fail_read:
        with pytest.raises(RuntimeError,match='robot read failed'): asyncio.run(coro)
        assert not finished
        output=capsys.readouterr().out
        assert '[robot-io]' not in output
        assert 'elapsed_ms=' not in output
    else:
        asyncio.run(coro)
        assert sum(name=='execute' for name,_ in events)==(5 if execute_action is None else 0)
        output=capsys.readouterr().out
        assert output.count('Received action chunk')==2
        assert '[robot-io]' not in output
        assert 'elapsed_ms=' not in output
        assert '[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]' not in output
        assert len(inferences)==2
        assert sum(name=='read' for name,_ in events)==(6 if recording else 5)
        assert len(recordings)==(6 if recording else 0)
        assert (tmp_path/'recording-context.json').exists() == recording
        assert not (tmp_path/'recording-disabled.json').exists()
        assert finished[0][1]=='/runs/run-test/finish'
    assert events[0][0]=='create' and events[-1][0]=='close'
    assert len({thread for _,thread in events})==1
    assert events[0][1]!=main_thread



@pytest.mark.parametrize('ending', ['done', 'timeout'])
def test_navigation_trials_end_on_done_or_wall_clock_timeout(tmp_path, monkeypatch, capsys, ending):
    events = []
    class Robot:
        joint_count=0; has_gripper=False; action_dim=2; action_space_name='base_velocity'
        def __init__(self, config): pass
        def prepare_for_task(self, task): events.append(('task', task['goal']['type'], threading.get_ident()))
        def get_observation(self): return object()
        def execute(self, action): events.append(('execute', tuple(action), threading.get_ident()))
        def close(self): events.append(('close', None, threading.get_ident()))
    plans = []
    class Client:
        def __init__(self, *args, **kwargs): pass
        async def connect(self, **kwargs): return SimpleNamespace(action_spaces=['base_velocity'])
        async def infer(self, *args, **kwargs):
            plan = SimpleNamespace(done=ending == 'done' and len(plans) >= 1)
            plans.append(plan)
            return plan
        async def close(self): pass
    finished = []
    api = SimpleNamespace(request=lambda *args: finished.append(args))
    monkeypatch.setattr(e, 'ColosseumClient', Client)
    monkeypatch.setattr(e, 'protobuf_observation', lambda *a, **kw: object())
    monkeypatch.setattr(e, 'action_chunk', lambda plan, **kw: np.zeros((1, 2)) if plan.done else np.tile([[0.3, 0.0]], (2, 1)))
    config = RobotClientConfig(url='ws://test', token='test', cameras={}, control_hz=1000, recording=False)
    task = {'instruction': 'Drive', 'cameras': [], 'max_steps': 10000,
            'goal': {'type': 'gps', 'latitude': 0, 'longitude': 0}, 'navigation': {'timeout_s': 0.1 if ending == 'timeout' else 300}}
    asyncio.run(e.run_trial(config, {'task': task}, {'id': 'run-nav'}, tmp_path, api, robot_factory=Robot))
    output = capsys.readouterr().out
    executed = [event for event in events if event[0] == 'execute']
    if ending == 'done':
        assert 'declared the task done' in output and len(plans) == 2
        assert executed[-1][1] == (0.0, 0.0) and len(executed) == 3
    else:
        assert 'navigation timeout' in output and 3 <= len(executed) < 10000
    assert events[0][0] == 'task' and events[-1][0] == 'close' and finished[0][1] == '/runs/run-nav/finish'
    assert len({event[2] for event in events}) == 1
