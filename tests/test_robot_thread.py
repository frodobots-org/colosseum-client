import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from colosseum_client import evaluation as e
from colosseum_client.robot_config import RobotClientConfig


@pytest.mark.parametrize('execute_action', [None, False])
@pytest.mark.parametrize('fail_read', [False, True])
def test_robot_lifecycle_stays_on_one_worker_thread(tmp_path, monkeypatch, fail_read, capsys, execute_action):
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
        def __init__(self,*args): pass
        def add(self,*args,**kwargs): recordings.append(args)
        def close(self): pass
    finished=[]
    api=SimpleNamespace(request=lambda *args:finished.append(args))
    monkeypatch.setattr(e,'ColosseumClient',Client)
    monkeypatch.setattr(e,'TrialRecorder',Recorder)
    monkeypatch.setattr(e,'protobuf_observation',lambda *a,**kw:object())
    monkeypatch.setattr(e,'action_chunk',lambda *a,**kw:np.tile(np.arange(8, dtype=float), (3, 1)))
    config=RobotClientConfig(url='ws://test',token='test',cameras={},control_hz=1000)
    assignment={'task':{'instruction':'Close laptop','cameras':['head_image'],'max_steps':5}}
    options={} if execute_action is None else {'execute_action':execute_action}
    coro=e.run_trial(config,assignment,{'id':'run-test'},tmp_path,api,robot_factory=Robot,**options)
    if fail_read:
        with pytest.raises(RuntimeError,match='robot read failed'): asyncio.run(coro)
        assert not finished
        output=capsys.readouterr().out
        assert 'operation=get_observation:step_start event=start' in output
        assert 'event=error elapsed_ms=' in output
        assert "error='robot read failed' type=RuntimeError" in output
        assert 'operation=execute' not in output
    else:
        asyncio.run(coro)
        assert sum(name=='execute' for name,_ in events)==(5 if execute_action is None else 0)
        output=capsys.readouterr().out
        assert output.count('Received action chunk')==2
        assert output.count('event=end elapsed_ms=')==(10 if execute_action is None else 5)
        assert output.count('operation=execute event=skipped')==(0 if execute_action is None else 5)
        assert '[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]' in output
        assert len(inferences)==2
        assert sum(name=='read' for name,_ in events)==6
        assert len(recordings)==6
        assert not (tmp_path/'recording-disabled.json').exists()
        assert finished[0][1]=='/runs/run-test/finish'
    assert events[0][0]=='create' and events[-1][0]=='close'
    assert len({thread for _,thread in events})==1
    assert events[0][1]!=main_thread
