import json
from pathlib import Path
import pytest
from colosseum_client import evaluation as e
from colosseum_client.robot_config import RobotClientConfig


def test_failed_upload_resumes_without_reexecuting_or_rescoring(tmp_path, monkeypatch):
    assignment={'id':'ev_test','track':'fine-tuning','robot_id':'franka','state':'pending',
        'task':{'instruction':'cup','setup':'start','success_criteria':'placed','partial_success_criteria':'fraction','cameras':['head_image'],'max_steps':1},
        'runs':[{'id':'run_one','side':'trial','state':'assigned'}]}
    executions=[]; submissions=[]; attempts=[]
    class API:
        def __init__(self, config): pass
        def close(self): pass
        def request(self,method,path,body=None):
            if path == '/next' and submissions:
                raise e.NoAssignment('No trial available')
            if path.endswith('/result'):
                submissions.append(body); return {'state':'completed'}
            return assignment
        def upload(self,run,camera,path):
            attempts.append(run)
            if len(attempts)==1: raise RuntimeError('network unavailable')
    async def run(config,a,run,path,api):
        executions.append(run['id']); path.mkdir(parents=True)
        (path/'frames.jsonl').write_text('{}\n')
        run['state']='finished'
    monkeypatch.setattr(e,'EvalAPI',API)
    monkeypatch.setattr(e,'run_trial',run)
    monkeypatch.setattr(e.TrialRecorder,'encode',lambda p,c:{'head_image':p/'video.mp4'})
    config=RobotClientConfig(url='ws://localhost:8443',token='test',cameras={'head_image':'test'},evaluation_dir=str(tmp_path))
    answers=iter(['','y','75'])
    monkeypatch.setattr('builtins.input',lambda _:next(answers))
    with pytest.raises(RuntimeError,match='network unavailable'):
        e.run_evaluation(config,track='fine-tuning')
    saved=json.loads((tmp_path/'ev_test/manifest.json').read_text())
    assert saved['outcomes']['trial']=={'success':True,'partial_success':.75}
    answers=iter([])
    monkeypatch.setattr('builtins.input',lambda _:next(answers))
    e.run_evaluation(config,resume='ev_test')
    assert executions==['run_one']
    assert attempts==['run_one','run_one']
    assert submissions==[{'outcomes':{'trial':{'success':True,'partial_success':.75}},'preference':None,'feedback':''}]
    assert json.loads((tmp_path/'ev_test/manifest.json').read_text())['submitted']
