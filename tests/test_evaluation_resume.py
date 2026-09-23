import json
from pathlib import Path
import pytest
from colosseum_client import evaluation as e
from colosseum_client.robot_config import RobotClientConfig


@pytest.mark.parametrize("recording", [True, False])
@pytest.mark.parametrize("track", ["fine-tuning", "open"])
@pytest.mark.parametrize("skip_upload", [True, False])
def test_publish_before_upload_and_resume_failed_submission(tmp_path, monkeypatch, recording, track, skip_upload):
    assignment={'id':'ev_test','recording':recording,'track':track,'robot_id':'franka','state':'pending',
        'task':{'id':'cup-task','instruction':'cup','setup':'start','success_criteria':'placed','partial_success_criteria':'fraction','cameras':['head_image'],'max_steps':1},
        'runs':[{'id':'run_one','side':'trial','state':'assigned'}]}
    if track == 'open':
        assignment['runs'] = [{'id':'run_one','side':'A','state':'assigned'},
                              {'id':'run_two','side':'B','state':'assigned'}]
    expected_outcomes = ({'A':{'success':False,'partial_success':.75},
                          'B':{'success':True,'partial_success':1.0}} if track == 'open'
                         else {'trial':{'success':True,'partial_success':.75}})
    expected_result = {'outcomes':expected_outcomes,'preference':'B' if track == 'open' else None,
                       'feedback':'saved before upload' if track == 'open' else ''}
    executions=[]; submissions=[]; attempts=[]
    def check_upload_ready():
        assert executions == [run['id'] for run in assignment['runs']]
        saved = json.loads((tmp_path/'ev_test/manifest.json').read_text())
        assert saved['result'] == expected_result

    class API:
        def __init__(self, config): pass
        def close(self): pass
        def request(self,method,path,body=None):
            if path == '/reset': return {'discarded':None}
            if path.startswith('/tasks?'):
                return {'tasks':[assignment['task']]}
            if path == '/next' and track == 'fine-tuning':
                assert body['task_id']=='cup-task'
            if path == '/next' and submissions:
                raise e.NoAssignment('No trial available')
            if path.endswith('/result'):
                if not recording:
                    check_upload_ready()
                    attempts.append('result')
                    if len(attempts)==1: raise RuntimeError('network unavailable')
                submissions.append(body); return {'state':'completed'}
            return assignment
        def upload_dataset(self, run, path):
            assert not skip_upload, 'Deferred datasets must not contact the upload API'
            assert recording
            assert (path / "meta/info.json").is_file()
            check_upload_ready()
            assert submissions == [expected_result]
            assert json.loads((tmp_path/'ev_test/manifest.json').read_text())['submitted']
            attempts.append(run)
            if len(attempts)==1: raise RuntimeError('network unavailable')
        def upload(self,*args):
            pytest.fail('Standalone videos must not be uploaded')
    async def run(config,a,run,path,api,*,execute_action=True):
        if run['side'] == 'B' and recording:
            assert (path.parent/'run_one/lerobot/meta/info.json').is_file()
            assert not attempts, 'Model A must be saved locally without uploading before B'
        executions.append(run['id']); path.mkdir(parents=True)
        if not recording:
            run['state']='finished'
            return
        import numpy as np
        from colosseum_client.droid_robot import RobotObservation
        recorder = e.TrialRecorder(path, ['head_image'])
        obs = RobotObservation({'head_image': np.zeros((16,16,3), dtype=np.uint8)},
                               np.zeros(7), np.zeros(1), np.zeros(6))
        recorder.add(obs, np.ones(8))
        recorder.add(obs)
        recorder.close()
        (path/'recording-context.json').write_text(json.dumps({
            'robot_type':'franka','fps':15,'test':True,
            'action_space':'joint_position','execution_enabled':False}))
        run['state']='finished'
    monkeypatch.setattr(e,'EvalAPI',API)
    monkeypatch.setattr(e,'run_trial',run)
    def encode(p, cameras):
        pytest.fail('Standalone review videos must not be encoded')
    monkeypatch.setattr(e.TrialRecorder,'encode',encode)
    config=RobotClientConfig(url='ws://localhost:8443',token='test',cameras={'head_image':'test'},evaluation_dir=str(tmp_path),recording=recording,skip_upload=skip_upload,instruction='cup',scene='desk')
    answers=iter(['','75','','100','B','saved before upload'] if track == 'open' else ['1','','y','75'])
    if recording and skip_upload and track == 'open':
        answers = iter(['','75','','100','B','saved before upload','n'])
    monkeypatch.setattr('builtins.input',lambda _:next(answers))
    if recording:
        e.run_evaluation(config,track=track)
    else:
        with pytest.raises(RuntimeError,match='network unavailable'):
            e.run_evaluation(config,track=track)
    saved=json.loads((tmp_path/'ev_test/manifest.json').read_text())
    assert saved['outcomes']==expected_outcomes
    assert saved['result']==expected_result
    answers=iter(['n'] if track == 'open' else [])
    monkeypatch.setattr('builtins.input',lambda _:next(answers))
    if not recording:
        e.run_evaluation(config,resume='ev_test')
    assert executions==[run['id'] for run in assignment['runs']]
    assert attempts==(([] if skip_upload else ['run_one']) if recording else ['result','result'])
    if skip_upload:
        assert not saved.get('datasets_uploaded')
    assert submissions==[expected_result]
    assert json.loads((tmp_path/'ev_test/manifest.json').read_text())['submitted']

    if recording:
        output = tmp_path / "ev_test/run_one/lerobot"
        assert json.loads((output / "meta/info.json").read_text())["total_frames"] == 1
        metadata = json.loads((output / "meta/colosseum.json").read_text())
        assert metadata["outcome"] == expected_outcomes[assignment['runs'][0]['side']]
        assert metadata["test"] is True


def test_interrupted_second_trial_keeps_first_score_without_uploading(tmp_path, monkeypatch):
    assignment = {'id':'ev_interrupted','track':'open','state':'pending',
                  'task':{'instruction':'cup','cameras':['head_image']},
                  'runs':[{'id':'run_a','side':'A','state':'assigned'},
                          {'id':'run_b','side':'B','state':'assigned'}]}
    class API:
        def __init__(self, config): pass
        def close(self): pass
        def request(self, method, path, body=None):
            if path == '/reset': return {'discarded':None}
            assert path == '/next', 'Interrupted evaluation must not submit a result'
            return assignment
        def upload_dataset(self, *args):
            pytest.fail('No dataset upload before all trials finish')
        def upload(self, *args):
            pytest.fail('No video upload before all trials finish')
    async def run(config, assignment, trial, path, api, **kwargs):
        path.mkdir(parents=True)
        import numpy as np
        from colosseum_client.droid_robot import RobotObservation
        recorder = e.TrialRecorder(path, ['head_image'])
        observation = RobotObservation({'head_image':np.zeros((16,16,3),dtype=np.uint8)},
                                       np.zeros(7),np.zeros(1),np.zeros(6))
        recorder.add(observation,np.zeros(8))
        recorder.close()
        (path/'recording-context.json').write_text(json.dumps({
            'robot_type':'franka','fps':15,'test':False,
            'action_space':'joint_position','execution_enabled':False}))
        if trial['side'] == 'B':
            assert (path.parent/'run_a/lerobot/meta/info.json').is_file()
            trial['state'] = 'interrupted'
            raise KeyboardInterrupt()
        trial['state'] = 'finished'
    monkeypatch.setattr(e, 'EvalAPI', API)
    monkeypatch.setattr(e, 'run_trial', run)
    answers = iter(['','75',''])
    monkeypatch.setattr('builtins.input', lambda _: next(answers))
    config = RobotClientConfig(url='ws://localhost', token='test', cameras={'head_image':'dummy'},
                               instruction='cup', scene='desk', evaluation_dir=str(tmp_path))
    with pytest.raises(KeyboardInterrupt):
        e.run_evaluation(config, track='open')
    saved = json.loads((tmp_path/'ev_interrupted/manifest.json').read_text())
    assert saved['outcomes'] == {'A':{'success':False,'partial_success':.75}}
    assert 'result' not in saved and not saved.get('submitted')
    assert (tmp_path/'ev_interrupted/run_b/frames.jsonl').is_file()
    assert (tmp_path/'ev_interrupted/run_a/lerobot/meta/info.json').is_file()
