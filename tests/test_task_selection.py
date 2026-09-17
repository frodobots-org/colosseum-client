import pytest
from colosseum_client import evaluation as e
from colosseum_client.robot_config import RobotClientConfig


def test_select_task_reprompts_and_uses_server_task_id(monkeypatch):
    class API:
        def request(self,method,path):
            assert method=='GET' and path=='/tasks?robot_id=franka'
            return {'tasks':[{'id':'a','instruction':'Cup','setup':'Tray'},
                             {'id':'b','instruction':'Block','setup':'Table'}]}
    answers=iter(['oops','0','3','2'])
    monkeypatch.setattr('builtins.input',lambda _:next(answers))
    assert e.select_task(API(),'franka')=='b'


def test_empty_task_list_does_not_create_evaluation(tmp_path,monkeypatch):
    class API:
        def __init__(self,config): pass
        def close(self): pass
        def request(self,method,path):
            if path == '/reset': return {'discarded':None}
            assert method=='GET' and path.startswith('/tasks?')
            return {'tasks':[]}
    monkeypatch.setattr(e,'EvalAPI',API)
    monkeypatch.setattr('builtins.input',lambda _:pytest.fail('No prompt for empty task list'))
    e.run_evaluation(RobotClientConfig(url='ws://localhost',token='test',cameras={},evaluation_dir=str(tmp_path)),track='fine-tuning')


@pytest.mark.parametrize('track', ['open', 'fine-tuning'])
def test_config_skips_track_scene_prompts_and_sends_metadata(tmp_path, monkeypatch, track):
    sent = []
    class API:
        def __init__(self, config): pass
        def close(self): pass
        def request(self, method, path, body=None):
            if path == '/reset': return {'discarded':None}
            if path.startswith('/tasks?'):
                return {'tasks':[{'id':'task-1','instruction':'Move cup'}]}
            sent.append(body)
            raise e.NoAssignment('No trial available')
    monkeypatch.setattr(e, 'EvalAPI', API)
    questions = []
    def answer(prompt):
        questions.append(prompt)
        return 'Move cup' if track == 'open' else '1'
    monkeypatch.setattr('builtins.input', answer)
    config = RobotClientConfig(url='ws://localhost', token='test', cameras={'head_image':'1'},
        scene='kitchen', evaluator='operator-1', track=track, evaluation_dir=str(tmp_path))
    e.run_evaluation(config)
    assert len(questions) == 1
    assert sent[0]['scene'] == 'kitchen' and sent[0]['evaluator'] == 'operator-1'
    assert sent[0]['track'] == track
