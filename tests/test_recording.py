import json
import subprocess
import numpy as np
from colosseum_client.recording import TrialRecorder
from colosseum_client.droid_robot import RobotObservation


def test_multicamera_trial_encodes_valid_mp4_and_preserves_source(tmp_path):
    cameras=['head_image','left_image','right_image']
    recorder=TrialRecorder(tmp_path,cameras)
    for i in range(3):
        obs=RobotObservation({c:np.full((25,31,3),i*50,dtype=np.uint8) for c in cameras},np.zeros(7),np.zeros(1),np.zeros(6))
        recorder.add(obs,np.zeros(8))
    recorder.close()
    videos=TrialRecorder.encode(tmp_path,cameras)
    assert set(videos)==set(cameras)
    for path in videos.values():
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(path)]))
        assert info['streams'][0]['codec_name']=='h264'
        assert info['streams'][0]['width']==32
    assert len((tmp_path/'frames.jsonl').read_text().splitlines())==3
    assert len(list((tmp_path/'head_image').glob('*.png')))==3
    # Encoding may be retried after a failed upload.
    assert TrialRecorder.encode(tmp_path,cameras)==videos


def test_background_writer_snapshots_buffers_and_preserves_capture_times(tmp_path,monkeypatch):
    import threading
    from PIL import Image
    entered=threading.Event(); release=threading.Event(); threads=[]
    original=TrialRecorder._write_frame
    def slow_write(self,*args):
        threads.append(threading.get_ident()); entered.set()
        assert release.wait(3)
        original(self,*args)
    monkeypatch.setattr(TrialRecorder,'_write_frame',slow_write)
    recorder=TrialRecorder(tmp_path,['head_image'],queue_size=2)
    pixels=np.full((16,16,3),7,dtype=np.uint8)
    obs=RobotObservation({'head_image':pixels},np.zeros(7),np.zeros(1),np.zeros(6))
    try:
        recorder.add(obs,np.ones(8),captured_at=recorder.started+1)
        assert entered.wait(2)
        # A second enqueue must return even while compression is blocked.
        recorder.add(obs,np.ones(8),captured_at=recorder.started+1.25)
        pixels[:]=99; obs.joints[:]=99
    finally:
        release.set();recorder.close()
    assert all(t!=threading.get_ident() for t in threads)
    assert np.all(np.asarray(Image.open(tmp_path/'head_image/000000.png'))==7)
    assert np.all(np.asarray(Image.open(tmp_path/'head_image/000001.png'))==7)
    records=[json.loads(line) for line in (tmp_path/'frames.jsonl').read_text().splitlines()]
    assert [r['frame'] for r in records]==[0,1]
    assert [r['seconds'] for r in records]==[1,1.25]
    assert records[0]['joints']==[0]*7
    assert json.loads((tmp_path/'recording-status.json').read_text())['complete']


def test_full_queue_fails_explicitly_without_blocking_or_hiding_lost_frames(tmp_path,monkeypatch):
    import threading
    import pytest
    entered=threading.Event();release=threading.Event()
    def blocked(self,*args):
        entered.set();assert release.wait(3)
    monkeypatch.setattr(TrialRecorder,'_write_frame',blocked)
    recorder=TrialRecorder(tmp_path,['head_image'],queue_size=1)
    obs=RobotObservation({'head_image':np.zeros((16,16,3),dtype=np.uint8)},np.zeros(7),np.zeros(1),np.zeros(6))
    try:
        recorder.add(obs);assert entered.wait(2)
        recorder.add(obs)
        with pytest.raises(RuntimeError,match='queue full'):recorder.add(obs)
    finally:
        release.set()
        with pytest.raises(RuntimeError,match='Recording failed'):recorder.close()
    assert not recorder._worker.is_alive()
    with pytest.raises(RuntimeError,match='incomplete'):TrialRecorder.encode(tmp_path,['head_image'])


def test_disk_failure_is_propagated_and_blocks_encoding(tmp_path,monkeypatch):
    import pytest
    def failed(self,*args):raise OSError('disk full')
    monkeypatch.setattr(TrialRecorder,'_write_frame',failed)
    recorder=TrialRecorder(tmp_path,['head_image'])
    obs=RobotObservation({'head_image':np.zeros((16,16,3),dtype=np.uint8)},np.zeros(7),np.zeros(1),np.zeros(6))
    recorder.add(obs)
    with pytest.raises(RuntimeError,match='Recording failed') as exc:recorder.close()
    assert isinstance(exc.value.__cause__,OSError)
    assert not recorder._worker.is_alive()
    with pytest.raises(RuntimeError,match='incomplete'):TrialRecorder.encode(tmp_path,['head_image'])
