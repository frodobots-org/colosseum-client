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
