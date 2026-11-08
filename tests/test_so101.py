from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from colosseum_client import RobotClientConfig, make_robot
from colosseum_client.robots import so101


def config(**settings):
    return RobotClientConfig(url='ws://localhost', token='test', robot_type='so101', control_hz=30,
        image_width=4, image_height=2,
        cameras={'head_image': '123456789012', 'left_image': '/dev/video0'},
        adapter_config={'port': '/dev/ttyACM0', 'robot_id': 'follower_arm',
                        'joint_low': [-100, -100, -100, -100, -170],
                        'joint_high': [100, 100, 100, 100, 170], **settings})


@pytest.fixture
def hardware(monkeypatch):
    opened = []

    class Follower:
        def __init__(self, port, robot_id, calibration_dir, cameras, max_relative_target):
            self.arguments = (port, robot_id, calibration_dir, max_relative_target)
            self.cameras, self.commands, self.disconnected = cameras, [], 0
            self.position = np.array([1., 2., 3., 4., 5., 40.])
            self.config = SimpleNamespace(num_read_retries=0)
            self.bus = SimpleNamespace(sync_read=lambda _, num_retry: dict(zip(so101.MOTORS, self.position)))
            opened.append(self)
        def get_observation(self):
            raw = {f'{motor}.pos': value for motor, value in zip(so101.MOTORS, self.position)}
            for index, role in enumerate(self.cameras):
                raw[role] = np.full((2, 4, 3), index + 1, np.uint8)
            return raw
        def send_action(self, action):
            self.commands.append([action[f'{motor}.pos'] for motor in so101.MOTORS])
            self.position = np.asarray(self.commands[-1])
        def disconnect(self):
            self.disconnected += 1

    monkeypatch.setattr(so101, '_open_follower', Follower)
    monkeypatch.setattr(so101.time, 'sleep', lambda _: None)
    return opened


def test_state_split_camera_types_and_action_order(hardware):
    robot = make_robot(config(max_relative_target=20))
    follower = hardware[0]
    assert follower.arguments == ('/dev/ttyACM0', 'follower_arm', None, 20.0)
    assert type(follower.arguments[3]) is float
    assert follower.cameras == {'head_image': ('realsense', '123456789012', 640, 480, 30),
                                'left_image': ('opencv', '/dev/video0', 640, 480, 30)}
    obs = robot.get_observation()
    np.testing.assert_allclose(obs.joints, [1, 2, 3, 4, 5])
    np.testing.assert_allclose(obs.gripper, [40])
    assert obs.joints.dtype == obs.gripper.dtype == np.float32
    assert obs.cartesian_position.size == 0
    assert obs.images['head_image'].shape == (2, 4, 3) and obs.images['left_image'][0, 0, 0] == 2
    robot.execute(np.array([6, 7, 8, 9, 10, 60], np.float32))
    assert follower.commands == [[6, 7, 8, 9, 10, 60]]


def test_action_is_clamped_to_configured_range(hardware):
    robot = make_robot(config())
    robot.execute([-100.4, 150, 0, 0, 171, 100.7])
    assert hardware[0].commands == [[-100, 100, 0, 0, 170, 100]]
    robot.execute([0, 0, 0, 0, 0, -3])
    assert hardware[0].commands[-1][5] == 0


@pytest.mark.parametrize('action', [np.zeros(5), np.zeros(7), [0, 0, float('nan'), 0, 0, 0],
                                    [0, 0, 0, 0, 0, float('inf')]])
def test_invalid_action_never_moves(hardware, action):
    robot = make_robot(config())
    with pytest.raises(ValueError, match='6 finite'):
        robot.execute(action)
    assert not hardware[0].commands


def test_close_returns_to_start_in_bounded_steps_then_releases(hardware):
    robot = make_robot(config(return_step_deg=2))
    robot.execute([11, 2, 3, 4, 5, 40])
    robot.close()
    robot.close()
    follower = hardware[0]
    steps = np.asarray(follower.commands[1:])
    assert len(steps) == 5 and np.max(np.abs(np.diff(steps[:, 0]))) <= 2 + 1e-9
    np.testing.assert_allclose(steps[-1], [1, 2, 3, 4, 5, 40])
    assert follower.disconnected == 1
    with pytest.raises(RuntimeError, match='closed'):
        robot.get_observation()
    with pytest.raises(RuntimeError, match='closed'):
        robot.execute(np.zeros(6))


def test_return_to_start_can_be_disabled(hardware):
    robot = make_robot(config(return_to_start=False))
    robot.execute([11, 2, 3, 4, 5, 40])
    robot.close()
    assert len(hardware[0].commands) == 1 and hardware[0].disconnected == 1


def test_failed_return_still_releases_and_is_reported(hardware):
    robot = make_robot(config())
    robot.execute([11, 2, 3, 4, 5, 40])
    def fail(_):
        raise RuntimeError('bus lost')
    hardware[0].send_action = fail
    with pytest.raises(RuntimeError, match='SO101 shutdown failed'):
        robot.close()
    assert hardware[0].disconnected == 1


OVERLOAD = "Failed to write 'Torque_Enable' on id_=6 with '0' after 6 tries. [RxPacketError] Overload error!"


class Bus:
    """Feetech bus whose servos carry out a torque-off write even when they answer with an alarm."""
    def __init__(self, stuck=(), read_fails=False):
        self.motors = dict.fromkeys(so101.MOTORS)
        self.torque = dict.fromkeys(so101.MOTORS, 1)
        self.is_connected, self.stuck, self.read_fails = True, stuck, read_fails
    def write(self, name, motor, value, num_retry):
        assert (name, value) == ('Torque_Enable', 0)
        if motor not in self.stuck:
            self.torque[motor] = 0
        if motor == 'gripper':
            raise RuntimeError(OVERLOAD)
    def sync_read(self, name, normalize, num_retry):
        assert name == 'Torque_Enable' and normalize is False
        if self.read_fails:
            raise ConnectionError('There is no status packet!')
        return dict(self.torque)
    def disconnect(self, disable_torque):
        assert disable_torque is False
        self.is_connected = False


def alarmed_disconnect(follower, bus):
    """LeRobot's disconnect raises on the gripper's alarm before closing the port and cameras."""
    def disconnect():
        raise RuntimeError(OVERLOAD)
    follower.disconnect, follower.bus = disconnect, bus
    follower.cameras = {'head_image': Camera(), 'left_image': Camera()}
    return follower.cameras.values()


def test_gripper_overload_at_shutdown_releases_torque_port_and_cameras(hardware, caplog):
    robot = make_robot(config(return_to_start=False))
    bus = Bus()
    cameras = alarmed_disconnect(hardware[0], bus)
    robot.close()
    assert not any(bus.torque.values()) and not bus.is_connected
    assert not any(camera.alive for camera in cameras)
    assert 'Overload error' in caplog.text and 'torque is confirmed off' in caplog.text


@pytest.mark.parametrize('bus, unconfirmed', [
    (Bus(stuck=('gripper',)), 'gripper'),
    (Bus(read_fails=True), ', '.join(so101.MOTORS)),
])
def test_unconfirmed_torque_release_is_reported_after_releasing_the_rest(hardware, bus, unconfirmed):
    robot = make_robot(config(return_to_start=False))
    cameras = alarmed_disconnect(hardware[0], bus)
    with pytest.raises(RuntimeError, match='SO101 shutdown failed') as failure:
        robot.close()
    assert str(failure.value.__cause__) == f'SO101 torque may still be on for {unconfirmed}'
    assert not bus.is_connected and not any(camera.alive for camera in cameras)


def test_closed_port_after_camera_failure_needs_no_torque_writes(hardware):
    robot = make_robot(config(return_to_start=False))
    bus = Bus()
    bus.is_connected = False
    bus.write = bus.sync_read = None
    cameras = alarmed_disconnect(hardware[0], bus)
    robot.close()
    assert not any(camera.alive for camera in cameras)


class Camera:
    def __init__(self, alive=True, connect_failures=0):
        self.alive, self.connect_failures, self.connects = alive, connect_failures, 0
    def read_latest(self):
        if not self.alive:
            raise RuntimeError('read thread is not running.')
        return np.zeros((2, 4, 3), np.uint8)
    def disconnect(self):
        self.alive = False
    def connect(self):
        self.connects += 1
        if self.connects <= self.connect_failures:
            raise ConnectionError('Failed to open camera')
        self.alive = True


def with_cameras(follower, cameras):
    read = follower.get_observation
    def observe():
        for camera in cameras.values():
            camera.read_latest()
        return read()
    follower.get_observation = observe
    follower.cameras = cameras


def test_dropped_camera_is_reconnected_and_read_again(hardware):
    robot = make_robot(config())
    head, wrist = Camera(), Camera(alive=False, connect_failures=2)
    with_cameras(hardware[0], {'head_image': head, 'left_image': wrist})
    obs = robot.get_observation()
    assert obs.images.keys() == {'head_image', 'left_image'}
    assert wrist.connects == 3 and head.connects == 0


def test_camera_that_never_returns_fails_and_still_homes(hardware, monkeypatch):
    robot = make_robot(config(return_step_deg=2))
    robot.execute([11, 2, 3, 4, 5, 40])
    with_cameras(hardware[0], {'head_image': Camera(), 'left_image': Camera(alive=False, connect_failures=10**6)})
    clock = iter(range(0, 10**6, 10))
    monkeypatch.setattr(so101.time, 'monotonic', lambda: next(clock))
    with pytest.raises(RuntimeError, match='left_image did not reconnect'):
        robot.get_observation()
    robot.close()
    np.testing.assert_allclose(hardware[0].commands[-1], [1, 2, 3, 4, 5, 40])


def test_motor_errors_are_not_treated_as_camera_drops(hardware):
    robot = make_robot(config())
    def fail():
        raise ConnectionError('Incorrect status packet!')
    hardware[0].get_observation = fail
    hardware[0].cameras = {'head_image': Camera()}
    with pytest.raises(ConnectionError, match='status packet'):
        robot.get_observation()


def test_images_are_resized_to_the_configured_size(hardware):
    robot = make_robot(replace(config(), image_width=2, image_height=1))
    assert robot.get_observation().images['left_image'].shape == (1, 2, 3)


@pytest.mark.parametrize('settings,match', [
    ({'port': ''}, 'port and robot_id'),
    ({'robot_id': None}, 'port and robot_id'),
    ({'joint_low': None}, 'joint_low'),
    ({'joint_high': [1, 2, 3]}, 'joint_high'),
    ({'joint_low': [100] * 5}, 'joint_low < joint_high'),
    ({'max_relative_target': 0}, 'max_relative_target'),
    ({'return_to_start': 'yes'}, 'return_to_start'),
    ({'return_step_deg': 50}, 'return_step_deg'),
    ({'camera_fps': 0}, 'positive integers'),
    ({'camera_types': {'left_image': 'zed'}}, 'camera_types'),
    ({'camera_types': {'right_image': 'opencv'}}, 'camera_types'),
    ({'baudrate': 1}, 'Unknown SO101 settings'),
])
def test_invalid_settings_open_nothing(hardware, settings, match):
    with pytest.raises(ValueError, match=match):
        make_robot(config(**settings))
    assert hardware == []


def test_duplicate_cameras_open_nothing(hardware):
    with pytest.raises(ValueError, match='distinct camera'):
        make_robot(replace(config(), cameras={'head_image': '0', 'left_image': '0'}))
    assert hardware == []


def test_camera_type_can_be_set_explicitly(hardware):
    make_robot(config(camera_types={'head_image': 'opencv'})).close()
    assert hardware[0].cameras['head_image'][0] == 'opencv'


def test_malformed_driver_state_is_not_padded_and_releases(hardware, monkeypatch):
    factory = so101._open_follower
    def broken(*args):
        follower = factory(*args)
        follower.get_observation = lambda: {'shoulder_pan.pos': 0.0}
        return follower
    monkeypatch.setattr(so101, '_open_follower', broken)
    with pytest.raises(ValueError, match='missing shoulder_lift.pos'):
        make_robot(config())
    assert hardware[0].disconnected == 1 and not hardware[0].commands


def test_missing_lerobot_is_reported_before_opening_hardware(monkeypatch):
    import builtins
    original = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name.startswith('lerobot'):
            raise ImportError(name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', blocked)
    with pytest.raises(RuntimeError, match=r'colosseum-client\[so101\]'):
        make_robot(config())


def test_synthetic_so101_uses_same_split_without_loading_hardware(monkeypatch):
    def fail(*_):
        raise AssertionError('Hardware must not load in test mode')
    monkeypatch.setattr(so101, '_open_follower', fail)
    robot = make_robot(replace(config(), test=True))
    robot.configure_model({'action_dim': 6, 'action_space': 'joint_position'})
    assert robot.get_observation().joints.shape == (5,)
    robot.execute(np.r_[np.arange(5), 60.])
    obs = robot.get_observation()
    np.testing.assert_array_equal(obs.joints, np.arange(5))
    np.testing.assert_array_equal(obs.gripper, [60])


@pytest.mark.asyncio
@pytest.mark.parametrize('execute_action', [False, True])
async def test_local_trial_so101_contract_and_hardware_open_after_confirmation(tmp_path, monkeypatch, hardware, execute_action):
    from types import SimpleNamespace
    from colosseum_client import evaluation, local_policy
    from colosseum_client import colosseum_pb2 as pb
    from colosseum_client.tensors import tensor_from_numpy
    events = []
    class Client:
        def __init__(self, config, *args, **kwargs):
            assert kwargs['joint_count'] == 5
            assert kwargs['action_spaces'] == {'joint_position': 6}
            self.model = {'action_space': 'joint_position', 'action_dim': 6}
        async def connect(self, **kwargs):
            assert not hardware
            assert kwargs['before_start'](self.model, {}) is True
            return SimpleNamespace(action_spaces=['joint_position'])
        async def infer(self, obs, **kwargs):
            assert list(obs.state['joint_position'].shape) == [5]
            assert list(obs.state['gripper_position'].shape) == [1]
            return pb.ActionPlan(start_step=0, valid_until_step=0, control_hz=30,
                                 actions=tensor_from_numpy(hardware[0].position.astype(np.float32)[None]))
        async def before_action(self, step):
            pass
        async def finish(self):
            events.append('finish')
        async def close(self):
            events.append('close')
    monkeypatch.setattr(local_policy, 'LocalPolicyClient', Client)
    def confirm(*_):
        events.append('confirm')
        return True
    await evaluation.run_trial(replace(config(), recording=False), {'inference_mode': 'local', 'task': {
        'instruction': 'pick', 'cameras': ['head_image', 'left_image'], 'max_steps': 1}},
        {'id': 'so101-run'}, tmp_path, SimpleNamespace(), execute_action=execute_action, confirm_live=confirm)
    assert events == ['confirm', 'finish', 'close']
    assert hardware[0].disconnected == 1
    assert len(hardware[0].commands) == int(execute_action)


def test_transient_bus_error_during_connection_is_retried_and_released(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace
    created = []

    class Follower:
        def __init__(self, config):
            self.cameras, self.bus, self.released = {}, SimpleNamespace(disconnect=lambda: None), 0
            created.append(self)
        def connect(self, calibrate):
            assert calibrate is False
            if len(created) < 3:
                raise ConnectionError('[TxRxResult] Incorrect status packet!')
        is_calibrated = True
        def disconnect(self):
            self.released += 1

    modules = {name: ModuleType(name) for name in (
        'lerobot', 'lerobot.cameras', 'lerobot.cameras.opencv', 'lerobot.cameras.opencv.configuration_opencv',
        'lerobot.cameras.realsense', 'lerobot.cameras.realsense.configuration_realsense',
        'lerobot.robots', 'lerobot.robots.so_follower')}
    modules['lerobot.cameras.opencv.configuration_opencv'].OpenCVCameraConfig = lambda **kwargs: kwargs
    modules['lerobot.cameras.realsense.configuration_realsense'].RealSenseCameraConfig = lambda **kwargs: kwargs
    modules['lerobot.robots.so_follower'].SO101Follower = Follower
    modules['lerobot.robots.so_follower'].SO101FollowerConfig = lambda **kwargs: kwargs
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(so101.time, 'sleep', lambda _: None)
    follower = so101._open_follower('/dev/ttyACM0', 'arm', None, {'head_image': ('opencv', '0', 640, 480, 30)}, None)
    assert follower is created[2] and [f.released for f in created] == [1, 1, 0]

    created.clear()
    Follower.connect = lambda self, calibrate: (_ for _ in ()).throw(ConnectionError('bus down'))
    with pytest.raises(ConnectionError, match='bus down'):
        so101._open_follower('/dev/ttyACM0', 'arm', None, {}, None)
    assert len(created) == 3 and all(f.released == 1 for f in created)

    created.clear()
    Follower.connect = lambda self, calibrate: None
    Follower.is_calibrated = False
    with pytest.raises(RuntimeError, match='not calibrated'):
        so101._open_follower('/dev/ttyACM0', 'arm', None, {}, None)
    assert len(created) == 1 and created[0].released == 1


G05 = {'url': 'https://huggingface.co/OpenGalaxea/G05', 'subfolder': 'g05-so101', 'runtime_profile': 'g05-so101-v1'}
HOME = [3.1, -34.3, 31.5, 55.9, -12.3, 13.4]


@pytest.mark.parametrize('key', ['g05-so101-v1', 'g05-so101', 'https://huggingface.co/OpenGalaxea/G05'])
def test_start_pose_for_the_assigned_model_is_reached_in_bounded_steps(hardware, key):
    robot = make_robot(config(start_poses={key: HOME}, return_step_deg=2))
    assert robot.prepare_for_model(G05) == key
    steps = np.asarray(hardware[0].commands)
    np.testing.assert_allclose(steps[-1], HOME, atol=1e-5)
    assert np.max(np.abs(np.diff(np.vstack([[1, 2, 3, 4, 5, 40], steps]), axis=0))) <= 2 + 1e-6
    before = len(steps)
    robot.close()   # shutdown still returns to the pose measured at start-up
    np.testing.assert_allclose(hardware[0].commands[-1], [1, 2, 3, 4, 5, 40], atol=1e-5)
    assert len(hardware[0].commands) > before and hardware[0].disconnected == 1


def test_models_without_a_start_pose_do_not_move(hardware):
    robot = make_robot(config(start_poses={'g05-so101-v1': HOME}))
    assert robot.prepare_for_model({'url': 'https://huggingface.co/x/y', 'subfolder': '', 'runtime_profile': 'pi05-so101-v1'}) is None
    assert make_robot(config()).prepare_for_model(G05) is None
    assert all(not follower.commands for follower in hardware)


@pytest.mark.parametrize('poses,match', [
    ({'g05': [0, 0, 0]}, 'start_poses'),
    ({'g05': [0, 0, 0, 0, 0, float('nan')]}, 'start_poses'),
    ({'g05': [0, -150, 0, 0, 0, 10]}, 'outside the joint limits'),
    ({'g05': [0, 0, 0, 0, 0, 120]}, 'outside the joint limits'),
    ({'': HOME}, 'start_poses must map'),
    ([HOME], 'start_poses must map'),
])
def test_invalid_start_poses_open_nothing(hardware, poses, match):
    with pytest.raises(ValueError, match=match):
        make_robot(config(start_poses=poses))
    assert hardware == []


@pytest.mark.asyncio
@pytest.mark.parametrize('execute_action,moved', [(True, True), (False, False)])
async def test_local_trial_moves_to_the_model_start_pose_only_when_executing(tmp_path, monkeypatch, hardware, execute_action, moved):
    from types import SimpleNamespace
    from colosseum_client import evaluation, local_policy
    from colosseum_client import colosseum_pb2 as pb
    from colosseum_client.tensors import tensor_from_numpy
    seen = {}

    class Client:
        def __init__(self, config, *args, **kwargs):
            self.model = {'action_space': 'joint_position', 'action_dim': 6, **G05}
        async def connect(self, **kwargs):
            return SimpleNamespace(action_spaces=['joint_position'])
        async def infer(self, obs, **kwargs):
            seen['first_state'] = np.frombuffer(obs.state['joint_position'].data, np.float32).copy()
            return pb.ActionPlan(start_step=0, valid_until_step=0, control_hz=30,
                                 actions=tensor_from_numpy(hardware[0].position.astype(np.float32)[None]))
        async def before_action(self, step):
            pass
        async def finish(self):
            pass
        async def close(self):
            pass
    monkeypatch.setattr(local_policy, 'LocalPolicyClient', Client)
    cfg = replace(config(start_poses={'g05-so101-v1': HOME}), recording=False)
    await evaluation.run_trial(cfg, {'inference_mode': 'local', 'task': {
        'instruction': 'pick', 'cameras': ['head_image', 'left_image'], 'max_steps': 1}},
        {'id': 'so101-run'}, tmp_path, SimpleNamespace(), execute_action=execute_action, confirm_live=lambda *_: True)
    # The model's first observation is taken at the home pose only when the arm was allowed to move.
    np.testing.assert_allclose(seen['first_state'], HOME[:5] if moved else [1, 2, 3, 4, 5], atol=1e-4)
    assert hardware[0].disconnected == 1

