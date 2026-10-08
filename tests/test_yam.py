from dataclasses import replace
import json
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from colosseum_client import RobotClientConfig, make_robot
from colosseum_client.robots import yam


def test_close_waits_for_inflight_can_exchange():
    entered, stop_requested, exited = (threading.Event() for _ in range(3))
    events = []

    class Chain:
        running = True

        def exchange(self):
            entered.set()
            assert stop_requested.wait(2)
            # A final exchange may still be in flight when running becomes false.
            events.append('last_can_send')
            exited.set()

    chain = Chain()
    def socket_close():
        assert exited.is_set()
        events.append('socket_close')
    chain.motor_interface = SimpleNamespace(close=socket_close)
    def sdk_close():
        events.append('robot_loop_stopped')
        chain.running = False
        stop_requested.set()
        chain.motor_interface.close()
    arm = SimpleNamespace(motor_chain=chain, close=sdk_close)
    worker = threading.Thread(target=chain.exchange)
    worker.start()
    try:
        assert entered.wait(2)
        yam._close_arm(arm)
        assert not worker.is_alive()
        assert events == ['robot_loop_stopped', 'last_can_send', 'socket_close']
        assert chain.motor_interface.close is socket_close
    finally:
        stop_requested.set()
        worker.join(2)


def test_close_timeout_is_reported_and_socket_shutdown_attempted(monkeypatch):
    events = []
    chain = SimpleNamespace(running=True)
    def socket_close():
        events.append('socket_close')
    chain.motor_interface = SimpleNamespace(close=socket_close)
    worker = SimpleNamespace(name='stuck', _target=SimpleNamespace(__self__=chain),
        join=lambda **kwargs: events.append('join'), is_alive=lambda: True)
    monkeypatch.setattr(threading, 'enumerate', lambda: [worker])
    arm = SimpleNamespace(motor_chain=chain, close=lambda: chain.motor_interface.close())
    with pytest.raises(RuntimeError, match='did not stop'):
        yam._close_arm(arm)
    assert events == ['join', 'socket_close']
    assert chain.motor_interface.close is socket_close


def test_shutdown_failure_still_closes_other_arm_and_cameras(hardware, monkeypatch):
    robot = make_robot(config())
    original = yam._close_arm
    def fail_left(arm):
        if arm is robot.arms[0]:
            raise RuntimeError('shutdown failed')
        original(arm)
    monkeypatch.setattr(yam, '_close_arm', fail_left)
    with pytest.raises(RuntimeError, match='YAM resource shutdown failed'):
        robot.close()
    assert robot.arms[1].closed == 1
    assert all(camera.closed == 1 for camera in robot.cameras.values())


def config():
    return RobotClientConfig(url='ws://localhost', token='test', robot_type='yam', control_hz=30,
        cameras=dict(zip(yam.CAMERAS, ('top', 'left', 'right'))),
        adapter_config=dict(left_channel='can_left', right_channel='can_right',
                            left_gripper_limits=[0, 1], right_gripper_limits=[0, 1],
                            joint_low=[-3]*12, joint_high=[3]*12, joint_max_step=[.1]*12))


@pytest.mark.parametrize('position,normalized,running,valid', [
    (.1951247425, 1.185, True, False),
    (1.2, (1.2 - 6.49) / (1.18 - 6.49), True, True),
    (1.17, (1.17 - 6.49) / (1.18 - 6.49), True, True),
    (1.2, 1.0, False, False),
    (1.2, 2.0, True, False),
    (float('nan'), 1.0, True, False),
])
def test_sdk_startup_checks_gripper_before_position_hold(
        monkeypatch, capsys, position, normalized, running, valid):
    events = []
    current = np.r_[np.zeros(6), normalized]
    chain = SimpleNamespace(
        read_states=lambda: [SimpleNamespace(id=7, pos=position)],
        motor_offset=np.r_[np.zeros(6), -2*np.pi],
        motor_direction=np.ones(7), running=running)
    def command(value):
        events.append('hold')
        assert 0 <= value[6] <= 1
        np.testing.assert_array_equal(value[:6], current[:6])
    arm = SimpleNamespace(motor_chain=chain, get_joint_pos=lambda: current.copy(),
                          command_joint_pos=command, close=lambda: events.append('close'))
    def factory(**kwargs):
        assert kwargs['zero_gravity_mode'] is True
        np.testing.assert_array_equal(kwargs['gripper_limits_override'], [6.49, 1.18])
        events.append('gravity_start')
        return arm
    monkeypatch.setitem(sys.modules, 'i2rt.robots.get_robot', SimpleNamespace(get_yam_robot=factory))
    monkeypatch.setitem(sys.modules, 'i2rt.robots.utils',
                        SimpleNamespace(GripperType=SimpleNamespace(LINEAR_4310='linear_4310')))
    if valid:
        assert yam._open_arm('can_left', [6.49, 1.18]) is arm
        assert events == ['gravity_start', 'hold']
        entry = json.loads(capsys.readouterr().out)
        assert entry['motor_offset_rad'] == pytest.approx(-2*np.pi)
        assert entry['measured_position_rad'] == position
    else:
        with pytest.raises((RuntimeError, ValueError)):
            yam._open_arm('can_left', [6.49, 1.18])
        assert events == ['gravity_start', 'close']


@pytest.fixture
def hardware(monkeypatch):
    opened = []

    class Camera:
        def __init__(self, serial, timeout):
            self.serial, self.closed = serial, 0
            opened.append(self)
        def read(self):
            return np.full((2, 3, 3), ('top', 'left', 'right').index(self.serial), np.uint8)
        def close(self):
            self.closed += 1

    class Arm:
        def __init__(self, channel, limits):
            self.channel, self.closed, self.commands = channel, 0, []
            self.position = np.r_[np.arange(6)*.01, .2 if channel == 'can_left' else .8]
            opened.append(self)
        def get_joint_pos(self):
            return self.position.copy()
        def command_joint_pos(self, value):
            self.commands.append(value.copy())
        def close(self):
            self.closed += 1

    monkeypatch.setattr(yam, '_RealSenseCamera', Camera)
    monkeypatch.setattr(yam, '_open_arm', Arm)
    return opened


@pytest.mark.parametrize('runtime_offset', [0., -2*np.pi, 2*np.pi])
@pytest.mark.parametrize('direction', [1., -1.])
@pytest.mark.parametrize('reference_error', [0., 100.])
def test_calibration_offset_preserves_physical_endpoints(
        monkeypatch, capsys, runtime_offset, direction, reference_error):
    raw_limits = np.array([.20695048447394493, -5.102807660028995])
    calibration_offset = -2*np.pi
    calibrated = (raw_limits - calibration_offset) * direction
    original = calibrated.copy()
    raw_position = .0211718929
    measured = (raw_position - runtime_offset) * direction
    states = [SimpleNamespace(id=i+1, pos=.01*i) for i in range(6)]
    states.append(SimpleNamespace(id=7, pos=measured))
    class Mapper:
        def __init__(self, index_range_map, total_dofs):
            assert total_dofs == 7
            self.limits = np.array(index_range_map[6], copy=True)
        def to_command_joint_pos_space(self, values):
            value = np.array(values, copy=True)
            value[6] = (value[6]-self.limits[0])/(self.limits[1]-self.limits[0])
            return value
        def to_robot_joint_pos_space(self, values):
            value = np.array(values, copy=True)
            value[6] = self.limits[0]+value[6]*(self.limits[1]-self.limits[0])
            return value
    class Chain:
        running = True
        motor_offset = np.r_[np.zeros(6), runtime_offset]
        motor_direction = np.r_[np.ones(6), direction]
        def __len__(self): return 7
        def read_states(self): return states
    class Arm:
        motor_chain = Chain()
        _gripper_index = 6
        _command_lock = threading.Lock()
        _state_lock = threading.Lock()
        _gripper_limits = calibrated.copy()
        remapper = Mapper({6: calibrated}, 7)
        _commands = SimpleNamespace(**{name: np.zeros(7) for name in ('kp', 'kd', 'torques', 'pos')})
        _gripper_force_limiter = SimpleNamespace(_gripper_adjusted_qpos=None, _is_clogged=False)
        closed = False
        def _motor_state_to_joint_state(self, values):
            return SimpleNamespace(pos=self.remapper.to_command_joint_pos_space([s.pos for s in values]))
        def get_joint_pos(self): return self._joint_state.pos.copy()
        def command_joint_pos(self, values):
            self.target = self.remapper.to_robot_joint_pos_space(values)
            self._commands.kp[6] = 8  # Hold starts only after the rebase/check.
        def close(self): self.closed = True
    arm = Arm()
    arm._joint_state = arm._motor_state_to_joint_state(states)
    def factory(**kwargs):
        assert kwargs['zero_gravity_mode'] is True
        return arm
    monkeypatch.setitem(sys.modules, 'i2rt.robots.get_robot', SimpleNamespace(get_yam_robot=factory))
    monkeypatch.setitem(sys.modules, 'i2rt.robots.utils', SimpleNamespace(
        GripperType=SimpleNamespace(LINEAR_4310='linear_4310'), JointMapper=Mapper))
    if reference_error:
        with pytest.raises(RuntimeError, match='startup mismatch'):
            yam._open_arm('can_left', calibrated,
                          calibration_offset=calibration_offset + reference_error)
        assert arm.closed
        assert not hasattr(arm, 'target')
        return
    result = yam._open_arm('can_left', calibrated, calibration_offset=calibration_offset)
    assert result is arm and not arm.closed
    np.testing.assert_allclose(arm.target[:6], [s.pos for s in states[:6]])
    assert arm.target[6]*direction + runtime_offset == pytest.approx(raw_position)
    np.testing.assert_allclose(arm._gripper_limits*direction + runtime_offset, raw_limits)
    for opening in (0., .5, 1.):
        target = arm.remapper.to_robot_joint_pos_space(np.r_[np.zeros(6), opening])[6]
        assert target*direction + runtime_offset == pytest.approx(
            raw_limits[0] + opening*(raw_limits[1]-raw_limits[0]))
    np.testing.assert_array_equal(calibrated, original)
    entry = json.loads(capsys.readouterr().out)
    assert entry['calibration_offset_rad'] == calibration_offset
    assert entry['motor_offset_rad'] == runtime_offset
    with pytest.raises(RuntimeError, match='control is active'):
        yam._rebase_gripper_limits(arm, calibrated)


@pytest.mark.parametrize('offset', [float('nan'), float('inf'), True, '-6.28'])
def test_invalid_calibration_offset_opens_nothing(hardware, offset):
    cfg = config()
    with pytest.raises(ValueError, match='calibration_offset'):
        make_robot(replace(cfg, adapter_config={**cfg.adapter_config,
            'left_gripper_calibration_offset': offset}))
    assert hardware == []


def test_calibration_offset_is_routed_to_correct_arm(hardware, monkeypatch):
    factory = yam._open_arm
    calls = []
    def record(channel, limits, **kwargs):
        calls.append((channel, kwargs))
        return factory(channel, limits)
    monkeypatch.setattr(yam, '_open_arm', record)
    cfg = config()
    robot = make_robot(replace(cfg, adapter_config={**cfg.adapter_config,
        'left_gripper_calibration_offset': -2*np.pi}))
    assert calls == [('can_left', {'calibration_offset': -2*np.pi}), ('can_right', {})]
    robot.close()


def test_state_and_action_order_and_close(hardware):
    robot = make_robot(config())
    obs = robot.get_observation()
    np.testing.assert_allclose(obs.joints, np.tile(np.arange(6)*.01, 2))
    np.testing.assert_allclose(obs.gripper, [.2, .8])
    assert obs.cartesian_position.size == 0
    assert obs.images['right_image'][0, 0, 0] == 2
    target = np.r_[obs.joints[:6]+.02, .3, obs.joints[6:]-.03, .7]
    robot.execute(target)
    np.testing.assert_allclose(robot.arms[0].commands[0], target[:7])
    np.testing.assert_allclose(robot.arms[1].commands[0], target[7:])
    robot.close()
    robot.close()
    assert all(item.closed == 1 for item in hardware)
    with pytest.raises(RuntimeError, match='closed'):
        robot.get_observation()


def _temperature_chain(channel, running=True):
    state = SimpleNamespace(id=7, temp_mos=61.0, temp_rotor=64.0,
                            eff=.25, pos=1.2, vel=0.0, error_code='0x1')
    # ID 7 need not be the last element in a diagnostic snapshot.
    return SimpleNamespace(channel=channel, running=running,
                           read_states=lambda: [state, SimpleNamespace(id=1)])


def test_gripper_temperatures_read_sdk_cache_and_mark_stopped_chain(hardware, capsys):
    robot = make_robot(config())
    for arm in robot.arms:
        arm.motor_chain = _temperature_chain(arm.channel)
    robot.arms[0].motor_chain.running = False
    robot._print_gripper_telemetry()
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [r['side'] for r in rows] == ['left', 'right']
    assert [r['channel'] for r in rows] == ['can_left', 'can_right']
    assert rows[0]['source'] == 'sdk_feedback_cache'
    assert rows[0]['chain_running'] is False
    assert rows[0]['possibly_stale'] is True
    assert rows[1]['chain_running'] is True
    assert rows[0]['mos_temperature_c'] == 61
    assert rows[0]['rotor_temperature_c'] == 64
    assert rows[0]['effort_nm'] == .25
    assert all(not arm.commands for arm in robot.arms)
    robot.close()


def test_gripper_telemetry_failure_does_not_hide_other_arm(hardware, capsys, caplog):
    robot = make_robot(config())
    robot.arms[0].motor_chain = _temperature_chain('can_left')
    def failed():
        raise RuntimeError('feedback unavailable')
    robot.arms[0].motor_chain.read_states = failed
    robot.arms[1].motor_chain = _temperature_chain('can_right')
    robot._print_gripper_telemetry()
    assert 'left gripper telemetry unavailable' in caplog.text
    assert json.loads(capsys.readouterr().out)['side'] == 'right'
    assert not robot.closed
    robot.close()


def test_gripper_monitor_runs_without_actions_and_stops_before_sdk_close(hardware, monkeypatch):
    robot = make_robot(config())
    for arm in robot.arms:
        arm.motor_chain = _temperature_chain(arm.channel)
    sampled = threading.Event()
    samples = []
    def report():
        samples.append(1)
        if len(samples) >= 2:
            sampled.set()
    monkeypatch.setattr(robot, '_print_gripper_telemetry', report)
    monkeypatch.setattr(yam, '_GRIPPER_LOG_INTERVAL', .01)
    original_close = yam._close_arm
    def close_after_monitor(arm):
        assert not robot._telemetry_thread.is_alive()
        original_close(arm)
    monkeypatch.setattr(yam, '_close_arm', close_after_monitor)
    robot._start_gripper_telemetry()
    try:
        assert sampled.wait(2)  # No observe()/execute() calls are needed.
        assert all(not arm.commands for arm in robot.arms)
    finally:
        robot.close()
    assert not robot._telemetry_thread.is_alive()
    assert all(resource.closed == 1 for resource in hardware)


@pytest.mark.parametrize('index,value', [(0, 4), (7, .3), (6, 1.1), (13, -.1), (1, float('nan'))])
def test_invalid_action_never_moves_either_arm(hardware, index, value):
    robot = make_robot(config())
    action = np.r_[robot.arms[0].position, robot.arms[1].position]
    action[index] = value
    with pytest.raises(ValueError):
        robot.execute(action)
    assert all(not arm.commands for arm in robot.arms)
    robot.close()


def test_invalid_settings_open_nothing(hardware):
    with pytest.raises(ValueError, match='joint_low'):
        make_robot(replace(config(), adapter_config={}))
    assert hardware == []


def test_clip_uses_fresh_feedback_preserves_grippers_and_policy_action(hardware, caplog):
    cfg = config()
    cfg = replace(cfg, adapter_config={**cfg.adapter_config,
        'joint_step_mode': 'clip', 'joint_max_step': [.01]*12})
    robot = make_robot(cfg)
    for arm in robot.arms:
        arm.position[:6] = 0
    action = np.array([.16, -.16, .005, 0, 0, 0, .3,
                       -.16, .16, -.005, 0, 0, 0, .7])
    original = action.copy()
    robot.execute(action)
    np.testing.assert_allclose(robot.arms[0].commands[-1], [.01, -.01, .005, 0, 0, 0, .3])
    np.testing.assert_allclose(robot.arms[1].commands[-1], [-.01, .01, -.005, 0, 0, 0, .7])
    # A stalled arm must not accumulate larger targets on successive calls.
    robot.execute(action)
    for arm in robot.arms:
        np.testing.assert_array_equal(arm.commands[-1], arm.commands[-2])
    robot.arms[0].position[0] = .008
    robot.execute(action)
    assert robot.arms[0].commands[-1][0] == pytest.approx(.018)
    np.testing.assert_array_equal(action, original)
    assert 'left_joint1: measured=0.000000, target=0.160000' in caplog.text
    assert 'right_joint1' in caplog.text
    robot.close()


@pytest.mark.parametrize('index,value', [(0, 4), (7, -4), (6, 1.1), (13, -.1), (1, float('nan'))])
def test_clip_keeps_invalid_actions_from_moving_either_arm(hardware, index, value):
    cfg = config()
    robot = make_robot(replace(cfg, adapter_config={**cfg.adapter_config, 'joint_step_mode': 'clip'}))
    action = np.r_[robot.arms[0].position, robot.arms[1].position]
    action[index] = value
    with pytest.raises(ValueError):
        robot.execute(action)
    assert all(not arm.commands for arm in robot.arms)
    robot.close()


def test_clip_rejects_measured_position_outside_absolute_limits(hardware):
    cfg = config()
    robot = make_robot(replace(cfg, adapter_config={**cfg.adapter_config, 'joint_step_mode': 'clip'}))
    action = np.r_[robot.arms[0].position, robot.arms[1].position]
    robot.arms[1].position[0] = 3.01
    action[7] = 3.
    with pytest.raises(ValueError, match='measured position exceeds joint limits'):
        robot.execute(action)
    assert all(not arm.commands for arm in robot.arms)
    robot.close()


def test_invalid_step_mode_opens_no_hardware(hardware):
    cfg = config()
    with pytest.raises(ValueError, match='joint_step_mode'):
        make_robot(replace(cfg, adapter_config={**cfg.adapter_config, 'joint_step_mode': 'ignore'}))
    assert hardware == []


@pytest.mark.asyncio
async def test_interpolation_bounds_commands_and_reaches_original_target(hardware, monkeypatch):
    cfg = config()
    robot = make_robot(replace(cfg, adapter_config={**cfg.adapter_config,
        'joint_step_mode': 'interpolate', 'joint_max_step': [.01]*12}))
    starts = np.r_[robot.arms[0].position, robot.arms[1].position]
    target = starts.copy()
    target[0] += .1599
    target[7] -= .1221
    target[6] += .05
    original = target.copy()
    waits = []
    async def call(fn, *args): return fn(*args)
    async def sleep(delay): waits.append(delay)
    monkeypatch.setattr(yam.asyncio, 'sleep', sleep)
    await robot.execute_async(target, call)
    commands = np.c_[robot.arms[0].commands, robot.arms[1].commands]
    assert commands.shape == (16, 14)
    differences = np.abs(np.diff(np.vstack([starts, commands]), axis=0))
    assert np.max(differences) <= .01 + 1e-12
    np.testing.assert_allclose(commands[-1], original)
    np.testing.assert_array_equal(target, original)
    assert len(waits) == 16 and all(0 <= d <= 1/30 for d in waits)
    # Planning a later action must start from SDK feedback, not prior targets.
    robot.arms[0].position[0] = -.1
    points = robot._interpolation(target)
    assert points[0, 0] < -.09 + 1e-12
    robot.close()


@pytest.mark.asyncio
async def test_interpolation_cancellation_stops_subsequent_commands(hardware, monkeypatch):
    cfg = config()
    robot = make_robot(replace(cfg, adapter_config={**cfg.adapter_config,
        'joint_step_mode': 'interpolate', 'joint_max_step': [.01]*12}))
    target = np.r_[robot.arms[0].position, robot.arms[1].position]
    target[0] += .16
    async def call(fn, *args): return fn(*args)
    async def cancel(_): raise yam.asyncio.CancelledError
    monkeypatch.setattr(yam.asyncio, 'sleep', cancel)
    with pytest.raises(yam.asyncio.CancelledError):
        await robot.execute_async(target, call)
    assert all(len(arm.commands) == 1 for arm in robot.arms)
    robot.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('index,value', [(0, 4), (7, -4), (6, 1.1), (1, float('nan')), (0, 1.5)])
async def test_interpolation_rejects_invalid_or_excessive_plan_before_moving(hardware, index, value):
    cfg = config()
    robot = make_robot(replace(cfg, adapter_config={**cfg.adapter_config,
        'joint_step_mode': 'interpolate', 'joint_max_step': [.01]*12}))
    target = np.r_[robot.arms[0].position, robot.arms[1].position]
    target[index] = value
    async def call(fn, *args): return fn(*args)
    with pytest.raises(ValueError):
        await robot.execute_async(target, call)
    assert all(not arm.commands for arm in robot.arms)
    robot.close()


@pytest.mark.asyncio
async def test_async_execution_dispatch_and_disabled_mode(hardware, monkeypatch):
    from colosseum_client.diagnostics import execute_robot_action_async
    cfg = config()
    robot = make_robot(replace(cfg, adapter_config={**cfg.adapter_config,
        'joint_step_mode': 'interpolate', 'joint_max_step': [.01]*12}))
    action = np.r_[robot.arms[0].position, robot.arms[1].position]
    action[0] += .16
    async def call(fn, *args): return fn(*args)
    async def sleep(_): pass
    monkeypatch.setattr(yam.asyncio, 'sleep', sleep)
    await execute_robot_action_async(robot, action, 0, call, False)
    assert all(not arm.commands for arm in robot.arms)
    await execute_robot_action_async(robot, action, 0, call, True)
    assert all(len(arm.commands) == 16 for arm in robot.arms)
    robot.close()


def test_second_arm_failure_closes_first_arm_and_cameras(hardware, monkeypatch):
    factory = yam._open_arm
    def fail(channel, limits):
        if channel == 'can_right':
            raise RuntimeError('CAN unavailable')
        return factory(channel, limits)
    monkeypatch.setattr(yam, '_open_arm', fail)
    with pytest.raises(RuntimeError, match='CAN unavailable'):
        make_robot(config())
    assert len(hardware) == 4 and all(item.closed == 1 for item in hardware)


def test_command_failure_closes_both_arms(hardware):
    robot = make_robot(config())
    def fail(_):
        raise RuntimeError('CAN lost')
    robot.arms[1].command_joint_pos = fail
    with pytest.raises(RuntimeError, match='CAN lost'):
        robot.execute(np.r_[robot.arms[0].position, robot.arms[1].position])
    assert all(item.closed == 1 for item in hardware)


def test_malformed_sdk_state_is_not_padded(hardware):
    robot = make_robot(config())
    robot.arms[0].position = np.zeros(6)
    with pytest.raises(ValueError, match='7 finite'):
        robot.get_observation()
    robot.close()


def test_gripper_feedback_saturates_without_mutating_sdk_or_joints(hardware, caplog):
    robot = make_robot(config())
    robot.arms[0].position[6] = -.02
    robot.arms[1].position[6] = 1.03
    for arm in robot.arms:
        arm.get_joint_pos = lambda arm=arm: arm.position
    original = [arm.position.copy() for arm in robot.arms]
    for _ in range(2):
        obs = robot.get_observation()
        np.testing.assert_array_equal(obs.gripper, [0, 1])
        np.testing.assert_allclose(obs.joints, np.r_[original[0][:6], original[1][:6]])
    assert len(caplog.records) == 4
    assert 'left_gripper measured normalized value=-0.020000' in caplog.text
    assert 'right_gripper measured normalized value=1.030000' in caplog.text
    for arm, value in zip(robot.arms, original):
        np.testing.assert_array_equal(arm.position, value)
    # Interpolation and return-home read the same bounded gripper state.
    robot._validated_action(np.concatenate(robot._read_positions()))
    robot.close()


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_gripper_feedback_is_not_clipped(hardware, bad):
    robot = make_robot(config())
    robot.arms[1].position[6] = bad
    with pytest.raises(ValueError, match='7 finite'):
        robot.get_observation()
    robot.close()


def test_joint_observation_and_action_bounds_are_logged(hardware, caplog):
    robot = make_robot(config())
    robot.arms[1].position[2] = 3.25
    robot.get_observation()
    assert 'right_joint3: value=3.250000, low=-3.000000, high=3.000000' in caplog.text
    caplog.clear()
    with pytest.raises(ValueError, match='action exceeds joint limits'):
        robot.execute(np.r_[robot.arms[0].position, robot.arms[1].position])
    assert 'right_joint3: value=3.250000' in caplog.text
    robot.close()


def test_synthetic_yam_uses_same_split_without_loading_hardware(monkeypatch):
    def fail(*_):
        raise AssertionError('Hardware must not load in test mode')
    monkeypatch.setattr(yam, '_open_arm', fail)
    robot = make_robot(replace(config(), test=True))
    robot.configure_model({'action_dim': 14, 'action_space': 'joint_position'})
    action = np.r_[np.arange(6), .25, np.arange(6,12), .75]
    robot.execute(action)
    obs = robot.get_observation()
    np.testing.assert_array_equal(obs.joints, np.arange(12))
    np.testing.assert_array_equal(obs.gripper, [.25, .75])


@pytest.mark.asyncio
@pytest.mark.parametrize('infer_error', [False, True, 'cancel'])
@pytest.mark.parametrize('execute_action', [False, True])
@pytest.mark.parametrize('return_error', [False, True])
async def test_local_trial_yam_contract_and_hardware_open_after_confirmation(tmp_path, monkeypatch, hardware, execute_action, infer_error, return_error):
    from types import SimpleNamespace
    from colosseum_client import evaluation, local_policy
    from colosseum_client import colosseum_pb2 as pb
    from colosseum_client.tensors import tensor_from_numpy
    events = []
    class Client:
        def __init__(self, config, *args, **kwargs):
            assert kwargs['joint_count'] == 12
            assert kwargs['action_spaces'] == {'joint_position': 14}
            self.model = {'action_space': 'joint_position', 'action_dim': 14}
        async def connect(self, **kwargs):
            assert not hardware
            assert kwargs['before_start'](self.model, {}) is True
            return SimpleNamespace(action_spaces=['joint_position'])
        async def infer(self, obs, **kwargs):
            if infer_error == 'cancel':
                raise yam.asyncio.CancelledError('inference failed')
            if infer_error:
                raise RuntimeError('inference failed')
            assert len(hardware) == 5
            assert list(obs.state['joint_position'].shape) == [12]
            assert list(obs.state['gripper_position'].shape) == [2]
            target = np.r_[hardware[3].position, hardware[4].position].astype(np.float32)[None]
            return pb.ActionPlan(start_step=0, valid_until_step=0, control_hz=30,
                                 actions=tensor_from_numpy(target))
        async def before_action(self, step):
            pass
        async def finish(self):
            events.append('finish')
        async def close(self):
            events.append('close')
    monkeypatch.setattr(local_policy, 'LocalPolicyClient', Client)
    async def finish_robot(self, robot_call):
        assert all(not arm.closed for arm in self.arms)
        events.append('park')
        if return_error:
            raise RuntimeError('return failed')
    async def hold(self, robot_call):
        assert all(not arm.closed for arm in self.arms)
        events.append('hold_and_retry')
    monkeypatch.setattr(yam.YAMRobot, 'finish_trial', finish_robot)
    monkeypatch.setattr(yam.YAMRobot, 'hold_for_recovery', hold)
    cfg = replace(config(), recording=False)
    def confirm(*_):
        events.append('confirm')
        return True
    if infer_error:
        error_type = yam.asyncio.CancelledError if infer_error == 'cancel' else RuntimeError
        with pytest.raises(error_type, match='inference failed'):
            await evaluation.run_trial(cfg, {'inference_mode': 'local', 'task': {
                'instruction': 'pick', 'cameras': list(yam.CAMERAS), 'max_steps': 1}},
                {'id': 'yam-run'}, tmp_path, SimpleNamespace(), execute_action=execute_action, confirm_live=confirm)
        assert events == ['confirm', *(['park'] if execute_action else []),
                          *(['hold_and_retry'] if execute_action and return_error else []), 'close']
    elif execute_action and return_error:
        with pytest.raises(RuntimeError, match='return failed'):
            await evaluation.run_trial(cfg, {'inference_mode': 'local', 'task': {
                'instruction': 'pick', 'cameras': list(yam.CAMERAS), 'max_steps': 1}},
                {'id': 'yam-run'}, tmp_path, SimpleNamespace(), execute_action=execute_action, confirm_live=confirm)
        assert events == ['confirm', 'park', 'hold_and_retry', 'close']
    else:
        await evaluation.run_trial(cfg, {'inference_mode': 'local', 'task': {
            'instruction': 'pick', 'cameras': list(yam.CAMERAS), 'max_steps': 1}},
            {'id': 'yam-run'}, tmp_path, SimpleNamespace(), execute_action=execute_action, confirm_live=confirm)
        assert events == ['confirm', *(['park'] if execute_action else []), 'finish', 'close']
    assert all(resource.closed == 1 for resource in hardware)
    assert len(hardware[3].commands) == int(execute_action and not infer_error)


@pytest.mark.asyncio
async def test_failed_return_holds_power_until_operator_retry(hardware, monkeypatch):
    robot = make_robot(config())
    starts = [arm.position.copy() for arm in robot.arms]
    checks = iter([None, None, 'retry'])
    monkeypatch.setattr(robot, '_zero_recovery_choice', lambda: next(checks))
    async def call(fn, *args): return fn(*args)
    async def sleep(_):
        assert all(arm.closed == 0 for arm in robot.arms)
        for arm, start in zip(robot.arms, starts):
            np.testing.assert_array_equal(arm.commands[-1], start)
    retried = []
    async def finish(_): retried.append(True)
    monkeypatch.setattr(yam.asyncio, 'sleep', sleep)
    monkeypatch.setattr(robot, 'finish_trial', finish)
    await robot.hold_for_recovery(call)
    assert retried == [True]
    assert all(arm.closed == 0 for arm in robot.arms)
    robot.close()


@pytest.mark.asyncio
async def test_operator_can_skip_failed_return_without_retry(hardware, monkeypatch, capsys):
    robot = make_robot(config())
    monkeypatch.setattr(robot, '_zero_recovery_choice', lambda: 'skip')
    async def call(fn, *args): return fn(*args)
    async def forbidden(_): pytest.fail('Skip must not retry return')
    monkeypatch.setattr(robot, 'finish_trial', forbidden)
    await robot.hold_for_recovery(call)
    assert not robot.closed  # Caller owns final shutdown, after explicit release.
    assert 'zero position NOT confirmed' in capsys.readouterr().out
    robot.close()
    assert all(arm.closed == 1 for arm in robot.arms)


@pytest.mark.parametrize('line,expected', [('\n', 'retry'), ('s\n', 'skip'),
    ('skip\n', 'skip'), ('wrong\n', None), ('', None)])
def test_zero_recovery_input(line, expected, monkeypatch):
    import io
    stream = io.StringIO(line)
    monkeypatch.setattr(stream, 'isatty', lambda: True)
    monkeypatch.setattr(yam.sys, 'stdin', stream)
    monkeypatch.setattr(yam.select, 'select', lambda *args: ([stream], [], []))
    assert yam.YAMRobot._zero_recovery_choice() == expected


@pytest.mark.parametrize('value', [0, -.01, .2, float('nan'), float('inf'), True, '0.05'])
def test_invalid_zero_tolerance_opens_no_hardware(hardware, value):
    cfg = config()
    with pytest.raises(ValueError, match='zero_position_tolerance'):
        make_robot(replace(cfg, adapter_config={**cfg.adapter_config, 'zero_position_tolerance': value}))
    assert hardware == []


@pytest.mark.asyncio
async def test_default_zero_tolerance_accepts_reported_residual(hardware, monkeypatch):
    robot = make_robot(config())
    for arm in robot.arms:
        arm.position[:6] = 0
    robot.arms[1].position[3] = -.027657
    async def call(fn, *args): return fn(*args)
    async def sleep(_): pass
    monkeypatch.setattr(yam.asyncio, 'sleep', sleep)
    await robot.finish_trial(call)
    assert robot.zero_position_tolerance == .05
    np.testing.assert_array_equal(robot.arms[1].commands[-1][:6], np.zeros(6))
    # Still requires three confirmations after sending zero, not just one.
    assert sum(np.all(c[:6] == 0) for c in robot.arms[1].commands) >= 3
    robot.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('max_step', [.001, .01, .05])
async def test_finish_zero_ramp_preserves_grippers_and_limits(hardware, monkeypatch, max_step):
    cfg = config()
    cfg = replace(cfg, adapter_config={**cfg.adapter_config, 'joint_max_step': [max_step]*12})
    robot = make_robot(cfg)
    starts = [arm.position.copy() for arm in robot.arms]
    for arm in robot.arms:
        original = arm.command_joint_pos
        def follow(value, arm=arm, original=original):
            original(value)
            arm.position = value.copy()
        arm.command_joint_pos = follow
    async def call(fn, *args): return fn(*args)
    async def sleep(_): pass
    monkeypatch.setattr(yam.asyncio, 'sleep', sleep)
    await robot.finish_trial(call)
    for arm, start in zip(robot.arms, starts):
        actions = np.vstack([start, *arm.commands])
        assert np.max(np.abs(np.diff(actions[:, :6], axis=0))) <= min(max_step, .15/30) + 1e-9
        np.testing.assert_array_equal(actions[:, 6], start[6])
        np.testing.assert_array_equal(arm.position[:6], np.zeros(6))
        assert arm.closed == 0
    robot.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['reject', 'clip', 'interpolate'])
async def test_zero_ramp_reaches_zero_despite_feedback_deadband(hardware, monkeypatch, mode):
    cfg = config()
    robot = make_robot(replace(cfg, adapter_config={**cfg.adapter_config,
        'joint_step_mode': mode, 'joint_max_step': [.01]*12}))
    starts = []
    for arm in robot.arms:
        arm.position[:6] = [-.06, .06, -.04, .04, -.02, .02]
        starts.append(arm.position.copy())
        original = arm.command_joint_pos
        def lag(value, arm=arm, original=original):
            # A simulated deadband larger than the old half-step guard would
            # keep that ramp stuck forever, even though larger targets work.
            original(value)
            error = value[:6] - arm.position[:6]
            arm.position[:6] += np.where(np.abs(error) > .012, .25 * error, 0)
        arm.command_joint_pos = lag
    now = [0.]
    monkeypatch.setattr(yam.time, 'monotonic', lambda: now[0])
    async def call(fn, *args): return fn(*args)
    async def sleep(delay): now[0] += delay
    monkeypatch.setattr(yam.asyncio, 'sleep', sleep)
    await robot.finish_trial(call)
    assert now[0] < 60
    for arm, start in zip(robot.arms, starts):
        commands = np.vstack([start, *arm.commands])
        assert len(arm.commands) >= 14  # Ramp plus measured settling checks.
        assert np.max(np.abs(np.diff(commands[:, :6], axis=0))) <= .005 + 1e-12
        assert np.all(np.diff(np.abs(commands[:, :6]), axis=0) <= 1e-12)
        np.testing.assert_array_equal(commands[:, 6], start[6])
        np.testing.assert_array_equal(commands[-1, :6], np.zeros(6))
        assert np.all(np.abs(arm.position[:6]) <= .02)
    robot.close()


@pytest.mark.asyncio
async def test_finish_zero_timeout_does_not_report_success(hardware, monkeypatch, capsys):
    cfg = config()
    cfg = replace(cfg, adapter_config={**cfg.adapter_config, 'joint_max_step': [.01]*12,
                                      'joint_step_mode': 'interpolate', 'zero_position_tolerance': .02})
    robot = make_robot(cfg)
    now = [0.]
    monkeypatch.setattr(yam.time, 'monotonic', lambda: now[0])
    async def call(fn, *args): return fn(*args)
    async def sleep(delay): now[0] += delay
    monkeypatch.setattr(yam.asyncio, 'sleep', sleep)
    with pytest.raises(RuntimeError, match='timed out') as error:
        await robot.finish_trial(call)
    assert 'zero_target_sent=True' in str(error.value)
    assert 'left_joint6: measured=0.050000, target=0.000000' in str(error.value)
    assert 'right_joint6' in str(error.value)
    assert 'zero position reached' not in capsys.readouterr().out
    for arm in robot.arms:
        np.testing.assert_array_equal(arm.commands[-1][:6], np.zeros(6))
    robot.close()


def test_zero_outside_limits_rejected_before_hardware(hardware):
    cfg = config()
    with pytest.raises(ValueError, match='zero within'):
        make_robot(replace(cfg, adapter_config={**cfg.adapter_config,
            'joint_low': [.1]*12}))
    assert hardware == []


@pytest.mark.asyncio
async def test_zero_ramp_cancellation_stops_commands(hardware, monkeypatch):
    import asyncio
    cfg = config()
    robot = make_robot(cfg)
    async def call(fn, *args): return fn(*args)
    async def cancel(_): raise asyncio.CancelledError
    monkeypatch.setattr(yam.asyncio, 'sleep', cancel)
    with pytest.raises(asyncio.CancelledError):
        await robot.finish_trial(call)
    assert all(len(arm.commands) == 1 for arm in robot.arms)
    robot.close()
