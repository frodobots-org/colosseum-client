# Bimanual YAM with MolmoAct2

This Client implementation targets **two six-joint YAM arms with LINEAR_4310
grippers**, driven by I2RT, and three directly connected RealSense cameras.
It does not support single-arm YAM, other gripper variants, or upstream's ZMQ
camera service. Software tests use fake hardware; CAN, camera timing, motor
shutdown, calibration, GPU inference and closed-loop task performance still
require validation on your rig.

Reference: [MolmoAct2 YAM example](https://github.com/allenai/molmoact2/tree/6070080a20321b4f498ab30f28e1d09ac465edb7/examples/yam),
and [I2RT SDK](https://github.com/i2rt-robotics/i2rt/tree/120c3c81400171174604e503943f8d1ebc891058).
These are the source revisions inspected for this integration.

## Contract

- `robot_type: yam`, absolute `joint_position`, 14 values, 30 Hz.
- Action/state model order: `[left_joint_1..6, left_gripper, right_joint_1..6, right_gripper]`.
- Joint angles are radians. The I2RT gripper remapper uses **0 closed, 1 open**;
  there is no inversion or binarization in this adapter.
- Transport observations split `joint_position` into 12 values (left then right)
  and `gripper_position` into two values (left then right). The Policy Server
  interleaves them before inference. LeRobot export uses the same interleaved
  order for `observation.state` and `action`, with named dimensions.
- `head_image` maps to model `top_cam`, `left_image` to `left_cam`, and
  `right_image` to `right_cam`, in that order. Native capture is RGB uint8,
  640x480, 30 FPS; the driver uses these native dimensions regardless of the
  generic Client image_width/image_height settings. Cameras are read
  sequentially, not hardware synchronized.
- Cartesian pose is unavailable: the driver returns an empty vector and export
  omits `observation.cartesian_position`. It never fabricates a zero pose.

## Install and configure the robot host

From the `colosseum-client` checkout:

```bash
uv sync --extra yam
# I2RT is installed separately because its native/motor dependencies are rig-specific.
# Its ruckig source build requires the upstream build constraint shown here.
uv pip install --python .venv/bin/python --build-constraint <(echo 'scikit-build-core<0.10') \
  'i2rt @ git+https://github.com/i2rt-robotics/i2rt.git@120c3c81400171174604e503943f8d1ebc891058'
cp configs/robot.yam.yaml.example configs/robot.yam.yaml
```

Native I2RT dependencies may require system build tools. The install command
above is a reference setup, not a hardware installation tested by this change.
Use `.venv/bin/colosseum-robot` (or `uv run --no-sync`) afterwards so a sync does
not remove the separately installed SDK. Install ffmpeg for recording/export.

Fill in the Router URL/token, camera serials, two CAN channel names, and your
calibrated gripper `[closed, open]` raw motor positions. Fill the three
12-element arrays `joint_low`, `joint_high`, and `joint_max_step` from your
rig's validated operating range. Placeholder nulls deliberately fail before
opening hardware. Both arms' proposed commands are checked before either is
sent; out-of-range values are rejected, not silently clipped or interpolated.
`joint_max_step` bounds target-minus-measured position, not a hardware velocity
or torque limit. CAN writes are sequential, not atomic.

Bring up the CAN interfaces using the I2RT instructions for your hardware.
The adapter does not change CAN configuration or disable motor watchdogs.
Startup enables the SDK's position control at the measured pose; it does not
home or automatically calibrate the grippers. Even `--no-execute-action` opens
and enables the robot, so it is not a passive or torque-free inspection mode.

I2RT maintains its motor control loop while inference is pending. There is no
additional application watchdog in this adapter, and a blocked native SDK call
cannot be cancelled by Python. SDK initialization/close can block. Validate
motor watchdog and emergency-stop behavior on the rig before unattended use.
On normal exit or command failure, the adapter calls SDK `close()` for both
arms and stops cameras. For the pinned I2RT version, the adapter waits for each
arm's CAN control workers before the SDK closes its motor interface, avoiding
sends against a closed socket. A worker that does not stop within five seconds
is reported as a shutdown failure; SDK shutdown is still attempted and the
other arm and cameras are still closed. This does not make blocked native SDK
calls cancellable. **I2RT close releases motor torque; it does not hold
against gravity. Support/park the arms appropriately before shutdown.**

## Connect the Policy Server

Follow [the Policy Server YAM guide](../../colosseum-policy-server/docs/yam.md)
to run the model worker and Colosseum Local Policy Server. Set
`policy_server_url` to that server; if it runs on another host, use an SSH
forward for the loopback-only local service:

```bash
ssh -N -L 8000:127.0.0.1:8000 GPU_HOST
```

Your Router must have the matching YAM task/camera roles and policy deployment.
The example revision is a pinned model reference, not proof of current Router
state. Open Track needs two eligible policies; this change implements only
MolmoAct2 YAM. For one-model validation use a configured Fine Tune assignment
(`track: 2`) or a standalone adapter/model test. No Router catalog is modified.

From the Client checkout, after filling configuration:

```bash
# Real observations and inference; no policy actions sent to the arms.
.venv/bin/colosseum-robot configs/robot.yam.yaml --no-execute-action
# After checking joint order, camera roles, gripper polarity and action limits:
.venv/bin/colosseum-robot configs/robot.yam.yaml
```

`test: true` bypasses the I2RT and camera drivers. With a compatible assignment
it checks synthetic 12-joint / 2-gripper transport, not physical integration.
