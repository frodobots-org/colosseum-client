# Integrating a robot

The evaluation command loads a robot adapter independently of the Client's
network, scoring, recording and upload implementation. External integrations can
live in their own Python package; no Client source edits are required.

## Hardware and inference are separate integrations

```text
Robot SDK / cameras
    ↕ Robot (read observations / execute actions)
Client evaluation loop
    ↕ observation / action messages
Policy Server (model loading / preprocessing / inference / action decoding)
```

The Router assigns compatible models and coordinates evaluations. For remote
inference it also relays messages; local inference uses the configured
`policy_server_url`. The Client does not load model weights in a robot adapter.
Adding an adapter does not implement the model runtime or register a compatible
policy on the Router.

| File | Responsibility |
| --- | --- |
| `src/colosseum_client/robot_interface.py` | Shared `RobotObservation` and `Robot` contract |
| `src/colosseum_client/adapters.py` | Select a built-in or external adapter |
| `src/colosseum_client/robots/franka.py` | Existing Franka driver (`DroidRobot`, using DROID/R2D2) |
| `src/colosseum_client/robots/so101.py` | SO101 hardware integration TODOs |
| `src/colosseum_client/robots/yam.py` | Bimanual YAM I2RT driver (see [YAM setup](yam.md)) |
| `src/colosseum_client/robots/g1.py` | G1 hardware integration TODOs |
| `src/colosseum_client/evaluation.py` | Shared trial, observation and action loop |
| `src/colosseum_client/local_policy.py` | Local Policy Server transport and lifecycle |

Public imports remain `from colosseum_client import Robot,
RobotObservation`. The legacy `droid_robot.RobotObservation` import still refers
to the same class.

All hardware implementations live in `robots/`. The old `droid_robot.py` module
only re-exports compatibility imports; it contains no separate driver logic.

## Integration stages when inference is not implemented yet

1. Copy the shared example config, set `robot_type`, and keep the TODO adapter. With `test: false`,
   these fail explicitly before hardware is opened. Do not advertise the robot
   as operational at this stage.
2. Agree on camera roles, state ordering, action ordering, units, dimensions and
   control rate with the model implementer. For SO101, specify servo calibration
   and gripper mapping; for YAM, specify arm configuration and ordering; for G1,
   specify the controlled joints/hands and ownership with its balance controller.
3. Implement and check `get_observation`, `execute` and `close` against the
   selected hardware SDK. Observation acquisition can be checked directly via
   `make_robot(config)` and `get_observation()` without a Policy Server. Always
   close the adapter in a `finally` block. This still opens real hardware.
4. Implement the model runtime separately in the Policy Server integration:
   weight loading, image/state preprocessing, inference and action decoding.
   Register the matching model contract on the Router.
5. Run the evaluation command with `--no-execute-action` to inspect observations
   and returned actions before enabling execution. This requires a working
   Policy Server; it is not an inference-free hardware check.

`test: true` can exercise the synthetic Client workflow when the corresponding
Router/policy setup is available. It bypasses the hardware adapter entirely and
does not validate real robot state/action mappings. Keep real hardware and
real inference validation as separate milestones.

## Available adapters

| `robot_type` | Status |
| --- | --- |
| `franka` | Existing DROID/R2D2 hardware driver |
| `so101` | TODO template; no hardware driver |
| `yam` | Bimanual I2RT + three RealSense cameras; [setup and validation limits](yam.md) |
| `g1` | TODO template; no hardware driver |

YAM requires calibrated limits and connection settings before hardware can open.

Templates are in `src/colosseum_client/robots/`. They raise `NotImplementedError`
at construction, before opening hardware. Their joint/action dimensions are
deliberately unspecified: choose the actual controlled joints and policy contract.
For YAM use `configs/robot.yam.yaml.example` and [the YAM guide](yam.md).
Other robots share `configs/robot.yaml.example`. Copy it to `configs/robot.yaml`
and change `robot_type` to `franka`, `so101`, `yam`, `g1`, or an installed external
robot type. Set the camera IDs and hardware settings for the selected robot.

`test: true` always selects the existing synthetic TestRobot without importing
the chosen hardware adapter. It tests the Client workflow, not your integration.
Router still needs the robot identity, task and compatible models configured.
Adding a Client adapter does not register anything on Router.

## Loading your implementation

Install your driver package into the **same Python environment** as the Client.
Its constructor or factory accepts one `RobotClientConfig` and returns a
`Robot`. Register the robot type in your driver's `pyproject.toml`:

```toml
[project.entry-points."colosseum_client.adapters"]
my_robot = "my_robot_driver:Robot"
```

Select it using only `robot_type`:

```yaml
robot_type: my_robot
adapter_config:
  address: 192.0.2.10
  camera_device: /dev/video0
cameras:
  head_image: camera-id
```

`robot_type` is both the Router/catalog identity and the local implementation
selector. There is no `adapter` selection field. Built-in types (`franka`, `test`,
`so101`, `yam`, `g1`) are reserved; implement their existing driver files
when adding support for those robots. Other types resolve an installed entry
point of the same name. Duplicate entry point names are rejected.

Use `robot_type: franka` for Franka; `DROID` and `droid` are not accepted robot
types. DROID/R2D2 remains the underlying Franka SDK. Remove the `adapter:` line
from older YAML files; it now produces an unsupported-key error.
Identifiers are case-sensitive. `adapter_config` remains an optional mapping
of hardware connection/calibration settings, not a second robot selector.
Adapters are executable Python code: load only packages you trust.

Camera roles may use letters, digits, underscores and hyphens. They must match
the task/model's expected roles; accepting a name locally does not establish
server/model compatibility. Real configurations still require at least one camera.

The current Router evaluation API still restricts cameras to `head_image`,
`left_image`, and `right_image` (at most three). Use those roles for evaluation
with that Router. Custom roles are accepted by the Client extension interface,
but require a corresponding Router API change before evaluation can use them.

## Adapter interface

`Robot` is an ABC. Inherit from it and implement `get_observation`,
`execute` and `close`; Python rejects instantiation if any abstract method is
missing. Constructors remain driver-specific. Hardware metadata fields must also
be supplied by the driver; ABC does not validate their values or action semantics.
The SO101/G1 skeletons override these methods with `NotImplementedError` and
still fail explicitly in their constructors until hardware support is implemented.

```python
from colosseum_client import Robot, RobotClientConfig, RobotObservation

class MyRobot(Robot):
    # Implement get_observation(), execute(action), close(), and hardware metadata.
    pass  # Abstract: cannot be instantiated yet.

def create_robot(config: RobotClientConfig) -> Robot:
    return MyRobot(config)
```

Implement these members directly in `robots/so101.py`, `robots/yam.py`, or
`robots/g1.py` (each contains its own TODO skeleton):

- `joint_count: int`, `has_gripper: bool`, `action_dim: int`,
  `action_space_name: str`: describe the actual policy-facing hardware interface.
- `get_observation() -> RobotObservation`: RGB uint8 HWC images keyed by camera
  role, plus finite joint, gripper and Cartesian state vectors.
- `execute(action) -> None`: consume one action, validating dimensions, units,
  joint order and hardware limits before converting it to SDK commands.
- `close() -> None`: stop/hold as appropriate and release hardware resources;
  cleanup should be safe to repeat.

Current observation fields remain `images`, `joints`, `gripper`, and
`cartesian_position`; recording/export use these fields too. This extension does
not introduce a generalized state/action schema. Define arm ordering, gripper
semantics and Cartesian representation together with the model integration.

Evaluation constructs, reads, executes and closes the adapter on one dedicated
worker thread, preserving thread affinity for SDKs. I/O must be bounded; provide
hardware watchdog/stop behavior. Clean up partially opened resources if the
constructor fails. Client resource cleanup alone is not a hardware stop contract.

## Run

After implementing and installing your adapter, copy the shared example, set the Router
Client token and institution, and set `robot_type`:

```bash
cp configs/robot.yaml.example configs/robot.yaml
# Edit robot_type, token, institution, cameras and hardware settings first.
colosseum-robot configs/robot.yaml --no-execute-action
# After validating the observation and action mapping:
colosseum-robot configs/robot.yaml
```

`--no-execute-action` still constructs the adapter and reads real hardware.
The TODO templates intentionally fail in this mode too. The legacy
`--inference-only` runner remains real-DROID-only; other adapters and dummy mode
are rejected there instead of silently opening DROID.
