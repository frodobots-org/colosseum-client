# Colosseum Client

Robot-side client for token-authenticated policy inference through a Colosseum Router.

## DROID robot

The robot runner follows an open-loop action-chunk cycle: read RobotEnv, send the
current images and robot state through the WSS Router, then execute each returned action
at 15 Hz before requesting the next chunk. Joint and gripper observations stay separate;
the action uses 7 joint values followed by one gripper value.

```bash
uv sync
cp configs/robot.yaml.example configs/robot.yaml
chmod 600 configs/robot.yaml
uv run colosseum-robot
```

`configs/robot.yaml` contains the WSS URL, Client token and the DROID camera serial
numbers assigned to `left_image`, `right_image`, and `head_image`. DROID or R2D2 must
already be installed on the robot computer. Production deployments must use `wss://`.

For every action, the runner rejects incorrect dimensions and non-finite values,
binarizes the final gripper value at `0.5`, and maintains the registered control rate.
Press Ctrl-C to close the Router session and RobotEnv.

## Robot registration

The Client declares its hardware and action dimensions in the initial `HELLO`:

```python
client = ColosseumClient(
    router_url,
    token,
    client_id="robot-01",
    robot_type="DROID",
    joint_count=7,
    has_gripper=True,
    control_hz=15,
    action_spaces={
        "joint_position": 8,
        "joint_velocity": 8,
        "cartesian_position": 7,
    },
)
```

The Router attaches this `RobotSpec` to the matched session. Policy Servers can use it
to select a compatible action space and validate every action chunk dimension.
