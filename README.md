# Colosseum Client

Robot-side client for token-authenticated policy inference through a Colosseum Router.

## DROID robot

The `--inference-only` robot runner follows an open-loop action-chunk cycle: read RobotEnv, send the
current images and robot state through the WSS Router, then execute each returned action
at 15 Hz before requesting the next chunk. Joint and gripper observations stay separate;
the action uses 7 joint values followed by one gripper value.

```bash
uv sync
cp configs/robot.yaml.example configs/robot.yaml
chmod 600 configs/robot.yaml
uv run colosseum-robot --inference-only
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

## Evaluation mode (default)

```bash
uv sync --extra dev
# Install ffmpeg on the robot workstation before recording.
uv run colosseum-robot configs/robot.yaml
```

Choose `1` for Open Track or `2` for Fine-tuning. Open requests a task instruction,
executes server-assigned A and B, then collects success/progress and preference.
Fine-tuning obtains the predefined task and anonymous model entirely from the server;
the operator confirms scene setup, runs a trial, and records success/failure and
partial success (0–100). Fine-tuning automatically requests the next assignment;
press Ctrl-C at the next setup prompt to stop. Open asks whether to continue.
No model name is printed by evaluation mode. New evaluation connections verify TLS
normally; the older `--inference-only` runner retains its existing temporary workaround.

The server must have task protocols and policy deployments configured first; see the
router's `EVALUATION.md`. Starting with an empty server produces a no-assignment message,
not invented tasks/models. Use `robot_type: franka` and `adapter: droid` for the existing
Franka/DROID hardware bridge. A standalone Franka without DROID/R2D2 is not yet supported.
`adapters.py` defines the observation, action and dimension interface for future robot
backends; adding a YAML robot name alone does not implement its hardware control.

Optional command-line shortcuts:

```bash
uv run colosseum-robot --track fine-tuning
uv run colosseum-robot --track open
uv run colosseum-robot --resume ev_ASSIGNMENT_ID
uv run colosseum-robot --abort ev_ASSIGNMENT_ID
```

`api_url` defaults to the HTTP(S) equivalent of the configured WS(S) endpoint, sharing
the same port. Evaluation uses a **Client token**, not an Admin or Policy token.
Each trial starts only after the operator confirms the initial scene; reset objects
and robot pose as appropriate before A, B, or the next Fine-tuning trial. Press Enter
during execution to end a trial normally; Ctrl-C marks an interrupted session.
`max_trial_steps` controls Open trials and defaults to 2700 steps (about three minutes
at 15 Hz, plus inference and recording overhead). Fine-tuning limits come from the
server task configuration, whose default is also 2700 steps. These are step limits,
not wall-clock timeouts.

Recordings are saved in `evaluation_dir` (default `eval_runs`) with separate assignment,
run, and camera folders. Every action step records RGB images plus timestamp/state/action
metadata. PNG recording and hardware camera reads consume time; confirm the achievable
control rate on the real robot. MP4 encoding uses capture intervals, including inference
pauses, instead of pretending every frame was captured at 30 Hz.

The local `manifest.json` records scores, uploaded cameras, and pending final submission.
An upload or submission failure can be resumed without reexecuting a finished trial or
asking for scores again. Source PNGs, timing records, MP4s, and the manifest are retained.
If an execution itself was interrupted, resume will report that state; use `--abort`
with a reason, then request another assignment. This prevents silently rerunning the
same assigned attempt until it succeeds. An interrupted recording can be encoded and
uploaded through the run video endpoint before aborting when needed for review.

The end-to-end tests use simulated hardware and real local HTTP/WebSocket/ffmpeg video
uploads. They are not validation of physical robot behavior or real S3 credentials.


With `COLOSSEUM_VIDEO_STORAGE=s3` on the router, evaluation MP4s upload directly from
this client to S3 using short-lived presigned URLs. No AWS credentials are needed here.
The client computes SHA-256, uploads the bytes without forwarding its Colosseum token,
then requests server verification. An uploaded camera is marked complete in the local
manifest only after verification. Local storage mode continues to upload to the router.
