## Test data with a real robot identity

```yaml
robot_type: franka # or yam
test: true        # dummy data/actions; false uses the hardware adapter
track: 1          # 1 Open, 2 Fine-tuning
```

`test` must be a YAML boolean. The normal Client command and evaluation workflow
are shared. Dummy local action dimensions follow the Router-assigned model contract;
this is a transport fixture, not a validated YAM hardware interface. Model loading
is still simulated by `colosseum-policy-verify`.

Router persists an immutable test flag per assignment, exposes it in Open and
Fine-tuning reviews, and excludes synthetic results from formal ranking, difficulty
fitting and Fine-tuning summaries. Test attempts do not consume formal trial quotas.
Test-only deployments are unavailable to `test: false` clients. Ready messages for
simulation cannot start an assignment marked real. Changing mode during a pending
assignment is supported by restarting normally; the prior unfinished assignment is closed automatically.

Web review is available at `/web/evals` on the Router, with Test badges under
the original robot identity. Historical `robot_type: test` records are retained,
but Legacy test is no longer offered in the robot selector.
The old robot_type test syntax remains accepted for compatibility. New configs
should always use a real robot_type with a separate test flag.

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
# Edit token and institution before running.
uv run colosseum-robot configs/robot.yaml
```

The example defaults to Franka with dummy data (`test: true`) and a local Policy
Server at `ws://localhost:8000`. Start `uv run colosseum-policy-verify` in the
Policy Server project first. Set the Client token and matching Institution.

For real DROID/Franka hardware, set `test: false`, configure camera serial numbers
for `left_image`, `right_image`, and `head_image`, and use a real inference server.
DROID or R2D2 must already be installed on the robot computer. Production
deployments must use `wss://`.

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

Actions execute by default in evaluation and `--inference-only` modes. To receive
and print complete action chunks without executing them:

```bash
colosseum-robot configs/robot.yaml --no-execute-action
```

Camera reads, RobotEnv initialization and cleanup still run. Per-step robot I/O
timing logs are disabled; exceptions still propagate to the caller.

The evaluation loop reads one observation at the start of each control step and
requests inference only when the previous action chunk is exhausted. Observation,
inference and execution time all count toward the control period; steps that exceed
the period do not add a sleep. There is no additional read at chunk boundaries. One final observation is recorded
after the last action to capture the ending scene.

Recording uses a dedicated writer thread and a bounded 32-frame queue. The control
loop copies image/state/action buffers and enqueues them without waiting for PNG
compression or disk writes. Actual capture timestamps are retained separately
from the nominal-FPS LeRobot video timeline. At 512×288 RGB, queued pixel data is about 13.5 MiB per camera.
The final frame and all queued writes finish before marking the run finished,
scoring, encoding, or uploading. A full queue or disk error interrupts the trial
with an explicit error instead of silently dropping evidence or stalling control.
Incomplete recordings cannot be encoded by the upload flow.

Older diagnostic trials with `recording-disabled.json` still have no video; use
a normal restart to close those assignments automatically. Explicit `--resume` can retry pending uploads before starting a new evaluation.

Choose `1` for Open Track or `2` for Fine-tuning. Open requests a task instruction,
executes server-assigned A, asks for its partial success (0–100), then prompts to
restore the scene and execute B. After B partial success, select A/B/tie and enter
optional feedback. For Open Track, 100% maps to success; lower values map to not
fully successful in the stored result. No separate success/failure prompt is shown.
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
control rate on the real robot. LeRobot MP4s align one frame per recorded action at
the configured control FPS; capture timestamps preserve actual inference pauses.

The local `manifest.json` records scores, uploaded cameras, and pending final submission.
An upload or submission failure can be resumed without reexecuting a finished trial or
asking for scores again. Source PNGs, timing records, MP4s, and the manifest are retained.
Starting the client normally always starts a fresh evaluation: it automatically
closes any previous unfinished assignment owned by the same token, including
runs left behind by Ctrl-C, connection loss or a killed process. No manual abort
is required. Completed results are unchanged. Unfinished evaluations stay hidden
from public review and do not consume completed-trial capacity. Local recordings
remain available. Explicit `--resume` still retries a pending finished trial's
uploads/results before a fresh start; it cannot revive a replaced assignment.


The end-to-end tests use simulated hardware and real local HTTP/WebSocket/ffmpeg video
uploads. They are not validation of physical robot behavior or real S3 credentials.


With `COLOSSEUM_VIDEO_STORAGE=s3` on the router, evaluation MP4s upload directly from
this client to S3 using short-lived presigned URLs. No AWS credentials are needed here.
The client computes SHA-256, uploads the bytes without forwarding its Colosseum token,
then requests server verification. An uploaded camera is marked complete in the local
manifest only after verification. Local storage mode continues to upload to the router.

For Fine-tuning, select a predefined task from the server-provided menu after choosing
the track. Only tasks for this robot with remaining trials (or your pending trial)
are listed. The server selects the model; successful rounds continue on the same task.
`--resume FT-ID` keeps the original task. Stop and start again to select another
task; any previous unfinished assignment is automatically closed. This menu requires
a router version with `GET /api/eval/tasks` and `task_id` support.

## Configured evaluation and local inference

`configs/robot.yaml.example` now includes:

```yaml
router_url: wss://router.example.com
token: clt_replace_me
scene: kitchen_01
evaluator: operator_01
track: 1 # 1 = Open, 2 = Fine-tuning
policy_server_url: null # or wss://local-policy.example.com:8000
```

Keep the existing `cameras`, `robot_type`, adapter and control settings in that same
file. `url` remains a supported alias for `router_url`; conflicting values are rejected.
`--track` overrides config; resume uses the saved assignment's track and metadata.
With track and scene set, Open asks only for the instruction before setup confirmation;
Fine-tuning asks for the task. Scoring remains interactive. Scene/evaluator are saved
on Router and in the local assignment manifest; evaluator text does not replace token
ownership. The Router must be upgraded alongside Client for the new request fields.

With `policy_server_url` set, Client queries supported runtime profiles over WS(S),
then Router chooses the model. Client receives its HF link and pinned revision from
Router and prepares the model through the local WS connection. Observations/actions
travel directly to the local service. Recordings start after readiness, and each
new Trial must reset local model state. Local inference requires evaluation mode;
`--inference-only` with a local URL is rejected.

Run `uv run colosseum-robot configs/robot.yaml`. Existing remote inference remains
available by omitting the local URL. Model loading timeout defaults to 1800 seconds
and can be shortened with `prepare_timeout`. WSS verifies server certificates normally.
The Router token is sent only to Router. This version has no local token setting.

The local server must implement [Local inference protocol v1](../colosseum-router/docs/local-policy.md).
The existing Policy Server outbound SDK does not yet implement this inbound protocol.
Client/Router are implemented and mock-tested; a real model-serving implementation
and hardware validation are still needed. Each action currently waits for a Router
control acknowledgment, so WAN latency can limit the control rate.

### Verify WSS model delivery without a robot

```bash
.venv/bin/python -m colosseum_client.local_check configs/robot.yaml
```

This uses binary Protobuf through WSS, checks the receipt from the verification-only
Policy Server, and leaves the assignment unexecuted. See
[verification setup](../colosseum-policy-server/docs/local-verification.md).

### Robot type and a hardware-free local test

Set `robot_type: franka` or `robot_type: yam`. The type is passed to Router for task
and deployment selection. Omitted adapter defaults to `droid` for Franka and `yam`
for YAM; the YAM hardware adapter is not yet installed. Receipt-only WSS verification
supports either type without instantiating any robot or assuming its action dimensions.
Legacy `robot_type: DROID` remains accepted.

From the workspace root, run the automated local test:

```bash
colosseum-router/.venv/bin/python colosseum-router/scripts/run_local_verification.py --robot-type yam
```

Use `--robot-type franka` for Franka. The script starts isolated Router, TLS Policy
Server and Client, registers synthetic task/model data, checks Protobuf delivery,
and shuts everything down. No hardware, GPU, internet download or inference is used.
The three existing development virtual environments and `openssl` are required.
Config, receipt and verification summary paths are printed under a fresh `/tmp`
directory. YAM model URL/action dimensions are synthetic transport fixtures, not a
real YAM model or hardware specification.

### Test robot adapter

All robot types use the same evaluation entry point:

```bash
uv run colosseum-robot configs/robot.yaml
```

`robot_type: test` selects a dummy data/action adapter. It generates RGB and state,
and applies returned actions in memory. Without camera configuration it provides
`head_image`. It uses the same recording, scoring, upload, resume and result flow as
physical robots. `track: 1` runs A then B with partial success and preference;
`track: 2` selects a Router task and runs Fine-tuning evaluation.

Start `uv run colosseum-policy-verify` in Policy Server for simulated model preparation
and actions. Use `policy_server_url: ws://localhost:8000`. Restart the server after
updating. Router records these evaluations under `robot_id: test`; scores and videos
are submitted normally, so filter by robot when viewing results. No hardware executes.
Open Track duration uses `max_trial_steps` (the prepared test config sets 3);
Fine-tuning uses the Router task limit. Enter can finish a trial early.

The separate `python -m colosseum_client.local_check` command remains an explicit
receipt-only diagnostic and is not the normal Client entry point.

### Complete local A/B test (no hardware)

From `colosseum-router`:

```bash
uv run python scripts/run_ab_eval.py
```

Requires the sibling Client development environment and Router test dependencies.
The command launches an isolated Router and mock Policy Server, invokes the actual
Client evaluation workflow with a fake robot, and supplies the operator responses
automatically: A=75%, B=100%, preference=B. It verifies preparation before observations,
different A/B revisions, action reporting, actual MP4 uploads, persisted scores and
idempotent result submission. Services stop automatically; the printed artifact
folder retains the database, prompts, model events, recordings and summary.
No cloud scores or real robot actions are generated.

For real local inference, Policy Server `ready` directly enables Client evaluation.
Router start/step acknowledgments are handled in the background. Completion waits
for Router to persist the trial before scoring and switching models. Real model loading/inference still requires
a Policy Server implementation that emits `ready` after loading the assigned model.

### Institution binding

Set `institution` to the exact value registered for your Client token in Router's
`/admin` token manager. Evaluation requests (including resume/uploads and the
inference control connection) carry this value; Router rejects unbound tokens,
missing values and mismatches. Whitespace around the value is trimmed, but case is
preserved. The stored evaluation institution comes from the token binding.

Every Client token has an Institution. The admin form uses a single Institution
field; new tokens bind it automatically. Existing unbound tokens are migrated using
their original names, while existing bindings are preserved.
To change an existing binding, create a new token. Never infer institution from the
operator-provided scene or evaluator name.

### LeRobot recording

With recording enabled, evaluations export a **LeRobot v3.0** dataset per trial. Update
dependencies with `uv sync`, then run the usual Client command. Configuration:

```yaml
recording: true # default; false disables recording and dataset uploads
```

Each A/B run is a separate one-episode dataset:

```text
eval_runs/<assignment>/<run>/lerobot/
  data/chunk-000/file-000.parquet
  videos/observation.images.<camera>/chunk-000/file-000.mp4
  meta/info.json
  meta/stats.json
  meta/tasks.parquet
  meta/episodes/chunk-000/file-000.parquet
  meta/colosseum.json
```

- `observation.state`: joint positions followed by gripper state.
- `observation.cartesian_position`: the adapter's raw Cartesian pose vector.
- `action`: the selected command for that observation, before execution; this does
  not prove physical execution. Action-space and execution-enabled metadata are
  recorded separately.
- `observation.images.<camera>`: RGB video aligned one frame per recorded action.
- `timestamp`: frame index divided by configured control FPS. This is a nominal
  step timeline, not the actual elapsed wall time; observations are not resampled.
- `observation.capture_timestamp`: original capture time in seconds since the
  recorder started, preserving inference stalls and variable control timing.
- `meta/colosseum.json`: run/assignment identity, task, model (local mode),
  institution, scene, evaluator, per-trial score and synthetic-data flag.

The final observation with no associated action remains in the raw recording; no
action is invented for it. Odd image sizes are padded to even dimensions for H.264.
Statistics are computed from recorded states/actions and source RGB pixels
(including padding); decoded video can differ slightly due to lossy encoding.

During each trial, a background writer saves image frames and state/action
records locally. After that trial finishes and its score is saved, the Client
finalizes its LeRobot dataset before starting the next model. Only uploads wait
until all trials, scores, A/B preference and feedback are complete. Files are
then uploaded sequentially before final result submission. Until that upload
phase, recordings exist only on the Client machine. Completed exports are
validated and reused on resume, and failed exports can be retried. Source
PNG/JSONL recordings remain available. The complete dataset is uploaded before
result submission. Web review reuses the dataset's videos, so LeRobot mode does
not encode or upload separate review MP4s. With Router S3 storage
enabled, files go directly to the private bucket using signed URLs; the Router
verifies every file's size and SHA-256 before publishing replay. Failed uploads
can be resumed and verified files are skipped. Datasets are not published to
Hugging Face.

The evaluation page retains the standard evaluation video players, using the
LeRobot camera videos at the dataset's nominal FPS. The final observation without
an action remains only in the raw recording. With `recording: false`, inference
and scoring continue, but no images, state/action records, videos or datasets are
saved or uploaded. Assignment metadata and scores still persist locally for
resume. Published results have no playback footage. The flag is bound to the
assignment and cannot change on resume.

Normal usage is unchanged:

```bash
uv run colosseum-robot configs/robot.yaml
```

To upload an already exported dataset without rerunning the robot or changing
its score, use the same owner token and institution:

```bash
uv run colosseum-upload-dataset configs/robot.yaml --evaluation-id <evaluation-id>
```

Omit `--evaluation-id` to upload all exported evaluations in `evaluation_dir`.
The Router must support dataset endpoints, the assignment recording flag, and
`POST /api/eval/reset` before using this Client version.

Older recordings lack gripper state and recording context and cannot be faithfully
exported with the current client. Preserve those source files; do not fabricate
missing state or switch recording off to bypass an existing assignment requirement.

Load a run in a separate environment with LeRobot installed:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    "local/colosseum-run",
    root="eval_runs/<assignment>/<run>/lerobot",
    video_backend="pyav",
)
frame = dataset[0]
```

Format reference: https://huggingface.co/docs/lerobot/lerobot-dataset-v3
