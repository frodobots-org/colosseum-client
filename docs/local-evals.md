# YAM evaluations with local VLAs: record A/B, review, then submit

The five VLAs (G05, GR00T, MolmoAct2, LingBot, pi05) run as local checkpoints
on this machine's GPU. The three LLM policies (GPT, Grok, Claude) run through
their APIs, using Inspect Robots end-effector (EE) control through the local
Policy Server. The Router assigns the model and run IDs. Scores, videos and
datasets are submitted only when you click **Save + Submit** on the review page.

> [!IMPORTANT]
> Every new instruction starts a **new evaluation**. The Router allows one
> pending evaluation per evaluator, so submit a recording before starting the
> next one if you want it uploaded.

This guide assumes `~/colosseum-client` and
[`~/colosseum-policy-server`](https://github.com/frodobots-org/colosseum-policy-server)
are checked out side by side.

- Client config: `configs/robot.yam-eef-llm.yaml`, used for recording and review.
- Policy Server config: `configs/local-runtime-yam-local.yaml`. It holds the 5
  VLA and 3 LLM entries. Each VLA has a `launcher` that starts its checkpoint
  on a local port (8202–8206).
- GPT and Grok use your official OpenAI and xAI API keys. Claude uses the Claude
  Code login on the Policy Server host.

A VLA worker starts **on demand** when the Router assigns it. Only one fits on
a 24 GB GPU at a time. The first step of a VLA run waits while the checkpoint
loads, for up to 10 minutes (`start_timeout_seconds: 600`). Worker logs go to
the Policy Server's `local-policy-logs/`.

> [!WARNING]
> Before a physical LLM trial, verify that the three LLM `name`, `url` and
> `revision` values in the Policy Server config exactly match the current Router
> registrations. Also check the EE bounds against the rig. Keep the robot
> attended, with the emergency stop within reach.

## One-time setup

### Client

1. Install the Client, YAM extra and I2RT as in [YAM setup](yam.md#install-and-configure-the-robot-host).
2. Install Inspect YAM EE control, which LLM trials need:
   ```bash
   cd ~/colosseum-client
   uv pip install --python .venv/bin/python \
     'inspect-robots @ git+https://github.com/supernoooo/inspect-robots.git@0865dcac2f38dfdd0a4768a68bf4146ef394b69f' \
     'inspect-robots-yam-eef @ git+https://github.com/supernoooo/inspect-robots.git@0865dcac2f38dfdd0a4768a68bf4146ef394b69f#subdirectory=plugins/inspect-robots-yam-eef'
   ```
3. Create the config and fill in the Router URL, token, institution, camera
   serials, CAN channels, gripper calibration (`scripts/calibrate_yam_gripper.py`),
   joint limits and `inspect_eef` bounds:
   ```bash
   cp configs/robot.yam-eef-llm.yaml.example configs/robot.yam-eef-llm.yaml
   ```
   Optionally set `evaluation_dir: local/eval_runs` to keep recordings in the
   git-ignored `local/` folder.
4. Put the LLM API keys in a private file:
   ```bash
   mkdir -p local && install -m 600 /dev/null local/llm-keys.env
   ```
   ```bash
   export OPENAI_API_KEY=...
   export XAI_API_KEY=...
   ```
5. Install `ffmpeg`, which the review page uses to encode preview videos.

### Policy Server

1. Install the Inspect agent:
   ```bash
   cd ~/colosseum-policy-server
   uv sync --extra agent
   ```
2. Set up each VLA's worker environment and checkpoint snapshot. See the Policy
   Server's [YAM guide](https://github.com/frodobots-org/colosseum-policy-server/blob/main/docs/yam.md)
   and [LingBot guide](https://github.com/frodobots-org/colosseum-policy-server/blob/main/docs/lingbot-yam.md).
3. Create the config. Fill in the launcher paths, then the joint limits and EE
   bounds, matching the Client config, and `robot_notes`:
   ```bash
   cp configs/local-runtime-yam-local.yaml.example configs/local-runtime-yam-local.yaml
   ```
4. Log in to Claude Code as the same operating-system user that starts the
   Policy Server. The `claude` CLI must be on that user's `PATH`:
   ```bash
   claude auth login --claudeai
   ```

## 0. Check I2RT in the Client environment

```bash
cd ~/colosseum-client
.venv/bin/python -c 'from i2rt.robots.get_robot import get_yam_robot; print("I2RT ready")'
```

If a `uv sync` removed it, reinstall it as in [YAM setup](yam.md#install-and-configure-the-robot-host).

## 1. Start CAN and check the GPU (Terminal 1)

Bring up the CAN interfaces, for example with I2RT's `scripts/reset_all_can.sh`, then:

```bash
ip -br link | grep can
nvidia-smi   # should show no leftover model workers
```

> [!WARNING]
> Stop any teleoperation servers that use the same CAN channels before recording.

> [!CAUTION]
> If there are leftover workers, stop the Policy Server and recorder first.
> Then stop all GPU compute processes with SIGINT, which lets them free GPU
> memory, and check again.

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -INT
sleep 5; nvidia-smi   # if one is still listed: kill -TERM <pid>, and kill -KILL <pid> only as a last resort
```

This only lists compute processes (model workers, Python), not the desktop/Xorg display.

## 2. Claude login and Policy Server (Terminal 2)

```bash
cd ~/colosseum-policy-server
claude auth status --text
uv run --no-sync colosseum-policy-local --config configs/local-runtime-yam-local.yaml
```

> [!TIP]
> If the Claude session has expired, run `claude auth login --claudeai` in this
> terminal, then restart the Policy Server.

## 3. Record a new evaluation (Terminal 3)

Load the LLM API keys in the terminal that launches the recorder. The Client
sends only the assigned model's key to the Policy Server. Claude uses the login
from step 2.

```bash
cd ~/colosseum-client
source local/llm-keys.env          # sets OPENAI_API_KEY and XAI_API_KEY
.venv/bin/python scripts/record_evals.py configs/robot.yam-eef-llm.yaml --scene '<scene>'
```

Enter the task at the `Instruction:` prompt and confirm robot initialization.
Record A, restore the scene, then record B. EE trials need action execution,
so don't pass `--no-execute-action`.

Recordings go to `<evaluation_dir>/<evaluation ID>/<run ID>/`. For LLM EE runs,
`action` is the post-IK joint command and `action.eef_target` is the EE target.
VLA runs record joint actions.

## 4. Start the review server (Terminal 4)

```bash
cd ~/colosseum-client
.venv/bin/python scripts/review_evals.py --config configs/robot.yam-eef-llm.yaml
```

> [!TIP]
> Leave it open, then open <http://127.0.0.1:8765>. **Save locally** only
> writes local files. **Save + Submit to Router** submits scores and uploads
> videos and the dataset. Don't run `colosseum-robot` separately.

## 5. Review and submit

Refresh the review page, open the evaluation, watch the A/B videos, and enter
both scores (0–100), a preference, and feedback. Then choose:

- **Save locally:** keep a draft.
- **Save + Submit to Router:** publish the result and upload videos and the dataset.
- **Discard:** close the evaluation on the Router without publishing.

## Interrupted runs and upload retry

- A first VLA step that times out usually means the checkpoint is still
  loading. Check the Policy Server's `local-policy-logs/`. Once the worker is
  up, continue the same evaluation:
  ```bash
  .venv/bin/python scripts/record_evals.py configs/robot.yam-eef-llm.yaml --scene '<scene>' --resume ev_<id>
  ```
- If you want a fresh evaluation instead, rerun step 3 without `--resume`. The
  Router closes the old one.
- If you hit CUDA out of memory, stop the Policy Server with Ctrl-C, check
  `nvidia-smi`, and run `kill -INT <pid>` on any leftover worker.
- If the website is missing videos or dataset files after a submit, retry the upload:
  ```bash
  .venv/bin/colosseum-upload-dataset configs/robot.yam-eef-llm.yaml --evaluation-id ev_<id>
  ```

> [!NOTE]
> When you're done, press Ctrl-C in each terminal and confirm `nvidia-smi`
> shows the GPU is free.
