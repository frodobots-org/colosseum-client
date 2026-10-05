# Colosseum Client

Run robot evaluations with [RoboColosseum](https://frodobots-org.github.io/robo-colosseum/) and upload the recorded results.

## 1. Install

Requirements: **Python 3.10 or newer**, **uv**, and **FFmpeg** for recording export.

```bash
git clone https://github.com/frodobots-org/colosseum-client.git
cd colosseum-client
```

## 2. Getting Started

```bash
cp configs/robot.yaml.example configs/robot.yaml
```

Edit `configs/robot.yaml` for your setup. See the
[Configuration example](configs/robot.yaml.example) for configuration details.

Install first, then run:

```bash
uv sync --extra <robot_type>
.venv/bin/colosseum-robot configs/robot.yaml
```

Or install and run in one command:

```bash
uv run --extra <robot_type> colosseum-robot configs/robot.yaml
```

Follow the prompts to enter an instruction or select a task, start the trial,
and submit the evaluation. Recordings are saved under `eval_runs/` by default.

## Supported robots

| Robot | `robot_type` | Status |
| --- | --- | --- |
| [Franka](src/colosseum_client/robots/franka.py) | `franka` | Implemented |
| [SO-ARM101](src/colosseum_client/robots/so101.py) | `so101` | TODO |
| [Bimanual YAM](src/colosseum_client/robots/yam.py) | `yam` | TODO |
| [Unitree G1](src/colosseum_client/robots/g1.py) | `g1` | TODO |

[Adding a new robot](docs/robot-adapters.md)

## Reference

[Evaluation flow & API reference](docs/client-reference.md)

### LLM relay credentials

The Router supplies each LLM's API URL; no Client base-URL setting is needed.
`llm_api_keys.openai`, `.xai`, and `.anthropic` select credentials by the assigned
model name (`gpt-*`, `grok-*`, `claude-*`), including when all three use one relay
host. Use that relay's respective keys, not unrelated official-provider keys.
For example:

```yaml
llm_api_keys:
  openai: env:YHLXJ_OPENAI_KEY
  xai: env:YHLXJ_XAI_KEY
  anthropic: env:YHLXJ_ANTHROPIC_KEY
deadline_ms: 120000
```

`env:` resolves process environment variables; it does not load `.env` by
itself. Literal key values in a private YAML config are also supported.
The Client sends only capability names/legacy URLs to Router, never keys.
Upgrade Router before this Client: `/next` now includes `llm_providers`.
