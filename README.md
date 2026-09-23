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
uv run --no-sync colosseum-robot configs/robot.yaml
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
