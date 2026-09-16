"""Test Router-selected model delivery and synthetic inference without robot access."""
import argparse
import asyncio
import json

from .evaluation import EvalAPI, api_url, select_task
from .local_policy import LocalPolicyClient, capabilities
from .robot_config import RobotClientConfig


async def deliver(config, run_id):
    client = LocalPolicyClient(config, api_url(config), robot_type=config.robot_type,
        control_hz=config.control_hz)
    try:
        return await client.connect(evaluation_run=run_id, verification_only=True, send_dummy=config.robot_type == 'test')
    finally:
        await client.close()


def main():
    parser = argparse.ArgumentParser(description='Verify local WSS model delivery; no robot or inference')
    parser.add_argument('config')
    parser.add_argument('--run-id', help='Verify an existing assigned local run')
    parser.add_argument('--task-id')
    parser.add_argument('--prompt')
    args = parser.parse_args()
    config = RobotClientConfig.from_yaml(args.config)
    run_check(config, run_id=args.run_id, task_id=args.task_id, prompt=args.prompt)


def run_check(config, *, run_id=None, task_id=None, prompt=None):
    if not config.policy_server_url.startswith(('ws://', 'wss://')):
        raise SystemExit('Verification requires a ws:// or wss:// policy_server_url')
    assignment_id = None
    if not run_id:
        if config.track not in {'open', 'fine-tuning'}:
            raise SystemExit('Set track: 1 or track: 2 in config')
        api = EvalAPI(config)
        try:
            profiles = asyncio.run(capabilities(config.policy_server_url))
            task_id = (task_id or select_task(api, config.robot_type)) if config.track == 'fine-tuning' else ''
            prompt = (prompt or config.instruction or input('Instruction: ').strip()) if config.track == 'open' else ''
            assignment = api.request('POST', '/next', dict(robot_id=config.robot_type, track=config.track,
                inference_mode='local', runtime_profiles=profiles, test=config.test, task_id=task_id,
                instruction=prompt, institution=config.institution.strip(), scene=config.scene, evaluator=config.evaluator,
                max_steps=config.max_trial_steps, cameras=list(config.cameras) or ['head_image']))
            run_id = next((r['id'] for r in assignment['runs'] if r['state'] == 'assigned'), None)
            if not run_id:
                raise SystemExit('No assigned run is available to verify')
            if config.robot_type == 'test':
                assignment_id = assignment['id']
                print('Test assignment: synthetic inference only; no evaluation score will be submitted.', flush=True)
            else:
                print(f"Assignment {assignment['id']} remains pending; verification does not execute it.", flush=True)
        finally:
            api.close()
    try:
        receipt = asyncio.run(deliver(config, run_id))
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
    finally:
        if assignment_id:
            api = EvalAPI(config)
            try:
                api.request('POST', f'/assignments/{assignment_id}/abort',
                    dict(reason='Synthetic communication test ended; no physical trial or score.'))
                print('Test assignment released; you can select another task next time.', flush=True)
            except Exception as exc:
                print(f'Test assignment cleanup failed ({assignment_id}): {exc}', flush=True)
            finally:
                api.close()


if __name__ == '__main__':
    main()
