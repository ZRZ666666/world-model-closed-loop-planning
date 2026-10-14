"""Isolated persistent JEPA PushT worker. No Temporal or model imports.

Only --execute worker invocations initialize a simulator. Commands and observed
feedback use a JSON line protocol; frames/full state are saved in output files.
Full block state is for reset/evaluation only, never an action optimizer input.
"""
import argparse
import json
import os
import math
from pathlib import Path
import sys
import traceback
from common import WM03, read_environment_reference, protocol_entry, save, sha

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def physics_snapshot(core):
    """Actual Pymunk properties, not ineffectual Body.friction user attributes."""
    def number(value):
        value = float(value)
        return value if math.isfinite(value) else ('positive_infinity' if value > 0 else 'negative_infinity')
    def shape_record(shape):
        record = dict(type=type(shape).__name__, friction=number(shape.friction),
            elasticity=number(shape.elasticity), sensor=bool(shape.sensor),
            collision_type=int(shape.collision_type), radius=number(shape.radius),
            filter=dict(group=int(shape.filter.group), categories=int(shape.filter.categories), mask=int(shape.filter.mask)))
        if hasattr(shape, 'get_vertices'):
            record['vertices'] = [[number(v.x), number(v.y)] for v in shape.get_vertices()]
        if hasattr(shape, 'offset'):
            record['offset'] = [number(v) for v in shape.offset]
        if hasattr(shape, 'a') and hasattr(shape, 'b'):
            record['a'] = [number(v) for v in shape.a]
            record['b'] = [number(v) for v in shape.b]
        return record
    def body_record(body):
        shapes = [shape_record(shape) for shape in body.shapes]
        return dict(body_type=int(body.body_type), mass=number(body.mass), moment=number(body.moment),
            center_of_gravity=[number(v) for v in body.center_of_gravity],
            shapes=sorted(shapes, key=lambda x: json.dumps(x, sort_keys=True)))
    space = core.space
    return dict(sim_hz=int(core.sim_hz), control_hz=int(core.control_hz),
        microstep_dt=1/float(core.sim_hz), k_p=number(core.k_p), k_v=number(core.k_v),
        action_scale=number(core.action_scale), relative=bool(core.relative), shape=core.shape,
        with_velocity=bool(core.with_velocity), with_target=bool(core.with_target),
        render_size=int(core.render_size), window_size=int(core.window_size),
        space=dict(gravity=[number(v) for v in space.gravity], damping=number(space.damping),
            iterations=int(space.iterations), collision_slop=number(space.collision_slop),
            collision_bias=number(space.collision_bias), collision_persistence=int(space.collision_persistence),
            sleep_time_threshold=number(space.sleep_time_threshold), idle_speed_threshold=number(space.idle_speed_threshold)),
        agent=body_record(core.agent), block=body_record(core.block), static_body=body_record(space.static_body))


def main(args):
    import numpy as np
    sys.path.insert(0, str(WM03))
    import run_native_pusht as native
    import torch
    from omegaconf import OmegaConf
    from evals.simu_env_planning.envs.pusht_gym_wrap import make_env
    from evals.simu_env_planning.envs.wrappers.tensor import TensorWrapper
    from evals.simu_env_planning.envs.wrappers.pixels import PixelWrapper
    torch.set_num_threads(2)
    protocol_entry(args.reference_cache, args.moment_factor, seed=1)
    task, realization, hashes = read_environment_reference(args.reference_cache)
    cfg = OmegaConf.create({'task_specification': dict(task='pusht-base', obs='rgb_state',
        obs_concat_channels=False, max_episode_steps=args.max_actions, num_frames=1,
        num_proprios=1, img_size=224, env=dict(with_velocity=True, with_target=True))})
    env = PixelWrapper(cfg, TensorWrapper(make_env(cfg)))
    env.update_env({'shape': task['shape']})
    obs, info = env.prepare(task['env_seed'], task['initial_state'])
    rgb, state, prop = native.observation_arrays(obs, info)
    for value, key in zip((rgb, state, prop), ('initial_rgb', 'initial_state', 'initial_proprio')):
        assert np.array_equal(value, realization[key]), f'Initial {key} changed'
    core = env.unwrapped
    physics_before = physics_snapshot(core)
    original_moment = float(core.block.moment)
    assert float(core.block.mass) == 1.0 and math.isfinite(original_moment) and original_moment > 0
    if args.moment_factor != 1.0:
        core.block.moment = original_moment * args.moment_factor
    assert float(core.block.moment) == original_moment * args.moment_factor
    physics_after = physics_snapshot(core)
    expected_after = json.loads(json.dumps(physics_before))
    expected_after['block']['moment'] = original_moment * args.moment_factor
    assert physics_after == expected_after, 'Physics parameters other than block moment changed'
    # Fresh render/state after the setter, with no physics step. Do not compare
    # only stale cached obs/infos, which could hide unintended reset changes.
    after_rgb = env.render(mode='rgb_array', width=224, height=224)
    after_state = core._get_obs()
    after_prop = np.concatenate((after_state[:2], after_state[5:]))
    assert np.array_equal(after_rgb, rgb) and np.array_equal(after_state, state) and np.array_equal(after_prop, prop)
    physics_receipt = dict(action_gain=args.action_gain, moment_factor=args.moment_factor,
        original_block_moment=original_moment, actual_block_moment=float(core.block.moment),
        before=physics_before, after=physics_after, only_block_moment_may_change=True,
        post_setter_initial_rgb_state_proprio_bitwise_equal=True,
        gain_implementation='Applied only immediately before env.step: applied_command = command * float32(gain), before original action_scale100 PD target.',
        action_log_convention='action and action_command are pre-gain physical-unit commands; action_applied is the true input passed to unchanged env.step.',
        perturbation_visible_to_planner=False,
        source_hashes={str(Path(__file__).resolve()):sha(__file__),
            str(WM03/'jepa-wms/evals/simu_env_planning/envs/pusht_env/pusht_env.py'):sha(WM03/'jepa-wms/evals/simu_env_planning/envs/pusht_env/pusht_env.py')})
    save(args.output/'physics_receipt.json', physics_receipt)
    physics_receipt_sha = sha(args.output/'physics_receipt.json')
    rgbs, states, proprios = [rgb], [state], [prop]
    actions, applied_actions, rewards, dones, truncations = [], [], [], [], []

    def persist(tag):
        path = args.output / f'{tag}.npz'
        assert not path.exists()
        np.savez_compressed(path, rgb=np.stack(rgbs), state=np.stack(states), proprio=np.stack(proprios),
            action=np.asarray(actions, dtype=np.float32).reshape(-1, 2),
            action_command=np.asarray(actions, dtype=np.float32).reshape(-1, 2),
            action_applied=np.asarray(applied_actions, dtype=np.float32).reshape(-1, 2), reward=np.asarray(rewards),
            done=np.asarray(dones, bool), truncated=np.asarray(truncations, bool))
        joint = float(np.linalg.norm(states[-1][:4] - realization['goal_state'][:4]))
        delta = states[-1][4] - realization['goal_state'][4]
        angle = float(abs(np.arctan2(np.sin(delta), np.cos(delta))))
        result = dict(path=str(path), actions=len(actions), simulator_resets=1,
            new_actions=0, joint_position_error=joint, angle_error=angle,
            pusher_error=float(np.linalg.norm(states[-1][:2] - realization['goal_state'][:2])),
            block_error=float(np.linalg.norm(states[-1][2:4] - realization['goal_state'][2:4])),
            success=bool(joint < 20 and angle < np.pi / 9), gpu_initialized=torch.cuda.is_initialized(),
            physics_receipt_sha256=physics_receipt_sha)
        return result

    initial = persist('initial')
    initial['reference_hashes'] = hashes
    print('WM11_JSON ' + json.dumps(initial), flush=True)
    try:
        for line in sys.stdin:
            command = json.loads(line)
            if command['command'] == 'close':
                break
            assert command['command'] == 'step'
            physical = np.asarray(command['actions'], dtype=np.float32)
            assert physical.shape == (5, 2) and np.isfinite(physical).all()
            assert len(actions) + 5 <= args.max_actions
            for action in physical:
                applied = action.copy() if args.action_gain == 1.0 else np.asarray(action * np.float32(args.action_gain), dtype=np.float32)
                assert np.isfinite(applied).all()
                obs, reward, done, truncated, info = env.step(torch.from_numpy(applied.copy()))
                rgb, state, prop = native.observation_arrays(obs, info)
                actions.append(action.copy()); applied_actions.append(applied.copy()); rewards.append(float(reward))
                dones.append(bool(done)); truncations.append(bool(truncated))
                rgbs.append(rgb); states.append(state); proprios.append(prop)
                # Every physical step is durable, including a failed block.
                result = persist(f'step_{len(actions):03d}')
                assert not done or len(actions) == args.max_actions
            result['new_actions'] = 5
            print('WM11_JSON ' + json.dumps(result), flush=True)
    finally:
        env.close()
        save(args.output / 'worker_terminal.json', dict(status='closed', physical_actions=len(actions),
            simulator_resets=1, gpu_initialized=torch.cuda.is_initialized(), reference_hashes=hashes,
            physics_receipt_sha256=physics_receipt_sha))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-actions', type=int, required=True)
    parser.add_argument('--execute', action='store_true', required=True)
    parser.add_argument('--action-gain', type=float, default=1.0)
    parser.add_argument('--moment-factor', type=float, default=1.0)
    parser.add_argument('--reference-cache', type=Path, required=True)
    args = parser.parse_args()
    assert args.action_gain == 1.0 and args.moment_factor in (1.0, 2.0)
    assert args.max_actions == 100
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        main(args)
    except BaseException as error:
        save(args.output / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc()))
        raise
