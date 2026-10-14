"""Run one preregistered JEPA-WMs native PushT episode, without loading a dataset.

Task NPZ (allow_pickle=False): initial_state [7], expert_actions [30,2] in
environment units (raw relative dataset actions / 100, NOT standardized),
env_seed scalar integer, optional shape scalar string and task_id scalar string.
The 30 expert actions construct the goal by actual replay, then are discarded
from the planning interface. Only current/goal RGB and proprio enter the model.

Example: python run_native_pusht.py --task-path task.npz --output run_001
Requires adjacent load_native.py::load_model(device)->(model,preprocessor,config).
Preserves official CEM plan(), final-elite mean, and diagnostic mean unrolls;
only its candidate cost evaluation is split on action tensor dimension 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
import types


HERE = Path(__file__).resolve().parent
REPO = HERE / "jepa-wms"
for candidate in reversed((HERE / "runtime", HERE.parent / "wm01_20261002" / "runtime", REPO)):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2), encoding="utf-8")


def log_event(output, event, **fields):
    row = dict(event=event, wall_time_unix=time.time(), **fields)
    text = json.dumps(jsonable(row), ensure_ascii=False)
    print(text, flush=True)
    with (output / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def load_task(path):
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        for key in ("initial_state", "expert_actions", "env_seed"):
            if key not in data.files:
                raise ValueError(f"Task requires {key}; supplied keys: {data.files}")
        initial_state = np.array(data["initial_state"], copy=True)
        actions = np.array(data["expert_actions"], copy=True)
        env_seed = np.asarray(data["env_seed"])
        if initial_state.shape != (7,) or actions.shape != (30, 2):
            raise ValueError("Native task requires initial_state[7] and exactly 30 expert actions[30,2]")
        if not np.issubdtype(initial_state.dtype, np.floating) or not np.issubdtype(actions.dtype, np.floating):
            raise ValueError("Initial state and expert actions must be floating arrays")
        if not np.isfinite(initial_state).all() or not np.isfinite(actions).all():
            raise ValueError("Non-finite task values")
        if env_seed.size != 1 or not np.issubdtype(env_seed.dtype, np.integer):
            raise ValueError("env_seed must be an integer scalar")
        shape = str(data["shape"].item()) if "shape" in data.files else "T"
        task_id = str(data["task_id"].item()) if "task_id" in data.files else path.stem
        if "action_units" in data.files and str(data["action_units"].item()) != "relative_pixels_div_100":
            raise ValueError("Task action_units must be relative_pixels_div_100")
    return dict(initial_state=initial_state, expert_actions=actions,
                env_seed=int(env_seed.item()), shape=shape, task_id=task_id)


def make_native_env(shape):
    from omegaconf import OmegaConf
    from evals.simu_env_planning.envs.pusht_gym_wrap import make_env
    from evals.simu_env_planning.envs.wrappers.tensor import TensorWrapper
    from evals.simu_env_planning.envs.wrappers.pixels import PixelWrapper

    # Exact PushT branch of native envs.init, avoiding unrelated environment imports.
    cfg = OmegaConf.create({"task_specification": {
        "task": "pusht-base", "obs": "rgb_state", "obs_concat_channels": False,
        "max_episode_steps": 30, "num_frames": 1, "num_proprios": 1,
        "img_size": 224, "env": {"with_velocity": True, "with_target": True}}})
    env = PixelWrapper(cfg, TensorWrapper(make_env(cfg)))
    env.update_env({"shape": shape})
    return env


def observation_arrays(obs, info):
    import numpy as np
    import torch

    rgb = obs[-1].detach().cpu().numpy().transpose(1, 2, 0).copy()
    state = np.asarray(info["state"]).copy()
    prop = info["proprio"]
    if isinstance(prop, torch.Tensor):
        prop = prop.detach().cpu().numpy()
    return rgb, state, np.asarray(prop)[-1].copy()


def execute_sequence(env, actions, initial_obs, initial_info, output, name):
    """Continuous native physics; persist every step, including partial failures."""
    import numpy as np
    import torch
    from PIL import Image

    directory = output / name
    directory.mkdir()
    rgbs, states, proprios, rewards, dones, executed = [], [], [], [], [], []
    obs, info = initial_obs, initial_info

    def record(index):
        rgb, state, prop = observation_arrays(obs, info)
        rgbs.append(rgb)
        states.append(state)
        proprios.append(prop)
        Image.fromarray(rgb).save(directory / f"frame_{index:03d}.png")
        np.savez_compressed(directory / "trajectory_partial.npz", rgb=np.stack(rgbs),
                            state=np.stack(states), proprio=np.stack(proprios),
                            action=np.asarray(executed), reward=np.asarray(rewards),
                            done=np.asarray(dones, dtype=bool))

    record(0)
    for index, action in enumerate(actions):
        physical_action = torch.as_tensor(action).detach().cpu().clone()
        obs, reward, done, truncated, info = env.step(physical_action)
        executed.append(physical_action.numpy().copy())
        rewards.append(float(reward))
        dones.append(bool(done))
        record(index + 1)
        if done and index + 1 != len(actions):
            raise RuntimeError(f"Native time limit terminated {name} after {index+1}/{len(actions)} actions")
        log_event(output, "physical_step", phase=name, step=index+1, total=len(actions))
    if len(executed) != 30 or not dones[-1]:
        raise RuntimeError("Native episode must terminate at action 30")
    payload = dict(rgb=np.stack(rgbs), state=np.stack(states), proprio=np.stack(proprios),
                   action=np.stack(executed), reward=np.asarray(rewards), done=np.asarray(dones, dtype=bool))
    np.savez_compressed(directory / "trajectory.npz", **payload)
    return obs, info, payload


def main(args):
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {output}")
    output.mkdir(parents=True)
    env = None
    start = time.perf_counter()
    try:
        task_path = args.task_path.resolve(strict=True)
        task = load_task(task_path)
        torch.set_num_threads(args.threads)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        source_files = [Path(__file__), HERE / "load_native.py"]
        source_files += [REPO / p for p in (
            "evals/simu_env_planning/envs/pusht_env/pusht_env.py",
            "evals/simu_env_planning/envs/pusht_gym_wrap.py",
            "evals/simu_env_planning/envs/wrappers/pixels.py",
            "evals/simu_env_planning/envs/wrappers/tensor.py",
            "evals/simu_env_planning/envs/wrappers/time_limit.py",
            "evals/simu_env_planning/planning/planning/planner.py",
            "evals/simu_env_planning/planning/planning/objectives.py",
            "app/vjepa_wm/modelcustom/simu_env_planning/vit_enc_preds.py")]
        lock = dict(command=sys.argv, task_path=str(task_path), task_sha256=sha256(task_path),
                    task_id=task["task_id"], env_seed=task["env_seed"], planner_seed=args.seed,
                    source_hashes={str(p): sha256(p) for p in source_files if p.exists()},
                    protocol={"planning_calls":1,"iterations":30,"num_samples":300,"num_elites":10,
                              "model_horizon":6,"frameskip":5,"physical_budget":30,
                              "goal_replay_actions":30,"objective":"terminal L2 visual + 0.1 proprio",
                              "candidate_chunk":args.chunk,"dtype":"float32","done_at_success":False,
                              "heads":False,"diagnostic_mean_unrolls":30},
                    task_access="One explicit task NPZ; no dataset sampling in this runner",
                    privileged_info="Expert actions for goal construction; state for final evaluation only",
                    version={"python":sys.version,"torch":torch.__version__,"numpy":np.__version__})
        save_json(output / "pre_run_lock.json", lock)
        np.savez(output / "task_input_copy.npz", **task)
        log_event(output, "task_loaded", task_id=task["task_id"])
        env = make_native_env(task["shape"])
        obs, info = env.prepare(task["env_seed"], task["initial_state"])
        initial_rgb, initial_state, initial_prop = observation_arrays(obs, info)
        goal_obs, goal_info, expert_trace = execute_sequence(
            env, task["expert_actions"], obs, info, output, "expert_replay")
        goal_rgb, goal_state, goal_prop = observation_arrays(goal_obs, goal_info)
        goal_model_obs = {"visual":goal_obs.detach().clone().unsqueeze(0),
                          "proprio":goal_info["proprio"].detach().clone().unsqueeze(0)}
        obs, info = env.prepare(task["env_seed"], task["initial_state"])
        reset_rgb, reset_state, reset_prop = observation_arrays(obs, info)
        if not (np.array_equal(initial_rgb, reset_rgb) and np.array_equal(initial_state, reset_state)
                and np.array_equal(initial_prop, reset_prop)):
            raise RuntimeError("Expert-to-planner reset did not reproduce initial observation/state/proprio exactly")
        np.savez_compressed(output / "task_realization.npz", initial_rgb=initial_rgb,
                            initial_state=initial_state, initial_proprio=initial_prop,
                            goal_rgb=goal_rgb, goal_state=goal_state, goal_proprio=goal_prop)
        log_event(output, "goal_replay_complete", steps=30)
        if args.environment_only:
            save_json(output / "environment_only.json", {"status":"complete","reset_exact":True,
                      "task_id":task["task_id"],"model_run":False,"elapsed_seconds":time.perf_counter()-start})
            return

        from load_native import load_model
        from evals.simu_env_planning.planning.planning.planner import CEMPlanner
        from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective

        load_start = time.perf_counter()
        model, preprocessor, config = load_model(device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if len(model.heads) != 0:
            raise ValueError("Loader must disable optional decoder heads")
        if model.action_dim != 10 or model.action_skip != 1 or model.ctxt_window != 2:
            raise ValueError("Unexpected native PushT model action/context configuration")
        config_value = OmegaConf.to_container(config, resolve=False) if OmegaConf.is_config(config) else config
        save_json(output / "loaded_config.json", config_value)
        log_event(output, "model_loaded", elapsed_seconds=time.perf_counter()-load_start)
        with torch.no_grad():
            goal_encoding = model.encode(goal_model_obs, act=False).detach()
            z_init = model.encode({"visual":obs.unsqueeze(0),
                                  "proprio":info["proprio"].unsqueeze(0)}, act=True).detach()
        objective = ReprTargetDistMPCObjective(config, target_enc=goal_encoding,
                                              sum_all_diffs=False, alpha=0.1)
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)
        planner = CEMPlanner(unroll=model.unroll, iterations=30, num_samples=300, horizon=6,
                             action_dim=10, num_elites=10, var_scale=1.0,
                             momentum_mean=0.0, momentum_std=0.0, max_norms=None,
                             distribute_planner=False, local_generator=generator,
                             num_act_stepped=6, decode_each_iteration=False, decode_unroll=None)
        planner.device = device
        planner.set_objective(objective)
        trace_dir = output / "cem"
        trace_dir.mkdir()
        cost_calls = []

        def sync():
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        def chunked_cost(this, actions, encoding):
            sync()
            tick = time.perf_counter()
            costs = []
            for begin in range(0, actions.shape[1], args.chunk):
                part = actions[:, begin:begin+args.chunk]
                predicted = this.unroll(encoding, part)
                costs.append(this.objective(predicted, part))
                del predicted
            result = torch.cat(costs, dim=0)
            if not torch.isfinite(result).all():
                np.savez_compressed(trace_dir / f"iteration_{len(cost_calls):02d}_nonfinite.npz",
                                    actions=actions.detach().cpu().numpy(), cost=result.detach().cpu().numpy())
                raise FloatingPointError("Non-finite CEM candidate costs")
            sync()
            index = len(cost_calls)
            np.savez_compressed(trace_dir / f"iteration_{index:02d}.npz",
                                actions=actions.detach().cpu().numpy(), cost=result.detach().cpu().numpy())
            timing = {"iteration":index,"candidate_count":actions.shape[1],
                      "cost_seconds":time.perf_counter()-tick,
                      "minimum_cost":result.min().item()}
            cost_calls.append(timing)
            log_event(output, "cem_iteration", **timing)
            return result

        planner.cost_function = types.MethodType(chunked_cost, planner)
        np.savez(output / "rng_before_planning.npz", cpu=torch.get_rng_state().numpy(),
                 planner=generator.get_state().cpu().numpy(),
                 cuda=torch.cuda.get_rng_state(device).cpu().numpy() if device.type=="cuda" else np.array([],dtype=np.uint8))
        if args.audit_chunk:
            # Independent fixed candidates; does not advance the planning generator.
            probe_generator = torch.Generator(device=device).manual_seed(args.seed+100000)
            probe_count = max(16, min(64, args.chunk))
            probe_chunk = min(8, args.chunk)
            probe = torch.randn(6, probe_count, 10, generator=probe_generator, device=device)
            with torch.no_grad():
                direct = objective(model.unroll(z_init, probe), probe)
                pieces = [objective(model.unroll(z_init, probe[:,i:i+probe_chunk]), probe[:,i:i+probe_chunk])
                          for i in range(0,probe_count,probe_chunk)]
                split = torch.cat(pieces)
            diff = (direct-split).abs()
            same_top10 = torch.equal(torch.topk(-direct,10).indices,torch.topk(-split,10).indices)
            close = torch.allclose(direct, split, atol=1e-5, rtol=1e-5)
            save_json(output / "chunk_equivalence.json", {
                "candidates":probe_count,"split_batch":probe_chunk,"diagnostic_only":True,
                "tolerance_atol":1e-5,"tolerance_rtol":1e-5,"costs_within_tolerance":close,
                "max_abs_cost_difference":diff.max().item(),
                "max_relative_cost_difference":(diff/direct.abs().clamp_min(1e-12)).max().item(),
                "same_top10_order":same_top10,
                "direct":direct.cpu().numpy(),"split":split.cpu().numpy()})
            np.savez(output / "chunk_equivalence_inputs.npz", actions=probe.cpu().numpy())
            if not (close and same_top10):
                raise RuntimeError("Candidate-batching audit failed cost/ranking tolerance; investigate before planning")
            del direct, split, pieces, probe
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        sync()
        planning_start = time.perf_counter()
        result = planner.plan(z_init, steps_left=6)
        sync()
        planning_seconds = time.perf_counter()-planning_start
        if len(cost_calls) != 30 or any(row["candidate_count"] != 300 for row in cost_calls):
            raise RuntimeError("CEM did not execute native 300x30 candidate budget")
        standardized_actions = result.actions.detach().cpu().reshape(30,2)
        physical_actions = preprocessor.denormalize_actions(standardized_actions).detach().cpu()
        np.savez_compressed(output / "plan.npz", actions_model=result.actions.detach().cpu().numpy(),
                            actions_standardized=standardized_actions.numpy(),
                            actions_environment=physical_actions.numpy(),
                            losses=result.losses.detach().cpu().numpy(),
                            elite_loss_mean=result.prev_elite_losses_mean.detach().cpu().numpy(),
                            elite_loss_std=result.prev_elite_losses_std.detach().cpu().numpy())
        save_json(output / "cem_timing.json", cost_calls)
        log_event(output, "planning_complete", seconds=planning_seconds)
        final_obs, final_info, agent_trace = execute_sequence(
            env, physical_actions, obs, info, output, "agent_execution")
        final_state = np.asarray(final_info["state"]).copy()
        native_result = env.eval_state(goal_state, final_state)
        pos_error = float(np.linalg.norm(goal_state[:4].astype(np.float64)-final_state[:4].astype(np.float64)))
        angle_error = float(abs((float(goal_state[4])-float(final_state[4])+np.pi)%(2*np.pi)-np.pi))
        independent_success = pos_error < 20 and angle_error < np.pi/9
        if bool(native_result["success"]) != independent_success:
            raise RuntimeError("Native terminal criterion disagrees with independent float64 check")
        summary = {"status":"complete","task_id":task["task_id"],"unique_tasks":1,
                   "success":bool(native_result["success"]),"position_first4_l2":pos_error,
                   "wrapped_angle_error":angle_error,"native_state_distance":float(native_result["state_dist"]),
                   "planning_seconds":planning_seconds,"total_seconds":time.perf_counter()-start,
                   "physical_actions":30,"expert_goal_replay_actions":30,"planning_calls":1,
                   "candidate_sequences_evaluated":9000,"diagnostic_mean_unrolls":30,
                   "chunk_audit_extra_sequences":2*probe_count if args.audit_chunk else 0,
                   "cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
                   "cuda_peak_reserved_bytes":torch.cuda.max_memory_reserved(device) if device.type=="cuda" else None,
                   "final_state":final_state,"goal_state":goal_state,"reset_exact":True,
                   "scope":"Single native one-plan/30-action development episode; not full paper replication"}
        save_json(output / "summary.json", summary)
        log_event(output, "complete", success=summary["success"], position_l2=pos_error)
    except BaseException as error:
        save_json(output / "failure.json", {"error":repr(error),"traceback":traceback.format_exc(),
                                           "elapsed_seconds":time.perf_counter()-start})
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--audit-chunk", action="store_true")
    parser.add_argument("--environment-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.chunk < 1 or arguments.chunk > 300:
        parser.error("--chunk must be between 1 and 300")
    main(arguments)
