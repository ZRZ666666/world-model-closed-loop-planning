"""WM13 unchanged frozen Temporal GD/MPC on eight cached development targets.

Default invocation is source-only --audit-only. --execute requires explicit
selection and performs the locked consumed-development task, no new targets.
No native JEPA model imports enter this process; the simulator is a subprocess.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import pickletools
import random
import shutil
import subprocess
import sys
import time
import traceback
import types
import zipfile
from common import (BASE, WORK, REPO, ASSETS, CONFIG, CHECKPOINT, COMMIT, CKPT_SHA,
    CFG_SHA, COMPONENTS, ACTION_MEAN, ACTION_STD, PROPRIO_MEAN, PROPRIO_STD,
    sha, jr, save, provenance, read_planner_reference, protocol_entry, PROTOCOL, PROTOCOL_SHA)

sys.dont_write_bytecode = True
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')


def state_digest(module):
    import hashlib
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        value = value.detach().contiguous().cpu()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_model(output, device):
    import torch
    from omegaconf import OmegaConf
    assert sha(CHECKPOINT) == CKPT_SHA and sha(CONFIG) == CFG_SHA
    assert 'models' not in sys.modules and 'utils' not in sys.modules
    sys.path.insert(0, str(REPO))
    # Reuse the already installed small Hydra runtime, after site packages.
    # Do not import JEPA entrypoints, which would reorder generic model packages.
    sys.path.append(str(WORK / 'wm01_20261002/runtime'))
    from planning.mpc import MPCPlanner
    from planning.gd import GDPlanner
    import hydra
    save(output / 'planning_dependency_preflight.json', dict(hydra_module=hydra.__file__,
        hydra_version=hydra.__version__, official_mpc_module=sys.modules['planning.mpc'].__file__,
        official_gd_module=sys.modules['planning.gd'].__file__, installs=0))
    with zipfile.ZipFile(CHECKPOINT) as archive:
        name = next(n for n in archive.namelist() if n.endswith('data.pkl'))
        operations = list(pickletools.genops(archive.read(name)))
    assert not any(op.name == 'STACK_GLOBAL' for op, _, _ in operations)
    for entry in sorted({arg for op, arg, _ in operations if op.name == 'GLOBAL'}):
        module_name = entry.split(' ')[0]
        importlib.import_module('builtins' if module_name == '__builtin__' else module_name)
    from models.visual_world_model import VWorldModel
    config = OmegaConf.load(CONFIG)
    kwargs = dict(OmegaConf.to_container(config.model, resolve=True))
    assert kwargs.pop('_target_') == 'models.visual_world_model.VWorldModel'
    save(output / 'checkpoint_load_started.json', dict(load_number=1, map_location='cpu', sha256=CKPT_SHA))
    payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    parts = {k: payload[k] for k in COMPONENTS}
    before = {k: state_digest(v) for k, v in parts.items()}
    kwargs.update(parts)
    kwargs.update(proprio_dim=int(config.proprio_emb_dim), action_dim=int(config.action_emb_dim),
        concat_dim=int(config.concat_dim), num_action_repeat=int(config.num_action_repeat),
        num_proprio_repeat=int(config.num_proprio_repeat))
    model = VWorldModel(**kwargs)
    model.eval(); model.requires_grad_(False)
    assert before == {k: state_digest(v) for k, v in parts.items()}
    model.to(device)
    # Author attention's plain Tensor bias is not a registered buffer. Move it
    # without reconstructing attention or changing any value/mask semantics.
    plain_tensors = []
    for module_name, module in model.named_modules():
        for name, value in list(vars(module).items()):
            if isinstance(value, torch.Tensor) and name not in module._parameters and name not in module._buffers:
                checksum = state_digest(_TensorHolder(value))
                setattr(module, name, value.to(device))
                assert state_digest(_TensorHolder(getattr(module, name))) == checksum
                plain_tensors.append(dict(module=module_name, attribute=name, shape=list(value.shape), sha256=checksum))
    assert all(p.device == device and not p.requires_grad for p in model.parameters())
    assert not any(m.training for m in model.modules())
    assert before == {k: state_digest(v) for k, v in parts.items()}
    save(output / 'load_receipt.json', dict(checkpoint_loads=1, checkpoint_sha256=CKPT_SHA,
        config_sha256=CFG_SHA, repo_commit=COMMIT, model_parameter_count=sum(p.numel() for p in model.parameters()),
        component_state_hashes=before, plain_tensor_device_migrations=plain_tensors,
        state_unchanged=True, parameters_frozen=True, device=str(device), epoch=int(payload['epoch'])))
    return model, config


def _TensorHolder(tensor):
    # Small hash-only helper; no forward or model construction.
    import torch
    holder = torch.nn.Module()
    holder.register_buffer('value', tensor)
    return holder


def verify_plain_tensors(model, output, device):
    modules = dict(model.named_modules())
    records = jr(output / 'load_receipt.json')['plain_tensor_device_migrations']
    for record in records:
        value = getattr(modules[record['module']], record['attribute'])
        assert value.device == device
        assert list(value.shape) == record['shape']
        assert state_digest(_TensorHolder(value)) == record['sha256']
    return len(records)


class LocalLog:
    def __init__(self, output):
        self.output = output
    def log(self, row):
        with (self.output / 'official_logs.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row) + '\n')


class PersistentEvaluator:
    """MPC evaluator contract; cumulative plans execute only unexecuted suffix.

Official MPC sets its evaluator's initial condition back to the origin each
round. Those assignments are metadata only here: the real simulator remains
alive. Prefix identity assertions make accidental replay/branching impossible.
"""
    frameskip = 5
    def __init__(self, output, max_actions, preprocessor, expected_realization, reference_cache, action_gain=1.0, moment_factor=1.0):
        import numpy as np
        self.output, self.preprocessor = output, preprocessor
        self.executed_model = np.empty((1, 0, 10), dtype=np.float32)
        self.last = None
        self.stderr = (output / 'env_stderr.log').open('w', encoding='utf-8')
        self.process = subprocess.Popen([sys.executable, str(BASE / 'env_worker.py'), '--execute',
            '--max-actions', str(max_actions), '--action-gain', str(action_gain),
            '--moment-factor', str(moment_factor), '--reference-cache', str(reference_cache),
            '--output', str(output / 'environment')],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, text=True,
            encoding='utf-8', bufsize=1, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.last = self.receive()
        obs, state = self.read_trajectory(self.last['path'])
        for key, ref_key in (('visual', 'initial_rgb'), ('proprio', 'initial_proprio')):
            assert np.array_equal(obs[key][0, 0], expected_realization[ref_key])
        # Full initial state and goal state are cache-checked inside the CPU
        # worker; reference loading here decodes only the four RGB/proprio fields.
        self.obs_0, self.state_0 = obs, state[:, 0]
        self.initial_obs, self.initial_state = obs, state[:, 0]
    def receive(self):
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f'Environment worker ended with {self.process.poll()}; inspect env_stderr.log')
            if line.startswith('WM11_JSON '):
                return json.loads(line[len('WM11_JSON '):])
            with (self.output / 'env_stdout.log').open('a', encoding='utf-8') as handle:
                handle.write(line)
    def read_trajectory(self, path):
        import numpy as np
        with np.load(path, allow_pickle=False) as data:
            return dict(visual=data['rgb'][None].copy(), proprio=data['proprio'][None].copy()), data['state'][None].copy()
    def assign_init_cond(self, obs_0, state_0):
        self.obs_0, self.state_0 = obs_0, state_0
    def get_init_cond(self):
        return self.obs_0, self.state_0
    def eval_actions(self, actions, action_len=None, filename='output', save_video=False):
        import numpy as np
        actions_cpu = actions.detach().cpu()
        actions_np = actions_cpu.numpy().copy()
        old_len = self.executed_model.shape[1]
        assert actions_np.shape == (1, old_len + 1, 10)
        assert np.array_equal(actions_np[:, :old_len], self.executed_model)
        physical = self.preprocessor.denormalize_actions(actions_cpu[:, -1:].reshape(1, 5, 2)).numpy()[0]
        assert np.isfinite(physical).all()
        self.process.stdin.write(json.dumps(dict(command='step', actions=physical.tolist())) + '\n')
        self.process.stdin.flush()
        self.last = self.receive()
        self.executed_model = actions_np
        assert self.last['actions'] == actions_np.shape[1] * 5 and self.last['new_actions'] == 5
        obs, states = self.read_trajectory(self.last['path'])
        save(self.output / f'feedback_{actions_np.shape[1]-1:02d}.json', self.last)
        print(json.dumps(dict(event='real_feedback', round=actions_np.shape[1]-1, **self.last)), flush=True)
        logs = {key: value for key, value in self.last.items() if key in
            ('joint_position_error', 'angle_error', 'pusher_error', 'block_error')}
        logs['success_rate'] = float(self.last['success'])
        return logs, np.array([self.last['success']], dtype=bool), obs, states
    def close(self):
        if self.stderr.closed:
            return
        if self.process.poll() is None:
            self.process.stdin.write(json.dumps({'command': 'close'}) + '\n')
            self.process.stdin.flush()
            self.process.wait(timeout=30)
        self.stderr.close()


def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    evaluator = None
    hooks = []
    try:
        source_hashes = provenance()
        queue_entry = protocol_entry(args.reference_cache, args.moment_factor, args.seed)
        source_directory = output / 'source_snapshot'
        source_directory.mkdir()
        source_index = {}
        for i, (path, digest) in enumerate(source_hashes.items()):
            dest = source_directory / f'{i:02d}_{Path(path).name}'
            shutil.copyfile(path, dest)
            assert sha(dest) == digest
            source_index[path] = dict(snapshot=str(dest), sha256=digest)
        save(output / 'source_snapshot_index.json', source_index)
        budget = dict(raw_horizon=25, model_horizon=5, frameskip=5, executed_raw_actions_per_round=5,
            maximum_rounds=args.max_rounds, maximum_physical_actions=5*args.max_rounds,
            gd_steps_per_round=args.gd_steps, maximum_rollout_calls=args.gd_steps*args.max_rounds,
            maximum_selected_model_transitions=5*args.gd_steps*args.max_rounds,
            maximum_backward_calls=args.gd_steps*args.max_rounds, simulator_resets=1,
            extra_evaluation_model_rollouts=0, expert_replay_actions=0)
        lock = dict(schema='wm13-temporal-spatial-paired-development-v1', command=[sys.executable, *sys.argv],
            source_hashes=source_hashes, checkpoint_sha256=CKPT_SHA, config_sha256=CFG_SHA, repo_commit=COMMIT,
            preregistration_path=str(PROTOCOL), preregistration_sha256=PROTOCOL_SHA, queue_entry=queue_entry,
            seed=args.seed, budget=budget, official_gd=dict(lr=0.1, sample_type='zero', action_noise=0,
                optimizer='adam', use_cosine_scheduler=True, scheduler_eta_min=0, eval_every=-1),
            objective=dict(alpha=1, base=2, mode='staged', stage1_rounds='0..4', stage2_rounds='5..19 if run',
                all_mode_extra_mean_over_time='preserved verbatim from official objective'),
            perturbation=dict(action_gain=args.action_gain, block_moment_factor=args.moment_factor,
                initialization='Raw reset and cached initial match first; moment set after reset before first real action, then fresh render/state/proprio must remain byte-identical.',
                gain_applied='Physical command multiplied only inside environment worker before unchanged env.step/PD action_scale100.',
                action_logging='action and action_command are original pre-gain physical-unit commands; action_applied is the actual perturbed env.step input.',
                hidden_from_planner=True, proprio='True observed pusher xyvxvy, unmodified.', rgb='Original renderer, unmodified.',
                other_physics='Block mass, geometry/center-of-gravity, actual shape friction, damping, PD gains and all other fixed physics remain original; runtime values and hash recorded.',
                interpretation='Frozen strong-model deployment failure probe, not an adaptation method or original contribution.'),
            observations=dict(current_frames=1, goal_frames=1, initial_history_padding=False,
                feedback_history_padding=False, action_history_padding=False, max_recursive_context=3,
                image_normalization='uint8/255 then Resize224 CenterCrop224 Normalize(mean=.5,std=.5)',
                action_mean=ACTION_MEAN, action_std=ACTION_STD, proprio_mean=PROPRIO_MEAN, proprio_std=PROPRIO_STD),
            warmstart='remaining four optimized action tokens + one physical-zero token appended by official GD init_actions',
            differences=['Preselected train development target, cached by actual nominal25-action reference replay; no source/reference action array read by this runner.',
                'JEPA simulator process reused; matching physics source, 100-step TimeLimit at default max_rounds20.',
                'Persistent real execution replaces author cumulative reset/replay; no replayed-prefix interactions.',
                'Omit author evaluator diagnostic model rollout/decoding; unchanged action optimization and feedback rule.',
                'Eight preselected development targets across three exact initial layouts; author benchmark sampling/independence not reproduced.'],
            permissions='Reference loading by model process deserializes only current/goal RGB and pusher xyvxvy; CPU worker alone decodes raw reset metadata and cached full goal state for reset/evaluation. Actual simulator full states remain evaluator bookkeeping and never enter world-model inputs/objective. No source_tasks/expert/reference trajectories/train tensors opened.',
            training_exposure='Unverified; consumed development/interface experiment, not independent held-out evidence.',
            model_updated=False, status='pre_execution_lock', source_only=args.audit_only,
            cpu_preflight=bool(args.cpu_preflight))
        save(output / 'pre_run_lock.json', lock)
        if args.audit_only:
            save(output / 'audit_only.json', dict(status='source_ready', source_hashes=source_hashes,
                model_loads=0, forwards=0, environment_steps=0, task_reads=0, dataset_reads=0))
            print('Source-only audit complete; no task, model, CUDA or simulator loaded.', flush=True)
            return
        import numpy as np
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        device = torch.device(args.device)
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda', 0)
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        if device.type == 'cuda':
            torch.cuda.set_device(device); torch.cuda.manual_seed_all(args.seed)
            torch.cuda.reset_peak_memory_stats(device)
        task, realization, reference_hashes = read_planner_reference(args.reference_cache)
        save(output / 'reference_receipt.json', dict(task_id=task['task_id'], reference_hashes=reference_hashes,
            expert_action_arrays_read=0, current_goal_from_cached_realization=True))
        model, config = load_model(output, device)
        before = state_digest(model)
        from preprocessor import Preprocessor
        from datasets.img_transforms import default_transform
        from planning.objectives import create_objective_fn
        from planning.mpc import MPCPlanner
        preprocessor = Preprocessor(action_mean=torch.tensor(ACTION_MEAN, dtype=torch.float32),
            action_std=torch.tensor(ACTION_STD, dtype=torch.float32), state_mean=None, state_std=None,
            proprio_mean=torch.tensor(PROPRIO_MEAN, dtype=torch.float32),
            proprio_std=torch.tensor(PROPRIO_STD, dtype=torch.float32), transform=default_transform(224))
        evaluator = PersistentEvaluator(output, 5*args.max_rounds, preprocessor, realization,
            reference_cache=args.reference_cache, action_gain=args.action_gain, moment_factor=args.moment_factor)
        goal = dict(visual=realization['goal_rgb'][None, None], proprio=realization['goal_proprio'][None, None])
        if args.cpu_preflight:
            from planning.gd import GDPlanner
            assert device.type == 'cpu' and not torch.cuda.is_initialized()
            trans_obs = {k: v.to(device) for k, v in preprocessor.transform_obs(evaluator.initial_obs).items()}
            trans_goal = {k: v.to(device) for k, v in preprocessor.transform_obs(goal).items()}
            objective = create_objective_fn(alpha=1, base=2, mode='staged')
            dummy_planner = GDPlanner(horizon=5, action_noise=0, sample_type='zero', lr=.1,
                opt_steps=args.gd_steps, eval_every=-1, wm=model, action_dim=10,
                objective_fn=objective, preprocessor=preprocessor, evaluator=evaluator,
                wandb_run=LocalLog(output), log_filename=None)
            actions = dummy_planner.init_actions(evaluator.initial_obs).to(device).requires_grad_(True)
            assert actions.shape == (1, 5, 10)
            zero_roundtrip_error = preprocessor.denormalize_actions(actions.reshape(1, 25, 2)).abs().max().item()
            assert zero_roundtrip_error <= 1e-8, 'Float32 physical-zero roundtrip exceeded numerical tolerance'
            with torch.no_grad():
                target = model.encode_obs(trans_goal)
            context_log = []
            native_predict = model.predict
            def cpu_predict(self, z):
                context_log.append(list(z.shape))
                return native_predict(z)
            model.predict = types.MethodType(cpu_predict, model)
            pred, joined = model.rollout(trans_obs, actions)
            loss = objective(pred, target, step=0)
            gradient = torch.autograd.grad(loss.sum(), actions)[0]
            assert pred['visual'].shape == (1, 6, 196, 8) and pred['proprio'].shape == (1, 6, 10)
            assert all(torch.isfinite(v).all() for v in (*pred.values(), loss, gradient))
            assert [x[1] for x in context_log] == [1, 2, 3, 3, 3]
            assert gradient.norm().item() > 0
            assert all(p.grad is None and not p.requires_grad for p in model.parameters())
            after = state_digest(model)
            assert before == after and not torch.cuda.is_initialized()
            assert verify_plain_tensors(model, output, device) > 0
            evaluator.close()
            terminal = jr(output / 'environment/worker_terminal.json')
            assert terminal['physical_actions'] == 0 and terminal['simulator_resets'] == 1
            np.savez_compressed(output / 'preflight_tensors.npz', actions=actions.detach().numpy(),
                gradient=gradient.detach().numpy(), loss=loss.detach().numpy(),
                predicted_visual=pred['visual'].detach().numpy(), predicted_proprio=pred['proprio'].detach().numpy(),
                target_visual=target['visual'].numpy(), target_proprio=target['proprio'].numpy())
            receipt = dict(status='passed_cpu_preflight', checkpoint_loads=1, model_rollouts=1,
                predictor_calls=5, predictor_context_frames=sum(x[1] for x in context_log),
                context_shapes=context_log, observation_encodes=2, autograd_grad_calls=1,
                gradient_norm=float(gradient.norm()), loss=float(loss.detach()),
                physical_zero_initialization_roundtrip_error=zero_roundtrip_error,
                image_input_min=float(trans_obs['visual'].min()), image_input_max=float(trans_obs['visual'].max()),
                current_input_shapes={k: list(v.shape) for k, v in trans_obs.items()},
                goal_input_shapes={k: list(v.shape) for k, v in trans_goal.items()},
                output_shapes={k: list(v.shape) for k, v in pred.items()},
                state_hash_before=before, state_hash_after=after, gpu_initialized=False,
                physical_actions=0, simulator_resets=1, new_targets=0, independent_targets=0,
                expert_action_arrays_read=0, duration_seconds=time.perf_counter()-started,
                limitation='One CPU gradient interface check, not task performance or full planner execution.')
            save(output / 'cpu_preflight_receipt.json', receipt)
            print(json.dumps(receipt), flush=True)
            return
        counts = dict(rollout_calls=0, selected_model_transitions=0, predictor_context_frames=0,
            predict_calls=0, backward_calls=0, encode_obs_calls=0, encoded_observation_frames=0)
        phase = dict(round=-1, iteration=-1)
        rows = []; proposals = []; targets = []; predictions = []; predictor_rows = []
        native_rollout, native_predict, native_encode = model.rollout, model.predict, model.encode_obs
        objective = create_objective_fn(alpha=1, base=2, mode='staged')

        def instrument_rollout(self, obs_0, act):
            assert obs_0['visual'].shape == (1, 1, 3, 224, 224)
            assert obs_0['proprio'].shape == (1, 1, 4) and act.shape == (1, 5, 10)
            assert act.requires_grad and torch.isfinite(act).all()
            counts['rollout_calls'] += 1
            phase['iteration'] = len(proposals)
            proposals.append(act.detach().cpu().numpy().copy())
            return native_rollout(obs_0, act)

        def instrument_predict(self, z):
            counts['predict_calls'] += 1
            counts['selected_model_transitions'] += int(z.shape[0])
            counts['predictor_context_frames'] += int(z.shape[0]*z.shape[1])
            predictor_rows.append(dict(round=phase['round'], iteration=phase['iteration'],
                batch=int(z.shape[0]), context=int(z.shape[1]), grad_enabled=torch.is_grad_enabled()))
            return native_predict(z)

        def instrument_encode(self, obs):
            counts['encode_obs_calls'] += 1
            counts['encoded_observation_frames'] += int(obs['visual'].shape[0]*obs['visual'].shape[1])
            return native_encode(obs)

        def instrument_objective(pred, target, step=None):
            loss = objective(pred, target, step=step)
            assert torch.isfinite(loss).all()
            def backward_hook(grad):
                counts['backward_calls'] += 1
                assert torch.isfinite(grad).all()
                return grad
            loss.register_hook(backward_hook)
            rows.append(dict(iteration=len(rows), loss=float(loss.detach().item()),
                stage='last' if step < pred['visual'].shape[1]-1 else 'all',
                terminal_visual_mse=float((pred['visual'][:, -1:] - target['visual']).square().mean().detach()),
                terminal_proprio_mse=float((pred['proprio'][:, -1:] - target['proprio']).square().mean().detach())))
            predictions.append({k: v.detach().cpu().numpy().copy() for k, v in pred.items()})
            if not targets:
                targets.append({k: v.detach().cpu().numpy().copy() for k, v in target.items()})
            return loss

        model.rollout = types.MethodType(instrument_rollout, model)
        model.predict = types.MethodType(instrument_predict, model)
        model.encode_obs = types.MethodType(instrument_encode, model)
        planner = MPCPlanner(max_iter=args.max_rounds, n_taken_actions=1,
            sub_planner=dict(target='planning.gd.GDPlanner', lr=0.1, sample_type='zero', action_noise=0,
                horizon=5, var_scale=1, opt_steps=args.gd_steps, eval_every=-1),
            wm=model, env=None, action_dim=10, objective_fn=instrument_objective,
            preprocessor=preprocessor, evaluator=evaluator, wandb_run=LocalLog(output),
            log_filename=str(output / 'official_mpc_logs.jsonl'))
        native_plan = planner.sub_planner.plan

        def instrument_plan(self, obs_0, obs_g, actions=None, step=None):
            phase['round'] = int(step)
            rows.clear(); proposals.clear(); predictions.clear(); targets.clear()
            directory = output / f'round_{step:02d}'
            directory.mkdir()
            np.savez_compressed(directory / 'input_observations.npz',
                current_rgb=obs_0['visual'], current_proprio=obs_0['proprio'],
                goal_rgb=obs_g['visual'], goal_proprio=obs_g['proprio'],
                memo_actions=np.empty((1, 0, 10), np.float32) if actions is None else actions.detach().cpu().numpy())
            result, lengths = native_plan(obs_0=obs_0, obs_g=obs_g, actions=actions, step=step)
            assert len(rows) == len(proposals) == args.gd_steps
            assert torch.isfinite(result).all()
            np.savez_compressed(directory / 'optimization.npz',
                proposals_before_update=np.stack(proposals), final_actions_model=result.detach().cpu().numpy(),
                visual_predictions=np.stack([x['visual'] for x in predictions]),
                proprio_predictions=np.stack([x['proprio'] for x in predictions]),
                goal_visual=targets[0]['visual'], goal_proprio=targets[0]['proprio'])
            save(directory / 'losses.json', rows)
            save(directory / 'counts_after_round.json', counts)
            print(json.dumps(dict(event='planning_complete', round=step, loss_first=rows[0]['loss'],
                loss_last_before_final_update=rows[-1]['loss'], counts=counts)), flush=True)
            return result, lengths

        planner.sub_planner.plan = types.MethodType(instrument_plan, planner.sub_planner)
        # The official sampling (including randn_like multiplied by zero) starts
        # from an explicit CPU/CUDA seed after checkpoint and environment setup.
        torch.manual_seed(args.seed)
        if device.type == 'cuda': torch.cuda.manual_seed_all(args.seed)
        np.savez_compressed(output / 'rng_before_plan.npz', cpu=torch.get_rng_state().numpy(),
            **({'cuda': torch.cuda.get_rng_state(device).cpu().numpy()} if device.type == 'cuda' else {}))
        result, lengths = planner.plan(evaluator.initial_obs, goal)
        np.savez_compressed(output / 'selected_actions.npz', actions_model=result.detach().cpu().numpy(), action_lengths=lengths)
        save(output / 'predictor_calls.json', predictor_rows)
        assert counts['rollout_calls'] == planner.iter * args.gd_steps
        assert counts['backward_calls'] == counts['rollout_calls']
        assert counts['selected_model_transitions'] == 5*counts['rollout_calls']
        assert counts['encode_obs_calls'] == planner.iter*(args.gd_steps+1)
        assert all(p.grad is None and not p.requires_grad for p in model.parameters())
        after = state_digest(model)
        assert before == after
        verified_plain_tensors = verify_plain_tensors(model, output, device)
        for path, digest in source_hashes.items(): assert sha(path) == digest
        evaluator.close()
        worker_terminal = jr(output / 'environment/worker_terminal.json')
        assert worker_terminal['physical_actions'] == evaluator.last['actions']
        summary = dict(status='complete', model_state_sha256_before=before, model_state_sha256_after=after,
            checkpoint_sha256=CKPT_SHA, source_hashes=source_hashes, rounds_completed=planner.iter,
            plain_tensor_final_hashes_and_devices_verified=verified_plain_tensors,
            physical_actions=int(evaluator.last['actions']), simulator_resets=1, counts=counts,
            metrics=evaluator.last, success=bool(planner.is_success[0]), original_method=False,
            perturbation=dict(action_gain=args.action_gain, moment_factor=args.moment_factor),
            physics_receipt_sha256=sha(output / 'environment/physics_receipt.json'),
            independent_targets=0, new_targets=0, development_target=task['task_id'],
            trajectory_id=task['trajectory_id'], env_seed=task['env_seed'],
            preregistration_sha256=PROTOCOL_SHA, queue_entry=queue_entry, expert_action_arrays_read=0,
            maximum_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
            maximum_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
            wall_seconds=time.perf_counter()-started, protocol=lock)
        save(output / 'summary.json', summary)
        print(json.dumps(dict(event='complete', success=summary['success'], physical_actions=summary['physical_actions'],
            metrics=evaluator.last, counts=counts)), flush=True)
    except BaseException as error:
        if output.exists():
            save(output / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
                elapsed_seconds=time.perf_counter()-started))
        raise
    finally:
        if evaluator is not None: evaluator.close()
        for hook in hooks: hook.remove()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--audit-only', action='store_true')
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--cpu-preflight', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--max-rounds', type=int, default=20)
    parser.add_argument('--gd-steps', type=int, default=100)
    parser.add_argument('--action-gain', type=float, default=1.0)
    parser.add_argument('--moment-factor', type=float, default=1.0)
    parser.add_argument('--reference-cache', type=Path, required=True)
    args = parser.parse_args()
    assert args.max_rounds == 20 and args.gd_steps == 100 and args.seed == 1
    assert args.action_gain == 1.0 and args.moment_factor in (1.0, 2.0)
    assert not args.cpu_preflight, 'WM13 permits only source audit or the preregistered full evaluation, no extra model/environment preflight'
    if args.cpu_preflight:
        assert args.device == 'cpu', 'CPU preflight must explicitly select --device cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    main(args)
