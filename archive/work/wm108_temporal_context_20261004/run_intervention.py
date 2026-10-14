"""WM108: same frozen GD/MPC with real macro history and/or fixed initial deadline.
Only source preparation; execution requires the root-owned protocol gate.
"""
from pathlib import Path
import argparse,json,os,random,sys,time,traceback,types,hashlib
from wm108_contract import B,BASE_PROTOCOL,sha,jr,save,verify_protocol
W=B.parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(W/'wm13_20261002/runner'))
sys.path.insert(0,str(W/'wm105_native_phase_baseline_20261004/runtime'))
from task_contract import read_planner_reference
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')

def main(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();evaluator=None;phase='planning';active_round=None
    counts=dict(model_loads=0,planning_predict_calls=0,planning_context_frames=0,planning_encode_calls=0,
        planning_encoded_frames=0,planning_action_encode_calls=0,planning_action_encoded_tokens=0,
        action_backward_calls=0,parameter_updates=0,physical_actions=0,post100_rollout_calls=0,
        post100_predict_calls=0,post100_context_frames=0,post100_encode_calls=0,post100_encoded_frames=0,
        post100_action_encode_calls=0,post100_action_encoded_tokens=0,terminal_encode_calls=0,
        terminal_encoded_frames=0,diagnostic_state_digest_calls=0,base_state_digest_calls=0)
    diagnostic_seconds=dict(post100_total=0.,terminal_total=0.)
    planning_seconds=[]
    try:
        p=verify_protocol(args.protocol);entry=next(e for e in p['queue'] if e['run_name']==args.run)
        assert entry['arm'] in ('history3','deadline','history3_deadline') and out==Path(entry['output']).resolve()
        history_frames=int(entry['history_frames']);deadline=bool(entry['deadline'])
        assert (history_frames,deadline)=={'history3':(3,False),'deadline':(1,True),'history3_deadline':(3,True)}[entry['arm']]
        baseline_output=Path(entry['baseline_output']).resolve()
        save(out/'pre_run_lock.json',dict(protocol_sha256=sha(args.protocol),source_hashes=p['source_hashes'],
            entry=entry,command=[sys.executable,*sys.argv]))
        import numpy as np
        import torch
        from common import ACTION_MEAN,ACTION_STD,PROPRIO_MEAN,PROPRIO_STD
        from run_temporal_mpc import load_model,state_digest,verify_plain_tensors,LocalLog
        from local_evaluator import KvPersistentEvaluator
        torch.set_num_threads(2);torch.set_num_interop_threads(2)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        seed=entry['planner_seed'];assert seed==100
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
        task,realization,refs=read_planner_reference(Path(entry['reference_cache']),BASE_PROTOCOL)
        save(out/'reference_receipt.json',dict(task=task,source_hashes=refs,expert_arrays=0))
        model,config=load_model(out,device);counts['model_loads']=1
        state_before=state_digest(model);counts['base_state_digest_calls']+=1
        assert state_before==p['expected_model_state_sha256'] and all(not m.training for m in model.modules())
        from preprocessor import Preprocessor
        from datasets.img_transforms import default_transform
        from planning.objectives import create_objective_fn
        from planning.mpc import MPCPlanner
        preprocessor=Preprocessor(action_mean=torch.tensor(ACTION_MEAN,dtype=torch.float32),
            action_std=torch.tensor(ACTION_STD,dtype=torch.float32),state_mean=None,state_std=None,
            proprio_mean=torch.tensor(PROPRIO_MEAN,dtype=torch.float32),
            proprio_std=torch.tensor(PROPRIO_STD,dtype=torch.float32),transform=default_transform(224))
        evaluator=KvPersistentEvaluator(out,100,preprocessor,realization,reference_cache=Path(entry['reference_cache']),kv_factor=1.,protocol=BASE_PROTOCOL)
        goal=dict(visual=realization['goal_rgb'][None,None],proprio=realization['goal_proprio'][None,None])
        native_rollout,native_predict,native_encode,native_encode_act=model.rollout,model.predict,model.encode_obs,model.encode_act
        objective=create_objective_fn(alpha=1,base=2,mode='staged')
        last_objective=create_objective_fn(alpha=1,base=2,mode='last')
        assert model.num_hist==3
        history_count=1;past_actions=None
        def intervention_objective(pred,target,step):
            assert pred['visual'].shape[1]==pred['proprio'].shape[1]==6
            if deadline and int(step)<5:
                q=5-int(step)
                return last_objective({k:v[:,q:q+1] for k,v in pred.items()},target,step=step)
            return objective(pred,target,step=step)
        def variant_rollout(obs,act):
            assert act.shape==(1,5,10) and act.requires_grad
            assert obs['visual'].shape[1]==obs['proprio'].shape[1]==history_count
            assert past_actions.shape==(1,history_count-1,10) and not past_actions.requires_grad
            # Keep the h=1 route identical, including no new concatenation.
            joined=act if history_count==1 else torch.cat((past_actions,act),dim=1)
            full,z=native_rollout(obs,joined)
            assert full['visual'].shape[1]==full['proprio'].shape[1]==history_count+5
            pred=full if history_count==1 else {k:v[:,history_count-1:] for k,v in full.items()}
            return pred,z,full
        proposals=[];predictions=[];losses=[];targets=[]
        def rollout(self,obs_0,act):
            assert phase=='planning' and act.shape==(1,5,10) and act.requires_grad
            proposals.append(act.detach().cpu().numpy().copy())
            pred,z,_=variant_rollout(obs_0,act)
            return pred,z[:,history_count-1:]
        def predict(self,z):
            assert phase in ('planning','post100') and z.shape[0]==1 and z.shape[-2:]==(196,28)
            counts[phase+'_predict_calls']+=1;counts[phase+'_context_frames']+=int(z.shape[1])
            return native_predict(z)
        def encode(self,obs):
            assert phase in ('planning','post100','terminal')
            counts[phase+'_encode_calls']+=1;counts[phase+'_encoded_frames']+=int(obs['visual'].shape[0]*obs['visual'].shape[1])
            return native_encode(obs)
        def encode_act(self,act):
            assert phase in ('planning','post100')
            counts[phase+'_action_encode_calls']+=1;counts[phase+'_action_encoded_tokens']+=int(act.shape[0]*act.shape[1])
            return native_encode_act(act)
        def score(pred,target,step=None):
            assert phase=='planning'
            loss=intervention_objective(pred,target,step=step);assert torch.isfinite(loss).all()
            def grad(g):
                counts['action_backward_calls']+=1;assert torch.isfinite(g).all();return g
            loss.register_hook(grad)
            predictions.append({k:v.detach().cpu().numpy().copy() for k,v in pred.items()})
            losses.append(float(loss.detach()))
            if not targets:targets.append({k:v.detach().cpu().numpy().copy() for k,v in target.items()})
            return loss
        model.rollout=types.MethodType(rollout,model);model.predict=types.MethodType(predict,model)
        model.encode_obs=types.MethodType(encode,model);model.encode_act=types.MethodType(encode_act,model)
        planner=MPCPlanner(max_iter=20,n_taken_actions=1,
            sub_planner=dict(target='planning.gd.GDPlanner',lr=.1,sample_type='zero',action_noise=0,
                horizon=5,var_scale=1,opt_steps=100,eval_every=-1),wm=model,env=None,action_dim=10,
            objective_fn=score,preprocessor=preprocessor,evaluator=evaluator,wandb_run=LocalLog(out),
            log_filename=str(out/'official_mpc_logs.jsonl'))
        native_plan=planner.sub_planner.plan;native_evaluate=evaluator.eval_actions
        def snapshot_rng():
            ns=np.random.get_state()
            return dict(cpu=torch.get_rng_state().numpy().copy(),cuda=torch.cuda.get_rng_state(device).cpu().numpy().copy(),
                numpy_keys=ns[1].copy(),numpy_meta=np.asarray(json.dumps([ns[0],ns[2],ns[3],ns[4]])),
                python=np.asarray(json.dumps(random.getstate())))
        def rng_receipt(path,before,after):
            np.savez_compressed(path,**{k+'_before':v for k,v in before.items()},**{k+'_after':v for k,v in after.items()})
            same={k:before[k].dtype==after[k].dtype and before[k].shape==after[k].shape and before[k].tobytes()==after[k].tobytes() for k in before}
            return dict(all_equal=all(same.values()),per_source_equal=same,file_sha256=sha(path),
                before_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in before.items()},
                after_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in after.items()})
        def diagnostic_hash():
            counts['diagnostic_state_digest_calls']+=1;return state_digest(model)
        def plan(self,obs_0,obs_g,actions=None,step=None):
            nonlocal phase,active_round,history_count,past_actions
            active_round=int(step);assert phase=='planning'
            assert int(evaluator.last['actions'])==5*active_round
            history_count=min(history_frames,active_round+1)
            context_start=active_round-history_count+1
            raw_indices=np.arange(context_start,active_round+1,dtype=np.int64)*5
            source_feedback=Path(evaluator.last['path'])
            with np.load(source_feedback,allow_pickle=False) as saved:
                history_obs=dict(visual=saved['rgb'][raw_indices][None].copy(),
                    proprio=saved['proprio'][raw_indices][None].copy())
            for k in history_obs:
                assert history_obs[k][:,-1:].dtype==obs_0[k].dtype
                assert history_obs[k][:,-1:].shape==obs_0[k].shape
                assert history_obs[k][:,-1:].tobytes()==obs_0[k].tobytes()
            executed=evaluator.executed_model.copy()
            assert executed.shape==(1,active_round,10) and executed.dtype==np.float32
            past_np=executed[:,context_start:active_round].copy()
            past_actions=torch.from_numpy(past_np.copy()).to(device).detach()
            planning_obs=obs_0 if history_count==1 else history_obs
            proposals.clear();predictions.clear();losses.clear();targets.clear()
            rd=out/f'round_{step:02d}';rd.mkdir();before=counts.copy()
            np.savez_compressed(rd/'input_observations.npz',current_rgb=obs_0['visual'],current_proprio=obs_0['proprio'],
                goal_rgb=obs_g['visual'],goal_proprio=obs_g['proprio'],
                memo_actions=np.empty((1,0,10),np.float32) if actions is None else actions.detach().cpu().numpy())
            np.savez_compressed(rd/'history_inputs.npz',history_rgb=history_obs['visual'],history_proprio=history_obs['proprio'],
                raw_action_indices=raw_indices,past_actions_model=past_np,executed_model_prefix=executed)
            save(rd/'history_receipt.json',dict(history_frames=history_count,configured_history_frames=history_frames,
                current_raw_action=5*active_round,history_start_macro=context_start,raw_action_indices=raw_indices.tolist(),
                source_feedback_path=str(source_feedback),source_feedback_sha256=sha(source_feedback),
                past_token_start=context_start,past_token_end_exclusive=active_round,past_tokens_are_commands_not_displacements=True,
                past_tokens_detached=True,future_optimized_tokens=5,crop_start=history_count-1,score_sequence_length=6,
                deadline=deadline,score_slot=5-active_round if deadline and active_round<5 else None,
                objective_mode='deadline_single_slot' if deadline and active_round<5 else 'native_staged',
                original_staged_step=active_round,no_source_episode_prehistory=True))
            gd_started=time.perf_counter()
            try:result,lengths=native_plan(obs_0=planning_obs,obs_g=obs_g,actions=actions,step=step)
            except BaseException:
                np.savez_compressed(rd/'partial_optimization.npz',proposals_before_update=np.asarray(proposals),
                    visual_predictions=np.asarray([x['visual'] for x in predictions]),proprio_predictions=np.asarray([x['proprio'] for x in predictions]),losses=np.asarray(losses))
                raise
            assert len(proposals)==len(predictions)==len(losses)==100
            returned=result.detach().cpu().numpy().copy()
            gd_seconds=time.perf_counter()-gd_started;planning_seconds.append(gd_seconds)
            np.savez_compressed(rd/'optimization.npz',proposals_before_update=np.stack(proposals),final_actions_model=returned,
                visual_predictions=np.stack([x['visual'] for x in predictions]),proprio_predictions=np.stack([x['proprio'] for x in predictions]),
                goal_visual=targets[0]['visual'],goal_proprio=targets[0]['proprio'])
            save(rd/'losses.json',losses)
            delta={k:counts[k]-before[k] for k in counts}
            assert delta['planning_predict_calls']==500 and delta['action_backward_calls']==100 and delta['planning_encode_calls']==101
            assert delta['planning_context_frames']==100*sum(min(history_count+i,3) for i in range(5))
            assert delta['planning_encoded_frames']==100*history_count+1
            assert delta['planning_action_encode_calls']==500
            assert delta['planning_action_encoded_tokens']==100*(history_count+4)
            assert past_actions.detach().cpu().numpy().tobytes()==past_np.tobytes()
            save(rd/'counts.json',dict(delta=delta,cumulative=counts.copy(),
                planning_call_wall_seconds_including_return_transfer=gd_seconds,
                peak_allocated_bytes_so_far=torch.cuda.max_memory_allocated(device)))
            # Exact post100 plan; bypass rollout/score recording hooks. Preserve grad
            # execution path through action inputs without ever invoking backward.
            started_query=time.perf_counter();phase='post100';cb=counts.copy();rb=snapshot_rng();mb=diagnostic_hash()
            obs_copy={k:np.array(v,copy=True) for k,v in planning_obs.items()}
            transformed={k:v.to(device) for k,v in preprocessor.transform_obs(obs_copy).items()}
            a=result.detach().clone().requires_grad_(True)
            counts['post100_rollout_calls']+=1
            with torch.enable_grad():
                pred,z,full_pred=variant_rollout(transformed,a)
                target={k:torch.as_tensor(v,device=device) for k,v in targets[0].items()}
                returned_loss=intervention_objective(pred,target,step=step)
            assert pred['visual'].shape==(1,6,196,8) and pred['proprio'].shape==(1,6,10)
            assert torch.isfinite(returned_loss).all() and all(torch.isfinite(v).all() for v in pred.values())
            np.savez_compressed(rd/'post100_prediction.npz',current_rgb=obs_copy['visual'][:,-1:],current_proprio=obs_copy['proprio'][:,-1:],
                history_rgb=obs_copy['visual'],history_proprio=obs_copy['proprio'],past_actions_model=past_np,
                full_H_history_future_model_actions=np.concatenate((past_np,returned),axis=1),
                returned_H5_model_actions=returned,full_visual_prediction=pred['visual'].detach().cpu().numpy(),
                full_proprio_prediction=pred['proprio'].detach().cpu().numpy(),
                uncropped_visual_prediction=full_pred['visual'].detach().cpu().numpy(),
                uncropped_proprio_prediction=full_pred['proprio'].detach().cpu().numpy())
            loss_value=float(returned_loss.detach());del pred,z,full_pred,a,transformed,target,returned_loss
            ma=diagnostic_hash();ra=snapshot_rng();rng=rng_receipt(rd/'post100_rng.npz',rb,ra)
            dd={k:counts[k]-cb[k] for k in counts}
            action_same=result.detach().cpu().numpy().tobytes()==returned.tobytes()
            obs_same=all(np.array_equal(obs_copy[k],planning_obs[k]) for k in obs_copy)
            past_same=past_actions.detach().cpu().numpy().tobytes()==past_np.tobytes()
            elapsed=time.perf_counter()-started_query;diagnostic_seconds['post100_total']+=elapsed
            save(rd/'post100_receipt.json',dict(status='recorded',round=int(step),step=int(step),
                before_model_state_sha256=mb,after_model_state_sha256=ma,rng=rng,counts_delta=dd,
                returned_actions_byte_equal=action_same,current_observation_unchanged=obs_same,past_actions_byte_equal=past_same,
                history_frames=history_count,crop_start=history_count-1,returned_loss=loss_value,
                objective=dict(mode='deadline_single_slot' if deadline and active_round<5 else 'staged',
                    alpha=1,base=2,step=int(step),score_sequence_length=6,
                    score_slot=5-active_round if deadline and active_round<5 else None),
                initial_logged_loss=losses[0],last_logged_pre_update_loss=losses[-1],minimum_logged_pre_update_loss=min(losses),
                minimum_logged_index=int(np.argmin(losses)),returned_is_post100_not_argmin=True,
                backward_calls=0,slot1_only_has_actual_first5_truth=True,suffix_truth_available=False,wall_seconds=elapsed))
            assert mb==ma==state_before and rng['all_equal'] and action_same and obs_same and past_same
            assert dd['post100_predict_calls']==dd['post100_action_encode_calls']==5
            assert dd['post100_action_encoded_tokens']==history_count+4
            assert dd['post100_context_frames']==sum(min(history_count+i,3) for i in range(5))
            assert dd['post100_rollout_calls']==dd['post100_encode_calls']==1
            assert dd['post100_encoded_frames']==history_count
            assert all(dd[k]==0 for k in dd if k.startswith('planning_') or k in ('action_backward_calls','parameter_updates','physical_actions'))
            assert len(proposals)==len(predictions)==len(losses)==100
            if active_round==0:
                # Read the old plan only after this arm's plan and diagnostic are durable.
                old_plan=baseline_output/'round_00/optimization.npz'
                old_input=baseline_output/'round_00/input_observations.npz'
                def saved_array_identity(old_path,new_path,fields):
                    detail={}
                    with np.load(old_path,allow_pickle=False) as old,np.load(new_path,allow_pickle=False) as new:
                        for name in fields:
                            a,b=old[name],new[name]
                            detail[name]=dict(dtype_equal=a.dtype==b.dtype,shape_equal=a.shape==b.shape,
                                bytes_equal=a.tobytes()==b.tobytes(),baseline_dtype=str(a.dtype),actual_dtype=str(b.dtype),
                                baseline_shape=list(a.shape),actual_shape=list(b.shape))
                    return detail
                optimization_fields=('proposals_before_update','final_actions_model','visual_predictions',
                    'proprio_predictions','goal_visual','goal_proprio')
                bank_identity=saved_array_identity(old_plan,rd/'optimization.npz',optimization_fields)
                old_rng=baseline_output/'rng_before_plan.npz';new_rng=out/'rng_before_plan.npz'
                initial_rng_identity=saved_array_identity(old_rng,new_rng,('cpu','cuda'))
                old_losses_path=baseline_output/'round_00/losses.json';new_losses_path=rd/'losses.json'
                old_losses,new_losses=jr(old_losses_path),jr(new_losses_path)
                loss_identity=[type(a) is type(b) and a==b for a,b in zip(old_losses,new_losses)]
                losses_exact=len(old_losses)==len(new_losses)==len(loss_identity)==100 and all(loss_identity)
                bank_exact=all(all(row[k] for k in ('dtype_equal','shape_equal','bytes_equal')) for row in bank_identity.values())
                rng_exact=all(all(row[k] for k in ('dtype_equal','shape_equal','bytes_equal')) for row in initial_rng_identity.values())
                first_h5_same=all(bank_identity['final_actions_model'][k] for k in ('dtype_equal','shape_equal','bytes_equal'))
                with np.load(old_input,allow_pickle=False) as old:
                    first_inputs_same=all(old[name].dtype==a.dtype and old[name].shape==a.shape and old[name].tobytes()==a.tobytes() for name,a in
                        [('current_rgb',obs_0['visual']),('current_proprio',obs_0['proprio']),('goal_rgb',obs_g['visual']),('goal_proprio',obs_g['proprio'])])
                first_round_same=first_h5_same and first_inputs_same and bank_exact and rng_exact and losses_exact
                save(out/'first_round_identity.json',dict(status='passed' if first_round_same else 'failed_retained',
                    final_H5_byte_equal=first_h5_same,planner_inputs_byte_equal=first_inputs_same,
                    optimization_field_identity=bank_identity,all_optimization_fields_byte_equal=bank_exact,
                    initial_rng_field_identity=initial_rng_identity,initial_rng_byte_equal=rng_exact,
                    losses_json_exact_equal=losses_exact,loss_count_baseline=len(old_losses),loss_count_actual=len(new_losses),
                    loss_value_exact_equal=loss_identity,no_numerical_tolerance=True,before_first_actual_action=True,
                    baseline_plan_path=str(old_plan),baseline_plan_sha256=sha(old_plan),baseline_input_path=str(old_input),baseline_input_sha256=sha(old_input),
                    actual_plan_path=str(rd/'optimization.npz'),actual_plan_sha256=sha(rd/'optimization.npz'),
                    baseline_rng_path=str(old_rng),baseline_rng_sha256=sha(old_rng),actual_rng_path=str(new_rng),actual_rng_sha256=sha(new_rng),
                    baseline_losses_path=str(old_losses_path),baseline_losses_sha256=sha(old_losses_path),
                    actual_losses_path=str(new_losses_path),actual_losses_sha256=sha(new_losses_path),
                    baseline_read_only_after_returned_plan_saved=True))
                assert first_round_same
            phase='planning';return result,lengths
        def evaluate(self,actions,action_len=None,filename='output',save_video=False):
            result=native_evaluate(actions,action_len,filename,save_video)
            counts['physical_actions']=int(evaluator.last['actions'])
            if counts['physical_actions']==5:
                old=baseline_output/'environment/step_005.npz';new=Path(evaluator.last['path'])
                with np.load(old,allow_pickle=False) as oz,np.load(new,allow_pickle=False) as nz:
                    same={k:oz[k].dtype==nz[k].dtype and oz[k].shape==nz[k].shape and oz[k].tobytes()==nz[k].tobytes()
                        for k in ('rgb','state','proprio','action_command','action_applied')}
                save(out/'first5_identity.json',dict(status='passed' if all(same.values()) else 'failed_retained',
                    per_field_byte_equal=same,baseline_path=str(old),baseline_sha256=sha(old),actual_path=str(new),actual_sha256=sha(new)))
                assert all(same.values())
            return result
        planner.sub_planner.plan=types.MethodType(plan,planner.sub_planner);evaluator.eval_actions=types.MethodType(evaluate,evaluator)
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        np.savez_compressed(out/'rng_before_plan.npz',cpu=torch.get_rng_state().numpy(),cuda=torch.cuda.get_rng_state(device).cpu().numpy())
        result,lengths=planner.plan(evaluator.initial_obs,goal)
        np.savez_compressed(out/'selected_actions.npz',actions_model=result.detach().cpu().numpy(),action_lengths=lengths)
        assert counts['planning_predict_calls']==planner.iter*500 and counts['physical_actions']==planner.iter*5
        assert counts['post100_rollout_calls']==planner.iter and counts['post100_predict_calls']==planner.iter*5
        # One real terminal observation encode; other actual endpoint codes can
        # reuse the next round's post100 slot0, after saved identity checks.
        phase='terminal';started_query=time.perf_counter();cb=counts.copy();rb=snapshot_rng();mb=diagnostic_hash()
        actual_obs,_=evaluator.read_trajectory(evaluator.last['path']);terminal_obs={k:v[:,-1:].copy() for k,v in actual_obs.items()}
        transformed={k:v.to(device) for k,v in preprocessor.transform_obs(terminal_obs).items()}
        with torch.enable_grad():encoded=model.encode_obs(transformed)
        assert all(torch.isfinite(v).all() for v in encoded.values())
        np.savez_compressed(out/'terminal_observation_encoding.npz',raw_rgb=terminal_obs['visual'],raw_proprio=terminal_obs['proprio'],
            visual=encoded['visual'].detach().cpu().numpy(),proprio=encoded['proprio'].detach().cpu().numpy())
        del encoded,transformed
        ma=diagnostic_hash();ra=snapshot_rng();rng=rng_receipt(out/'terminal_encoding_rng.npz',rb,ra)
        dd={k:counts[k]-cb[k] for k in counts};elapsed=time.perf_counter()-started_query;diagnostic_seconds['terminal_total']=elapsed
        save(out/'terminal_encoding_receipt.json',dict(status='recorded',before_model_state_sha256=mb,after_model_state_sha256=ma,
            rng=rng,counts_delta=dd,actual_actions=counts['physical_actions'],source_feedback_path=evaluator.last['path'],wall_seconds=elapsed))
        assert mb==ma==state_before and rng['all_equal'] and dd['terminal_encode_calls']==dd['terminal_encoded_frames']==1
        assert all(dd[k]==0 for k in dd if k.startswith(('planning_','post100_')) or k in ('action_backward_calls','parameter_updates','physical_actions'))
        after=state_digest(model);counts['base_state_digest_calls']+=1;assert after==state_before
        assert all(not v.requires_grad and v.grad is None for v in model.parameters())
        verify_plain_tensors(model,out,device)
        evaluator.close();worker=jr(out/'environment/worker_terminal.json')
        assert worker['simulator_resets']==1 and worker['physical_actions']==counts['physical_actions'] and not worker['gpu_initialized']
        verify_protocol(args.protocol)
        bank_bytes=sum((out/f'round_{r:02d}/optimization.npz').stat().st_size for r in range(planner.iter))
        s=dict(status='complete',dataset='val',arm=entry['arm'],draw_index=entry['draw_index'],trajectory_id=entry['trajectory_id'],offset=entry['offset'],
            initial_group_id=entry['initial_group_id'],trajectory_group_id=entry['trajectory_group_id'],
            goal_receipt_sha256=p['goal_receipt_sha256'],task=task,kv_factor=1,planner_seed=seed,env_seed=entry['env_seed'],
            success=bool(planner.is_success[0]),physical_actions=counts['physical_actions'],rounds=planner.iter,metrics=evaluator.last,counts=counts,
            diagnostic_wall_seconds=diagnostic_seconds,optimization_bank_bytes=bank_bytes,
            planning_call_wall_seconds=planning_seconds,planning_total_wall_seconds=sum(planning_seconds),
            total_case_bytes_before_summary=sum(f.stat().st_size for f in out.rglob('*') if f.is_file()),
            base_state_before=state_before,base_state_after=after,physical_resets=1,base_updates=0,new_targets_generated_in_run=0,
            prefix_checks='WM105 same-draw first H5 and first5 trajectory byte equality; all later history from own execution',
            history_frames=history_frames,deadline=deadline,baseline_output=str(baseline_output),
            planning_context_frames_formula='sum_r 100*sum_i0to4 min(min(history_frames,r+1)+i,3)',
            planning_encoded_frames_formula='sum_r (100*min(history_frames,r+1)+1)',
            encoder_count_semantics='Each native encode_obs call invokes one visual encoder and one frozen proprio encoder on the recorded number of frames.',
            matched_forward_and_GD_counts_not_equal_FLOPs=True,
            no_independent_confirmation=True,protocol_sha256=sha(args.protocol),wall_seconds=time.perf_counter()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        save(out/'summary.json',s);save(out/'terminal.json',dict(status='complete',summary_sha256=sha(out/'summary.json')))
        print(json.dumps(dict(event='complete',run=entry['run_name'],success=s['success'],actions=s['physical_actions'])),flush=True)
    except BaseException as e:
        save(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),phase=phase,round=active_round,counts=counts))
        save(out/'terminal.json',dict(status='failed_retained',failure_sha256=sha(out/'failure.json')));raise
    finally:
        if evaluator is not None:evaluator.close()
if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--protocol',type=Path,required=True);a.add_argument('--run',required=True);a.add_argument('--output',type=Path,required=True)
    main(a.parse_args())
