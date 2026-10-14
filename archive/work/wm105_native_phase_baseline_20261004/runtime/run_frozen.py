"""WM105 mechanical native GD/MPC derivative plus read-only forecast records."""
from pathlib import Path
import argparse,json,os,random,sys,time,traceback,types,hashlib
B=Path(__file__).resolve().parent;W=B.parent.parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(W/'wm13_20261002/runner'))
from task_contract import sha,jr,save,verify_protocol,read_planner_reference
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
    try:
        p=verify_protocol(args.protocol);entry=next(e for e in p['queue'] if e['run_name']==args.run)
        assert entry['arm']=='native_frozen' and out==Path(entry['output']).resolve()
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
        task,realization,refs=read_planner_reference(Path(entry['reference_cache']),args.protocol)
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
        evaluator=KvPersistentEvaluator(out,100,preprocessor,realization,reference_cache=Path(entry['reference_cache']),kv_factor=1.,protocol=args.protocol)
        goal=dict(visual=realization['goal_rgb'][None,None],proprio=realization['goal_proprio'][None,None])
        native_rollout,native_predict,native_encode,native_encode_act=model.rollout,model.predict,model.encode_obs,model.encode_act
        objective=create_objective_fn(alpha=1,base=2,mode='staged')
        proposals=[];predictions=[];losses=[];targets=[]
        def rollout(self,obs_0,act):
            assert phase=='planning' and act.shape==(1,5,10) and act.requires_grad
            proposals.append(act.detach().cpu().numpy().copy())
            return native_rollout(obs_0,act)
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
            loss=objective(pred,target,step=step);assert torch.isfinite(loss).all()
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
            nonlocal phase,active_round
            active_round=int(step);assert phase=='planning'
            proposals.clear();predictions.clear();losses.clear();targets.clear()
            rd=out/f'round_{step:02d}';rd.mkdir();before=counts.copy()
            np.savez_compressed(rd/'input_observations.npz',current_rgb=obs_0['visual'],current_proprio=obs_0['proprio'],
                goal_rgb=obs_g['visual'],goal_proprio=obs_g['proprio'],
                memo_actions=np.empty((1,0,10),np.float32) if actions is None else actions.detach().cpu().numpy())
            try:result,lengths=native_plan(obs_0=obs_0,obs_g=obs_g,actions=actions,step=step)
            except BaseException:
                np.savez_compressed(rd/'partial_optimization.npz',proposals_before_update=np.asarray(proposals),
                    visual_predictions=np.asarray([x['visual'] for x in predictions]),proprio_predictions=np.asarray([x['proprio'] for x in predictions]),losses=np.asarray(losses))
                raise
            assert len(proposals)==len(predictions)==len(losses)==100
            returned=result.detach().cpu().numpy().copy()
            np.savez_compressed(rd/'optimization.npz',proposals_before_update=np.stack(proposals),final_actions_model=returned,
                visual_predictions=np.stack([x['visual'] for x in predictions]),proprio_predictions=np.stack([x['proprio'] for x in predictions]),
                goal_visual=targets[0]['visual'],goal_proprio=targets[0]['proprio'])
            save(rd/'losses.json',losses)
            delta={k:counts[k]-before[k] for k in counts}
            assert delta['planning_predict_calls']==500 and delta['action_backward_calls']==100 and delta['planning_encode_calls']==101
            assert delta['planning_action_encode_calls']==delta['planning_action_encoded_tokens']==500
            save(rd/'counts.json',dict(delta=delta,cumulative=counts.copy()))
            # Exact post100 plan; bypass rollout/score recording hooks. Preserve grad
            # execution path through action inputs without ever invoking backward.
            started_query=time.perf_counter();phase='post100';cb=counts.copy();rb=snapshot_rng();mb=diagnostic_hash()
            obs_copy={k:np.array(v,copy=True) for k,v in obs_0.items()}
            transformed={k:v.to(device) for k,v in preprocessor.transform_obs(obs_copy).items()}
            a=result.detach().clone().requires_grad_(True)
            counts['post100_rollout_calls']+=1
            with torch.enable_grad():
                pred,z=native_rollout(transformed,a)
                target={k:torch.as_tensor(v,device=device) for k,v in targets[0].items()}
                returned_loss=objective(pred,target,step=step)
            assert pred['visual'].shape==(1,6,196,8) and pred['proprio'].shape==(1,6,10)
            assert torch.isfinite(returned_loss).all() and all(torch.isfinite(v).all() for v in pred.values())
            np.savez_compressed(rd/'post100_prediction.npz',current_rgb=obs_copy['visual'],current_proprio=obs_copy['proprio'],
                returned_H5_model_actions=returned,full_visual_prediction=pred['visual'].detach().cpu().numpy(),
                full_proprio_prediction=pred['proprio'].detach().cpu().numpy())
            loss_value=float(returned_loss.detach());del pred,z,a,transformed,target,returned_loss
            ma=diagnostic_hash();ra=snapshot_rng();rng=rng_receipt(rd/'post100_rng.npz',rb,ra)
            dd={k:counts[k]-cb[k] for k in counts}
            action_same=result.detach().cpu().numpy().tobytes()==returned.tobytes()
            obs_same=all(np.array_equal(obs_copy[k],obs_0[k]) for k in obs_copy)
            elapsed=time.perf_counter()-started_query;diagnostic_seconds['post100_total']+=elapsed
            save(rd/'post100_receipt.json',dict(status='recorded',round=int(step),step=int(step),
                before_model_state_sha256=mb,after_model_state_sha256=ma,rng=rng,counts_delta=dd,
                returned_actions_byte_equal=action_same,current_observation_unchanged=obs_same,
                returned_loss=loss_value,objective=dict(mode='staged',alpha=1,base=2,step=int(step)),
                initial_logged_loss=losses[0],last_logged_pre_update_loss=losses[-1],minimum_logged_pre_update_loss=min(losses),
                minimum_logged_index=int(np.argmin(losses)),returned_is_post100_not_argmin=True,
                backward_calls=0,slot1_only_has_actual_first5_truth=True,suffix_truth_available=False,wall_seconds=elapsed))
            assert mb==ma==state_before and rng['all_equal'] and action_same and obs_same
            assert dd['post100_predict_calls']==dd['post100_action_encode_calls']==dd['post100_action_encoded_tokens']==5
            assert dd['post100_rollout_calls']==dd['post100_encode_calls']==dd['post100_encoded_frames']==1
            assert all(dd[k]==0 for k in dd if k.startswith('planning_') or k in ('action_backward_calls','parameter_updates','physical_actions'))
            assert len(proposals)==len(predictions)==len(losses)==100
            phase='planning';return result,lengths
        def evaluate(self,actions,action_len=None,filename='output',save_video=False):
            result=native_evaluate(actions,action_len,filename,save_video)
            counts['physical_actions']=int(evaluator.last['actions'])
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
            total_case_bytes_before_summary=sum(f.stat().st_size for f in out.rglob('*') if f.is_file()),
            base_state_before=state_before,base_state_after=after,physical_resets=1,base_updates=0,new_targets_generated_in_run=0,
            prefix_checks='old17 cross-arm gates intentionally removed; current/issued/actual identities audited for each new draw',
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
