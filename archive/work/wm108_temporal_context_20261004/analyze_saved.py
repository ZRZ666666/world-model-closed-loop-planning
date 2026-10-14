"""All150 WM108 completions plus the fixed50 reused native baselines, saved-only.

No partial effect analysis, model, encoder, physics, bootstrap or IID claim.
"""
from pathlib import Path
import argparse,csv,json,sys,traceback
import numpy as np
from wm108_contract import B,BASE_PROTOCOL,sha,jr,save,verify_protocol
sys.dont_write_bytecode=True
ARMS=('deadline','history3','history3_deadline')

def write_csv(path,rows):
    with Path(path).open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def describe(rows):
    return dict(draws=len(rows),successes_by100=sum(r['success_by100'] for r in rows),successes_by25=sum(r['success_by25'] for r in rows),
        mean_capped100_cost=float(np.mean([r['capped100_cost'] for r in rows])),mean_capped25_cost=float(np.mean([r['capped25_cost'] for r in rows])),
        physical_actions=sum(r['actual_actions'] for r in rows),resets=sum(r['resets'] for r in rows),
        S100=sum(r['success_by100'] and r['first_success_action']==100 for r in rows),F100=sum(not r['success_by100'] for r in rows),
        initial_success_descriptive=sum(r['initial_success'] for r in rows),failed_draws=[r['draw_index'] for r in rows if not r['success_by100']])
def main(args):
    p=verify_protocol(args.protocol);old=jr(BASE_PROTOCOL)
    assert len(p['queue'])==150 and len(old['queue'])==50
    assert [(e['draw_index'],e['arm']) for e in p['queue']]==[(i,arm) for i in range(50) for arm in ARMS]
    out=B/'final_v1';assert not out.exists(), 'One complete final analysis; no overwrite'
    pins={};pairs=[]
    def pin(path,expected=None):
        path=str(Path(path));h=sha(path)
        assert expected is None or h==expected,('hash',path)
        pins[path]=h;return h
    def read(path,expected=None):pin(path,expected);return jr(path)
    pin(__file__,p['source_hashes'][str(Path(__file__).resolve())]);pin(args.protocol);pin(BASE_PROTOCOL)
    controller=read(B/'controller_v1/terminal.json')
    assert controller['status']=='complete' and controller['cases']==150 and controller['draws']==50 and controller['protocol_sha256']==sha(args.protocol)
    base_controller=read(BASE_PROTOCOL.parent/'candidate_actual_controller_v1/terminal.json')
    assert base_controller['status']=='complete' and base_controller['draws']==50 and base_controller['protocol_sha256']==sha(BASE_PROTOCOL)
    # Check every new completion first. No survivor subset if even one is absent.
    for e in p['queue']:
        run=Path(e['output']);ad=B/'audit_v1'/e['run_name']
        assert (run/'terminal.json').is_file() and (ad/'terminal.json').is_file(),e['run_name']
        actual=jr(run/'terminal.json');audit=jr(ad/'terminal.json')
        assert actual['status']==audit['status']=='complete',e['run_name']
    def load_case(entry,ad,links,protocol_path,arm):
        run=Path(entry['output']);name=entry['run_name']
        paths=dict(actual_summaries=run/'summary.json',actual_terminals=run/'terminal.json',actual_audits=ad/'audit.json',audit_terminals=ad/'terminal.json')
        for key,path in paths.items():
            link=links[key][name];assert Path(link['path']).resolve()==path.resolve();pin(path,link['sha256'])
        at=jr(paths['actual_terminals']);au=jr(paths['audit_terminals'])
        assert at['status']==au['status']=='complete'
        pin(paths['actual_summaries'],at['summary_sha256']);pin(paths['actual_audits'],au['audit_sha256'])
        for path,h in au['output_hashes'].items():pin(path,h)
        summary=jr(paths['actual_summaries']);audit=jr(paths['actual_audits'])
        assert summary['status']=='complete' and audit['status']=='passed' and summary['protocol_sha256']==sha(protocol_path)
        assert summary['draw_index']==audit['draw_index']==entry['draw_index'] and summary['arm']==arm
        assert summary['physical_actions']==audit['actual_actions']==5*audit['rounds'] and summary['physical_resets']==audit['resets']==1
        first=audit['first_native_success_action'];success=bool(audit['success'])
        assert success==summary['success'] and (first==audit['actual_actions'] if success else first is None and audit['actual_actions']==100)
        geo=audit['final_geometry'];counts=summary['counts'];assert counts['parameter_updates']==0
        row=dict(draw_index=entry['draw_index'],arm=arm,trajectory_id=entry['trajectory_id'],offset=entry['offset'],task_id=entry['task_id'],
            initial_group_id=entry['initial_group_id'],trajectory_group_id=entry['trajectory_group_id'],
            history_frames=1 if arm=='native_frozen' else entry['history_frames'],deadline=False if arm=='native_frozen' else entry['deadline'],
            success_by100=success,success_by25=success and first<=25,first_success_action=first,
            capped100_cost=first if success else 100,capped25_cost=first if success and first<=25 else 25,
            actual_actions=audit['actual_actions'],resets=1,rounds=audit['rounds'],initial_success=audit['initial_geometry_success'],
            wall_seconds=summary['wall_seconds'],planning_total_wall_seconds=summary.get('planning_total_wall_seconds'),peak_allocated_bytes=summary['peak_allocated_bytes'],
            final_joint_position_error=geo['joint_position_error'],final_angle_error_rad=geo['angle_error'],
            final_pusher_error=geo['pusher_error'],final_block_error=geo['block_error'])
        return row,counts
    # Hash-bound audit receipts are the numerical source; no raw trajectories or E queries.
    base_rows=[];new_rows=[];count_groups={arm:[] for arm in ('native_frozen',*ARMS)}
    for e in old['queue']:
        row,c=load_case(e,BASE_PROTOCOL.parent/'candidate_audit_v1'/e['run_name'],base_controller,BASE_PROTOCOL,'native_frozen')
        base_rows.append(row);count_groups['native_frozen'].append(c)
    assert [r['draw_index'] for r in base_rows]==list(range(50))
    for e in p['queue']:
        row,c=load_case(e,B/'audit_v1'/e['run_name'],controller,args.protocol,e['arm'])
        ref=base_rows[e['draw_index']]
        for key in ('trajectory_id','offset','task_id','initial_group_id','trajectory_group_id'):assert row[key]==ref[key]
        assert Path(e['baseline_output']).resolve()==Path(old['queue'][e['draw_index']]['output']).resolve()
        new_rows.append(row);count_groups[e['arm']].append(c)
        pairs.append(dict(draw_index=row['draw_index'],arm=row['arm'],trajectory_id=row['trajectory_id'],offset=row['offset'],
            native_success_by100=ref['success_by100'],new_success_by100=row['success_by100'],
            native_success_by25=ref['success_by25'],new_success_by25=row['success_by25'],
            native_capped100_cost=ref['capped100_cost'],new_capped100_cost=row['capped100_cost'],new_minus_native_capped100_cost=row['capped100_cost']-ref['capped100_cost'],
            native_capped25_cost=ref['capped25_cost'],new_capped25_cost=row['capped25_cost'],new_minus_native_capped25_cost=row['capped25_cost']-ref['capped25_cost']))
    out.mkdir(exist_ok=False)
    try:
        all_rows=base_rows+new_rows;groups={arm:[r for r in all_rows if r['arm']==arm] for arm in count_groups}
        arm_summary={arm:describe(rows) for arm,rows in groups.items()};comparisons={}
        for arm in ARMS:
            pp=[r for r in pairs if r['arm']==arm];assert len(pp)==50
            cc={}
            for cap in (100,25):
                key=f'new_minus_native_capped{cap}_cost';delta=[r[key] for r in pp]
                cc[f'by{cap}']=dict(paired_table={f'new_{a}_native_{b}':sum(int(r[f'new_success_by{cap}'])==a and int(r[f'native_success_by{cap}'])==b for r in pp) for a in (0,1) for b in (0,1)},
                    mean_new_minus_native_cost=float(np.mean(delta)),new_lower_cost=sum(d<0 for d in delta),equal_cost=sum(d==0 for d in delta),new_higher_cost=sum(d>0 for d in delta),
                    recovered_draws=[r['draw_index'] for r in pp if r[f'new_success_by{cap}'] and not r[f'native_success_by{cap}']],
                    lost_success_draws=[r['draw_index'] for r in pp if not r[f'new_success_by{cap}'] and r[f'native_success_by{cap}']])
            comparisons[arm]=cc
        dependencies={};trajectory_rows=[]
        for row in base_rows:dependencies.setdefault(str(row['trajectory_id']),[]).append(dict(draw_index=row['draw_index'],offset=row['offset'],initial_group_id=row['initial_group_id']))
        for arm,rows in groups.items():
            for tid in sorted({r['trajectory_id'] for r in rows}):trajectory_rows.append(dict(arm=arm,trajectory_id=tid,**describe([r for r in rows if r['trajectory_id']==tid])))
        budgets={arm:{k:sum(c[k] for c in counts) for k in counts[0]} for arm,counts in count_groups.items()}
        timing={}
        for arm,rows in groups.items():
            planning=[r['planning_total_wall_seconds'] for r in rows if r['planning_total_wall_seconds'] is not None]
            timing[arm]=dict(case_wall_seconds_sum=sum(r['wall_seconds'] for r in rows),case_wall_seconds_mean=float(np.mean([r['wall_seconds'] for r in rows])),
                planning_total_wall_seconds_sum=sum(planning) if planning else None,planning_total_wall_seconds_mean=float(np.mean(planning)) if planning else None,
                planning_time_available_cases=len(planning),peak_allocated_bytes_max=max(r['peak_allocated_bytes'] for r in rows),
                interpretation='Measured execution time and allocated CUDA peak, not matched FLOPs. Baseline planning-only time is absent rather than zero; history and deadline change computation.')
        # Descriptive two-factor contrasts on exactly the same50 draws. Binary
        # nonadditivity is not a mechanistic or statistical significance claim.
        lookups={arm:{r['draw_index']:r for r in rows} for arm,rows in groups.items()}
        effect_names=('D_at_H0','D_at_H1','H_at_D0','H_at_D1','interaction')
        factorial_rows=[];factorial={}
        for metric in ('success_by100','success_by25','capped100_cost','capped25_cost'):
            metric_rows=[]
            for i in range(50):
                b,d,h,hd=[int(lookups[arm][i][metric]) for arm in ('native_frozen','deadline','history3','history3_deadline')]
                rr=dict(draw_index=i,trajectory_id=base_rows[i]['trajectory_id'],offset=base_rows[i]['offset'],outcome=metric,
                    B=b,D=d,H=h,HD=hd,D_at_H0=d-b,D_at_H1=hd-h,H_at_D0=h-b,H_at_D1=hd-d,interaction=hd-h-d+b)
                metric_rows.append(rr);factorial_rows.append(rr)
            per_trajectory=[]
            for tid in sorted({r['trajectory_id'] for r in metric_rows}):
                selected=[r for r in metric_rows if r['trajectory_id']==tid]
                per_trajectory.append(dict(trajectory_id=tid,draws=len(selected),**{k:float(np.mean([r[k] for r in selected])) for k in effect_names}))
            factorial[metric]=dict(draw_equal_mean={k:float(np.mean([r[k] for r in metric_rows])) for k in effect_names},
                trajectory_equal_after_draw_aggregation={k:float(np.mean([r[k] for r in per_trajectory])) for k in effect_names},
                per_trajectory=per_trajectory,sign='Positive success contrasts mean more hits; negative cost contrasts mean fewer capped actions.',
                limitation='Additive-scale descriptive contrasts; binary nonadditivity does not establish an interaction mechanism, causation, or significance.')
        new_actions=sum(r['actual_actions'] for r in new_rows);assert len(new_rows)==150 and 750<=new_actions<=15000
        result=dict(status='complete_all150_temporal_context_descriptions',draws=50,new_cases=150,reused_native_cases=50,arms=arm_summary,paired_vs_native=comparisons,
            trajectory_dependencies=dependencies,unique_trajectories=len(dependencies),trajectory_descriptions=trajectory_rows,
            descriptive_factorial_contrasts=factorial,budget_groups=budgets,timing_and_memory_by_arm=timing,new_physical_actions=new_actions,new_resets=150,new_analysis_model_calls=0,new_analysis_encoder_calls=0,new_analysis_physics_steps=0,
            all_consumed_development_data=True,independent_confirmation=False,
            definitions=dict(primary='By100 native joint4D position<20 and wrapped angle<pi/9, checked every5 raw; first success, no action0 relabeling.',
                secondary='By25 is a prespecified secondary horizon. Capped25 costs25 for both failures and successes exactlyat25; success flags remain separate.',
                units='One model token contains5 raw commands. Future horizon stays5 tokens; real history has1/2/3 available frames, not future observations.',
                weighting='Each arm uses the same50 windows. Baseline reused once;150 new runs are not150 independent tasks. Trajectory summaries retain dependence; no bootstrap/p-values.',
                information='History arms have more own-episode observed frames and fixed executed action tokens; no extra physical interactions. More encoded/context frames mean unequal compute despite equal GD100/horizon.',
                deadline='For rounds0..4 deadline uses relative forecast slot5-round; from round5 onwards native all-horizon weighting. History alone keeps original staged switch.',
                factorial='B=native, D=deadline, H=history3, HD=history3_deadline. D_at_H0=D-B; D_at_H1=HD-H; H_at_D0=H-B; H_at_D1=HD-D; interaction=HD-H-D+B. Four outcomes per same draw, not200 independent samples.',
                limitations='Methods are deployment diagnostics on consumed development data. Neither success differences nor descriptive paired counts establish originality, SOTA, or a unique mechanism.'))
        write_csv(out/'all200_tasks.csv',all_rows);write_csv(out/'all150_paired.csv',pairs)
        write_csv(out/'all50_factorial_contrasts.csv',factorial_rows)
        failures=[r for r in all_rows if not r['success_by100']]
        if failures:write_csv(out/'all_failed_cases.csv',failures)
        save(out/'summary.json',result);save(out/'input_hashes.json',pins)
        save(out/'terminal.json',dict(status=result['status'],summary_sha256=sha(out/'summary.json'),output_hashes={str(f):sha(f) for f in out.iterdir() if f.is_file()}))
        print(json.dumps(dict(status=result['status'],draws=50,new_cases=150,new_physical_actions=new_actions,analysis_model_calls=0)))
    except BaseException as error:
        save(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),no_automatic_retry=True));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--final',action='store_true',required=True);parser.add_argument('--protocol',type=Path,required=True);main(parser.parse_args())
