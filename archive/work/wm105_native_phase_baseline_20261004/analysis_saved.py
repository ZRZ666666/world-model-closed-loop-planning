"""WM105 saved-only native outcomes and factual slot1 prediction descriptions.

--final requires all50 controller/actual/audit completions. --snapshot includes
only TASK OUTCOMES of the audited contiguous queue prefix and requires a separate
create-only output under analysis_snapshots/. Factual error statistics are final
only, as required by the scientific lock. Neither mode imports a model/encoder.
"""
from pathlib import Path
import argparse
import csv
import datetime
import hashlib
import json
import sys
import traceback
import numpy as np

sys.dont_write_bytecode = True
B = Path(__file__).resolve().parent
R = B/'runtime'
PROTOCOL = R/'candidate_protocol.json'
CONTROLLER = R/'candidate_actual_controller_v1'
MODALITIES = ('visual', 'proprio', 'combined')
MEASURES = ('prediction_MSE', 'persistence_MSE', 'prediction_minus_persistence_MSE',
    'Dcurrent', 'Dpredicted', 'Dactual', 'bias', 'predicted_progress', 'actual_progress',
    'cross_term', 'decomposition_sum', 'identity_residual', 'tau')
FLAGS = ('optimistic', 'pessimistic', 'near_zero_bias', 'robust_opposite', 'reverse_opposite',
    'near_zero_predicted_progress', 'near_zero_actual_progress')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def jr(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    assert rows
    with Path(path).open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def bool_csv(value):
    assert value in ('True', 'False')
    return value == 'True'


def numerical_terms(current, prediction, actual, goal):
    c, p, a, g = [x.astype(np.float64).ravel() for x in (current, prediction, actual, goal)]
    def mean_square(x):
        return float(np.dot(x, x)/x.size)
    e, residual = p-a, a-g
    return dict(prediction_MSE=mean_square(e), persistence_MSE=mean_square(c-a),
        Dcurrent=mean_square(c-g), Dpredicted=mean_square(p-g), Dactual=mean_square(residual),
        cross_term=float(2*np.dot(e, residual)/e.size))


def derive(terms):
    t = dict(terms)
    t['prediction_minus_persistence_MSE'] = t['prediction_MSE']-t['persistence_MSE']
    t['bias'] = t['Dpredicted']-t['Dactual']
    t['predicted_progress'] = t['Dcurrent']-t['Dpredicted']
    t['actual_progress'] = t['Dcurrent']-t['Dactual']
    t['decomposition_sum'] = t['prediction_MSE']+t['cross_term']
    t['identity_residual'] = t['bias']-t['decomposition_sum']
    assert abs(t['identity_residual']) <= 1e-10*max(1, abs(t['bias']))
    t['tau'] = 1e-6+1e-5*max(abs(t[k]) for k in ('Dcurrent', 'Dpredicted', 'Dactual'))
    tau = t['tau']
    t.update(optimistic=t['bias'] < -tau, pessimistic=t['bias'] > tau,
        near_zero_bias=abs(t['bias']) <= tau,
        robust_opposite=t['predicted_progress'] > tau and t['actual_progress'] < -tau,
        reverse_opposite=t['predicted_progress'] < -tau and t['actual_progress'] > tau,
        near_zero_predicted_progress=abs(t['predicted_progress']) <= tau,
        near_zero_actual_progress=abs(t['actual_progress']) <= tau)
    assert sum(t[k] for k in ('optimistic', 'pessimistic', 'near_zero_bias')) == 1
    return t


def task_description(rows):
    if not rows:
        return dict(draws=0, success_count=0, success_fraction=None, capped100_cost_mean=None,
                    successful_step_mean=None, successful_step_median=None, S100=0, F100=0)
    successes = [r['physical_actions'] for r in rows if r['success']]
    return dict(draws=len(rows), success_count=len(successes), success_fraction=len(successes)/len(rows),
        capped100_cost_mean=float(np.mean([r['capped100_cost'] for r in rows])),
        successful_step_mean=float(np.mean(successes)) if successes else None,
        successful_step_median=float(np.median(successes)) if successes else None,
        physical_actions=sum(r['physical_actions'] for r in rows),
        initial_success_descriptive_count=sum(r['initial_geometry_success'] for r in rows),
        S100=sum(r['success'] and r['physical_actions'] == 100 for r in rows),
        F100=sum(not r['success'] and r['physical_actions'] == 100 for r in rows))


def main(args):
    pins = {}
    def pin(path, expected=None):
        path = Path(path).resolve()
        h = sha(path)
        assert expected is None or h == expected, str(path)
        assert str(path) not in pins or pins[str(path)] == h, str(path)
        pins[str(path)] = h
        return h
    def read(path, expected=None):
        pin(path, expected)
        return jr(path)
    p = read(PROTOCOL)
    assert p['schema'] == 'wm105-native-random-phase-v1' and p['draws'] == len(p['queue']) == 50
    assert [e['draw_index'] for e in p['queue']] == list(range(50))
    pin(B/'scientific_question.txt', p['scientific_question_sha256'])
    for path in (R/'run_frozen.py', R/'audit_case.py', R/'task_contract.py'):
        pin(path, p['source_hashes'][str(path.resolve())])
    pin(__file__)
    controller = None
    if args.final:
        terminal_path = CONTROLLER/'terminal.json'
        if not terminal_path.is_file():
            print(json.dumps(dict(status='not_ready', reason='Final controller terminal absent; no case performance read')))
            return
        controller = read(terminal_path)
        if controller.get('status') != 'complete' or controller.get('draws') != 50:
            print(json.dumps(dict(status='not_ready', reason='All50 final gate not satisfied; no case performance read')))
            return
        assert controller['protocol_sha256'] == sha(PROTOCOL)
        selected = p['queue']
        out = B/'analysis_v1'
        assert args.output is None, '--final output is the fixed create-only analysis_v1'
    else:
        assert args.output is not None, '--snapshot requires a new explicit output directory'
        out = args.output.resolve()
        assert out.parent == (B/'analysis_snapshots').resolve()
        selected = []
        for e in p['queue']:
            if not (CONTROLLER/(e['run_name']+'.complete.json')).is_file():
                break
            selected.append(e)
        assert not any((CONTROLLER/(e['run_name']+'.complete.json')).exists() for e in p['queue'][len(selected):]), 'Noncontiguous completion prefix'
        if not selected:
            print(json.dumps(dict(status='not_ready', reason='No complete audited prefix; no output written')))
            return

    # Establish completion, successful process exit and receipt hashes for every
    # selected draw BEFORE reading any case outcome or feature arrays.
    accepted = []
    for e in selected:
        run = Path(e['output'])
        ad = R/'candidate_audit_v1'/e['run_name']
        complete = read(CONTROLLER/(e['run_name']+'.complete.json'))
        assert complete['draw_index'] == e['draw_index'] and complete['protocol_sha256'] == sha(PROTOCOL)
        assert complete['actual_exit_code'] == complete['audit_exit_code'] == 0
        rt = read(run/'terminal.json')
        at = read(ad/'terminal.json')
        assert rt['status'] == at['status'] == 'complete'
        assert not (run/'failure.json').exists() and not (ad/'failure.json').exists()
        links = dict(actual_summary=run/'summary.json', actual_terminal=run/'terminal.json',
                     audit=ad/'audit.json', audit_terminal=ad/'terminal.json')
        plural = dict(actual_summary='actual_summaries', actual_terminal='actual_terminals',
                      audit='actual_audits', audit_terminal='audit_terminals')
        for kind, path in links.items():
            receipt = complete['receipts'][kind]
            assert Path(receipt['path']).resolve() == path.resolve()
            pin(path, receipt['sha256'])
            if controller is not None:
                assert controller[plural[kind]][e['run_name']] == receipt
        pin(run/'summary.json', rt['summary_sha256'])
        pin(ad/'audit.json', at['audit_sha256'])
        for path, digest in at['output_hashes'].items():
            pin(path, digest)
        accepted.append((e, run, ad))
    assert len(accepted) == (50 if args.final else len(selected))
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'entry.json', dict(status='started_saved_only', mode='final' if args.final else 'snapshot',
        fixed_queue_draws=50, completed_prefix=len(accepted), pending_not_failures=50-len(accepted),
        included_draw_indices=[e['draw_index'] for e, _, _ in accepted],
        command=[sys.executable, *sys.argv], source_sha256=sha(__file__), protocol_sha256=sha(PROTOCOL),
        models=0, encoders=0, physics=0, fits=0, no_retry=True))
    tasks, blocks, draw_metrics, optimization = [], [], [], []
    budgets, elapsed = {}, dict(case_wall_seconds=0., post100_total_seconds=0., terminal_total_seconds=0.)
    current_draw = None
    try:
        for e, run, ad in accepted:
            current_draw = e['draw_index']
            s, a = read(run/'summary.json'), read(ad/'audit.json')
            audit_pins = read(ad/'input_hashes.json')
            assert a['status'] == 'passed' and a['draw_index'] == s['draw_index'] == current_draw
            assert s['protocol_sha256'] == sha(PROTOCOL) and s['success'] == a['success']
            assert s['rounds'] == a['rounds'] == a['factual_blocks']
            assert s['physical_actions'] == a['actual_actions'] == 5*a['rounds']
            assert s['physical_resets'] == a['resets'] == 1
            assert s['counts'] == a['counts']
            first = a['first_native_success_action']
            assert (first == s['physical_actions']) if s['success'] else (first is None and s['physical_actions'] == 100)
            boundaries = read_csv(ad/'native_boundaries.csv')
            assert [int(r['actions']) for r in boundaries] == list(range(0, s['physical_actions']+1, 5))
            first_from_rows = next((int(r['actions']) for r in boundaries[1:] if bool_csv(r['success'])), None)
            assert first_from_rows == first
            task = dict(draw_index=current_draw, task_id=e['task_id'], trajectory_id=e['trajectory_id'],
                offset=e['offset'], initial_group_id=e['initial_group_id'], trajectory_group_id=e['trajectory_group_id'],
                success=bool(s['success']), physical_actions=s['physical_actions'], rounds=s['rounds'],
                first_native_success_action=first, capped100_cost=s['physical_actions'] if s['success'] else 100,
                S100=bool(s['success'] and s['physical_actions'] == 100),
                F100=bool(not s['success'] and s['physical_actions'] == 100),
                initial_geometry_success=bool(a['initial_geometry_success']),
                final_joint_position_error_px=a['final_geometry']['joint_position_error'],
                final_angle_error_rad=a['final_geometry']['angle_error'],
                final_pusher_error_px=a['final_geometry']['pusher_error'],
                final_block_error_px=a['final_geometry']['block_error'],
                overlaps_WM106_first8=current_draw < 8, all_consumed_development_data=True,
                historical_WM106_round0_action_equal=None if a['historical_WM106'] is None else a['historical_WM106']['round0_returned_H5_byte_equal'])
            tasks.append(task)
            for key, value in s['counts'].items():
                assert isinstance(value, int) and value >= 0
                budgets[key] = budgets.get(key, 0)+value
            elapsed['case_wall_seconds'] += s['wall_seconds']
            elapsed['post100_total_seconds'] += s['diagnostic_wall_seconds']['post100_total']
            elapsed['terminal_total_seconds'] += s['diagnostic_wall_seconds']['terminal_total']
            if args.snapshot:
                # The pre-execution question defers factual-error computation
                # until the entire baseline is complete, even for snapshots.
                continue
            cache = {}
            def saved_arrays(path, keys):
                path = Path(path).resolve()
                if path not in cache:
                    pin(path, audit_pins[str(path)])
                    with np.load(path, allow_pickle=False) as z:
                        cache[path] = {k: z[k].copy() for k in keys}
                assert set(keys).issubset(cache[path])
                return cache[path]
            def code(a, shape):
                assert a.shape == shape and a.dtype == np.float32 and np.isfinite(a).all()
                return a
            opt = saved_arrays(run/'round_00/optimization.npz', ('goal_visual', 'goal_proprio'))
            goal = dict(visual=code(opt['goal_visual'], (1,1,196,8)), proprio=code(opt['goal_proprio'], (1,1,10)))
            alignment = read_csv(ad/'factual_alignment.csv')
            assert len(alignment) == s['rounds']
            draw_rows = []
            for r, link in enumerate(alignment):
                assert int(link['round']) == r and int(link['start_raw']) == 5*r and int(link['end_raw']) == 5*r+5
                assert int(link['predicted_future_slot']) == 1 and bool_csv(link['no_truth_for_future_slots_2_to_5'])
                predicted_path = run/f'round_{r:02d}/post100_prediction.npz'
                assert Path(link['prediction_file']).resolve() == predicted_path.resolve()
                actual_path = Path(link['actual_code_file']).resolve()
                post_keys = ('full_visual_prediction', 'full_proprio_prediction')
                post = saved_arrays(predicted_path, post_keys)
                forecast = dict(visual=code(post[post_keys[0]], (1,6,196,8)),
                                proprio=code(post[post_keys[1]], (1,6,10)))
                current = {k: v[:, :1] for k, v in forecast.items()}
                prediction = {k: v[:, 1:2] for k, v in forecast.items()}
                if r+1 < s['rounds']:
                    assert link['actual_code_kind'] == 'next_current_slot0'
                    assert actual_path == (run/f'round_{r+1:02d}/post100_prediction.npz').resolve()
                    z = saved_arrays(actual_path, post_keys)
                    actual = dict(visual=code(z[post_keys[0]], (1,6,196,8))[:, :1],
                                  proprio=code(z[post_keys[1]], (1,6,10))[:, :1])
                else:
                    assert link['actual_code_kind'] == 'direct_terminal_encode'
                    assert actual_path == (run/'terminal_observation_encoding.npz').resolve()
                    z = saved_arrays(actual_path, ('visual', 'proprio'))
                    actual = dict(visual=code(z['visual'], (1,1,196,8)), proprio=code(z['proprio'], (1,1,10)))
                terms = {k: numerical_terms(current[k], prediction[k], actual[k], goal[k]) for k in ('visual', 'proprio')}
                terms['combined'] = {k: terms['visual'][k]+terms['proprio'][k] for k in terms['visual']}
                for modality in MODALITIES:
                    row = dict(draw_index=current_draw, trajectory_id=e['trajectory_id'], offset=e['offset'],
                        task_success=task['success'], round=r, start_raw=5*r, end_raw=5*r+5,
                        modality=modality, actual_code_kind=link['actual_code_kind'],
                        prediction_file=str(predicted_path), actual_code_file=str(actual_path), **derive(terms[modality]))
                    blocks.append(row)
                    draw_rows.append(row)
                receipt_path = run/f'round_{r:02d}/post100_receipt.json'
                receipt = read(receipt_path, audit_pins[str(receipt_path)])
                assert receipt['step'] == r and receipt['returned_is_post100_not_argmin']
                optimization.append(dict(draw_index=current_draw, trajectory_id=e['trajectory_id'], task_success=task['success'],
                    round=r, objective_phase='terminal_only' if r < 5 else 'all_horizon_weighted',
                    initial_logged_loss=receipt['initial_logged_loss'],
                    last_logged_pre_update_loss=receipt['last_logged_pre_update_loss'],
                    minimum_logged_pre_update_loss=receipt['minimum_logged_pre_update_loss'],
                    minimum_logged_index=receipt['minimum_logged_index'], returned_post100_loss=receipt['returned_loss'],
                    post100_minus_last_preupdate=receipt['returned_loss']-receipt['last_logged_pre_update_loss'],
                    post100_minus_best_preupdate=receipt['returned_loss']-receipt['minimum_logged_pre_update_loss']))
            for modality in MODALITIES:
                rr = [r for r in draw_rows if r['modality'] == modality]
                row = dict(draw_index=current_draw, trajectory_id=e['trajectory_id'], task_success=task['success'],
                           modality=modality, blocks=len(rr))
                row.update({k: float(np.mean([r[k] for r in rr])) for k in MEASURES})
                row.update({'count_'+k: sum(r[k] for r in rr) for k in FLAGS})
                row.update({'fraction_'+k: sum(r[k] for r in rr)/len(rr) for k in FLAGS})
                draw_metrics.append(row)

        trajectory_rows = []
        for trajectory in sorted({r['trajectory_id'] for r in tasks}):
            rr = [r for r in tasks if r['trajectory_id'] == trajectory]
            trajectory_rows.append(dict(trajectory_id=trajectory, draw_indices=[r['draw_index'] for r in rr], **task_description(rr)))
        def describe_errors(subset):
            indices = {r['draw_index'] for r in subset}
            result = {}
            fields = (*MEASURES, *('fraction_'+k for k in FLAGS))
            for modality in MODALITIES:
                dr = [r for r in draw_metrics if r['draw_index'] in indices and r['modality'] == modality]
                br = [r for r in blocks if r['draw_index'] in indices and r['modality'] == modality]
                ts = sorted({r['trajectory_id'] for r in dr})
                tm = [{k: float(np.mean([r[k] for r in dr if r['trajectory_id'] == t])) for k in fields} for t in ts]
                result[modality] = dict(draws=len(dr), trajectories=len(ts), blocks=len(br),
                    draw_equal={k: float(np.mean([r[k] for r in dr])) for k in fields} if dr else None,
                    trajectory_equal_after_draw_aggregation={k: float(np.mean([r[k] for r in tm])) for k in fields} if tm else None,
                    block_weighted_descriptive={k: float(np.mean([r[k] for r in br])) for k in MEASURES} if br else None,
                    raw_block_counts={k: sum(r[k] for r in br) for k in FLAGS},
                    draws_with_each_flag={k: sum(r['count_'+k] > 0 for r in dr) for k in FLAGS})
            return result
        strata = dict(all=tasks, succeeded=[r for r in tasks if r['success']], failed=[r for r in tasks if not r['success']])
        duplicates = {}
        for name, key in (('same_trajectory_offset', lambda r: f"{r['trajectory_id']}:{r['offset']}"),
                          ('same_registered_initial_group', lambda r: r['initial_group_id'])):
            groups = {}
            for task in tasks:
                groups.setdefault(key(task), []).append(task['draw_index'])
            duplicates[name] = {k: v for k, v in groups.items() if len(v)>1}
        if controller is not None:
            assert len(tasks) == 50 and controller['physical_resets'] == 50
            assert controller['physical_actions'] == sum(r['physical_actions'] for r in tasks) == budgets['physical_actions']
            assert controller['successes'] == sum(r['success'] for r in tasks)
            assert controller['rounds'] == len(optimization)
        assert len(blocks) == 3*len(optimization)
        assert len(draw_metrics) == (3*len(tasks) if args.final else 0)
        status = 'complete_all50_native_descriptions' if args.final else 'partial_audited_prefix_descriptions'
        result = dict(status=status, mode='final' if args.final else 'snapshot', registered_draws=50,
            completed_audited_draws=len(tasks), pending_not_failures=50-len(tasks),
            registered_trajectory_count=len({e['trajectory_id'] for e in p['queue']}),
            analyzed_trajectory_count=len(trajectory_rows),
            task_performance={k: task_description(v) for k, v in strata.items()},
            trajectory_equal_success_fraction=float(np.mean([r['success_fraction'] for r in trajectory_rows])),
            trajectory_equal_capped100_cost=float(np.mean([r['capped100_cost_mean'] for r in trajectory_rows])),
            trajectory_descriptions=trajectory_rows, repeated_registered_groups=duplicates,
            geometric_near_neighbors='Not assessed: no new post-hoc proximity threshold or exclusion introduced.',
            executed_macrosteps=sum(t['rounds'] for t in tasks), factual_blocks_analyzed=len(optimization),
            factual_blocks=len(optimization), modality_rows=len(blocks), draw_modality_rows=len(draw_metrics),
            factual_statistics_deferred=not args.final,
            factual_errors_by_task_outcome={k: describe_errors(v) for k, v in strata.items()} if args.final else None,
            actual_code_sources={kind: sum(r['actual_code_kind'] == kind for r in blocks if r['modality'] == 'combined')
                                for kind in ('next_current_slot0', 'direct_terminal_encode')},
            new_execution_budgets=dict(physical_actions=budgets['physical_actions'], resets=len(tasks), instrumented_model_counts=budgets,
                measured_wall_seconds=elapsed, analysis_model_calls=0, analysis_encoder_calls=0, analysis_physics_calls=0),
            historical_goal_generation=dict(draws=50, resets=50, raw_actions=1250, counted_as_new_execution=False),
            WM106_overlap_draws=[r['draw_index'] for r in tasks if r['overlaps_WM106_first8']],
            all_consumed_development_data=True, independent_confirmation=False,
            definitions=dict(primary='Native first success at the 5raw check grid by100; S100 distinct from F100; failures cost100.',
                errors='Float64 arithmetic on saved float32 native codes, slot1 only; no truth for planned future slots2..5.',
                actual_codes='Next round CURRENT slot0 or direct terminal encode; terminal proprio is used as saved, not forced identical to rollout separate_emb.',
                combined='Sum of visual element MSE and proprio element MSE, not a concatenated mean.',
                direction='WM103 tau=1e-6+1e-5 max(abs(Dcurrent),abs(Dpredicted),abs(Dactual)); robust opposite requires predicted progress>tau and actual progress<-tau.',
                weighting='Primary error summaries average blocks within draw then draws equally; trajectory equal averages draw summaries within trajectory first. Pooled block means are descriptive only.',
                uncertainty='No bootstrap, p-values, IID assumptions or causal success/failure contrast.',
                optimization='Only within-round exact post100 versus logged preupdate losses; objective changes after round4. No actual outcome for the best preupdate candidate.'),
            limitations=['Outcome-conditioned error differences are descriptive and confounded by visited states and early stopping.',
                'A one-macrostep retreat can be a useful H5 detour; this distance is not the full native staged objective.',
                'Observed-policy predictions do not identify candidate ranking, optimizer exploitation or a beneficial intervention.',
                'Batch1 planner seed100 perdraw differs from the author full-batch RNG trajectory; no bitwise author-evaluation claim.'])
        write_csv(out/'all_tasks.csv', tasks)
        if args.final:
            write_csv(out/'all_factual_blocks.csv', blocks)
            write_csv(out/'per_draw_error_summaries.csv', draw_metrics)
            write_csv(out/'optimization_path_descriptions.csv', optimization)
        write_json(out/'summary.json', result)
        write_json(out/'input_hashes.json', pins)
        text = ('WM105 native source baseline — saved-only descriptive analysis\n'
            + json.dumps(dict(status=status, task_performance=result['task_performance'],
                trajectory_count=result['analyzed_trajectory_count'], factual_blocks=result['factual_blocks'],
                first8_overlap=result['WM106_overlap_draws'], budget=result['new_execution_budgets']), ensure_ascii=False, indent=2)
            + '\nAll source trajectories are consumed development data. Draws and variable-duration blocks are not IID. '
            + 'Success/failure strata are descriptions, not causes. Only factual slot1 has realized truth; no H5 suffix truth or method benefit is claimed.\n')
        (out/'report.txt').write_text(text, encoding='utf-8')
        for path, digest in pins.items():
            assert sha(path) == digest
        write_json(out/'terminal.json', dict(status=status, summary_sha256=sha(out/'summary.json'),
            source_sha256=sha(__file__), protocol_sha256=sha(PROTOCOL),
            created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            output_hashes={str(f): sha(f) for f in out.iterdir() if f.is_file()},
            analysis_model_calls=0, analysis_encoder_calls=0, analysis_physics_calls=0))
        print(json.dumps(dict(status=status, draws=len(tasks), factual_blocks=len(optimization), output=str(out))))
    except BaseException as error:
        write_json(out/'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), draw_index=current_draw,
            analyzed_draws=len(tasks), partial_modality_rows=len(blocks), no_cases_dropped_or_retried=True))
        write_json(out/'partial_input_hashes.json', pins)
        write_json(out/'terminal.json', dict(status='failed_retained', failure_sha256=sha(out/'failure.json')))
        raise


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--final', action='store_true')
    mode.add_argument('--snapshot', action='store_true')
    ap.add_argument('--output', type=Path)
    main(ap.parse_args())
