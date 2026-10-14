"""Prepare all150 WM108 metadata cells. Never loads a model or simulator."""
import json
from pathlib import Path
from wm108_contract import B, BASE, BASE_PROTOCOL, BASE_PROTOCOL_SHA, PROTOCOL, ARMS, sha, jr, save, verify_protocol

def main():
    assert not PROTOCOL.exists(), 'Never overwrite a prepared protocol'
    assert sha(BASE_PROTOCOL) == BASE_PROTOCOL_SHA
    old = jr(BASE_PROTOCOL)
    baseline_terminal = BASE/'runtime/candidate_actual_controller_v1/terminal.json'
    baseline_analysis = BASE/'analysis_v1/terminal.json'
    fixed = B.parent/'wm107_native_commitment_20261004'
    assert jr(baseline_terminal)['status'] == 'complete' and jr(baseline_terminal)['draws'] == 50
    assert jr(baseline_analysis)['status'] == 'complete_all50_native_descriptions'
    assert jr(fixed/'controller_v1/terminal.json')['status'] == 'complete'
    assert jr(fixed/'final_v1/terminal.json')['status'] == 'complete_all50_fixed_initial_plan_descriptions'
    assert jr(baseline_analysis)['summary_sha256'] == sha(BASE/'analysis_v1/summary.json')
    assert jr(fixed/'final_v1/terminal.json')['summary_sha256'] == sha(fixed/'final_v1/summary.json')
    source_hashes = dict(old['source_hashes'])
    for name in ('scientific_lock.txt', 'wm108_contract.py', 'build_protocol.py', 'controller.py',
                 'run_intervention.py', 'audit_case.py', 'analyze_saved.py'):
        source_hashes[str(B/name)] = sha(B/name)
    source_hashes[str(BASE_PROTOCOL)] = sha(BASE_PROTOCOL)
    source_hashes[str(BASE/'runtime/root_execution_gate.json')] = sha(BASE/'runtime/root_execution_gate.json')
    inputs = {}
    for path in (baseline_terminal, baseline_analysis, BASE/'analysis_v1/summary.json',
                 fixed/'controller_v1/terminal.json', fixed/'final_v1/terminal.json', fixed/'final_v1/summary.json'):
        inputs[str(path)] = sha(path)
    queue = []
    for source in old['queue']:
        d = source['draw_index']
        run = Path(source['output'])
        audit = BASE/'runtime/candidate_audit_v1'/source['run_name']
        terminal = jr(run/'terminal.json')
        assert terminal['status'] == 'complete' and terminal['summary_sha256'] == sha(run/'summary.json')
        assert jr(audit/'terminal.json')['status'] == 'complete'
        for path in (run/'summary.json', run/'terminal.json', run/'round_00/optimization.npz', run/'round_00/input_observations.npz',
                     run/'round_00/losses.json', run/'rng_before_plan.npz', run/'environment/initial.npz', run/'environment/step_005.npz',
                     audit/'audit.json', audit/'terminal.json'):
            inputs[str(path)] = sha(path)
        for arm in ARMS:
            e = dict(source)
            name = f'draw_{d:03d}_{arm}'
            e.update(arm=arm, run_name=name, output=str(B/'execution_v1'/name),
                     baseline_output=source['output'], history_frames=3 if 'history3' in arm else 1,
                     deadline='deadline' in arm)
            queue.append(e)
    p = dict(schema='wm108-temporal-context-v1', status='prepared_for_root_review',
        dataset='val', draws=50, cases=150, arms=list(ARMS), queue=queue,
        goal_receipt_sha256=old['goal_receipt_sha256'], selection_receipt_sha256=old['selection_receipt_sha256'],
        data_protocol_sha256=old['data_protocol_sha256'], scientific_lock_sha256=sha(B/'scientific_lock.txt'),
        baseline_protocol=str(BASE_PROTOCOL), baseline_protocol_sha256=sha(BASE_PROTOCOL),
        source_hashes=source_hashes, input_hashes=inputs, python=old['python'],
        expected_model_state_sha256=old['expected_model_state_sha256'], checkpoint_sha256=old['checkpoint_sha256'],
        budget=dict(max_model_loads=150, max_physical_resets=150, max_raw_actions=15000,
                    max_planning_predict_calls=1500000, parameter_updates=0, new_targets=0),
        scope='Posthoc development diagnostic of deadline/history and their interaction; original native100 baseline reused.',
        no_automatic_retry=True, pausing='STOP_AFTER_CURRENT_CASE at a complete actual+audit boundary.')
    save(PROTOCOL, p)
    verify_protocol(PROTOCOL, require_gate=False)
    print(json.dumps(dict(status=p['status'], protocol_sha256=sha(PROTOCOL), cases=len(queue), new_model_calls=0, new_physics=0)))

if __name__ == '__main__':
    main()
