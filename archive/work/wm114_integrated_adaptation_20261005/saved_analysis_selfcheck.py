"""Synthetic-file checks only; no research data, model, physics or torch."""
from pathlib import Path
import json
import tempfile
from saved_analysis import analyze, digest, write_json


def main():
    with tempfile.TemporaryDirectory(prefix='wm114_saved_analysis_check_') as temporary:
        root = Path(temporary)
        queue = [dict(draw_index=d, arm=a, output=str(root/f'case_{d}_{a}'),
                      trajectory_group_id=f'trajectory_{d//2}')
                 for d in range(3) for a in ('B','BR','BG','BRG')]
        protocol = root/'protocol.json'
        write_json(protocol, dict(queue=queue))
        outcomes = [(False,100),(True,25),(True,30),(True,20),
                    (True,25),(False,100),(True,25),(True,25),(True,25)]
        for entry, (success, actions) in zip(queue, outcomes):
            case = Path(entry['output']); case.mkdir()
            rounds = actions//5
            reuse = 1 if 'G' in entry['arm'] else 0
            summary = dict(status='complete', success=success, physical_actions=actions,
                rounds=rounds, counts=dict(predict_calls=10*rounds),
                planning_total_wall_seconds=2.0, training_total_wall_seconds=0.25,
                monitor_total_wall_seconds=0.1, wall_seconds=3.0,
                residual_updates=rounds-1 if 'R' in entry['arm'] else 0,
                replan_rounds=rounds-reuse, reuse_rounds=reuse,
                frozen_state_unchanged=True, protocol_sha256=digest(protocol),
                trajectory_group_id=entry['trajectory_group_id'])
            write_json(case/'summary.json', summary)
            write_json(case/'terminal.json', dict(status='complete', summary_sha256=digest(case/'summary.json')))
            if entry['arm'] == 'B':
                write_json(case/'baseline_regression.json', dict(status='passed'))
        result = analyze(protocol, root/'analysis')
        assert result['status'] == 'partial_saved_cases'
        assert result['completed_cases'] == result['completed_queue_prefix'] == 9
        assert result['common_completed_draws'] == [0,1]
        assert result['common_trajectory_groups'] == ['trajectory_0']
        assert len(result['pending']) == 3
        assert [result['matched_main'][a]['mean_capped100'] for a in ('B','BR','BG','BRG')] == [62.5,62.5,27.5,22.5]
        assert [result['matched_main'][a]['success_by25'] for a in ('B','BR','BG','BRG')] == [1,1,1,2]
        assert result['comparisons_vs_B']['BR']['recovered_by100'] == [0]
        assert result['comparisons_vs_B']['BR']['new_failure_by100'] == [1]
        assert result['factorial_interaction']['mean_capped100'] == -5
        assert result['all_completed_budget']['B']['cases'] == 3
        assert result['matched_budget']['B']['cases'] == 2
        assert result['trajectory_weighted_descriptions']['B']['equal_trajectory_success_rate_by100'] == 0.5
        assert result['independent_confirmation'] is False
        before = digest(root/'analysis/summary.json')
        try:
            analyze(protocol, root/'analysis')
        except FileExistsError:
            pass
        else:
            raise AssertionError('existing output not refused')
        assert digest(root/'analysis/summary.json') == before
        # Corrupt the saved summary without resealing: audit must fail before a report.
        corrupt = root/'case_0_BR/summary.json'
        bad = json.loads(corrupt.read_text()); bad['physical_actions'] = 30
        corrupt.write_text(json.dumps(bad))
        try:
            analyze(protocol, root/'corrupt_analysis')
        except ValueError as error:
            assert 'summary_sha256 mismatch' in str(error)
        else:
            raise AssertionError('tampered receipt accepted')
        assert not (root/'corrupt_analysis/report.txt').exists()
        assert json.loads((root/'corrupt_analysis/terminal.json').read_text())['status'] == 'failed_retained'
    print('PASSED: paired/unmatched separation, recoveries/harms, costs, trajectory grouping, output refusal, hash rejection; synthetic files only.')


if __name__ == '__main__':
    main()
