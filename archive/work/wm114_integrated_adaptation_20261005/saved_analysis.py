"""Audit and summarize saved WM114 cases. Standard library; no model execution.

python saved_analysis.py --protocol protocol.json --output analysis_v1
The output directory must not exist. Partial queues are permitted; paired
statistics use only draws with all four audited, completed arms.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import traceback

ARMS = ('B', 'BR', 'BG', 'BRG')
TIME_FIELDS = ('planning_total_wall_seconds', 'training_total_wall_seconds',
               'monitor_total_wall_seconds', 'wall_seconds')
SCALAR_FIELDS = ('residual_updates', 'replan_rounds', 'reuse_rounds', 'rounds')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path, manifest):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    value = json.loads(raw.decode('utf-8-sig'))
    manifest[str(path)] = hashlib.sha256(raw).hexdigest()
    return value


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def write_csv(path, rows, fields):
    with Path(path).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name, minimum=0):
    check(type(value) is int and value >= minimum, f'{name}: invalid integer {value!r}')
    return value


def number(value, name):
    check(type(value) in (int, float) and math.isfinite(value) and value >= 0,
          f'{name}: invalid finite nonnegative number {value!r}')
    return value


def summarize(rows):
    if not rows:
        return dict(cases=0, success_by100=0, success_by25=0,
                    success_rate_by100=None, success_rate_by25=None,
                    mean_capped100=None, mean_capped25=None, physical_actions=0)
    return dict(cases=len(rows), success_by100=sum(r['success_by100'] for r in rows),
                success_by25=sum(r['success_by25'] for r in rows),
                success_rate_by100=statistics.mean(r['success_by100'] for r in rows),
                success_rate_by25=statistics.mean(r['success_by25'] for r in rows),
                mean_capped100=statistics.mean(r['capped100'] for r in rows),
                mean_capped25=statistics.mean(r['capped25'] for r in rows),
                physical_actions=sum(r['physical_actions'] for r in rows))


def budget(rows):
    keys = sorted(set().union(*(r['counts'].keys() for r in rows))) if rows else []
    return dict(cases=len(rows),
                counts_total={k: sum(r['counts'].get(k, 0) for r in rows) for k in keys},
                counts_cases_reporting={k: sum(k in r['counts'] for r in rows) for k in keys},
                time_totals={k: sum(r[k] for r in rows) for k in TIME_FIELDS},
                time_means={k: statistics.mean(r[k] for r in rows) if rows else None for k in TIME_FIELDS},
                totals={k: sum(r[k] for r in rows) for k in SCALAR_FIELDS},
                physical_actions=sum(r['physical_actions'] for r in rows))


def analyze(protocol_path, output):
    protocol_path = Path(protocol_path).resolve(strict=True)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {}
    try:
        protocol = read_json(protocol_path, manifest)
        protocol_sha = manifest[str(protocol_path)]
        queue = protocol['queue']
        check(isinstance(queue, list) and queue, 'queue must be a nonempty list')
        keys, identities, roots = set(), {}, set()
        for entry in queue:
            draw = integer(entry['draw_index'], 'draw_index')
            arm = entry['arm']
            check(arm in ARMS, f'unknown arm {arm}')
            check((draw, arm) not in keys, f'duplicate draw/arm {draw}/{arm}')
            keys.add((draw, arm))
            group = str(entry['trajectory_group_id'])
            check(draw not in identities or identities[draw] == group, 'trajectory grouping mismatch')
            identities[draw] = group
            root = Path(entry['output']).resolve()
            check(root not in roots, 'duplicate output directory')
            roots.add(root)
            check(root != output and root not in output.parents and output not in root.parents,
                  'analysis output overlaps a case directory')
        draws = sorted(identities)
        check(keys == {(d, a) for d in draws for a in ARMS}, 'every planned draw must have four arms')
        rows, pending, audited = [], [], {}
        completed_prefix = 0
        prefix_open = True
        for entry in queue:
            draw, arm = entry['draw_index'], entry['arm']
            root = Path(entry['output']).resolve()
            base = dict(draw_index=draw, arm=arm, trajectory_group_id=str(entry['trajectory_group_id']),
                        output=str(root))
            terminal_path = root / 'terminal.json'
            if not terminal_path.is_file():
                state = 'in_progress_or_unsealed' if root.exists() else 'not_started'
                pending.append(dict(base, status=state))
                rows.append(dict(base, status=state))
                prefix_open = False
                continue
            terminal = read_json(terminal_path, manifest)
            # All referenced terminal receipts must be verifiable, including failures.
            for field, value in terminal.items():
                if field in ('summary_sha256', 'failure_sha256'):
                    named = root / ('summary.json' if field == 'summary_sha256' else 'failure.json')
                    read_json(named, manifest)
                    check(manifest[str(named.resolve())] == value, f'{root}: {field} mismatch')
            if terminal.get('status') != 'complete':
                state = terminal.get('status', 'unknown_terminal_status')
                pending.append(dict(base, status=state, terminal_sha256=manifest[str(terminal_path)]))
                rows.append(dict(base, status=state))
                prefix_open = False
                continue
            check('summary_sha256' in terminal, f'{root}: complete terminal has no summary hash')
            summary_path = root / 'summary.json'
            saved = read_json(summary_path, manifest)
            check(manifest[str(summary_path)] == terminal['summary_sha256'],
                  f'{root}: summary changed between receipt reads')
            check(saved.get('status') == 'complete', f'{root}: summary not complete')
            check(saved.get('protocol_sha256') == protocol_sha, f'{root}: protocol hash mismatch')
            check(str(saved['trajectory_group_id']) == base['trajectory_group_id'], f'{root}: group mismatch')
            check(type(saved['success']) is bool, f'{root}: success is not boolean')
            check(saved.get('frozen_state_unchanged') is True, f'{root}: frozen state not verified')
            actions = integer(saved['physical_actions'], f'{root}: physical_actions', 1)
            check(actions <= 100 and actions % 5 == 0, f'{root}: invalid macro action count')
            check(saved['success'] or actions == 100, f'{root}: completed failure not run to 100 actions')
            scalar = {k: integer(saved[k], f'{root}: {k}') for k in SCALAR_FIELDS}
            check(scalar['rounds'] * 5 == actions, f'{root}: rounds/actions mismatch')
            check(scalar['replan_rounds'] + scalar['reuse_rounds'] == scalar['rounds'],
                  f'{root}: replan/reuse accounting mismatch')
            check('R' in arm or scalar['residual_updates'] == 0,
                  f'{root}: residual updates in an arm without R')
            check('G' in arm or scalar['reuse_rounds'] == 0,
                  f'{root}: plan reuse in an arm without G')
            times = {k: number(saved[k], f'{root}: {k}') for k in TIME_FIELDS}
            counts = saved['counts']
            check(isinstance(counts, dict), f'{root}: counts is not a dictionary')
            for name, value in counts.items():
                number(value, f'{root}: counts.{name}')
            if arm == 'B':
                regression_path = root / 'baseline_regression.json'
                regression = read_json(regression_path, manifest)
                check(regression.get('status') == 'passed', f'{root}: B baseline regression did not pass')
            success = saved['success']
            row = dict(base, status='complete', success_by100=success,
                       success_by25=success and actions <= 25,
                       first_success_action=actions if success else None,
                       capped100=actions if success else 100, capped25=min(actions, 25),
                       physical_actions=actions, counts=counts, **scalar, **times,
                       summary_sha256=manifest[str(summary_path)],
                       terminal_sha256=manifest[str(terminal_path)])
            rows.append(row)
            audited[(draw, arm)] = row
            if prefix_open:
                completed_prefix += 1
        common = [d for d in draws if all((d, a) in audited for a in ARMS)]
        matched = {a: [audited[(d, a)] for d in common] for a in ARMS}
        by_arm = {a: [r for r in audited.values() if r['arm'] == a] for a in ARMS}
        paired, comparisons = [], {}
        for arm in ARMS[1:]:
            for d in common:
                base, new = audited[(d, 'B')], audited[(d, arm)]
                paired.append(dict(draw_index=d, trajectory_group_id=identities[d], arm=arm,
                    B_success_by100=base['success_by100'], arm_success_by100=new['success_by100'],
                    B_success_by25=base['success_by25'], arm_success_by25=new['success_by25'],
                    recovered_by100=not base['success_by100'] and new['success_by100'],
                    new_failure_by100=base['success_by100'] and not new['success_by100'],
                    recovered_by25=not base['success_by25'] and new['success_by25'],
                    new_failure_by25=base['success_by25'] and not new['success_by25'],
                    B_capped100=base['capped100'], arm_capped100=new['capped100'],
                    capped100_delta=new['capped100']-base['capped100'],
                    capped25_delta=new['capped25']-base['capped25'],
                    wall_seconds_delta=new['wall_seconds']-base['wall_seconds']))
            subset = [r for r in paired if r['arm'] == arm]
            comparisons[arm] = dict(paired_draws=len(common),
                recovered_by100=[r['draw_index'] for r in subset if r['recovered_by100']],
                new_failure_by100=[r['draw_index'] for r in subset if r['new_failure_by100']],
                recovered_by25=[r['draw_index'] for r in subset if r['recovered_by25']],
                new_failure_by25=[r['draw_index'] for r in subset if r['new_failure_by25']],
                mean_capped100_delta=statistics.mean(r['capped100_delta'] for r in subset) if subset else None,
                mean_capped25_delta=statistics.mean(r['capped25_delta'] for r in subset) if subset else None,
                interaction_cost_wins=sum(r['capped100_delta'] < 0 for r in subset),
                interaction_cost_ties=sum(r['capped100_delta'] == 0 for r in subset),
                interaction_cost_losses=sum(r['capped100_delta'] > 0 for r in subset))
        trajectory_groups = sorted({identities[d] for d in common})
        grouped = {}
        for arm in ARMS:
            per_group = {g: summarize([r for r in matched[arm] if r['trajectory_group_id'] == g])
                         for g in trajectory_groups}
            grouped[arm] = dict(per_group=per_group,
                equal_trajectory_success_rate_by100=statistics.mean(v['success_rate_by100'] for v in per_group.values()) if per_group else None,
                equal_trajectory_success_rate_by25=statistics.mean(v['success_rate_by25'] for v in per_group.values()) if per_group else None,
                equal_trajectory_mean_capped100=statistics.mean(v['mean_capped100'] for v in per_group.values()) if per_group else None)
        interactions = [dict(draw_index=d,
            capped100=audited[(d, 'BRG')]['capped100']-audited[(d, 'BR')]['capped100']-audited[(d, 'BG')]['capped100']+audited[(d, 'B')]['capped100'],
            success_by100=int(audited[(d, 'BRG')]['success_by100'])-int(audited[(d, 'BR')]['success_by100'])-int(audited[(d, 'BG')]['success_by100'])+int(audited[(d, 'B')]['success_by100'])) for d in common]
        result = dict(schema='wm114-saved-analysis-v1',
            status='complete_all_cases' if not pending else 'partial_saved_cases',
            protocol_sha256=protocol_sha, planned_cases=len(queue), planned_draws=len(draws),
            planned_trajectory_groups=len(set(identities.values())), completed_cases=len(audited),
            completed_queue_prefix=completed_prefix, common_completed_draws=common,
            common_trajectory_groups=trajectory_groups, pending=pending,
            matched_main={a: summarize(matched[a]) for a in ARMS},
            unmatched_completion_descriptions={a: summarize(by_arm[a]) for a in ARMS},
            comparisons_vs_B=comparisons, matched_budget={a: budget(matched[a]) for a in ARMS},
            all_completed_budget={a: budget(by_arm[a]) for a in ARMS},
            trajectory_weighted_descriptions=grouped,
            factorial_interaction=dict(definition='BRG - BR - BG + B; descriptive only',
                per_draw=interactions,
                mean_capped100=statistics.mean(r['capped100'] for r in interactions) if interactions else None,
                mean_success_by100=statistics.mean(r['success_by100'] for r in interactions) if interactions else None),
            independent_confirmation=False, model_calls_in_analysis=0, physical_actions_in_analysis=0,
            limits=['All windows are consumed development data; repeated trajectories are dependent.',
                    'Matched main tables include only draws with all four completed and audited arms.',
                    'All-completed tables can have unmatched task mixtures and are not performance comparisons.',
                    'Success by 100 and capped-100 interactions are primary; success by 25 is secondary.',
                    'No IID confidence interval, novelty, SOTA or acceptance-probability claim.',
                    'Input hashes verify saved receipts, not an independent model or simulator rerun.',
                    'Count keys absent in a case are shown in reporting coverage, not silently verified as zero.',
                    'Partial/failed cases are listed; their consumed compute is not included in completed budgets.'])
        write_json(output/'summary.json', result)
        case_fields = ['draw_index','arm','trajectory_group_id','status','success_by100','success_by25',
                       'first_success_action','capped100','capped25','physical_actions',*SCALAR_FIELDS,*TIME_FIELDS,
                       'summary_sha256','terminal_sha256','output']
        write_csv(output/'all_cases.csv', rows, case_fields)
        pair_fields = ['draw_index','trajectory_group_id','arm','B_success_by100','arm_success_by100',
                       'B_success_by25','arm_success_by25','recovered_by100','new_failure_by100',
                       'recovered_by25','new_failure_by25','B_capped100','arm_capped100','capped100_delta',
                       'capped25_delta','wall_seconds_delta']
        write_csv(output/'paired_vs_B.csv', paired, pair_fields)
        lines = ['WM114 整合适应：保存结果审核', '',
                 f"已审核完成 {len(audited)}/{len(queue)} 项；按队列连续完成前缀 {completed_prefix} 项。",
                 f"四臂共同完成 {len(common)}/{len(draws)} 个窗口，来自 {len(trajectory_groups)} 条轨迹。",
                 '下列主表只使用四臂共同完成窗口；全部是已消费开发数据，不能称独立确认。', '',
                 '配置 | 100步成功 | 平均封顶100步交互 | 25步成功 | 平均封顶25步交互']
        def fmt(value):
            return '尚无' if value is None else f'{value:.3f}'
        for arm in ARMS:
            s = result['matched_main'][arm]
            lines.append(f"{arm} | {s['success_by100']}/{s['cases']} | {fmt(s['mean_capped100'])} | {s['success_by25']}/{s['cases']} | {fmt(s['mean_capped25'])}")
        lines += ['', '与B逐窗口配对（负交互差值表示少用动作）']
        for arm, s in comparisons.items():
            lines.append(f"{arm}: 100步恢复{s['recovered_by100']}；新增失败{s['new_failure_by100']}；平均交互差值{fmt(s['mean_capped100_delta'])}；更快/相同/更慢={s['interaction_cost_wins']}/{s['interaction_cost_ties']}/{s['interaction_cost_losses']}。")
            lines.append(f"    25步恢复{s['recovered_by25']}；新增失败{s['new_failure_by25']}。")
        lines += ['', '同一批配对窗口的计算成本（独立核算，不能把交互节省称为计算节省）']
        for arm in ARMS:
            b = result['matched_budget'][arm]
            lines.append(f"{arm}: 模型残差更新{b['totals']['residual_updates']}；重规划{b['totals']['replan_rounds']}；复用{b['totals']['reuse_rounds']}；平均墙钟{fmt(b['time_means']['wall_seconds'])}秒。")
            lines.append('    规划/训练/监测累计秒=' + '/'.join(fmt(b['time_totals'][k]) for k in TIME_FIELDS[:3]))
            lines.append('    调用计数=' + json.dumps(b['counts_total'], ensure_ascii=False, sort_keys=True))
        lines += ['', '组合交互项BRG−BR−BG+B：平均封顶交互=' + fmt(result['factorial_interaction']['mean_capped100']) + '。这是描述，不能单独证明协同机制。',
                  '', '未完成/未封存记录：']
        lines.extend(f"draw{r['draw_index']}/{r['arm']}: {r['status']}" for r in pending)
        if not pending:
            lines.append('无。')
        lines += ['', '边界：每5原始动作检查；失败记100；25步为次指标。保留全部恢复和新增失败。',
                  '轨迹等权描述、逐项预算及全部已完成case（可能不配对）见summary.json。',
                  '输入哈希证明所读保存文件一致，未重新运行模型或物理环境。部分/失败case预算尚未纳入完整case预算。',
                  '本分析模型调用0、物理动作0；不作原创、SOTA、独立确认或录用结论。']
        with (output/'report.txt').open('x', encoding='utf-8', newline='\n') as stream:
            stream.write('\n'.join(lines)+'\n')
        write_json(output/'manifest.json', dict(inputs=manifest, analyzer_sha256=digest(__file__),
                                               command=[sys.executable, *sys.argv]))
        write_json(output/'terminal.json', dict(status='complete_saved_analysis',
            queue_status=result['status'], summary_sha256=digest(output/'summary.json'),
            report_sha256=digest(output/'report.txt'), manifest_sha256=digest(output/'manifest.json'),
            all_cases_sha256=digest(output/'all_cases.csv'), paired_vs_B_sha256=digest(output/'paired_vs_B.csv')))
        return result
    except BaseException as error:
        write_json(output/'failure.json', dict(status='failed_retained', error=repr(error),
                                               traceback=traceback.format_exc(), inputs_read=manifest))
        write_json(output/'terminal.json', dict(status='failed_retained', failure_sha256=digest(output/'failure.json')))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.protocol, args.output)
    print(json.dumps(dict(status=result['status'], completed_cases=result['completed_cases'],
                          common_completed_draws=len(result['common_completed_draws'])), ensure_ascii=False))
