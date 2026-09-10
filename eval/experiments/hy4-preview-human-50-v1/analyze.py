import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean

from hyscript.evaluation.human import spearman, _ranks
from hyscript.evaluation.rubric import load_rubric

root = Path.cwd()
out = root / 'eval/experiments/hy4-preview-human-50-v1'
base = root / 'eval/experiments/formal-100-v1'
rubric = load_rubric(root / 'eval/rubrics/script_quality_v1.json')
dims = rubric.judge_dimension_ids
files = sorted((base / 'validation/human_review').glob('*.csv'))
reviews = [{r['run_id']: r for r in csv.DictReader(p.open(encoding='utf-8-sig'))} for p in files]
tasks = json.loads((out / 'trace_manifest.json').read_text())['tasks']
ids = sorted(t['run_id'] for t in tasks)
hashes = {t['run_id']: t['trace_sha256'] for t in tasks}
cache = json.loads((out / 'cache_reuse.json').read_text())
cached = {item['run_id']: item for item in cache['cached']}
for review in reviews:
    assert sorted(review) == ids
    for run in ids:
        assert review[run]['trace_sha256'] == hashes[run]
human = {run: {d: mean(float(r[run][d]) for r in reviews) for d in rubric.dimension_ids} for run in ids}
human_totals = [sum(human[run][d] for d in dims) for run in ids]
rank_totals = [0.0] * len(ids)
for review in reviews:
    for d in dims:
        for i, rank in enumerate(_ranks([float(review[run][d]) for run in ids])):
            rank_totals[i] += rank / (len(reviews) * len(dims))
sources = {
    'hy4-preview': out / 'results-full',
    'luna': root / 'eval/experiments/formal-100-judge-comparison-v1/results/baseline/luna/pass-001',
    'glm': root / 'eval/experiments/formal-100-judge-comparison-glm-5.3-flash-v1/results/baseline/glm/pass-001',
}
result = {'sample_count': len(ids), 'dimension_count': len(dims), 'human_mean_total': mean(human_totals), 'rubric_sha256': rubric.sha256, 'human_sources': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, 'models': {}}
details = []
for name, directory in sources.items():
    records = {}
    record_paths = {}
    for run in ids:
        p = directory / 'items' / run / 'hy3_judge.json'
        if name == 'hy4-preview':
            if run in cached:
                p = root / cached[run]['path']
                assert hashlib.sha256(p.read_bytes()).hexdigest() == cached[run]['sha256']
            else:
                p = out / 'results-openrouter' / 'items' / run / 'hy3_judge.json'
        if not p.exists():
            raise RuntimeError(f'{name} not complete: {run}')
        rec = json.loads(p.read_text())
        assert rec['trace_sha256'] == hashes[run] and rec['rubric']['sha256'] == rubric.sha256
        assert rec['status'] == 'completed'
        if name == 'hy4-preview':
            expected_model = 'hy4-preview' if run in cached else 'tencent/hy4-preview'
            assert rec['metadata']['evaluator_fingerprint']['model'] == expected_model
        records[run] = rec
        record_paths[run] = str(p.relative_to(root))
    if len(records) != len(ids):
        continue
    scores = {run: {s['dimension_id']: s['score'] for s in rec['dimension_scores']} for run, rec in records.items()}
    totals = [sum(scores[run][d] for d in dims) for run in ids]
    flat = [scores[run][d] for run in ids for d in dims]
    stats = {'mean_total': mean(totals), 'mean_bias': mean(totals)-mean(human_totals), 'total_variance_population': mean((t-mean(totals))**2 for t in totals), 'score_distribution': dict(sorted(Counter(flat).items())), 'max_score_count': sum(s == rubric.score_max for s in flat), 'max_score_rate': mean(s == rubric.score_max for s in flat), 'spearman_raw_human_total': spearman(totals, human_totals), 'spearman_rank_normalized_human': spearman(totals, rank_totals), 'mae_dimensions': mean(abs(scores[run][d]-human[run][d]) for run in ids for d in dims), 'mae_total': mean(abs(a-b) for a,b in zip(totals,human_totals)), 'dimensions': {}, 'fingerprint': records[ids[0]]['metadata']['evaluator_fingerprint']}
    for d in dims:
        v = [scores[run][d] for run in ids]
        h = [human[run][d] for run in ids]
        stats['dimensions'][d] = {'name': next(x.name for x in rubric.dimensions if x.dimension_id == d), 'mean': mean(v), 'human_mean': mean(h), 'distribution': dict(sorted(Counter(v).items())), 'spearman': spearman(v,h), 'mae': mean(abs(a-b) for a,b in zip(v,h))}
    result['models'][name] = stats
    stats.pop('fingerprint')
    stats['fingerprints'] = list({rec['metadata']['evaluator_fingerprint']['sha256']: rec['metadata']['evaluator_fingerprint'] for rec in records.values()}.values())
    stats['source_model_counts'] = dict(Counter(rec['metadata']['evaluator_fingerprint']['model'] for rec in records.values()))
    stats['reported_usage_completed_records'] = {
        key: sum(rec['metadata'].get('usage', {}).get(key, 0) for rec in records.values())
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')
    }
    for i,run in enumerate(ids):
        details.append({'model': name, 'api_model': records[run]['metadata']['evaluator_fingerprint']['model'], 'result_path': record_paths[run], 'result_sha256': hashlib.sha256((root / record_paths[run]).read_bytes()).hexdigest(), 'run_id': run, 'trace_sha256': hashes[run], 'total': totals[i], 'human_mean_total': human_totals[i], 'scores': scores[run], 'human_mean_scores': human[run]})
model_totals = {name: [row['total'] for row in details if row['model'] == name] for name in result['models']}
rng = random.Random(20260910)
bootstrap_rho = []
bootstrap_delta = []
for _ in range(2000):
    selection = [rng.randrange(len(ids)) for _ in ids]
    h = [human_totals[i] for i in selection]
    a = spearman([model_totals['hy4-preview'][i] for i in selection], h)
    b = spearman([model_totals['luna'][i] for i in selection], h)
    if a is not None:
        bootstrap_rho.append(a)
    if a is not None and b is not None:
        bootstrap_delta.append(a-b)

def percentile(values, probability):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered)-1)*probability
    lower = int(position)
    upper = min(lower+1, len(ordered)-1)
    return ordered[lower] + (ordered[upper]-ordered[lower])*(position-lower)

result['bootstrap'] = {
    'seed': 20260910, 'replicates': 2000, 'resampling_unit': 'script (paired across judges)',
    'hy4_spearman_95_percentile_interval': [percentile(bootstrap_rho, q) for q in (0.025, 0.975)],
    'hy4_minus_luna_spearman_95_percentile_interval': [percentile(bootstrap_delta, q) for q in (0.025, 0.975)],
    'valid_hy4_replicates': len(bootstrap_rho), 'valid_delta_replicates': len(bootstrap_delta),
}
(out / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
(out / 'paired_scores.json').write_text(json.dumps(details, ensure_ascii=False, indent=2)+'\n')
for name, stats in result['models'].items():
    print(name, {k:v for k,v in stats.items() if k not in ('dimensions','fingerprints')})

def fmt(value):
    return '不可定义（无方差）' if value is None else f'{value:.4f}'

lines = [
    '# Hy4-preview、Luna 与 GLM：50 份双人 Rubric 对照',
    '',
    '结论：Hy4-preview 的七维总分与人工评分的 Spearman 为 −0.0033，接近零；四个维度在 50 份样本中全部满分。本次结果尚不足以支持用 Hy4-preview 可靠地替代人工质量排序。相对 Luna 的差异区间跨零，不应表述为已经证明 Hy4-preview 在总体上显著更差。',
    '',
    '日期：2026-09-10。以两位人工的逐维算术平均为参照，计算七个主观维度总分的 Spearman（含并列秩）；不含确定性长度分。',
    '',
    f'人工七维总分平均：{result["human_mean_total"]:.2f} / 21。50 份样本、350 个主观维度评分。',
    '',
    '| 裁判 | 七维均分 / 21 | 相对人工差 | 满分维度比例 | 总分 Spearman | 逐维 MAE |',
    '| --- | ---: | ---: | ---: | ---: | ---: |',
]
for name, s in result['models'].items():
    lines.append(f'| {name} | {s["mean_total"]:.2f} | {s["mean_bias"]:+.2f} | {s["max_score_count"]}/350 ({s["max_score_rate"]:.1%}) | {fmt(s["spearman_raw_human_total"])} | {s["mae_dimensions"]:.4f} |')
hy4 = result['models']['hy4-preview']
interval = result['bootstrap']['hy4_spearman_95_percentile_interval']
delta = result['bootstrap']['hy4_minus_luna_spearman_95_percentile_interval']
lines += ['', f'按稿件配对重采样 2,000 次（seed=20260910）的百分位 95% 区间：Hy4 Spearman [{fmt(interval[0])}, {fmt(interval[1])}]；Hy4 − Luna 的相关系数差 [{fmt(delta[0])}, {fmt(delta[1])}]。该区间不能消除样本选择偏差或验证跨批次泛化。']
lines += ['', f'Hy4 完整记录的输出 token 合计：{hy4["reported_usage_completed_records"]["completion_tokens"]:,}（包括推理 token；不含中断但未落盘的请求）。']
lines += ['', '## Hy4 分维度结果', '', '| 维度 | 人工均分 | Hy4 均分 | 分数分布 | Spearman | MAE |', '| --- | ---: | ---: | --- | ---: | ---: |']
for d, s in hy4['dimensions'].items():
    lines.append(f'| {s["name"]} | {s["human_mean"]:.2f} | {s["mean"]:.2f} | {s["distribution"]} | {fmt(s["spearman"])} | {s["mae"]:.4f} |')
lines += [
    '', '## 实验约束与复核', '',
    '- 使用 Rubric v1.1、Judge v3.3.0、提示词 script-quality-grounded-groups-v3.3；reasoning_effort=high、temperature=0。原接口显式发送 top_p=1；OpenRouter 未声明支持 top_p，因此省略中性默认值，并要求路由支持发送的参数。',
    '- 原始 50 份冻结 trace、两位人工的 run_id、SHA-256 和文案正文均已核对。人工分数不发送给模型。',
    '- Luna、GLM 使用第一轮记录；不取重复评分平均。复算得到 Luna 0.236212、GLM 0.032764，和原报告一致。',
    '- 本次是评分组件对照，不重新生成文案，不发起搜索，也不重新运行完整双门控流程。',
    f'- 按用户明确授权复用旧通道已完成的 {len(cached)} 份 hy4-preview 评分，其余 {len(ids)-len(cached)} 份使用 OpenRouter 的 tencent/hy4-preview。两通道记录的原始 fingerprint 均保留；不将旧记录伪装为 OpenRouter 结果。这是混合通道续跑，不是独立的纯 OpenRouter 50 份实验。',
    '- 中断时尚未落盘的部分请求可能已产生计费，其用量不完整。缓存授权与文件哈希见 cache_reuse.json；原始启动与切换信息见 experiment.json。',
    '- 50 份均为高质量基线样本，范围受限；单次评分不能证明跨批次稳定性。观察到更低满分率也不等于与人工排序更一致。',
    '- OpenRouter 首轮 22/25 成功，1 份结构化输出校验失败、2 份接口请求失败；只对缺失的 3 份补试，成功记录直接复用。首轮记录保存在 openrouter-first-attempt。',
    '- 默认单元测试：300 项通过。新增独立 OpenRouter 裁判配置和原生 reasoning 参数适配；未修改默认生成模型或历史评测记录。',
    '', '## 复现', '',
    '从仓库根目录运行。评分命令会校验并复用已完成记录；已有结果时无需再次调用模型。',
    '', '```bash',
    'uv run --no-sync python scripts/run_evaluation.py score --trace-manifest eval/experiments/hy4-preview-human-50-v1/openrouter_manifest.json --evaluators judge --judge-provider openrouter --judge-model-id tencent/hy4-preview --reasoning-effort high --concurrency 12 --output-dir eval/experiments/hy4-preview-human-50-v1/results-openrouter',
    'uv run --no-sync python eval/experiments/hy4-preview-human-50-v1/analyze.py',
    '```', '',
    'comparison.json 保存统计与各通道 fingerprint；paired_scores.json 保存逐样本对照与原记录路径；results-full/items 与 results-openrouter/items 保存带评分依据的原始结果。',
]
(out / 'report.md').write_text('\n'.join(lines)+'\n')
