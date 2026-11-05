#!/usr/bin/env python3
"""把 results/cs1_fs/cs1_fs.json 渲染成提案文档的 §3 结果章节（Markdown）。"""
import json
import sys

SHOTS = [1, 2, 4, 8, 16]
METHODS = ['proto', 'nn', 'logit', 'tree2', 'tree3', 'rf', 'ls', 'ls_rf', 'proto_rf']
LABEL = {'proto': '双原型(线性方向)', 'nn': '1-NN', 'logit': '逻辑回归(线性)',
         'tree2': '浅树 d2', 'tree3': '浅树 d3', 'rf': '**RF 直接训练**',
         'ls': 'kNN 图传播', 'ls_rf': '图传播+RF(自训练)', 'proto_rf': '原型+RF(自训练)',
         'metric_proto': '度量学习原型网'}


def main(path):
    D = json.load(open(path, encoding='utf-8'))
    L = []
    ref = D['reference']
    L += ['### 3.1 参照系（同一批数据、同一特征空间）', '',
          '| 模型 | pooled AUC | 说明 |', '|---|---:|---|']
    for k, v in ref.items():
        note = ''
        if 'global' in k:
            note = '主协议 5 折按学生分组（管线 sanity check，与论文 0.9258 一致）'
        elif 'inpart' in k:
            note = '**该段标签充足时的上限**'
        elif 'crosspart' in k:
            note = 'train 其它段全部标签 → test 目标段（跨段可迁移性）'
        L.append(f'| {k} | {v["pooled"]} | {note} |')
    L += ['', '逐段 AUC：', '']
    for k, v in ref.items():
        L.append(f'- **{k}**：`' + ' '.join(f'p{a}={b}' for a, b in v['per_part'].items()) + '`')

    L += ['', '### 3.2 FS-A 段内冷启动学习曲线（K 正 + K 负标注，12 次重复）', '']
    for fs in ['all29', 'peer10', 'peer_submit']:
        desc = {'all29': '全特征（A4+B3+C2 + 同伴相对10 + 自参照10 = 29 维）',
                'peer10': '只留同伴相对 10 维',
                'peer_submit': '**只留 peer_z_submit 一个维度**（全项目 7 个段里的第 1 重要特征）'}[fs]
        L += [f'**特征集 {fs}** —— {desc}', '',
              '| 方法 | ' + ' | '.join(f'K={k}' for k in SHOTS) + ' |',
              '|---|' + '---:|' * len(SHOTS)]
        for m in METHODS:
            vals = []
            for K in SHOTS:
                r = D['fs_a_inpart_coldstart'].get(f'{fs}|{m}|K={K}')
                vals.append(str(r['pooled']) if r else '—')
            L.append(f'| {LABEL[m]} | ' + ' | '.join(vals) + ' |')
        L.append('')

    L += ['**每个 K 下的最优配置**', '',
          '| K | 最优方法（特征集） | pooled AUC | 相当于全标签上限的 |', '|---:|---|---:|---:|']
    ceiling = ref.get('REF-inpart-full (段内全标签 RF, 少样本同空间)', {}).get('pooled')
    for K in SHOTS:
        best = None
        for fs in ['all29', 'peer10', 'peer_submit']:
            for m in METHODS:
                r = D['fs_a_inpart_coldstart'].get(f'{fs}|{m}|K={K}')
                if r and (best is None or r['pooled'] > best[1]['pooled']):
                    best = (f'{m}（{fs}）', r)
        if best:
            pct = f'{best[1]["pooled"]/ceiling*100:.1f}%' if ceiling else '—'
            L.append(f'| {K} | {LABEL[best[0].split("（")[0]]} {best[0]} | '
                     f'{best[1]["pooled"]} | {pct} |')
    L.append('')

    if 'fs_b_weak_parts' in D:
        L += ['### 3.3 FS-B 弱段救援（part5 / part6：正例 15 / 22，全项目最弱两段）', '',
              '| 方法 | ' + ' | '.join(f'K={k}' for k in [1, 2, 4, 8]) + ' |',
              '|---|' + '---:|' * 4]
        for m in METHODS:
            rec = D['fs_b_weak_parts'].get(m, {})
            vals = []
            for K in ['1', '2', '4', '8']:
                r = rec.get(K, {})
                vals.append(f'{r.get("pooled_all")} / {r.get("weak_pooled")}')
            L.append(f'| {LABEL[m]} | ' + ' | '.join(vals) + ' |')
        L += ['', '（每格 = 全段 pooled / 弱段 pooled）', '']

    if 'fs_c_crosspart_transfer' in D:
        L += ['### 3.4 FS-C 跨段原型迁移（支撑集只来自其它段）', '',
              '| K | pooled | 逐段 |', '|---:|---:|---|']
        for K, v in D['fs_c_crosspart_transfer'].items():
            L.append(f'| {K} | {v["pooled"]} | ' + ' '.join(
                f'p{a}={b}' for a, b in v['per_part'].items()) + ' |')
        L.append('')

    if 'm5_metric_proto_net' in D:
        L += ['### 3.5 度量学习原型网（MLP 编码器 + 原型损失，训练段 1–6）—— 对照', '',
              '| K | pooled |', '|---:|---:|']
        for K, v in D['m5_metric_proto_net'].items():
            L.append(f'| {K} | {v["pooled"]} |')
        L += ['', '（对照意义：给深度骨干一个公平机会——若它也打不过 RF/线性，'
              '说明 CS1 上的差距来自**样本量**而不是**架构**）', '']

    if 'fs_d_sitting_early' in D:
        L += ['### 3.6 FS-D 坐下级（段内更早）少样本', '']
        for k, v in D['fs_d_sitting_early'].items():
            L += [f'**{k}**（rows={v["rows"]}，同特征全标签 RF pooled = '
                  f'{v["ref_full_RF_pooled"]}）', '',
                  '| 方法 | K=2 | K=4 | K=8 |', '|---|---:|---:|---:|']
            for m in METHODS:
                L.append(f'| {LABEL[m]} | {v.get(f"{m}|K=2", "—")} | '
                         f'{v.get(f"{m}|K=4", "—")} | {v.get(f"{m}|K=8", "—")} |')
            L.append('')

    if 'fs_e_rare_class' in D:
        rec = D['fs_e_rare_class']
        L += ['### 3.7 FS-E 稀有子类：走完全程仍挂科（产品上另一类预警）', '',
              f'样本：到达 part7 的学生 n={rec["n"]}，其中最终挂科 {rec["pos"]} 人；'
              f'全标签 RF 参照 AUC = **{rec["ref_full_RF_auc"]}**', '',
              '| 方法 | ' + ' | '.join(f'K={k}' for k in [1, 2, 4, 8, 16]) + ' |',
              '|---|' + '---:|' * 5]
        for m in METHODS:
            vals = []
            for K in [1, 2, 4, 8, 16]:
                r = rec.get(f'{m}|K={K}')
                vals.append(str(r['auc_mean']) if r else '—')
            L.append(f'| {LABEL[m]} | ' + ' | '.join(vals) + ' |')
        L.append('')

    if 'significance' in D:
        L += ['### 3.8 显著性（配对 bootstrap 2000 次，part 内重采样）', '',
              '| 对比 | Δ pooled | 95% CI | P(Δ≤0) |', '|---|---:|---|---:|']
        for k, v in D['significance'].items():
            L.append(f'| {k} | {v["obs"]} | [{v["ci"][0]}, {v["ci"][1]}] | {v["p_le0"]} |')
        L.append('')
    print('\n'.join(L))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results/cs1_fs/cs1_fs.json')
