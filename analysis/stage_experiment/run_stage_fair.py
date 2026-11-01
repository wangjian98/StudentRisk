#!/usr/bin/env python3
"""
四项全上公平对照：MetaMamba vs RF（阶段级 + 去混淆 + 残差化）
==============================================================
目的：一次结掉"架构到底有没有价值"——给 MetaMamba 最公平的一次机会，
      如果它仍然输给同口径 RF，则阶段级任务上"新架构"这条线可以关闭。

四项的落地口径（两个 arm 完全对称）：
  ① 阶段级：数据 = stage_dataset.npz（2171 阶段样本 / 473 学生 / 258 掉队正例）
     标签 drop=1 ⇔ 该(生,part)是他到达过的最远 part 且未达 part7
  ② 去混淆：模型看不到"这是第几 part"
     · MetaMamba：删掉序列里的 part 通道（seq11 → seq11_np，保留 exercise 通道），
                  关闭 FiLM 的 part 调制（use_film=False），task_id 置 0
     · RF：特征里本就没有 part 身份（DER/SELF 绝对量 + 同 part 同伴 z 分位）
     · 附带：上下文特征用**同 part 同伴**统计量（peer_z），只给"相对同伴的位置"
  ③ 残差化（两层，公平起见两 arm 都做）
     · 特征层：剔除"活跃度方向"的线性成分（activity regressors = DER[0:4]），
                beta 只在 train 折上拟合 → X_resid = X − A·beta
     · 打分层（M1 两阶段残差）：score = p_model − w · p_prior_stage，
                p_prior_stage 为**折内**的同 part 掉队率先验；主口径 w=0.25
                （w 是 10-28 实验 D1 预先定下的最优值，不是事后挑的）
  ④ 架构：MetaMamba（S6/Mamba 骨干 + 可选 FiLM）vs RandomForest

主指标：part 内 pooled AUC（个体判别力口径）；配对 bootstrap 给 Δ 与 P(Δ≤0)。

用法：
  python3 run_stage_fair.py --data data/stage_dataset.npz --models rf_full rf_full_resid --tag cpu
  python3 run_stage_fair.py --data data/stage_dataset.npz --models mm11_np_ctx --tag mm11_np_ctx
  python3 run_stage_fair.py --data data/stage_dataset.npz --models mm11_np --tag mm11_np
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.stage_experiment import stage_common as sc
from models.base import set_seed
from models.meta_mamba.model import MetaMambaClassifier
from models.meta_mamba.train import train_one_fold

# seq11 通道：et_oh(0-6) | log_interval(7) | deadline_norm(8) | part_norm(9) | exercise_norm(10)
SEQ11_NO_PART = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]     # 去掉 part_norm → 10 维
ACT_REGRESSORS = [0, 1, 2, 3]                        # DER 里的活跃度方向
W_PRIOR_MAIN = 0.25                                 # M1 主口径（预先设定）


# ---------------------------------------------------------------- 特征
def peer_z_all(ds, rows, group):
    """用同 part **全部行**（无标签）算同伴统计量。

    留一 part（LOPO）时被留出的 part 在训练集里没有任何行，
    若只用训练行算同伴基线，该 part 的特征会全零 → 评估失真。
    这里改用该 part 自己的全部行算 median/MAD（纯无标签变换，不构成泄漏），
    两个 arm 口径一致。
    """
    RAW = ds['RAW']
    out = np.zeros((len(rows), RAW.shape[1]))
    for g in np.unique(group[rows]):
        m = group[rows] == g
        idx = np.where(group == g)[0]
        Xg = RAW[idx]
        med = np.median(Xg, 0)
        mad = np.maximum(np.median(np.abs(Xg - med), 0), 1e-3)
        out[m] = np.clip((RAW[rows][m] - med) / mad, -5, 5)
    return out


def build_context(ds, tr, te, residualize, lopo=False):
    """RF 与 MetaMamba 共用的上下文特征（rf_full 同源，29 维）。

    返回 (ctx_all, Xtr, Xte)：
      ctx_all 是整行对齐的 (N, D) 矩阵（只有 tr/te 行有值），
      方便 MetaMamba 按任意行子集广播上下文通道。
    """
    DER, SELF, part = ds['DER'], ds['SELF'], ds['part']
    if lopo:
        ztr = peer_z_all(ds, tr, part)
        zte = peer_z_all(ds, te, part)
    else:
        ztr = sc.peer_z(ds, tr, tr, part)
        zte = sc.peer_z(ds, tr, te, part)
    A, B = DER[:, [0, 1, 2, 3]], DER[:, [4, 5, 6]]
    C = np.column_stack([DER[:, 7], DER[:, 8]])
    Xtr = np.hstack([A[tr], B[tr], C[tr], ztr, SELF[tr]])
    Xte = np.hstack([A[te], B[te], C[te], zte, SELF[te]])
    if residualize:
        Atr = np.hstack([np.ones((len(tr), 1)), DER[tr][:, ACT_REGRESSORS]])
        Ate = np.hstack([np.ones((len(te), 1)), DER[te][:, ACT_REGRESSORS]])
        beta, *_ = np.linalg.lstsq(Atr, Xtr, rcond=None)
        Xtr = Xtr - Atr @ beta
        Xte = Xte - Ate @ beta
    Xtr = Xtr.astype(np.float32)
    Xte = Xte.astype(np.float32)
    ctx_all = np.zeros((len(ds['drop']), Xtr.shape[1]), dtype=np.float32)
    ctx_all[tr] = Xtr
    ctx_all[te] = Xte
    return ctx_all, Xtr, Xte


def build_seq(ds, variant, rows, ctx=None):
    """MetaMamba 输入序列：按变体选通道，可选把上下文广播成额外通道。"""
    if variant in ('mm11_np', 'mm11_np_ctx', 'mm11_np_ctxraw'):
        seq = ds['seq11'][:, :, SEQ11_NO_PART]
    elif variant == 'mm11':
        seq = ds['seq11']
    else:
        raise ValueError(variant)
    seq = seq[rows]
    if variant.endswith('_ctx') or variant.endswith('_ctxraw'):
        c = ctx[rows][:, None, :]
        seq = np.concatenate([seq, np.repeat(c, seq.shape[1], axis=1)], axis=2)
    return seq.astype(np.float32)


# ---------------------------------------------------------------- RF arm
def run_rf(ds, kind, folds, seeds):
    from sklearn.ensemble import RandomForestClassifier
    resid = kind.endswith('_resid')
    base = kind[:-len('_resid')] if resid else kind
    assert base == 'rf_full', kind
    n = len(ds['drop'])
    oof = np.zeros(n)
    for tr, te in folds:
        _, Xtr, Xte = build_context(ds, tr, te, residualize=resid)
        acc = np.zeros(len(te))
        for sd in seeds:
            m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                       max_features='sqrt', class_weight='balanced',
                                       random_state=sd, n_jobs=-1)
            m.fit(Xtr, ds['drop'][tr])
            acc += m.predict_proba(Xte)[:, 1]
        oof[te] = acc / len(seeds)
    return oof


# ---------------------------------------------------------------- MetaMamba arm
def inner_split(ds, tr, frac=0.15, seed=42):
    """从 train 折里按**学生**切一个内层验证集（只用于早停）。

    为什么要切：models/meta_mamba/train.py 的 train_one_fold 用传入的验证集做早停，
    而旧协议直接把评估折当验证集 → 评估折参与了模型选择（轻度乐观）。
    本脚本改为"内层验证集早停 + 评估折只做一次前向"，口径更严。
    """
    sid, y = ds['sid'], ds['drop']
    rng = np.random.default_rng(seed)
    students = np.unique(sid[tr])
    has_pos = {int(s): bool(y[sid == s].any()) for s in students}
    val_students = []
    for cls in (True, False):
        cand = [int(s) for s in students if has_pos[int(s)] == cls]
        if not cand:
            continue
        rng.shuffle(cand)
        rows_cls = int(sum((sid[tr] == s).sum() for s in cand))
        target_cls = frac * rows_cls
        acc_rows = 0
        for s in cand:
            if acc_rows >= target_cls:
                break
            val_students.append(s)
            acc_rows += int((sid[tr] == s).sum())
    val_set = set(val_students)
    m_val = np.array([int(sid[i]) in val_set for i in tr])
    return tr[~m_val], tr[m_val]


def run_metamamba(ds, variant, folds, seeds, args, device):
    n = len(ds['drop'])
    oof = np.zeros(n)
    fold_times = []
    use_ctx = variant.endswith('_ctx') or variant.endswith('_ctxraw')
    ctx_resid = variant.endswith('_ctx')      # _ctx = 上下文做活跃度残差化；_ctxraw = 不做
    for fi, (tr, te) in enumerate(folds):
        t0 = time.time()
        if use_ctx:
            ctx_all, _, _ = build_context(ds, tr, te, residualize=ctx_resid)
        else:
            ctx_all = None
        fit_idx, val_idx = inner_split(ds, tr, frac=args.val_frac, seed=42 + fi)
        seq_fit = build_seq(ds, variant, fit_idx, ctx_all)
        seq_val = build_seq(ds, variant, val_idx, ctx_all)
        seq_te = build_seq(ds, variant, te, ctx_all)
        d_event = seq_fit.shape[2]
        print(f'    [{variant}] fold{fi}: fit={len(fit_idx)} val={len(val_idx)} '
              f'te={len(te)} d_event={d_event}', flush=True)
        acc = np.zeros(len(te))
        for sd in seeds:
            set_seed(sd)
            model = MetaMambaClassifier(
                d_event=d_event, d_model=args.d_model, d_state=16, n_layers=2,
                n_tasks=7, dropout=args.dropout)
            model, _ = train_one_fold(
                model, seq_fit, ds['mask'][fit_idx], np.zeros(len(fit_idx), dtype=np.int64),
                ds['drop'][fit_idx],
                seq_val, ds['mask'][val_idx], np.zeros(len(val_idx), dtype=np.int64),
                ds['drop'][val_idx],
                epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                batch_size=args.batch_size, patience=args.patience,
                contrastive_weight=0.0, use_film=False, device=device)
            model.eval()
            with torch.no_grad():
                p = model(torch.from_numpy(seq_te).float().to(device),
                          torch.from_numpy(ds['mask'][te]).float().to(device),
                          torch.from_numpy(np.zeros(len(te), dtype=np.int64)).long().to(device))
            acc += torch.sigmoid(p).cpu().numpy()
        oof[te] = acc / len(seeds)
        dt = time.time() - t0
        fold_times.append(round(dt, 1))
        print(f'    [{variant}] fold{fi} done {dt:.0f}s', flush=True)
    return oof, fold_times


# ---------------------------------------------------------------- LOPO（跨 part 泛化）
def lopo_parts(ds):
    """可做留一评估的 part（正负都有）"""
    y, part = ds['drop'], ds['part']
    out = []
    for p in np.unique(part):
        m = part == p
        if y[m].sum() > 0 and (1 - y[m]).sum() > 0:
            out.append(int(p))
    return out


def run_lopo_rf(ds, kind, parts, seeds, args):
    from sklearn.ensemble import RandomForestClassifier
    resid = kind.endswith('_resid')
    n = len(ds['drop'])
    oof = np.zeros(n)
    per_part = {}
    for p in parts:
        te = np.where(ds['part'] == p)[0]
        tr = np.where(ds['part'] != p)[0]
        _, Xtr, Xte = build_context(ds, tr, te, residualize=resid, lopo=True)
        acc = np.zeros(len(te))
        for sd in seeds:
            m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                       max_features='sqrt', class_weight='balanced',
                                       random_state=sd, n_jobs=-1)
            m.fit(Xtr, ds['drop'][tr])
            acc += m.predict_proba(Xte)[:, 1]
        oof[te] = acc / len(seeds)
        y_te = ds['drop'][te]
        auc = float(roc_auc_score(y_te, oof[te])) if len(set(y_te.tolist())) > 1 else float('nan')
        per_part[p] = dict(n=int(len(te)), pos=int(y_te.sum()), auc=auc)
        print(f'    [lopo rf] part{p} held-out: n={len(te)} pos={int(y_te.sum())} AUC={auc:.4f}',
              flush=True)
    return oof, per_part


def run_lopo_mm(ds, variant, parts, seeds, args, device):
    n = len(ds['drop'])
    oof = np.zeros(n)
    per_part = {}
    use_ctx = variant.endswith('_ctx') or variant.endswith('_ctxraw')
    ctx_resid = variant.endswith('_ctx')
    for p in parts:
        te = np.where(ds['part'] == p)[0]
        tr = np.where(ds['part'] != p)[0]
        if use_ctx:
            ctx_all, _, _ = build_context(ds, tr, te, residualize=ctx_resid, lopo=True)
        else:
            ctx_all = None
        fit_idx, val_idx = inner_split(ds, tr, frac=args.val_frac, seed=42 + p)
        seq_fit = build_seq(ds, variant, fit_idx, ctx_all)
        seq_val = build_seq(ds, variant, val_idx, ctx_all)
        seq_te = build_seq(ds, variant, te, ctx_all)
        d_event = seq_fit.shape[2]
        acc = np.zeros(len(te))
        t0 = time.time()
        for sd in seeds:
            set_seed(sd)
            model = MetaMambaClassifier(
                d_event=d_event, d_model=args.d_model, d_state=16, n_layers=2,
                n_tasks=7, dropout=args.dropout)
            model, _ = train_one_fold(
                model, seq_fit, ds['mask'][fit_idx], np.zeros(len(fit_idx), dtype=np.int64),
                ds['drop'][fit_idx],
                seq_val, ds['mask'][val_idx], np.zeros(len(val_idx), dtype=np.int64),
                ds['drop'][val_idx],
                epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                batch_size=args.batch_size, patience=args.patience,
                contrastive_weight=0.0, use_film=False, device=device)
            model.eval()
            with torch.no_grad():
                pr = model(torch.from_numpy(seq_te).float().to(device),
                           torch.from_numpy(ds['mask'][te]).float().to(device),
                           torch.from_numpy(np.zeros(len(te), dtype=np.int64)).long().to(device))
            acc += torch.sigmoid(pr).cpu().numpy()
        oof[te] = acc / len(seeds)
        y_te = ds['drop'][te]
        auc = float(roc_auc_score(y_te, oof[te])) if len(set(y_te.tolist())) > 1 else float('nan')
        per_part[p] = dict(n=int(len(te)), pos=int(y_te.sum()), auc=auc)
        print(f'    [lopo {variant}] part{p} held-out: n={len(te)} pos={int(y_te.sum())} '
              f'AUC={auc:.4f} ({time.time()-t0:.0f}s)', flush=True)
    return oof, per_part


def weighted_auc(per_part):
    a = np.array([v['auc'] for v in per_part.values()], dtype=float)
    w = np.array([v['n'] for v in per_part.values()], dtype=float)
    ok = ~np.isnan(a)
    if not ok.any():
        return float('nan')
    return float(np.average(a[ok], weights=w[ok]))


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default='results/stage_experiment')
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 777])
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-3)
    ap.add_argument('--d-model', type=int, default=64)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--device', default=None)
    ap.add_argument('--val-frac', type=float, default=0.15,
                    help='内层验证集比例（早停用，从 train 折按学生切）')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--smoke-epochs', type=int, default=0,
                    help='>0 时 smoke 模式用这个 epoch 数（测速用）')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--lopo', action='store_true',
                    help='留一 part（跨 part 泛化）协议')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device} torch={torch.__version__} cuda={torch.cuda.is_available()} '
          f'tag={args.tag}', flush=True)

    ds = sc.load_dataset(args.data)
    ds['base_ex'] = sc.base_exercises(ds['agg10'], ds['part'])
    ds['DER'] = sc.derive(ds)
    ds['SELF'] = sc.self_ref(ds)
    y, part = ds['drop'], ds['part']

    folds = sc.group_folds(ds['sid'], ds['final'], n_splits=args.folds, seed=42)
    print(f'阶段样本 {len(y)} / 学生 {len(np.unique(ds["sid"]))} / 掉队正例 {int(y.sum())} '
          f'/ 折大小 {[len(te) for _, te in folds]}', flush=True)

    # ---------------- LOPO：跨 part 泛化 ----------------
    if args.lopo:
        parts = lopo_parts(ds)
        if args.smoke:
            parts = parts[:1]
            args.seeds = args.seeds[:1]
            args.epochs = args.smoke_epochs or 3
        print(f'LOPO 协议：留一 part {parts}（train = 其余 part，test = 留出 part）', flush=True)
        res = {'meta': dict(device=device, torch=torch.__version__, seeds=args.seeds,
                            epochs=args.epochs, d_model=args.d_model, tag=args.tag,
                            protocol='lopo', held_out_parts=parts,
                            n_stage=int(len(y))), 'models': {}}
        for name in args.models:
            t0 = time.time()
            print(f'\n  ==== LOPO {name} ====', flush=True)
            if name.startswith('rf'):
                oof, pp = run_lopo_rf(ds, name, parts, args.seeds, args)
            else:
                oof, pp = run_lopo_mm(ds, name, parts, args.seeds, args, device)
            entry = {'per_part': {str(k): v for k, v in pp.items()},
                     'weighted_auc': weighted_auc(pp),
                     'part4_auc': pp.get(4, {}).get('auc'),
                     'seconds': round(time.time() - t0, 1)}
            res['models'][name] = entry
            print(f'  LOPO {name}: 加权 AUC={entry["weighted_auc"]:.4f} | '
                  f'part4={entry["part4_auc"]} | {entry["seconds"]}s', flush=True)
            np.savez_compressed(os.path.join(args.out, f'stage_lopo_oof_{args.tag}.npz'),
                                **{name: oof})
        with open(os.path.join(args.out, f'stage_lopo_{args.tag}.json'), 'w') as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(f'\n写出 {args.out}/stage_lopo_{args.tag}.json', flush=True)
        return

    if args.smoke:
        folds = folds[:1]
        args.seeds = args.seeds[:1]
        if args.smoke_epochs:
            args.epochs = args.smoke_epochs

    prior = sc.stage_prior_oof(y, part, folds)
    res = {'meta': dict(device=device, torch=torch.__version__, seeds=args.seeds,
                        folds=args.folds, epochs=args.epochs, d_model=args.d_model,
                        tag=args.tag, w_prior_main=W_PRIOR_MAIN,
                        n_stage=int(len(y)), n_students=int(len(np.unique(ds['sid'])))),
           'models': {}}
    m = sc.metrics(y, prior, part, 'stage_part_prior')
    print(f'  stage_part_prior 总体={m["auc"]:.4f} | part内pooled={m["pooled"]:.4f}', flush=True)
    res['models']['stage_part_prior'] = m

    oofs = {'stage_part_prior': prior}
    for name in args.models:
        t0 = time.time()
        print(f'\n  ==== {name} ====', flush=True)
        if name.startswith('rf'):
            oof = run_rf(ds, name, folds, args.seeds)
            ft = []
        else:
            oof, ft = run_metamamba(ds, name, folds, args.seeds, args, device)
        oofs[name] = oof

        # 打分层残差化（M1）：p − w·p_prior，w 主口径 0.25，附带扫描
        ws = [0.0, 0.25, 0.5, 1.0]
        for w in ws:
            key = name if w == 0.0 else f'{name}_resid{w}'
            mm = sc.metrics(y, np.clip(oof - w * prior, 1e-6, 1 - 1e-6), part, key)
            mm['seconds'] = round(time.time() - t0, 1)
            res['models'][key] = mm
            oofs[key] = np.clip(oof - w * prior, 1e-6, 1 - 1e-6)
            print(f'  {key:26s} 总体={mm["auc"]:.4f} | part内pooled={mm["pooled"]:.4f} | '
                  f'judgeable={mm["judged"]:.4f}', flush=True)
        for k, v in sorted(res['models'][name]['per_part'].items(), key=lambda x: int(x[0])):
            print(f'      part{k}: n={v["n"]:4d} pos={v["pos"]:4d} AUC={v["auc"]:.4f}', flush=True)

    # 与 part-prior 的配对 bootstrap（w=0.25 主口径）
    boots = {}
    for name in args.models:
        for key in ([name] if name.startswith('rf') else [name, f'{name}_resid0.25']):
            if key in oofs:
                d = sc.paired_boot(y, oofs[key], oofs['stage_part_prior'], part)
                boots[f'{key}_minus_part_prior'] = d
                print(f'  [bootstrap] {key} − part_prior: Δpooled={d["obs"]:+.4f} '
                      f'[{d["lo"]:+.4f},{d["hi"]:+.4f}] P(≤0)={d["p_le0"]*100:.2f}%', flush=True)
    res['bootstrap'] = boots

    with open(os.path.join(args.out, f'stage_fair_{args.tag}.json'), 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    np.savez_compressed(os.path.join(args.out, f'stage_fair_oof_{args.tag}.npz'), **oofs)
    print(f'\n写出 {args.out}/stage_fair_{args.tag}.json', flush=True)


if __name__ == '__main__':
    main()
