# -*- coding: utf-8 -*-
"""评估基线诊断：量化"标注稀疏"对指标的影响。

背景：OpenTTGames 只标注事件前后少量帧的球坐标（训练集约 10%，test_1 仅 5.4%）。
      把未标注帧当作"无球"来算精确率，会系统性低估模型性能。

本脚本对比两种统计口径：
  A. 全帧口径：所有帧都参与，未标注帧视为"无球"（大多数论文/脚本的默认做法，但在此数据集上是错的）
  B. 标注区间口径：只在有事件标注的时间跨度内统计（更接近真实使用场景）

只用已有推理缓存 results/infer_<game>_full.json，不需要 GPU。

用法：
    python -m referee.eval_baseline            # 全部测试视频
    python -m referee.eval_baseline test_2     # 指定视频
"""

import argparse
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FPS = 120.0


def load_ball_gt(game, split='test'):
    """读取球坐标真值。返回 {frame_id: (x, y)}，坐标基准 1920x1080。"""
    p = os.path.join(ROOT, 'dataset', split, 'annotations', game, 'ball_markup.json')
    if not os.path.isfile(p):
        return None
    with open(p, encoding='utf-8') as f:
        d = json.load(f)
    out = {}
    for k, v in d.items():
        x, y = v.get('x', -1), v.get('y', -1)
        if x == -1 or y == -1:
            continue          # 标注为"无球"
        out[int(k)] = (float(x), float(y))
    return out


def load_event_span(game, split='test'):
    """事件标注的时间跨度（帧号区间）。这是"有效标注范围"。"""
    p = os.path.join(ROOT, 'dataset', split, 'annotations', game, 'events_markup.json')
    if not os.path.isfile(p):
        return None
    with open(p, encoding='utf-8') as f:
        d = json.load(f)
    if not d:
        return None
    ks = sorted(int(k) for k in d)
    return ks[0], ks[-1]


def load_cache(game, tag=''):
    suffix = '_pretrained' if tag == 'pretrained' else ''
    p = os.path.join(ROOT, 'results', 'infer_{}_full{}.json'.format(game, suffix))
    if not os.path.isfile(p):
        return None
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def analyse(game, cache, ball_gt, span, vis_thresh=0.05):
    """返回该视频的诊断数据。"""
    frames = cache['frames']                       # [[frame_id, ts, visible], ...]
    n_frames = len(frames)

    # 模型判为"可见"的帧
    pred_vis = {f[0] for f in frames if f[2] >= vis_thresh}
    gt_frames = set(ball_gt.keys())

    # --- 口径 A：全帧 ---
    n_all = n_frames
    a_vis = len(pred_vis)
    # 全帧口径下，"真值有球"= 标注帧；模型在这些帧上可见才算 TP
    a_tp = len(pred_vis & gt_frames)
    a_fp = a_vis - a_tp                            # 模型说有球、但该帧没标注 → 记为误报
    a_fn = len(gt_frames) - a_tp

    # --- 口径 B：仅标注区间（事件跨度） ---
    if span:
        lo, hi = span
        in_span = {f[0] for f in frames if lo <= f[0] <= hi}
        gt_in = {g for g in gt_frames if lo <= g <= hi}
    else:
        in_span, gt_in = set(pred_vis) | gt_frames, gt_frames

    b_vis = len(pred_vis & in_span)
    b_tp = len(pred_vis & gt_in)
    b_fp = b_vis - b_tp
    b_fn = len(gt_in) - b_tp

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f1

    a_p, a_r, a_f1 = prf(a_tp, a_fp, a_fn)
    b_p, b_r, b_f1 = prf(b_tp, b_fp, b_fn)

    return {
        'game': game,
        'n_frames': n_all,
        'n_gt_frames': len(gt_frames),
        'gt_density': len(gt_frames) / max(n_all, 1),
        'n_pred_vis': a_vis,
        'vis_rate': a_vis / max(n_all, 1),
        'span': span,
        'full': {'tp': a_tp, 'fp': a_fp, 'fn': a_fn,
                 'precision': a_p, 'recall': a_r, 'f1': a_f1},
        'span_only': {'tp': b_tp, 'fp': b_fp, 'fn': b_fn,
                      'precision': b_p, 'recall': b_r, 'f1': b_f1},
        # 模型在"有真值的帧"上实际检出多少 —— 这才是模型的真实召回
        'true_recall_on_gt': a_tp / max(len(gt_frames), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('games', nargs='*',
                    default=['test_1', 'test_2', 'test_3', 'test_4', 'test_5', 'test_6', 'test_7'])
    ap.add_argument('--tag', default='', help='pretrained 表示用预训练权重缓存')
    ap.add_argument('--json-out', default='')
    a = ap.parse_args()

    rows = []
    print('{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
        '视频', '标帧数', '标密度', '可见率', '全帧F1', '区间F1', 'GT帧召回'))
    print('-' * 78)

    for g in a.games:
        cache = load_cache(g, a.tag)
        gt = load_ball_gt(g)
        span = load_event_span(g)
        if cache is None:
            print('{:<10} 缺推理缓存'.format(g))
            continue
        if gt is None:
            print('{:<10} 缺球标注'.format(g))
            continue
        r = analyse(g, cache, gt, span)
        rows.append(r)
        print('{:<10}\t{:6d}\t{:5.1f}%\t{:5.1f}%\t{:6.1f}%\t{:6.1f}%\t{:6.1f}%'.format(
            g, r['n_gt_frames'], r['gt_density'] * 100, r['vis_rate'] * 100,
            r['full']['f1'] * 100, r['span_only']['f1'] * 100,
            r['true_recall_on_gt'] * 100))

    if not rows:
        return

    print('-' * 78)
    n = len(rows)
    avg = lambda k: sum(r[k] for r in rows) / n
    print('{:<10}\t{:6.0f}\t{:5.1f}%\t{:5.1f}%\t{:6.1f}%\t{:6.1f}%\t{:6.1f}%'.format(
        '均值', avg('n_gt_frames'), avg('gt_density') * 100, avg('vis_rate') * 100,
        sum(r['full']['f1'] for r in rows) / n * 100,
        sum(r['span_only']['f1'] for r in rows) / n * 100,
        avg('true_recall_on_gt') * 100))

    print()
    print('解读：')
    print('  标密度   = 有球坐标标注的帧 / 总帧数。低密度会让"全帧F1"失去意义')
    print('  区间F1   = 只在事件标注跨度内统计，未标注帧不参与')
    print('  GT帧召回 = 在有真值的帧中，模型成功检出的比例（不含误报，纯能力指标）')

    if a.json_out:
        with open(a.json_out, 'w', encoding='utf-8') as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        print('\n明细已写入: {}'.format(a.json_out))


if __name__ == '__main__':
    main()
