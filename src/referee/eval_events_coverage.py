# -*- coding: utf-8 -*-
"""在"实际标注覆盖"而非"首末事件跨度"上评估事件检测。

问题背景（实测发现）：
    OpenTTGames 的事件标注不是连续覆盖的。以 test_1 为例，
    bounce 标注从 1.6s 到 137.9s，但中间存在一段 109 秒的空档
    （19.8s -> 128.3s）。这段里模型预测了 69 个 bounce，
    全部被计为 FP —— 但一场乒乓球赛不可能 109 秒没有落台，
    这些几乎都是真实但未标注的事件。

    原评估脚本（evaluate_events.py / tune_events.py）只按"首末事件跨度"
    裁剪，无法排除内部空档，导致：
      - test_1 精确率被压到 29%
      - 在 test_2 上调出的阈值到 test_1 上看似"完全失效"
    实际上不是阈值问题，是评估口径问题。

本脚本的做法：
    用"标注块"定义有效评估区间。把真值事件按时间聚成块，
    块内视为已标注、块与块之间的大空档视为未标注并排除。

用法：
    python -m referee.eval_events_coverage                      # 全部视频
    python -m referee.eval_events_coverage test_1 --gap 10      # 指定视频和空档阈值
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from referee.tune_events import load_gt, detect, match

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def load_cache(game):
    p = os.path.join(ROOT, 'results', 'cache_{}.json'.format(game))
    if not os.path.isfile(p):
        return None
    with open(p, encoding='utf-8') as f:
        d = json.load(f)
    return d.get('rows', d) if isinstance(d, dict) else d


def coverage_spans(gt, gap_s):
    """把真值事件聚成标注块，返回若干 (start, end, label)。

    两个相邻真值事件间隔超过 gap_s，就认为中间是未标注区域，
    在该处切开。返回的每个区间都是"确定被标注过"的。
    """
    marks = []
    for kind in ('bounce', 'net'):
        for t in gt[kind]:
            marks.append((t, kind))
    if not marks:
        return []
    marks.sort()
    blocks = []
    cur_start, cur_end = marks[0][0], marks[0][0]
    cur_kinds = {marks[0][1]}
    for t, kind in marks[1:]:
        if t - cur_end > gap_s:
            blocks.append((cur_start, cur_end, cur_kinds))
            cur_start, cur_kinds = t, {kind}
        else:
            cur_kinds.add(kind)
        cur_end = t
    blocks.append((cur_start, cur_end, cur_kinds))
    return blocks


def evaluate_in_blocks(rows, gt, high, hyst, interval, mask, blocks, tol=0.1, pad=1.0):
    """只在标注块内统计，块外预测直接忽略（不判对也不判错）。"""
    tp_total = fp_total = fn_total = 0
    per_kind = {}
    for start, end, kinds in blocks:
        lo, hi = start - pad, end + pad
        sub = [r for r in rows if lo <= r['t'] <= hi]
        for key, gtkey in (('b', 'bounce'), ('n', 'net')):
            if gtkey not in kinds:
                continue
            truth = [t for t in gt[gtkey] if lo <= t <= hi]
            pred = detect(sub, key, high, hyst, interval, mask)
            tp, fp, fn, _ = match(pred, truth, tol)
            d = per_kind.setdefault(gtkey, {'tp': 0, 'fp': 0, 'fn': 0})
            d['tp'] += tp
            d['fp'] += fp
            d['fn'] += fn
    return per_kind


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('games', nargs='*',
                    default=['test_1', 'test_2', 'test_3', 'test_4', 'test_5', 'test_6', 'test_7'])
    ap.add_argument('--gap', type=float, default=10.0,
                    help='大于该秒数的真值间隔视为未标注空档（默认 10s）')
    ap.add_argument('--bounce-thresh', type=float, default=0.65)
    ap.add_argument('--net-thresh', type=float, default=0.5)
    ap.add_argument('--hyst', type=float, default=0.1)
    ap.add_argument('--interval', type=float, default=0.10)
    ap.add_argument('--tol', type=float, default=0.1)
    ap.add_argument('--json-out', default='')
    a = ap.parse_args()

    print('评估口径：只在"标注块"内统计，块间大空档视为未标注并排除')
    print('空档阈值: >{:.0f}s | 参数: bounce={} net={} hyst={} interval={}'.format(
        a.gap, a.bounce_thresh, a.net_thresh, a.hyst, a.interval))
    print()
    print('{:<9} {:<7} {:<28} {:<28}'.format('视频', '标注块', 'bounce P/R/F1', 'net P/R/F1'))
    print('-' * 78)

    agg = {'bounce': [], 'net': []}
    detail = []
    for g in a.games:
        rows = load_cache(g)
        if rows is None:
            print('{:<9} 缺缓存'.format(g))
            continue
        gt = load_gt(g)
        blocks = coverage_spans(gt, a.gap)
        pk = evaluate_in_blocks(rows, gt, a.bounce_thresh, a.hyst, a.interval, True,
                                blocks, a.tol)
        cells = {}
        for k in ('bounce', 'net'):
            d = pk.get(k, {'tp': 0, 'fp': 0, 'fn': 0})
            p, r, f1 = prf(d['tp'], d['fp'], d['fn'])
            cells[k] = (p, r, f1, d)
            agg[k].append((p, r, f1))
        print('{:<9} {:<7} {:<28} {:<28}'.format(
            g, len(blocks),
            '{:.0f}/{:.0f}/{:.1f}%'.format(cells['bounce'][0] * 100,
                                           cells['bounce'][1] * 100,
                                           cells['bounce'][2] * 100),
            '{:.0f}/{:.0f}/{:.1f}%'.format(cells['net'][0] * 100,
                                           cells['net'][1] * 100,
                                           cells['net'][2] * 100)))
        detail.append({'game': g, 'n_blocks': len(blocks),
                       'bounce': {k: cells['bounce'][3][k] for k in ('tp', 'fp', 'fn')},
                       'net': {k: cells['net'][3][k] for k in ('tp', 'fp', 'fn')}})

    print('-' * 78)
    for k in ('bounce', 'net'):
        if not agg[k]:
            continue
        xs = agg[k]
        print('{:<9} {:<7} {:.0f}/{:.0f}/{:.1f}%  ({} 视频均值)'.format(
            '均值' if k == 'bounce' else '', '',
            sum(x[0] for x in xs) / len(xs) * 100,
            sum(x[1] for x in xs) / len(xs) * 100,
            sum(x[2] for x in xs) / len(xs) * 100,
            len(xs)))

    if a.json_out:
        with open(a.json_out, 'w', encoding='utf-8') as f:
            json.dump(detail, f, ensure_ascii=False, indent=1)
        print('\n明细: {}'.format(a.json_out))


if __name__ == '__main__':
    main()
