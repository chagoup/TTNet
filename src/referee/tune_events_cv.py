# -*- coding: utf-8 -*-
"""跨视频联合调参：在多个视频上搜索一组通用事件阈值，并在留出视频上验证泛化。

为什么不能用 tune_events.py 的单视频结果：
    单视频网格搜索是在该视频上挑"最好的那一组"，必然过拟合——
    换一个视频，最优阈值可能完全不同。之前 bounce=0.65 就是在 test_2 上
    网格搜索得到的，这正是它在新视频上表现不稳的原因。

本脚本的做法：
    1. 在【调参视频】上联合优化：目标函数是所有调参视频的平均 F1
    2. 在【验证视频】上评估这组参数，检验是否真的泛化
    3. 同时报告"每个视频各自最优"的上界，用来衡量联合参数的损失有多大

用法：
    python -m referee.tune_events_cv                          # 默认 3 调参 + 4 验证
    python -m referee.tune_events_cv --tune test_2 test_3 test_5
"""

import argparse
import itertools
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from referee.tune_events import load_gt, labeled_span, match, detect

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def load_cache(game):
    p = os.path.join(ROOT, 'results', 'cache_{}.json'.format(game))
    if not os.path.isfile(p):
        return None
    with open(p, encoding='utf-8') as f:
        d = json.load(f)
    # 缓存格式为 {'meta': {...}, 'rows': [...]}
    return d.get('rows', d) if isinstance(d, dict) else d


def clip(rows, span):
    return [r for r in rows if span[0] <= r['t'] <= span[1]]


def evaluate(rows, gt, high, hyst, interval, mask, tol=0.1):
    """返回 bounce 与 net 的 (P, R, F1, tp, fp, fn)。"""
    out = {}
    for key, gtkey in (('b', 'bounce'), ('n', 'net')):
        pred = detect(rows, key, high, hyst, interval, mask)
        tp, fp, fn, errs = match(pred, gt[gtkey], tol)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        out[gtkey] = {'p': p, 'r': r, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn,
                      'n_pred': len(pred), 'n_gt': len(gt[gtkey])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tune', nargs='*', default=['test_2', 'test_3', 'test_5'])
    ap.add_argument('--holdout', nargs='*', default=['test_1', 'test_4', 'test_6', 'test_7'])
    ap.add_argument('--tol', type=float, default=0.1)
    ap.add_argument('--json-out', default='')
    a = ap.parse_args()

    data = {}
    for g in a.tune + a.holdout:
        c = load_cache(g)
        if c is None:
            print('[跳过] {} 缺缓存'.format(g))
            continue
        data[g] = {'rows': clip(c, labeled_span(g)), 'gt': load_gt(g)}

    tune_games = [g for g in a.tune if g in data]
    hold_games = [g for g in a.holdout if g in data]
    if not tune_games:
        raise SystemExit('没有可用的调参视频（先跑 cache_scores）')

    print('调参视频: {}'.format(tune_games))
    print('验证视频: {}'.format(hold_games))
    print()

    # ---- 网格搜索：联合优化调参视频的平均 F1
    grid_high = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    grid_hyst = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3]
    grid_int = [0.0, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]
    grid_mask = [True, False]

    best = {}
    for key in ('bounce', 'net'):
        best[key] = None
        for high, hyst, interval, mask in itertools.product(
                grid_high, grid_hyst, grid_int, grid_mask):
            if hyst >= high:
                continue
            f1s, ps, rs = [], [], []
            for g in tune_games:
                r = evaluate(data[g]['rows'], data[g]['gt'], high, hyst, interval, mask, a.tol)
                f1s.append(r[key]['f1'])
                ps.append(r[key]['p'])
                rs.append(r[key]['r'])
            mean_f1 = sum(f1s) / len(f1s)
            cand = {'high': high, 'hyst': hyst, 'interval': interval,
                    'mask': mask, 'mean_f1': mean_f1,
                    'min_f1': min(f1s),
                    'mean_p': sum(ps) / len(ps), 'mean_r': sum(rs) / len(rs),
                    'per_game_f1': f1s}
            # 主排序：平均 F1；平手时看最差视频，保证不牺牲某个视频
            if best[key] is None or (mean_f1, cand['min_f1']) > (best[key]['mean_f1'], best[key]['min_f1']):
                best[key] = cand

    print('=' * 76)
    print('第一步：在调参视频上联合优化的结果')
    print('=' * 76)
    for key in ('bounce', 'net'):
        b = best[key]
        print('{:<7} high={:<5} hyst={:<5} interval={:<5} mask={}'.format(
            key, b['high'], b['hyst'], b['interval'], b['mask']))
        print('        调参视频平均 F1={:.1f}%  最差={:.1f}%  P={:.1f}%  R={:.1f}%'.format(
            b['mean_f1'] * 100, b['min_f1'] * 100, b['mean_p'] * 100, b['mean_r'] * 100))
        print('        逐视频 F1: {}'.format(
            ', '.join('{}={:.1f}%'.format(g, f * 100) for g, f in zip(tune_games, b['per_game_f1']))))
    print()

    # ---- 在留出视频上验证
    if hold_games:
        print('=' * 76)
        print('第二步：在未参与调参的留出视频上验证（这才是真实泛化性能）')
        print('=' * 76)
        print('{:<9} {:<24} {:<24}'.format('视频', 'bounce P/R/F1', 'net P/R/F1'))
        print('-' * 76)
        agg = {'bounce': [], 'net': []}
        for g in hold_games:
            rb = evaluate(data[g]['rows'], data[g]['gt'],
                          best['bounce']['high'], best['bounce']['hyst'],
                          best['bounce']['interval'], best['bounce']['mask'], a.tol)['bounce']
            rn = evaluate(data[g]['rows'], data[g]['gt'],
                          best['net']['high'], best['net']['hyst'],
                          best['net']['interval'], best['net']['mask'], a.tol)['net']
            agg['bounce'].append(rb)
            agg['net'].append(rn)
            print('{:<9} {:<24} {:<24}'.format(
                g,
                '{:.0f}/{:.0f}/{:.1f}%'.format(rb['p'] * 100, rb['r'] * 100, rb['f1'] * 100),
                '{:.0f}/{:.0f}/{:.1f}%'.format(rn['p'] * 100, rn['r'] * 100, rn['f1'] * 100)))
        print('-' * 76)
        for key in ('bounce', 'net'):
            xs = agg[key]
            print('{:<9} {:<24} {:<24}'.format(
                '留出均值' if key == 'bounce' else '',
                '{:.0f}/{:.0f}/{:.1f}%'.format(
                    sum(x['p'] for x in xs) / len(xs) * 100,
                    sum(x['r'] for x in xs) / len(xs) * 100,
                    sum(x['f1'] for x in xs) / len(xs) * 100) if key == 'bounce' else '',
                '{:.0f}/{:.0f}/{:.1f}%'.format(
                    sum(x['p'] for x in xs) / len(xs) * 100,
                    sum(x['r'] for x in xs) / len(xs) * 100,
                    sum(x['f1'] for x in xs) / len(xs) * 100) if key == 'net' else ''))
        print()

    # ---- 与当前默认参数对比
    print('=' * 76)
    print('第三步：与当前默认参数对比（bounce=0.65/0.1/0.10, net=0.5/0.1/0.10）')
    print('=' * 76)
    for label, params in (('当前默认', {'bounce': (0.65, 0.1, 0.10, True),
                                        'net': (0.5, 0.1, 0.10, True)}),
                          ('联合调参', {'bounce': (best['bounce']['high'], best['bounce']['hyst'],
                                                   best['bounce']['interval'], best['bounce']['mask']),
                                        'net': (best['net']['high'], best['net']['hyst'],
                                                best['net']['interval'], best['net']['mask'])})):
        print('--- {} ---'.format(label))
        for key in ('bounce', 'net'):
            h, hy, iv, mk = params[key]
            for group_name, group in (('调参集', tune_games), ('留出集', hold_games)):
                if not group:
                    continue
                xs = [evaluate(data[g]['rows'], data[g]['gt'], h, hy, iv, mk, a.tol)[key]
                      for g in group]
                print('  {:<5} {:<6} P={:5.1f}% R={:5.1f}% F1={:5.1f}%  FP合计={:3d} FN合计={:3d}'.format(
                    key, group_name,
                    sum(x['p'] for x in xs) / len(xs) * 100,
                    sum(x['r'] for x in xs) / len(xs) * 100,
                    sum(x['f1'] for x in xs) / len(xs) * 100,
                    sum(x['fp'] for x in xs), sum(x['fn'] for x in xs)))

    if a.json_out:
        with open(a.json_out, 'w', encoding='utf-8') as f:
            json.dump({'tune_games': tune_games, 'holdout_games': hold_games,
                       'best': best}, f, ensure_ascii=False, indent=1)
        print('\n结果已写入: {}'.format(a.json_out))


if __name__ == '__main__':
    main()
