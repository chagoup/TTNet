# -*- coding: utf-8 -*-
"""对完整视频跑裁判流程，并把实时比分 UI 烧进结果视频。

与 run_referee.py 的区别
------------------------
run_referee.py 只做「离线分析」，输出文字统计；
本脚本做「在线可视化」，边推理边绘制比分 UI，最终产出可播放的 mp4。

关键实现细节
------------
1. **时间对齐**：TTNet 输入 9 帧、预测末帧，因此前 8 帧没有观测。
   为保证输出视频与原视频帧数一致、不丢开头，开头 8 帧沿用初始比分（0:0）绘制。
2. **事件滞后**：比分在事件发生的那一帧才更新，因此 UI 用「事件时间戳」判断
   当前帧应显示哪个比分，而不是用推理完成时刻。
3. **两遍式设计**：先完整推理一遍收集「帧号 → 比分/事件」，再渲染视频。
   这样避免在单遍流式推理中处理复杂的时间回溯，逻辑更可靠。
"""

import os
import sys
import json
import argparse
import time

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from referee.ttnet_adapter import TTNetAdapter
from referee.event_processing import TrajectoryProcessor, EventExtractor
from referee.score_state_machine import TableTennisStateMachine
from referee.unified_types import EventType, Side
from referee.scoreboard_ui import ScoreboardUI, load_font
from referee.run_referee import build_configs, DEFAULT_WEIGHTS


EVENT_TEXT = {
    EventType.BOUNCE: '落台',
    EventType.NET_HIT: '擦网',
    EventType.HIT: '击球',
}


def analyze(adapter, video_path, bounce_thresh, net_thresh, max_frames=None, verbose=True):
    """第一遍：推理并记录「逐帧观测 + 事件」。

    为了支持「一次推理、反复调判分规则」，这里只保存原始观测与事件，
    不在这一步跑状态机。判分由 build_score_timeline() 单独完成（毫秒级）。
    """
    tracker = TrajectoryProcessor()
    extractor = EventExtractor(bounce_thresh=bounce_thresh, net_thresh=net_thresh)

    frame_obs = []      # [(frame_id, ts, visible)]
    events = []         # [dict]
    detected = {'BOUNCE': 0, 'NET_HIT': 0, 'HIT': 0, 'BALL_LOST': 0}

    def _handle(evs):
        for ev in evs:
            detected[ev.type] = detected.get(ev.type, 0) + 1
            events.append(ev.to_dict())

    for obs in adapter.observations_from_video(video_path, max_frames=max_frames,
                                               progress_every=2000):
        traj = tracker.update(obs)
        _handle(extractor.process(obs, traj))
        frame_obs.append((obs.frame_id, round(obs.timestamp_s, 5),
                          1 if obs.ball.visible else 0))
    _handle(extractor.flush())

    if verbose:
        print('  推理完成：{} 帧，事件 {}'.format(len(frame_obs), detected))
    return {'frames': frame_obs, 'events': events, 'detected': detected,
            'low_conf_dropped': adapter.low_conf_dropped,
            'frozen_dropped': adapter.frozen_frames_dropped}


def build_score_timeline(cache, verbose=True, cumulative_mode=True):
    """第二遍（毫秒级）：在缓存的事件上跑状态机，产出「帧号 → 比分」快照。

    与推理完全解耦，因此可以随意调状态机参数并立即看到效果。

    :param cumulative_mode: 默认 True —— 测试片只有几十秒到几分钟，
        不存在「打完一局」，按 11 分制会在 11 分处归零，观众会看到比分突然清零。
        累计模式让比分单调上升，符合短片演示的直觉。正式比赛应设为 False。
    """
    from referee.unified_types import Event

    machine = TableTennisStateMachine(verbose=False, cumulative_mode=cumulative_mode)
    # 只送「有判罚意义」的事件进状态机。
    # BALL_LOST 只是断段信号（本视频出现 396 次），送入会让时间线错乱且无意义。
    # 事件必须按时间戳严格排序：BOUNCE/NET 在「驼峰结束时」才产出，但时间戳是
    # 峰值时刻，会比同帧的 BALL_LOST 更早；不排序会导致比分更新错位。
    evs = sorted([e for e in cache['events'] if e['type'] != EventType.BALL_LOST],
                 key=lambda e: (e['timestamp_s'], e['type']))

    timeline = []
    snapshots = {}
    ev_idx = 0

    for frame_id, ts, _vis in cache['frames']:
        while ev_idx < len(evs) and evs[ev_idx]['timestamp_s'] <= ts:
            d = evs[ev_idx]
            ev = Event(event_id=d['event_id'], timestamp_s=d['timestamp_s'],
                       type=d['type'], side=d['side'],
                       position_px=d.get('position_px'),
                       confidence=d.get('confidence'),
                       sources=d.get('sources', []))
            machine.consume(ev)
            if ev.type in (EventType.BOUNCE, EventType.NET_HIT):
                suffix = ' · {} 侧'.format(ev.side) if ev.side in (Side.A, Side.B) else ''
                timeline.append((ev.timestamp_s,
                                 EVENT_TEXT.get(ev.type, ev.type) + suffix))
            ev_idx += 1
        machine.tick_deep(ts)

        sb = machine.scoreboard
        snapshots[frame_id] = {
            'score_a': sb['score']['A'],
            'score_b': sb['score']['B'],
            'games_a': sb['games']['A'],
            'games_b': sb['games']['B'],
            'server': sb['server'],
        }

    # 视频末尾收尾：最后一个回合没有后续时间窗口，必须显式结算，
    # 否则末尾的落台永远无法判分。
    #
    # 注意：结算结果只写到「最后一帧」，绝不向前倒灌。
    # 曾经为了让终局比分更醒目，把结果覆盖到末帧前 2 秒的所有帧上，
    # 结果导致 27.99s 的画面（回合仍在进行）就显示了 2:1，
    # 而真实判分时刻是 29.99s —— 制造了 2 秒的时序错位。
    # 比分必须严格发生在判分那一刻，不能为「可读性」伪造时间。
    if cache['frames']:
        last_fid, last_ts = cache['frames'][-1][0], cache['frames'][-1][1]
        if machine.finalize(last_ts):
            sb = machine.scoreboard
            snapshots[last_fid] = {
                'score_a': sb['score']['A'],
                'score_b': sb['score']['B'],
                'games_a': sb['games']['A'],
                'games_b': sb['games']['B'],
                'server': sb['server'],
            }

    if verbose:
        print('  判分完成：{} 分（护栏拦下 {} 次）'.format(
            len(machine.points), machine.suppressed_count))
    return snapshots, timeline, machine


def render(video_path, out_path, snapshots, timeline, fps,
           ui=None, fade_s=1.6, progress_every=2000, max_frames=None,
           cumulative=True):
    """第二遍：逐帧回放原视频，叠加对应时刻的比分 UI。

    为保证「比分与画面同步」，用帧号查快照；事件提示用时间差淡出。

    注意：快照的键与视频帧号并不完全对齐 —— 适配器前 8 帧没有观测（9 帧窗口
    需要预热），因此快照从 frame_id=8 开始。这里统一按「帧号」查询，
    并在找不到时向前回溯最近一个快照，避免开头几帧与末尾帧丢掉比分。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError('无法打开视频: {}'.format(video_path))

    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        n_total = min(n_total, max_frames)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ui = ui or ScoreboardUI()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # 初始状态（前 8 帧还没有观测）
    cur = {'score_a': 0, 'score_b': 0, 'games_a': 0, 'games_b': 0, 'server': 'A'}
    written = 0
    ev_idx = 0
    last_ev_ts = None
    last_ev_text = None

    # 按时间戳排序的快照键，用于「找不到精确帧号时取最近一帧」
    snap_keys = sorted(snapshots.keys())
    snap_pos = 0

    while True:
        if max_frames is not None and written >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        ts = written / float(fps)

        # 推进快照指针：取所有 frame_id <= written 的最后一个快照
        while snap_pos < len(snap_keys) and snap_keys[snap_pos] <= written:
            cur = snapshots[snap_keys[snap_pos]]
            snap_pos += 1

        # 推进事件指针（timeline 按时间递增）
        while ev_idx < len(timeline) and timeline[ev_idx][0] <= ts:
            last_ev_ts, last_ev_text = timeline[ev_idx]
            ev_idx += 1

        age = (ts - last_ev_ts) if last_ev_ts is not None else None
        text = last_ev_text if (age is not None and age < fade_s) else None

        ui.draw(frame, score_a=cur['score_a'], score_b=cur['score_b'],
                games_a=cur['games_a'], games_b=cur['games_b'],
                server=cur['server'], event_text=text, event_age=age,
                cumulative=cumulative)
        writer.write(frame)
        written += 1
        if progress_every and written % progress_every == 0:
            print('  渲染 {}/{} 帧 ({:.0%})'.format(written, n_total, written / max(n_total, 1)))

    cap.release()
    writer.release()
    return written


def run(video_path, out_path, weights=DEFAULT_WEIGHTS,
        bounce_thresh=0.65, net_thresh=0.5, max_frames=None, device='cuda:0',
        cache_path=None):
    t0 = time.time()
    print('=' * 70)
    print('TTNet 裁判 · 带比分 UI 视频渲染')
    print('  输入 : {}'.format(video_path))
    print('  输出 : {}'.format(out_path))
    print('=' * 70)

    # ---- 第一遍：推理（可复用缓存）
    if cache_path and os.path.isfile(cache_path):
        print('[1/3] 载入推理缓存…')
        with open(cache_path, encoding='utf-8') as f:
            cache = json.load(f)
    else:
        configs = build_configs()
        adapter = TTNetAdapter(configs, weights, device=device)
        print('[1/3] 推理分析…')
        cache = analyze(adapter, video_path, bounce_thresh, net_thresh, max_frames)
        if cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False)

    # ---- 第二遍：判分（毫秒级，可反复调参）
    print('[2/4] 状态机判分…')
    snapshots, timeline, machine = build_score_timeline(cache)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    # ---- 第三遍：渲染
    print('[3/4] 渲染视频 + 比分 UI…')
    written = render(video_path, out_path, snapshots, timeline, fps,
                     max_frames=max_frames, cumulative=True)

    print('[4/4] 完成')
    print('  耗时 {:.1f}s | 输出 {} 帧 | 文件 {:.1f} MB'.format(
        time.time() - t0, written,
        os.path.getsize(out_path) / 1024 / 1024 if os.path.isfile(out_path) else 0))

    summary = {
        'video': video_path,
        'output': out_path,
        'frames': written,
        'fps': fps,
        'detected_events': cache['detected'],
        'final_score': machine.scoreboard,
        'points': machine.points,
        'suppressed_by_guard': machine.suppressed_count,
        'low_conf_dropped': cache.get('low_conf_dropped'),
        'frozen_dropped': cache.get('frozen_dropped'),
    }
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--bounce-thresh', type=float, default=0.65)
    ap.add_argument('--net-thresh', type=float, default=0.5)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--cache', default=None,
                    help='推理缓存路径；存在则直接复用，避免重复推理')
    a = ap.parse_args()

    s = run(a.video, a.out, bounce_thresh=a.bounce_thresh,
            net_thresh=a.net_thresh, max_frames=a.max_frames,
            device=a.device, cache_path=a.cache)
    print(json.dumps(s, ensure_ascii=False, indent=1))
