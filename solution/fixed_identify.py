#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""恒温台阶跃响应辨识：FOPDT（一阶惯性 + 纯延迟）参数估计。

命令行：
    python3 src/identify.py --run data/run.json --out out/report.json

口径：
  · 模型 y(t) = y0 + K·Δu·(1 - exp(-(t - t_step - θ)/τ))，t ≥ t_step + θ；之前保持 y0；
  · y0 取阶跃前 baseline_window_s 窗口内（剔除冻结段后）的均值；Δu 取阶跃前后实测输入之差；
  · 冻结段（连续 frozen_run_min 点以上读数完全相同）视为传感器卡死，整段剔除、不参与拟合；
  · θ 与 τ 先在网格上找残差平方和最小，再在 (θ, τ) 上做黄金分割细化；
    给定 (θ, τ) 时 K 有闭式最小二乘解；
  · rmse 按拟合窗口内（剔除冻结点后）的残差均方根算；
  · 输出先写临时文件再原子替换；非法输入以退出码 2 结束且不落文件。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile

EXIT_INVALID = 2


def _fail(msg):
    sys.stderr.write("非法输入：%s\n" % msg)
    raise SystemExit(EXIT_INVALID)


def load_run(path):
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        _fail("读不到试验记录 %s" % path)
    except ValueError as e:
        _fail("记录不是合法 JSON：%s" % e)
    if not isinstance(raw, dict):
        _fail("记录顶层要是对象")
    samples = raw.get("samples")
    if not isinstance(samples, list) or len(samples) < 100:
        _fail("samples 至少要 100 个采样点")
    fit_cfg = raw.get("fit") or {}
    base_win = fit_cfg.get("baseline_window_s", 30.0)
    horizon = fit_cfg.get("horizon_s", 840.0)
    run_min = fit_cfg.get("frozen_run_min", 5)
    for key, val in (("baseline_window_s", base_win), ("horizon_s", horizon)):
        if not isinstance(val, (int, float)) or isinstance(val, bool) or not (float(val) > 0):
            _fail("fit.%s 要是正数" % key)
    if not isinstance(run_min, int) or isinstance(run_min, bool) or run_min < 2:
        _fail("fit.frozen_run_min 要是不小于 2 的整数")

    times, us, ys = [], [], []
    for i, s in enumerate(samples):
        if not isinstance(s, dict):
            _fail("samples[%d] 不是对象" % i)
        try:
            t = float(s["t"])
            u = float(s["u"])
            y = float(s["y"])
        except (KeyError, TypeError, ValueError):
            _fail("samples[%d] 的 t/u/y 必须是数值" % i)
        if not (math.isfinite(t) and math.isfinite(u) and math.isfinite(y)):
            _fail("samples[%d] 有非有限数值" % i)
        if t < 0 or (times and t <= times[-1]):
            _fail("samples[%d].t 必须单调递增且非负" % i)
        times.append(t)
        us.append(u)
        ys.append(y)
    return {"times": times, "us": us, "ys": ys, "base_win": float(base_win),
            "horizon": float(horizon), "run_min": int(run_min), "unit": (raw.get("unit") or {})}


# ---------------------------------------------------------------- 冻结段与阶跃
def frozen_mask(ys, run_min):
    n = len(ys)
    mask = [False] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and ys[j] == ys[i]:
            j += 1
        if j - i > run_min - 1:
            for k in range(i, j):
                mask[k] = True
        i = j
    return mask


def find_step(us):
    idx = [i for i in range(1, len(us)) if us[i] != us[i - 1]]
    if not idx:
        _fail("输入序列里没有阶跃")
    if len(idx) > 1:
        _fail("输入序列里有多次跃变，单次阶跃辨识做不了")
    return idx[0]


# ---------------------------------------------------------------- 拟合
def _response(t, t_step, theta, tau):
    s = t - t_step - theta
    if s <= 0:
        return 0.0
    return 1.0 - math.exp(-s / tau)


def _gain_ls(base, ts, ys, t_step, theta, tau):
    num = den = 0.0
    for t, y in zip(ts, ys):
        f = _response(t, t_step, theta, tau)
        num += f * (y - base)
        den += f * f
    if den <= 1e-15:
        return 0.0
    return num / den


def _sse(base, ts, ys, t_step, theta, tau):
    gain = _gain_ls(base, ts, ys, t_step, theta, tau)
    acc = 0.0
    for t, y in zip(ts, ys):
        e = y - base - gain * _response(t, t_step, theta, tau)
        acc += e * e
    return acc, gain


def _golden_min(fn, lo, hi, iters=60):
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = fn(c), fn(d)
    for _ in range(iters):
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = fn(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = fn(d)
    return (a + b) / 2.0


def fit_fopdt(ts, ys, base, t_step):
    span = ts[-1] - t_step
    if span <= 0:
        _fail("阶跃发生在记录末尾，拟合窗口为空")
    theta_max = max(1.0, 0.5 * span)

    stride = max(1, len(ts) // 200)
    ts_s, ys_s = ts[::stride], ys[::stride]
    best = None
    for gi in range(81):
        theta = theta_max * gi / 80.0
        for gj in range(61):
            tau = 20.0 * (30.0 ** (gj / 60.0))
            sse, _ = _sse(base, ts_s, ys_s, t_step, theta, tau)
            if best is None or sse < best[0]:
                best = (sse, theta, tau)

    theta, tau = best[1], best[2]
    for _ in range(10):
        theta = _golden_min(lambda x: _sse(base, ts, ys, t_step, x, tau)[0],
                            max(0.0, theta - theta_max / 20.0),
                            min(theta_max, theta + theta_max / 20.0))
        tau = _golden_min(lambda x: _sse(base, ts, ys, t_step, theta, x)[0],
                          max(5.0, tau / 1.5), min(3000.0, tau * 1.5))
    sse, gain = _sse(base, ts, ys, t_step, theta, tau)
    rmse = math.sqrt(sse / len(ts)) if ts else float("inf")
    return gain, tau, theta, rmse


def analyse(scan):
    times, us, ys = scan["times"], scan["us"], scan["ys"]
    mask = frozen_mask(ys, scan["run_min"])
    if sum(mask) == len(ys):
        _fail("所有采样点都在冻结段里，没法拟合")

    step_idx = find_step(us)
    t_step = times[step_idx - 1]
    u_before, u_after = us[step_idx - 1], us[step_idx]
    d_u = u_after - u_before
    if d_u == 0:
        _fail("阶跃幅度为零")

    pre = [y for t, y, m in zip(times, ys, mask) if t <= times[0] + scan["base_win"] and not m]
    if len(pre) < 5:
        _fail("阶跃前基线窗口里的有效采样点太少")
    base = sum(pre) / len(pre)

    fit_ts, fit_ys = [], []
    for t, y, m in zip(times, ys, mask):
        if m:
            continue
        if t > times[0] + scan["base_win"] and t <= t_step + scan["horizon"]:
            fit_ts.append(t)
            fit_ys.append(y)
    if len(fit_ts) < 50:
        _fail("拟合窗口内的有效采样点太少")

    gain, tau, theta, rmse = fit_fopdt(fit_ts, fit_ys, base, t_step)
    return {
        "recorded_points": len(times), "fitted_points": len(fit_ts),
        "skipped_frozen": sum(1 for m in mask if m),
        "pre_step_level_c": base, "step_time_s": t_step, "step_size_pct": d_u,
        "steady_gain_c_per_pct": gain / d_u, "time_constant_s": tau, "dead_time_s": theta, "model_rms_c": rmse,
    }


def build_report(scan, res):
    return {
        "recorded_points": res["recorded_points"],
        "fitted_points": res["fitted_points"],
        "skipped_frozen": res["skipped_frozen"],
        "pre_step_level_c": round(res["pre_step_level_c"], 3),
        "step_time_s": round(res["step_time_s"], 1),
        "step_size_pct": round(res["step_size_pct"], 3),
        "steady_gain_c_per_pct": round(res["steady_gain_c_per_pct"], 3),
        "time_constant_s": round(res["time_constant_s"], 1),
        "dead_time_s": round(res["dead_time_s"], 1),
        "model_rms_c": round(res["model_rms_c"], 4),
        "model_note": {"model": "fopdt", "unit": scan["unit"]},
    }


def atomic_write(path, obj):
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    body = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=".idn-", suffix=".tmp", dir=out_dir or ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(body)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def main(argv=None):
    ap = argparse.ArgumentParser(description="台阶跃响应 FOPDT 辨识")
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    scan = load_run(args.run)
    res = analyse(scan)
    atomic_write(args.out, build_report(scan, res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
