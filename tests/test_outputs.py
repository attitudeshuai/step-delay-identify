#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""判分输出：10 条行为测试，按重要程度加权，权重和 = 1。

每条测试自己造一段台阶跃响应记录，并按 FOPDT 模型独立做一次网格 + 细化拟合
（与被测程序不同代码路径），再驱动被测程序的命令行入口比对报告。
期望值不读被测代码，也不读参考解的输出。
"""
import json
import math
import os
import subprocess
import sys
import tempfile

APP = os.environ.get("CT_APP_DIR", "/app")
ENTRY = os.path.join(APP, "src", "identify.py")

WEIGHTS = {
    "test_gain_and_tau_match_independent_fit": 0.16,
    "test_delay_estimated_from_grid_search": 0.15,
    "test_frozen_segment_removed_from_fit": 0.12,
    "test_second_run_with_other_parameters": 0.11,
    "test_short_delay_run_still_estimated": 0.10,
    "test_invalid_runs_rejected_without_output": 0.11,
    "test_rmse_reflects_fit_quality": 0.07,
    "test_report_schema_and_rounding": 0.07,
    "test_rerun_produces_identical_bytes": 0.06,
    "test_failed_run_keeps_previous_report": 0.05,
}


class _Lcg(object):
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def next_u(self):
        self.state = (1103515245 * self.state + 12345) & 0x7FFFFFFF
        return (self.state % 1000003) / 1000003.0


def make_run(gain=0.40, tau=120.0, delay=37.5, y0=25.0, u_before=20.0, u_after=70.0,
             dt_s=1.0, t_step=60.0, duration=900.0, noise=0.012, seed=101,
             frozen_from=400, frozen_len=20):
    rng = _Lcg(seed)
    n = int(round(duration / dt_s)) + 1
    samples = []
    for i in range(n):
        t = round(i * dt_s, 1)
        u = u_after if t >= t_step else u_before
        s = t - t_step - delay
        y = y0 if s <= 0 else y0 + gain * (u_after - u_before) * (1.0 - math.exp(-s / tau))
        y += (rng.next_u() - 0.5) * 2.0 * noise
        samples.append({"t": t, "u": u, "y": round(y, 4)})
    if frozen_len > 0:
        stuck = samples[frozen_from]["y"]
        for k in range(frozen_from, min(frozen_from + frozen_len, len(samples))):
            samples[k]["y"] = stuck
    return {"unit": {"time": "s", "temperature": "degC", "power": "%"},
            "samples": samples,
            "fit": {"baseline_window_s": 30.0, "horizon_s": 840.0, "frozen_run_min": 5}}


def _frozen(ys, run_min):
    mask = [False] * len(ys)
    i = 0
    while i < len(ys):
        j = i + 1
        while j < len(ys) and ys[j] == ys[i]:
            j += 1
        if j - i >= run_min:
            for k in range(i, j):
                mask[k] = True
        i = j
    return mask


def _resp(t, t_step, theta, tau):
    s = t - t_step - theta
    return 0.0 if s <= 0 else 1.0 - math.exp(-s / tau)


def _fit_gain(base, ts, ys, t_step, theta, tau):
    num = den = 0.0
    for t, y in zip(ts, ys):
        f = _resp(t, t_step, theta, tau)
        num += f * (y - base)
        den += f * f
    return 0.0 if den <= 1e-15 else num / den


def _sse(base, ts, ys, t_step, theta, tau):
    k = _fit_gain(base, ts, ys, t_step, theta, tau)
    acc = 0.0
    for t, y in zip(ts, ys):
        e = y - base - k * _resp(t, t_step, theta, tau)
        acc += e * e
    return acc, k


def _golden(fn, lo, hi, iters=70):
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


def independent_fit(data):
    times = [s["t"] for s in data["samples"]]
    us = [s["u"] for s in data["samples"]]
    ys = [s["y"] for s in data["samples"]]
    run_min = data["fit"]["frozen_run_min"]
    mask = _frozen(ys, run_min)
    step = next(i for i in range(1, len(us)) if us[i] != us[i - 1])
    t_step = times[step - 1]
    du = us[step] - us[step - 1]
    pre = [y for t, y, m in zip(times, ys, mask)
           if t <= times[0] + data["fit"]["baseline_window_s"] and not m]
    base = sum(pre) / len(pre)
    ts, ser = [], []
    for t, y, m in zip(times, ys, mask):
        if m:
            continue
        if t > times[0] + data["fit"]["baseline_window_s"] and t <= t_step + data["fit"]["horizon_s"]:
            ts.append(t)
            ser.append(y)

    span = ts[-1] - t_step
    theta_max = max(1.0, 0.5 * span)
    stride = max(1, len(ts) // 150)
    ts_s, ys_s = ts[::stride], ser[::stride]
    best = None
    for gi in range(61):
        theta = theta_max * gi / 60.0
        for gj in range(41):
            tau = 15.0 * (40.0 ** (gj / 40.0))
            sse, _ = _sse(base, ts_s, ys_s, t_step, theta, tau)
            if best is None or sse < best[0]:
                best = (sse, theta, tau)
    theta, tau = best[1], best[2]
    for _ in range(8):
        theta = _golden(lambda x: _sse(base, ts, ser, t_step, x, tau)[0],
                        max(0.0, theta - theta_max / 15.0),
                        min(theta_max, theta + theta_max / 15.0))
        tau = _golden(lambda x: _sse(base, ts, ser, t_step, theta, x)[0],
                      max(5.0, tau / 1.6), min(3000.0, tau * 1.6))
    sse, gain = _sse(base, ts, ser, t_step, theta, tau)
    return {"steady_gain_c_per_pct": gain / du, "time_constant_s": tau, "dead_time_s": theta,
            "model_rms_c": math.sqrt(sse / len(ts)), "skipped_frozen": sum(1 for m in mask if m),
            "fitted_points": len(ts), "step_size_pct": du}


def _run(path, out_path, timeout=240):
    cmd = [sys.executable, ENTRY, "--run", path, "--out", out_path]
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)


def run_case(data):
    tmp = tempfile.mkdtemp(prefix="id-case-")
    rpath = os.path.join(tmp, "run.json")
    opath = os.path.join(tmp, "report.json")
    with open(rpath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    proc = _run(rpath, opath)
    report = None
    if os.path.exists(opath):
        try:
            with open(opath, encoding="utf-8") as f:
                report = json.load(f)
        except ValueError:
            report = None
    return report, {"run": rpath, "out": opath, "proc": proc, "dir": tmp}


def _close(got, want, rel):
    try:
        v = float(got)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(v):
        return False
    return abs(v - want) <= rel * abs(want)


def test_gain_and_tau_match_independent_fit():
    """题面原文：「按 FOPDT 模型拟合增益与时间常数」。

    标准台架记录：增益相对误差不超过 1%，时间常数不超过 2%。
    """
    data = make_run(seed=101)
    report, _ = run_case(data)
    if not isinstance(report, dict):
        return False
    exp = independent_fit(data)
    return (_close(report.get("steady_gain_c_per_pct"), exp["steady_gain_c_per_pct"], 0.01)
            and _close(report.get("time_constant_s"), exp["time_constant_s"], 0.02))


def test_delay_estimated_from_grid_search():
    """题面原文：「纯延迟 θ 在 (θ, τ) 上搜出来，不能当零」。

    延迟真值 37.5 s：估计值与独立复算相差不超过 2 秒。
    """
    data = make_run(seed=211)
    report, _ = run_case(data)
    if not isinstance(report, dict):
        return False
    exp = independent_fit(data)
    try:
        got = float(report["dead_time_s"])
    except (KeyError, TypeError, ValueError):
        return False
    return abs(got - exp["dead_time_s"]) <= 2.0


def test_frozen_segment_removed_from_fit():
    """题面原文：「连续 5 点以上读数完全相同的整段按传感器卡死剔除」。

    记录里塞了 20 点卡死读数：报告要如实报出剔除数量，参数仍要准。
    """
    data = make_run(seed=307, frozen_from=380, frozen_len=20)
    report, _ = run_case(data)
    if not isinstance(report, dict):
        return False
    exp = independent_fit(data)
    return (report.get("skipped_frozen") == exp["skipped_frozen"]
            and _close(report.get("steady_gain_c_per_pct"), exp["steady_gain_c_per_pct"], 0.01)
            and _close(report.get("time_constant_s"), exp["time_constant_s"], 0.02))


def test_second_run_with_other_parameters():
    """题面原文：「复检会喂别的台架记录，增益、时间常数与采样步长都不同」。

    换增益 0.75、τ=260 s、dt=2 s 的记录后仍要吻合。
    """
    data = make_run(gain=0.75, tau=260.0, delay=20.0, dt_s=2.0, duration=1800.0,
                    frozen_from=300, seed=409)
    report, _ = run_case(data)
    if not isinstance(report, dict):
        return False
    exp = independent_fit(data)
    return (_close(report.get("steady_gain_c_per_pct"), exp["steady_gain_c_per_pct"], 0.01)
            and _close(report.get("time_constant_s"), exp["time_constant_s"], 0.02)
            and abs(float(report.get("dead_time_s", -99)) - exp["dead_time_s"]) <= 2.0)


def test_short_delay_run_still_estimated():
    """题面原文：「延迟小到几个采样周期时也要估出来，不能直接给零」。

    延迟真值 6 s：估计值与独立复算相差不超过 2 秒，且不得报 0。
    """
    data = make_run(delay=6.0, tau=90.0, seed=503, frozen_from=500)
    report, _ = run_case(data)
    if not isinstance(report, dict):
        return False
    exp = independent_fit(data)
    try:
        got = float(report["dead_time_s"])
    except (KeyError, TypeError, ValueError):
        return False
    return abs(got - exp["dead_time_s"]) <= 2.0 and got > 0.5


def test_invalid_runs_rejected_without_output():
    """题面原文：「记录不合法时以退出码 2 结束，不产出报告」。

    四种坏记录：采样点不足 100、时间不单调、没有阶跃、输入有两次跃变。
    """
    good = make_run(seed=601)
    few = json.loads(json.dumps(good))
    few["samples"] = few["samples"][:80]
    nonmono = json.loads(json.dumps(good))
    nonmono["samples"][10]["t"] = nonmono["samples"][9]["t"]
    nostep = json.loads(json.dumps(good))
    for s in nostep["samples"]:
        s["u"] = 20.0
    twostep = json.loads(json.dumps(good))
    for s in twostep["samples"]:
        if s["t"] >= 600.0:
            s["u"] = 40.0
    for data in (few, nonmono, nostep, twostep):
        tmp = tempfile.mkdtemp(prefix="id-bad-")
        rpath = os.path.join(tmp, "run.json")
        opath = os.path.join(tmp, "report.json")
        with open(rpath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        proc = _run(rpath, opath)
        if proc.returncode != 2 or os.path.exists(opath) or not (proc.stderr or "").strip():
            return False
        if [n for n in os.listdir(tmp) if n.endswith(".tmp")]:
            return False
    return True


def test_rmse_reflects_fit_quality():
    """题面原文：「报告给出拟合残差均方根，用于判断这次辨识是否可用」。

    标准台架记录噪声约 0.012：rmse 不超过 0.03，且与独立复算相差不超过 30%。
    """
    data = make_run(seed=701)
    report, _ = run_case(data)
    if not isinstance(report, dict):
        return False
    exp = independent_fit(data)
    try:
        got = float(report["model_rms_c"])
    except (KeyError, TypeError, ValueError):
        return False
    return got <= 0.03 and abs(got - exp["model_rms_c"]) <= 0.30 * exp["model_rms_c"]


def test_report_schema_and_rounding():
    """题面原文：「报告结构固定，增益三位小数、时间常数与延迟一位、rmse 四位」。

    校验顶层字段、位数与有限性，文件里不出现 NaN 或 Infinity。
    """
    data = make_run(seed=809)
    report, ctx = run_case(data)
    if not isinstance(report, dict):
        return False
    need = {"recorded_points", "fitted_points", "skipped_frozen", "pre_step_level_c", "step_time_s", "step_size_pct",
            "steady_gain_c_per_pct", "time_constant_s", "dead_time_s", "model_rms_c", "model_note"}
    if not need.issubset(set(report)):
        return False
    for key in ("pre_step_level_c", "step_size_pct", "steady_gain_c_per_pct"):
        v = float(report[key])
        if not math.isfinite(v) or abs(v - round(v, 3)) > 1e-12:
            return False
    for key in ("time_constant_s", "dead_time_s", "step_time_s"):
        v = float(report[key])
        if not math.isfinite(v) or abs(v - round(v, 1)) > 1e-12:
            return False
    v = float(report["model_rms_c"])
    if not math.isfinite(v) or abs(v - round(v, 4)) > 1e-12:
        return False
    raw = open(ctx["out"], encoding="utf-8").read()
    return "NaN" not in raw and "Infinity" not in raw


def test_rerun_produces_identical_bytes():
    """题面原文：「同一份记录重复运行，报告逐字节一致」。

    连跑两次，两个报告文件字节完全相同。
    """
    data = make_run(seed=907)
    first, ctx1 = run_case(data)
    if first is None:
        return False
    # 先确认跑出来的是一份合格报告（缺字段或参数不是有限的都不算）
    if not {"fitted_points", "skipped_frozen", "time_constant_s", "dead_time_s"}.issubset(set(first)):
        return False
    if not all(math.isfinite(float(first[k])) for k in ("time_constant_s", "dead_time_s", "steady_gain_c_per_pct")):
        return False
    body1 = open(ctx1["out"], "rb").read()
    second, ctx2 = run_case(data)
    if second is None:
        return False
    return body1 == open(ctx2["out"], "rb").read()


def test_failed_run_keeps_previous_report():
    """题面原文：「运行失败时不能覆盖上一次已经写好的报告」。

    先跑一次合格记录留下报告，再用没有阶跃的坏记录跑：退出码 2，旧报告字节不变。
    """
    good = make_run(seed=1009)
    tmp = tempfile.mkdtemp(prefix="id-keep-")
    rpath = os.path.join(tmp, "run.json")
    opath = os.path.join(tmp, "report.json")
    with open(rpath, "w", encoding="utf-8") as f:
        json.dump(good, f, ensure_ascii=False)
    first = _run(rpath, opath)
    if first.returncode != 0 or not os.path.exists(opath):
        return False
    before = open(opath, "rb").read()
    broken = json.loads(json.dumps(good))
    for s in broken["samples"]:
        s["u"] = 20.0
    with open(rpath, "w", encoding="utf-8") as f:
        json.dump(broken, f, ensure_ascii=False)
    second = _run(rpath, opath)
    if second.returncode != 2:
        return False
    return open(opath, "rb").read() == before


def main():
    total = 0.0
    for name, weight in WEIGHTS.items():
        fn = globals().get(name)
        try:
            passed = bool(fn and fn())
        except Exception as e:  # noqa: BLE001 —— 任何异常都按不通过计
            print("%s: 异常 %r" % (name, e), file=sys.stderr)
            passed = False
        total += weight * passed
        print("%s: %s (权重 %.2f)" % (name, "PASS" if passed else "FAIL", weight), file=sys.stderr)
    print("%.6f" % total)


if __name__ == "__main__":
    main()
