#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""判分输出：10 条行为测试，按重要程度加权，权重和 = 1。

每条测试自己造一段台阶跃响应记录，并按 FOPDT 模型独立做一次网格 + 细化拟合
（与被测程序不同代码路径），再驱动被测程序的命令行入口比对报告。
期望值不读被测代码，也不读参考解的输出。
"""
import json

import pytest
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


def test_gain_and_tau_match_independent_fit():
    """题面原文：「按 FOPDT 模型（一阶惯性 + 纯延迟）辨识」。

    标准台架记录：增益相对误差不超过 0.15%，时间常数不超过 0.3%。
    """
    data = make_run(seed=101)
    report, _ = run_case(data)
    assert isinstance(report, dict), "没有产出报告"
    exp = independent_fit(data)
    gain = float(report["steady_gain_c_per_pct"])
    tau = float(report["time_constant_s"])
    assert abs(gain - exp["steady_gain_c_per_pct"]) <= 0.0015 * abs(exp["steady_gain_c_per_pct"]), \
        "增益与独立复算差得太多：被测 %r，复算 %r" % (gain, exp["steady_gain_c_per_pct"])
    assert abs(tau - exp["time_constant_s"]) <= 0.003 * abs(exp["time_constant_s"]), \
        "时间常数与独立复算差得太多：被测 %r，复算 %r" % (tau, exp["time_constant_s"])


def test_delay_estimated_from_grid_search():
    """题面原文：「θ 与 τ 用残差平方和最小的方式估出来（先在网格上搜、再细化）」。

    延迟真值 37.5 s：估计值与独立复算相差不超过 0.5 秒，且必须明显大于零。
    """
    data = make_run(seed=211)
    report, _ = run_case(data)
    assert isinstance(report, dict), "没有产出报告"
    exp = independent_fit(data)
    got = float(report["dead_time_s"])
    assert abs(got - exp["dead_time_s"]) <= 0.5, \
        "纯滞后与独立复算差得太多：被测 %r，复算 %r" % (got, exp["dead_time_s"])
    assert got > 1.0, "纯滞后被当成零了：%r" % got


def test_frozen_segment_removed_from_fit():
    """题面原文：「连续 frozen_run_min 点以上读数完全相同的整段按传感器卡死处理」。

    记录里塞了 20 点卡死读数：报告要如实报出剔除数量，参数仍要准。
    """
    data = make_run(seed=307, frozen_from=380, frozen_len=20)
    report, _ = run_case(data)
    assert isinstance(report, dict), "没有产出报告"
    exp = independent_fit(data)
    assert int(report["skipped_frozen"]) == exp["skipped_frozen"], \
        "卡死点数不对：被测 %r，复算 %r" % (report["skipped_frozen"], exp["skipped_frozen"])
    gain = float(report["steady_gain_c_per_pct"])
    tau = float(report["time_constant_s"])
    assert abs(gain - exp["steady_gain_c_per_pct"]) <= 0.0015 * abs(exp["steady_gain_c_per_pct"]), \
        "剔除卡死段后增益偏了：%r 对 %r" % (gain, exp["steady_gain_c_per_pct"])
    assert abs(tau - exp["time_constant_s"]) <= 0.003 * abs(exp["time_constant_s"]), \
        "剔除卡死段后时间常数偏了：%r 对 %r" % (tau, exp["time_constant_s"])


def test_second_run_with_other_parameters():
    """题面原文：「复检会喂别的台架记录，增益、时间常数与采样步长都不同」。

    换增益 0.75、τ=260 s、dt=2 s 的记录后仍要吻合。
    """
    data = make_run(gain=0.75, tau=260.0, delay=20.0, dt_s=2.0, duration=1800.0,
                    frozen_from=300, seed=409)
    report, _ = run_case(data)
    assert isinstance(report, dict), "没有产出报告"
    exp = independent_fit(data)
    gain = float(report["steady_gain_c_per_pct"])
    tau = float(report["time_constant_s"])
    delay = float(report["dead_time_s"])
    assert abs(gain - exp["steady_gain_c_per_pct"]) <= 0.0015 * abs(exp["steady_gain_c_per_pct"]), \
        "换记录后增益偏了：%r 对 %r" % (gain, exp["steady_gain_c_per_pct"])
    assert abs(tau - exp["time_constant_s"]) <= 0.003 * abs(exp["time_constant_s"]), \
        "换记录后时间常数偏了：%r 对 %r" % (tau, exp["time_constant_s"])
    assert abs(delay - exp["dead_time_s"]) <= 0.5, \
        "换记录后纯滞后偏了：%r 对 %r" % (delay, exp["dead_time_s"])


def test_short_delay_run_still_estimated():
    """题面原文：「θ 与 τ 用残差平方和最小的方式估出来（先在网格上搜、再细化）」。

    延迟真值 6 s：估计值与独立复算相差不超过 0.5 秒，且不得报 0。
    """
    data = make_run(delay=6.0, tau=90.0, seed=503, frozen_from=500)
    report, _ = run_case(data)
    assert isinstance(report, dict), "没有产出报告"
    exp = independent_fit(data)
    got = float(report["dead_time_s"])
    assert abs(got - exp["dead_time_s"]) <= 0.5, \
        "短延迟与独立复算差得太多：被测 %r，复算 %r" % (got, exp["dead_time_s"])
    assert got > 0.5, "短延迟被当成零了：%r" % got


def _bad_records():
    """四类坏记录：点太少、时间不单调、没有阶跃、两次跃变。"""
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
    return [("采样点不足", few), ("时间不单调", nonmono),
            ("没有阶跃", nostep), ("两次跃变", twostep)]


@pytest.mark.parametrize("label,data", _bad_records())
def test_invalid_runs_rejected_without_output(label, data):
    """题面原文：「点数不足一百、时间倒序或为负、找不到阶跃、出现多次跃变、可用点太少、全部被判卡死，这些情况返回 2 且不写模型卡」。

    逐类坏记录跑一遍：退出码 2、stderr 有原因、不产出报告、不留临时文件。
    """
    tmp = tempfile.mkdtemp(prefix="id-bad-")
    rpath = os.path.join(tmp, "run.json")
    opath = os.path.join(tmp, "report.json")
    with open(rpath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    proc = _run(rpath, opath)
    assert proc.returncode == 2, "%s：退出码应为 2，实际 %r" % (label, proc.returncode)
    assert (proc.stderr or "").strip(), "%s：stderr 没有写原因" % label
    assert not os.path.exists(opath), "%s：不该产出报告文件" % label
    assert not [n for n in os.listdir(tmp) if n.endswith(".tmp")], "%s：留下临时文件" % label


def test_rmse_reflects_fit_quality():
    """题面原文：「报告给出拟合窗口内（剔除卡死点后）的残差均方根」。

    标准台架记录噪声约 0.012：rmse 不超过 0.03，且与独立复算相差不超过 30%。
    """
    data = make_run(seed=701)
    report, _ = run_case(data)
    assert isinstance(report, dict), "没有产出报告"
    exp = independent_fit(data)
    got = float(report["model_rms_c"])
    assert got <= 0.03, "残差均方根过大：%r" % got
    assert abs(got - exp["model_rms_c"]) <= 0.30 * exp["model_rms_c"], \
        "残差均方根与独立复算差得太多：%r 对 %r" % (got, exp["model_rms_c"])


def test_report_schema_and_rounding():
    """题面原文：「增益三位、时间常数一位、纯滞后一位、残差三位，位数错了按不通过算」。

    校验顶层字段、位数与有限性，文件里不出现 NaN 或 Infinity。
    """
    data = make_run(seed=809)
    report, ctx = run_case(data)
    assert isinstance(report, dict), "没有产出报告"
    need = {"recorded_points", "fitted_points", "skipped_frozen", "pre_step_level_c",
            "step_time_s", "step_size_pct", "steady_gain_c_per_pct", "time_constant_s",
            "dead_time_s", "model_rms_c", "model_note"}
    assert need.issubset(set(report)), "报告缺字段：%r" % (need - set(report))
    for key in ("pre_step_level_c", "step_size_pct", "steady_gain_c_per_pct"):
        v = float(report[key])
        assert math.isfinite(v), "%s 不是有限值" % key
        assert abs(v - round(v, 3)) <= 1e-12, "%s 不是三位小数：%r" % (key, v)
    for key in ("time_constant_s", "dead_time_s", "step_time_s"):
        v = float(report[key])
        assert math.isfinite(v), "%s 不是有限值" % key
        assert abs(v - round(v, 1)) <= 1e-12, "%s 不是一位小数：%r" % (key, v)
    v = float(report["model_rms_c"])
    assert math.isfinite(v) and abs(v - round(v, 4)) <= 1e-12, "model_rms_c 不是四位小数：%r" % v
    raw = open(ctx["out"], encoding="utf-8").read()
    assert "NaN" not in raw and "Infinity" not in raw, "报告里出现了 NaN 或 Infinity"


def test_rerun_produces_identical_bytes():
    """题面原文：「重复跑同一份实验记录，模型卡应逐字节相同，不要写入运行时刻之类的信息」。

    连跑两次，两个报告文件字节完全相同。
    """
    data = make_run(seed=907)
    first, ctx1 = run_case(data)
    assert isinstance(first, dict), "第一次运行没有产出报告"
    for key in ("time_constant_s", "dead_time_s", "steady_gain_c_per_pct"):
        assert math.isfinite(float(first[key])), "第一次运行的 %s 不是有限值" % key
    body1 = open(ctx1["out"], "rb").read()
    second, ctx2 = run_case(data)
    assert second is not None, "第二次运行没有产出报告"
    assert body1 == open(ctx2["out"], "rb").read(), "两次运行报告字节不一致"


def test_failed_run_keeps_previous_report():
    """题面原文：「一旦这次失败，先前写好的模型卡不能被冲掉」。

    先跑一次合格记录留下报告，再用没有阶跃的坏记录跑：退出码 2，旧报告字节不变。
    """
    good = make_run(seed=1009)
    tmp = tempfile.mkdtemp(prefix="id-keep-")
    rpath = os.path.join(tmp, "run.json")
    opath = os.path.join(tmp, "report.json")
    with open(rpath, "w", encoding="utf-8") as f:
        json.dump(good, f, ensure_ascii=False)
    first = _run(rpath, opath)
    assert first.returncode == 0, "合格记录应退出 0，实际 %r" % first.returncode
    assert os.path.exists(opath), "合格记录没有产出报告"
    before = open(opath, "rb").read()
    broken = json.loads(json.dumps(good))
    for s in broken["samples"]:
        s["u"] = 20.0
    with open(rpath, "w", encoding="utf-8") as f:
        json.dump(broken, f, ensure_ascii=False)
    second = _run(rpath, opath)
    assert second.returncode == 2, "坏记录应退出 2，实际 %r" % second.returncode
    assert open(opath, "rb").read() == before, "运行失败把上一次的报告覆盖了"
