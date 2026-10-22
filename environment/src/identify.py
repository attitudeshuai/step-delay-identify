#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""台阶跃响应辨识：从试验记录里估增益、时间常数与纯延迟。

用法：
    python3 src/identify.py --run data/run.json --out out/report.json
"""
import argparse
import datetime as dt
import json
import math
import os


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def analyse(raw):
    samples = raw["samples"]
    times = [float(s["t"]) for s in samples]
    us = [float(s["u"]) for s in samples]
    ys = [float(s["y"]) for s in samples]

    step = next(i for i in range(1, len(us)) if us[i] != us[i - 1])
    t_step = times[step - 1]
    du = us[step] - us[step - 1]

    base = ys[step - 1]
    steady = ys[-1]
    rise = steady - base
    gain = rise / du

    target = base + 0.632 * rise
    i63 = next(i for i, y in enumerate(ys) if i > step and y >= target)
    tau = times[i63] - t_step

    resid = []
    for i in range(step, len(ys)):
        model = base + rise * (1.0 - math.exp(-(times[i] - t_step) / tau))
        resid.append(ys[i] - model)
    rmse = math.sqrt(sum(r * r for r in resid) / len(resid))

    return {
        "n_samples": len(samples),
        "baseline_c": base,
        "step_at_s": t_step,
        "delta_u": du,
        "gain_c_per_pct": gain,
        "tau_s": tau,
        "delay_s": 0.0,
        "rmse_c": rmse,
        "generated_at": dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "meta": {"model": "fopdt"},
    }


def main():
    ap = argparse.ArgumentParser(description="台阶跃响应辨识")
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw = load(args.run)
    report = analyse(raw)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
