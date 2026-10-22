#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成恒温台阶跃响应试验记录（FOPDT：一阶惯性 + 纯延迟 + 测量噪声 + 传感器冻结段）。

数据口径：业务模型 + 确定性生成规则（固定伪随机序列，不用真随机）——
  y(t) = y0 + K·Δu·(1 - exp(-(t - t_step - θ)/τ))，t ≥ t_step + θ；之前保持 y0。
  噪声是固定序列的均匀伪随机；另有一段传感器卡死（读数逐点完全相同）需要剔除。
  时间固定步长采样，整份记录可逐字节复现。
"""
import json
import math
import sys


class Lcg(object):
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def next_u(self):
        self.state = (1103515245 * self.state + 12345) & 0x7FFFFFFF
        return (self.state % 1000003) / 1000003.0


def make_run(gain=0.40, tau=120.0, delay=37.5, y0=25.0, u_before=20.0, u_after=70.0,
             dt=1.0, t_step=60.0, duration=900.0, noise=0.012, seed=20261024,
             frozen_from=400, frozen_len=20):
    rng = Lcg(seed)
    n = int(round(duration / dt)) + 1
    samples = []
    for i in range(n):
        t = round(i * dt, 1)
        u = u_after if t >= t_step else u_before
        dt_since = t - t_step - delay
        if dt_since <= 0:
            y = y0
        else:
            y = y0 + gain * (u_after - u_before) * (1.0 - math.exp(-dt_since / tau))
        y += (rng.next_u() - 0.5) * 2.0 * noise
        samples.append({"t": t, "u": u, "y": round(y, 4)})
    if frozen_len > 0:
        stuck = samples[frozen_from]["y"]
        for k in range(frozen_from, min(frozen_from + frozen_len, len(samples))):
            samples[k]["y"] = stuck

    return {
        "unit": {"time": "s", "temperature": "degC", "power": "%"},
        "samples": samples,
        "fit": {"baseline_window_s": 30.0, "horizon_s": 840.0, "frozen_run_min": 5},
    }


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "run.json"
    data = make_run()
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    print("wrote %s  samples=%d" % (out, len(data["samples"])))
