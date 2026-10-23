# -*- coding: utf-8 -*-
"""pytest 汇总钩子：按权重把每条测试的结果折算成加权分，写进 weighted.txt。

口径：每条测试占一个权重、和 = 1（见 WEIGHTS）；只有 PASS 才拿到该条权重；
不在 WEIGHTS 里的测试名不给分。这里只写中间结果，reward 由 test.sh 校验结构化报告后落。
"""
import os

WEIGHTS = {
    "test_gain_and_tau_match_independent_fit": 0.18,
    "test_delay_estimated_from_grid_search": 0.17,
    "test_frozen_segment_removed_from_fit": 0.135,
    "test_second_run_with_other_parameters": 0.12,
    "test_short_delay_run_still_estimated": 0.11,
    "test_rmse_reflects_fit_quality": 0.08,
    "test_report_schema_and_rounding": 0.08,
    "test_rerun_produces_identical_bytes": 0.065,
    "test_failed_run_keeps_previous_report": 0.06,
}

_RESULTS = {}


def pytest_runtest_logreport(report):
    if report.when in ("setup", "call") and report.failed:
        _RESULTS[report.nodeid.split("::")[-1]] = False
    elif report.when == "call" and report.passed:
        _RESULTS.setdefault(report.nodeid.split("::")[-1], True)


def pytest_sessionfinish(session, exitstatus):
    missing = [n for n in WEIGHTS if n not in _RESULTS]
    total = 0.0
    if missing:
        print("[INCOMPLETE] 这些测试没有跑完：%s；记 0" % ", ".join(missing))
    else:
        for name, weight in WEIGHTS.items():
            if _RESULTS.get(name) is True:
                total += weight
    path = os.environ.get("CT_WEIGHTED_PATH", "/logs/verifier/weighted.txt")
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("%.6f\n" % total)
