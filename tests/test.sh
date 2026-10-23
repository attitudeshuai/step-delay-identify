#!/bin/bash
# 评分脚本：reward 只写 /logs/verifier/reward.txt；开头先覆盖为 0，崩溃/超时/信号都归零。
# 退出前再补一次 fail-closed（DONE 未置位说明没走到正常收尾），最后把 reward 置为只读。
set -u

# 评分脚本：先把评分环境隔离（不让被测目录里同名模块劫持 import），再跑测试。
#   · cd / 且 python3 -P：不让工作目录进 sys.path，挡住 /app/pytest.py 这类同名劫持
#   · env -u PYTHON*：挡住 PYTHONPATH / sitecustomize 注入
#   · 解释器走绝对路径：挡住 PATH 上的同名 python3

mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt
DONE=0
trap '[ "$DONE" = 1 ] || echo 0 > /logs/verifier/reward.txt' EXIT
trap 'exit 1' INT TERM HUP

cd /

SCORE=$(timeout 240 env -u PYTHONPATH -u PYTHONHOME -u PYTHONSTARTUP /usr/local/bin/python3 -P /tests/test_outputs.py) || SCORE=0
case "$SCORE" in
  ''|*[!0-9.]*) SCORE=0 ;;
esac
echo "$SCORE" > /logs/verifier/reward.txt
DONE=1
chmod 0444 /logs/verifier/reward.txt
