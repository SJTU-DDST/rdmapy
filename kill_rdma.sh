#!/bin/bash

# 查找并杀掉 client.py 和 server.py 进程
for script in client_mr_test.py server_mr_test.py client.py run_all.sh run_baseline.sh test_rw.py run_sequoia.py PromptDatabaseServer.py run_baseline.py test_specinfer.py; do
    PIDS=$(ps -ef | grep "$script" | grep -v "grep" | awk '{print $2}')
    if [ -z "$PIDS" ]; then
        echo "$script 未运行"
    else
        echo "正在终止 $script (PID: $PIDS)"
        kill $PIDS 2>/dev/null
        sleep 1
        # 如果还没杀掉就强制kill
        ps -p $PIDS > /dev/null 2>&1 && kill -9 $PIDS && echo "$script 已强制终止"
    fi
done
