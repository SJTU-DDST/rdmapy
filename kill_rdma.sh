#!/bin/bash

# 查找并杀掉 client.py 和 server.py 进程
for script in client_mr_test.py server_mr_test.py client.py server.py; do
    PIDS=$(ps -ef | grep "examples/$script" | grep -v "grep" | awk '{print $2}')
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
