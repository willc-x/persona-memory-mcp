#!/bin/sh
set -eu

if [ "$#" -ne 1 ] || [ ! -f "$1" ]; then
    echo "LLDB capture module unavailable" >&2
    exit 2
fi
capture_module=$1
lldb_path=$(/usr/bin/xcrun --find lldb)
attempt=0
while [ "$attempt" -lt 1800 ]; do
    chat_pid=$(/usr/bin/pgrep -x Chat | /usr/bin/head -n 1 || true)
    if [ -n "$chat_pid" ]; then
        echo "CHAT_PID $chat_pid"
        "$lldb_path" \
            -p "$chat_pid" -b \
            -o "command script import \"$capture_module\"" \
            -o chat-key-install -o continue -o chat-key-emit -o detach -o quit &
        debugger_pid=$!
        (
            /bin/sleep 120
            /bin/kill -KILL "$debugger_pid" 2>/dev/null || true
            /bin/kill -CONT "$chat_pid" 2>/dev/null || true
        ) >/dev/null 2>&1 &
        watchdog_pid=$!
        status=0
        wait "$debugger_pid" || status=$?
        /bin/kill "$watchdog_pid" 2>/dev/null || true
        /bin/kill -CONT "$chat_pid" 2>/dev/null || true
        exit "$status"
    fi
    /bin/sleep 0.1
    attempt=$((attempt + 1))
done

echo "Chat process did not start in time" >&2
exit 2
