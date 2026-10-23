#!/usr/bin/env bash
# Pod entrypoint: sshd for the tunnel, vLLM on pod loopback, then one foreground process
# whose exit deletes the pod.

set -euo pipefail

POD_MODE="${POD_MODE:-worker}"
EVAL_LIFETIME_MINUTES="${EVAL_LIFETIME_MINUTES:-120}"
DELETE_RETRY_SECONDS=60

: "${PUBLIC_KEY:?PUBLIC_KEY is not set}"
: "${RUNPOD_POD_ID:?RUNPOD_POD_ID is not set}"
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY is not set}"
MODEL="${VLLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"

case "$POD_MODE" in
    worker | eval) ;;
    *) echo "POD_MODE must be worker or eval, got '$POD_MODE'" >&2; exit 2 ;;
esac

mkdir -p /root/.ssh /run/sshd
chmod 700 /root/.ssh
printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
ssh-keygen -A
/usr/sbin/sshd

# 127.0.0.1 only: vLLM runs without --api-key, so it must never leave the pod.
# min_pixels 802,816 forces ~790 vision tokens per crop; max_pixels 1,310,720 caps
# the fallback page at 1,280 tokens inside max-model-len 1,536.
# Keep-alive 600 s (default 5 s matches the client's), so the client always closes idle
# connections first and never reuses one the server just dropped.
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600
vllm serve "$MODEL" \
    --host 127.0.0.1 --port 8000 \
    --dtype bfloat16 \
    --max-model-len 1536 \
    --gpu-memory-utilization 0.92 \
    --limit-mm-per-prompt '{"image": 1}' \
    --mm-processor-kwargs '{"min_pixels": 802816, "max_pixels": 1310720}' \
    2>&1 | tee /var/log/vllm.log &

if [ "$POD_MODE" = worker ]; then
    python3 -m eroll.service.workers gpu || echo "gpu worker exited with status $?" >&2
else
    echo "eval mode: pod deletes itself in ${EVAL_LIFETIME_MINUTES} minutes" >&2
    sleep "$((EVAL_LIFETIME_MINUTES * 60))"
fi

while true; do
    status="$(printf 'header = "Authorization: Bearer %s"\n' "$RUNPOD_API_KEY" |
        curl --silent --output /dev/null --write-out '%{http_code}' --max-time 30 \
            --config - -X DELETE "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" || true)"
    echo "delete pod: HTTP ${status:-none}" >&2
    case "$status" in 2??) exit 0 ;; esac
    sleep "$DELETE_RETRY_SECONDS"
done
