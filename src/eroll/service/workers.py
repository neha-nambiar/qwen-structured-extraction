"""Worker entry points: python -m eroll.service.workers cpu|gpu."""

import http.client
import logging
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import openai
import rq
from openai import OpenAI
from prometheus_client import start_http_server
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from eroll.extraction.qwen import MIN_PROBE_PROMPT_TOKENS, probe
from eroll.pages.classify import load_clip
from eroll.service import jobs
from eroll.service.db import check_head
from eroll.service.metrics import GPU_SECONDS, HOURLY_USD
from eroll.settings import Settings, load_settings, require

log = logging.getLogger(__name__)

ALEMBIC_INI = Path("alembic.ini")
CPU_QUEUE = "cpu"
GPU_QUEUE = "gpu"
USAGE = "usage: python -m eroll.service.workers cpu|gpu"

CPU_METRICS_PORT, GPU_METRICS_PORT = 9101, 9102
# A pod with no job for 15 continuous minutes deletes itself.
IDLE_TERMINATE_MINUTES = 15
IDLE_CHECK_SECONDS = 60
HEALTH_TIMEOUT_SECONDS = 5
DELETE_TIMEOUT_SECONDS = 30
RUNPOD_PODS_URL = "https://rest.runpod.io/v1/pods"
DELETE_USER_AGENT = "eroll-gpu-worker"


def cpu_main(settings: Settings) -> None:
    require(settings, "database_url", "redis_url")
    engine = create_engine(settings.database_url.get_secret_value())
    check_head(engine, ALEMBIC_INI)

    redis = Redis.from_url(settings.redis_url.get_secret_value())
    clip = load_clip()
    log.info("CLIP loaded")

    jobs.bind(
        jobs.Context(
            sessions=sessionmaker(engine),
            gpu_queue=rq.Queue(GPU_QUEUE, connection=redis),
            clip=clip,
            llm=None,
            vllm_model=settings.vllm_model,
            max_concurrent_requests=settings.max_concurrent_requests,
            confidence_threshold=settings.confidence_threshold,
        )
    )
    # Compose publishes no port for this, so only Prometheus on the compose network sees it.
    start_http_server(CPU_METRICS_PORT, addr="0.0.0.0")
    # SimpleWorker runs jobs without forking, so CLIP stays loaded.
    rq.SimpleWorker([rq.Queue(CPU_QUEUE, connection=redis)], connection=redis).work()


def idle_since(process_start: float, last_busy_end: float | None) -> float:
    """Start of the current idle span: the later of process start and the last job's end."""
    return process_start if last_busy_end is None else max(process_start, last_busy_end)


def should_terminate(idle_since: float, now: float, busy: bool) -> bool:
    return not busy and now - idle_since >= IDLE_TERMINATE_MINUTES * 60


def delete_pod(pod_id: str, api_key: str) -> int | None:
    """DELETE the pod through the RunPod REST API; the HTTP status, None on no response."""
    request = urllib.request.Request(
        f"{RUNPOD_PODS_URL}/{urllib.parse.quote(pod_id, safe='')}",
        method="DELETE",
        headers={"Authorization": f"Bearer {api_key}", "User-Agent": DELETE_USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=DELETE_TIMEOUT_SECONDS) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except (OSError, http.client.HTTPException) as exc:
        log.warning("delete pod: no response (%s)", type(exc).__name__)
        return None


class BurstWorker(rq.SimpleWorker):
    """A SimpleWorker whose dequeue gives up on a lost Redis connection."""

    # RQ retries a lost connection forever, which would freeze the idle timer and keep
    # a disconnected pod alive; one non-blocking pop lets gpu_main count it as idle.

    def dequeue_job_and_maintain_ttl(self, timeout, max_idle_time=None):
        self.heartbeat()
        return self.queue_class.dequeue_any(
            self._ordered_queues,
            None,
            connection=self.connection,
            job_class=self.job_class,
            serializer=self.serializer,
            death_penalty_class=self.death_penalty_class,
        )


def _vllm_healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=HEALTH_TIMEOUT_SECONDS) as response:
            return response.status == 200
    except (OSError, http.client.HTTPException):
        return False


def gpu_main(settings: Settings) -> None:
    require(settings, "database_url", "redis_url", "runpod_pod_id", "runpod_api_key")
    pod_id = settings.runpod_pod_id
    runpod_key = settings.runpod_api_key.get_secret_value()

    # Pod loopback only; Prometheus reaches it through the SSH tunnel (-L 19102).
    start_http_server(GPU_METRICS_PORT, addr="127.0.0.1")
    HOURLY_USD.set(settings.hourly_usd)

    engine = create_engine(settings.database_url.get_secret_value())
    redis = Redis.from_url(settings.redis_url.get_secret_value())
    # No retries: a failed request fails the job instead of being retried silently.
    client = OpenAI(
        base_url=settings.vllm_base_url,
        api_key="unused",
        max_retries=0,
        timeout=settings.request_timeout_seconds,
    )
    jobs.bind(
        jobs.Context(
            sessions=sessionmaker(engine),
            gpu_queue=None,
            clip=None,
            llm=client,
            vllm_model=settings.vllm_model,
            max_concurrent_requests=settings.max_concurrent_requests,
            confidence_threshold=settings.confidence_threshold,
        )
    )
    gpu_queue = rq.Queue(GPU_QUEUE, connection=redis)
    # vLLM serves /health at the root, beside /v1.
    health_url = settings.vllm_base_url.removesuffix("/v1") + "/health"

    # Each minute: delete the pod after 15 idle minutes (counted from start, so vLLM
    # loading counts too), else probe vLLM once, else run queued jobs.
    process_start = time.monotonic()
    last_tick = process_start
    last_busy_end: float | None = None
    ready = False
    probe_failed = False
    head_checked = False
    while True:
        now = time.monotonic()
        GPU_SECONDS.labels("idle").inc(now - last_tick)
        last_tick = now

        if should_terminate(idle_since(process_start, last_busy_end), now, busy=False):
            status = delete_pod(pod_id, runpod_key)
            log.info("idle for %d minutes; delete pod: HTTP %s", IDLE_TERMINATE_MINUTES, status)
            if status is not None and 200 <= status < 300:
                return
        elif not ready:
            if not probe_failed and _vllm_healthy(health_url):
                try:
                    tokens = probe(client, settings.vllm_model)
                except openai.APIError as exc:
                    log.warning("probe failed (%s); retrying next check", type(exc).__name__)
                else:
                    if tokens < MIN_PROBE_PROMPT_TOKENS:
                        log.error(
                            "probe %d prompt tokens, below MIN_PROBE_PROMPT_TOKENS %d; no job",
                            tokens,
                            MIN_PROBE_PROMPT_TOKENS,
                        )
                        probe_failed = True
                    else:
                        log.info("vLLM ready: probe %d prompt tokens", tokens)
                        ready = True
        elif not _vllm_healthy(health_url):
            # vLLM may have died; nothing restarts it, so the pod idles out.
            log.warning("vLLM health check failed; taking no job")
        else:
            ran = False
            burst_start = time.monotonic()
            try:
                # Checked on first use, since Postgres is only reachable once the tunnel is up.
                if not head_checked:
                    check_head(engine, ALEMBIC_INI)
                    head_checked = True
                ran = BurstWorker([gpu_queue], connection=redis).work(burst=True)
            except RedisConnectionError:
                log.warning("Redis unreachable (tunnel closed?); counted idle")
            except OperationalError:
                log.warning("Postgres unreachable (tunnel closed?); counted idle")
            if ran:
                end = time.monotonic()
                GPU_SECONDS.labels("busy").inc(end - burst_start)
                last_busy_end = end
                last_tick = end
                continue
        time.sleep(IDLE_CHECK_SECONDS)


if __name__ == "__main__":
    mode = sys.argv[1:]
    if mode == ["cpu"]:
        cpu_main(load_settings())
    elif mode == ["gpu"]:
        gpu_main(load_settings())
    else:
        raise SystemExit(USAGE)
