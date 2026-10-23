"""Every Prometheus metric; each process (API, CPU and GPU worker) serves the full set."""

import logging

import rq
from prometheus_client import Counter, Gauge, Histogram
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

from eroll.extraction.validators import LOW_CONFIDENCE, VALIDATOR_NAMES
from eroll.pages.classify import OUTCOME_CLIP_NON_VOTER, OUTCOME_CLIP_VOTER, OUTCOME_QWEN_FALLBACK

log = logging.getLogger(__name__)

JOB_FINAL_STATUSES = ("completed", "failed")
PAGE_OUTCOMES = (OUTCOME_CLIP_VOTER, OUTCOME_CLIP_NON_VOTER, OUTCOME_QWEN_FALLBACK)
FLAG_REASONS = (*VALIDATOR_NAMES, LOW_CONFIDENCE)
STAGES = ("cpu_stage", "gpu_stage", "clip", "fallback", "extract_page")
GPU_STATES = ("busy", "idle")

# Dense near 0: measured mean token log probabilities range -0.0190 to -0.0002.
CONFIDENCE_BUCKETS = (-1, -0.5, -0.2, -0.1, -0.05, -0.02, -0.01, -0.005, -0.002, -0.001, 0)
MARGIN_BUCKETS = tuple(step / 10 for step in range(11))  # 0.0, 0.1 ... 1.0
# Covers sub-second steps up to whole stages: extract_page is ~8 s (30 boxes at
# ~0.25 s/record); a 1,100-page gpu_stage is ~2.2 h.
STAGE_BUCKETS = (0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 300, 900, 3600, 10800, 21600)

JOBS_SUBMITTED = Counter("eroll_jobs_submitted_total", "CPU jobs enqueued by POST /jobs")
JOBS_FINISHED = Counter("eroll_jobs_finished_total", "Jobs ended, by final status", ["status"])
RECORDS_EXTRACTED = Counter("eroll_records_extracted_total", "Records stored by the GPU worker")
PAGE_DECISIONS = Counter(
    "eroll_page_decisions_total", "Cascade decisions by page outcome", ["outcome"]
)
CLIP_MARGIN = Histogram("eroll_clip_margin", "CLIP margin per page", buckets=MARGIN_BUCKETS)
RECORD_CONFIDENCE = Histogram(
    "eroll_record_confidence",
    "Mean token log probability per stored Record",
    buckets=CONFIDENCE_BUCKETS,
)
FLAGGED_RECORDS = Counter(
    "eroll_flagged_records_total", "Flag reasons set when a Job is finalised", ["reason"]
)
STAGE_SECONDS = Histogram(
    "eroll_stage_seconds", "Wall time by stage", ["stage"], buckets=STAGE_BUCKETS
)
GPU_SECONDS = Counter("eroll_gpu_seconds_total", "GPU worker seconds by state", ["state"])
HOURLY_USD = Gauge("eroll_hourly_usd", "Configured GPU price, USD per hour")

# Labels come only from these fixed sets (never job ids or field values), and are
# created up front so every series starts at 0.
for _status in JOB_FINAL_STATUSES:
    JOBS_FINISHED.labels(_status)
for _outcome in PAGE_OUTCOMES:
    PAGE_DECISIONS.labels(_outcome)
for _reason in FLAG_REASONS:
    FLAGGED_RECORDS.labels(_reason)
for _stage in STAGES:
    STAGE_SECONDS.labels(_stage)
for _state in GPU_STATES:
    GPU_SECONDS.labels(_state)

QUEUE_DEPTH_NAME = "eroll_queue_depth"
QUEUE_DEPTH_HELP = "Jobs waiting in each RQ queue"


class QueueDepthCollector(Collector):
    """Reads each RQ queue's length from Redis on every scrape."""

    def __init__(self, connection: Redis, queue_names: tuple[str, ...]) -> None:
        self.connection = connection
        self.queue_names = queue_names

    def describe(self):
        yield GaugeMetricFamily(QUEUE_DEPTH_NAME, QUEUE_DEPTH_HELP, labels=["queue"])

    def collect(self):
        try:
            depths = [
                (name, rq.Queue(name, connection=self.connection).count)
                for name in self.queue_names
            ]
        except RedisConnectionError as exc:
            log.warning("queue depth not collected: Redis unreachable (%s)", type(exc).__name__)
            return
        family = GaugeMetricFamily(QUEUE_DEPTH_NAME, QUEUE_DEPTH_HELP, labels=["queue"])
        for name, depth in depths:
            family.add_metric([name], depth)
        yield family
