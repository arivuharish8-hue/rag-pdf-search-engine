"""Celery configuration for the PDF RAG pipeline.

RabbitMQ is the broker.  Settings are chosen for a fault-tolerant queue:

- ``task_acks_late``       : ack a message only after the task finishes, so a
                             worker crash redelivers the message and the task
                             resumes instead of being lost.
- ``task_reject_on_worker_lost`` : reject (requeue) messages when the worker
                             process dies mid-task.
- ``worker_prefetch_multiplier=1`` : one message per worker at a time, which
                             keeps a slow worker from hoarding the queue and
                             distributes work fairly between workers.
- ``task_soft_time_limit`` / ``task_time_limit`` : hard caps replace the old
                             manual extraction timeout.
"""

import os

from dotenv import load_dotenv
from kombu import Queue

load_dotenv()

# RabbitMQ connection.  "//" vhost is the default.
broker_url = os.getenv(
    "RABBITMQ_URL", "amqp://guest:guest@localhost:5672//"
)

# Results are not consumed by the app (progress lives in processing_jobs),
# but keep an rpc backend for introspection / retry metadata.
result_backend = os.getenv("CELERY_RESULT_BACKEND", "rpc://")
result_expires = int(os.getenv("CELERY_RESULT_EXPIRES", "3600"))

# Modules Celery must import to discover tasks.
include = ["tasks"]

# ── Delivery / reliability ────────────────────────────────────────────────
task_acks_late = True
task_reject_on_worker_lost = True
task_track_started = True
worker_prefetch_multiplier = 1
broker_connection_retry_on_startup = True

# ── Time limits (replaces the old EXTRACT_TIMEOUT thread trick) ───────────
task_time_limit = int(os.getenv("CELERY_TASK_TIME_LIMIT", "1800"))
task_soft_time_limit = int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "1500"))

# ── Queues ────────────────────────────────────────────────────────────────
task_queues = (
    Queue("pdf_processing", routing_key="pdf_processing"),
    Queue("celery", routing_key="celery"),
)
task_default_queue = "pdf_processing"
task_default_exchange = "pdf_processing"
task_default_routing_key = "pdf_processing"

# ── Serialization ─────────────────────────────────────────────────────────
# JSON only: Celery runs as a non-root user and must not need pickle on the
# broker.  Every task argument is a job_id string and every result is a
# string / plain dict, so JSON covers the whole pipeline.  The pickle usage
# in tasks.py / faiss_db.py is local on-disk cache only and stays untouched.
task_serializer = "json"
result_serializer = "json"
accept_content = ["json"]
