"""Celery application for the PDF RAG pipeline.

Broker:  RabbitMQ  (see celery_config.py)
Workers: celery -A celery_app worker --loglevel=info

Lifecycle logging (Queue received / worker started / success / failure) is
wired through Celery signals so every environment — Flask, worker, Docker —
emits consistent, searchable log lines.
"""

import logging

from celery import Celery  # type: ignore[import-untyped]
from celery.signals import (  # type: ignore[import-untyped]
    task_failure,
    task_postrun,
    task_prerun,
    worker_process_init,
    worker_ready,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

celery_app = Celery("pdf_rag")
celery_app.config_from_object("celery_config")

celery_app.conf.update(
    task_default_queue="pdf_processing",
    task_default_exchange="pdf_processing",
    task_default_routing_key="pdf_processing",
)


@worker_process_init.connect
def on_worker_process_init(**kwargs):
    logger.info("[Celery] Worker process started (pid=%s)",
                kwargs.get("sender"))


@worker_ready.connect
def on_worker_ready(sender, **kwargs):
    logger.info("[Celery] Worker %s ready — consuming from RabbitMQ", sender)


@task_prerun.connect
def on_task_prerun(sender, task_id, task, args, kwargs, **kw):
    job_id = (args[0] if args else None) or ((kwargs or {}).get("job_id"))
    logger.info("[Queue] Received task %s (task_id=%s, job_id=%s)",
                task.name, task_id, job_id)


@task_postrun.connect
def on_task_postrun(sender, task_id, task, args, kwargs, retval, **kw):
    job_id = (args[0] if args else None) or ((kwargs or {}).get("job_id"))
    logger.info("[Celery] Task succeeded %s (task_id=%s, job_id=%s)",
                task.name, task_id, job_id)


@task_failure.connect
def on_task_failure(sender, task_id, exception, args, kwargs, traceback, **kw):
    job_id = (args[0] if args else None) or ((kwargs or {}).get("job_id"))
    task_name = getattr(sender, "name", None) or str(sender)
    logger.error(
        "[Celery] Task FAILED %s (task_id=%s, job_id=%s): %s",
        task_name, task_id, job_id, exception, exc_info=traceback,  # type: ignore[arg-type]
    )
