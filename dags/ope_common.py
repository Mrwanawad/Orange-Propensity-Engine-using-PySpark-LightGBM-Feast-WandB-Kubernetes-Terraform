"""Shared task factory. Every task is `python -m <module> ...` in the ope-jobs image:
local runs use BashOperator, Kubernetes runs use KubernetesPodOperator with the same command.
Set OPE_EXECUTION=k8s on the scheduler to switch."""

from __future__ import annotations

import os

from airflow.providers.standard.operators.bash import BashOperator

EXECUTION = os.environ.get("OPE_EXECUTION", "local")
IMAGE = os.environ.get("OPE_JOBS_IMAGE", "ope-jobs:local")
NAMESPACE = os.environ.get("OPE_NAMESPACE", "ope")


def step(task_id: str, command: str, **kw):
    if EXECUTION == "k8s":
        from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator

        return KubernetesPodOperator(
            task_id=task_id,
            name=f"ope-{task_id.replace('_', '-')}",
            namespace=NAMESPACE,
            image=IMAGE,
            cmds=["bash", "-c"],
            arguments=[command],
            service_account_name="ope-jobs",
            env_from=[{"configMapRef": {"name": "ope-config"}}],
            get_logs=True,
            on_finish_action="delete_succeeded_pod",
            **kw,
        )
    return BashOperator(task_id=task_id, bash_command=f"cd /app && {command}", **kw)
