"""ClearML tracking integration for news_google_challenge.

Follows the clearml-labkit conventions:
- early_init() called before ArgumentParser.parse_args()
- finalize_tracker() called after ArgumentParser.parse_args()
- write_metrics_json() adheres to the labkit registry contract
- resolve_clearml_dataset() resolves dataset paths seamlessly
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Iterable, Optional


class NullTracker:
    """No-op tracker used whenever --clearml is not set."""

    def report_scalar(self, title: str, series: str, value: float, iteration: int) -> None:
        pass

    def report_table(self, title: str, series: str, table: Any, iteration: int = 0) -> None:
        pass

    def upload_artifact(self, name: str, artifact_object: Any) -> None:
        pass

    def connect_configuration(self, config: dict, name: str = "General") -> None:
        pass

    def close(self) -> None:
        pass


class ClearMLTracker:
    """Wraps a clearml.Task, forwarding to its Logger and artifact APIs."""

    def __init__(self, task: Any) -> None:
        self.task = task
        self.logger = task.get_logger()

    def report_scalar(self, title: str, series: str, value: float, iteration: int) -> None:
        self.logger.report_scalar(title=title, series=series, value=value, iteration=iteration)

    def report_table(self, title: str, series: str, table: Any, iteration: int = 0) -> None:
        self.logger.report_table(title=title, series=series, iteration=iteration, table_plot=table)

    def upload_artifact(self, name: str, artifact_object: Any) -> None:
        self.task.upload_artifact(name=name, artifact_object=artifact_object)

    def connect_configuration(self, config: dict, name: str = "General") -> None:
        self.task.connect(config, name=name)

    def close(self) -> None:
        self.task.close()


def add_clearml_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("ClearML")
    group.add_argument(
        "--clearml", action=argparse.BooleanOptionalAction, default=False,
        help="Enable ClearML experiment tracking.",
    )
    group.add_argument(
        "--clearml_project", type=str, default="news_google_challenge",
        help="ClearML project name.",
    )
    group.add_argument(
        "--clearml_task_name", type=str, default=None,
        help="ClearML task name. Defaults to '<script>_<timestamp>'.",
    )
    group.add_argument(
        "--remote_queue", type=str, default=None,
        help="Enqueue this run onto a remote ClearML queue (e.g. 'seneca1').",
    )
    group.add_argument(
        "--clearml_dataset", type=str, default=None,
        help="Optional ClearML Dataset name to resolve data paths from.",
    )
    group.add_argument(
        "--clearml_template_only", action="store_true", default=False,
        help="Register hyperparameters with ClearML and exit immediately without running work.",
    )
    group.add_argument(
        "--clearml_tags", type=str, default=None,
        help="Comma-separated tags to set on task immediately after Task.init().",
    )
    group.add_argument(
        "--clearml_comment", type=str, default=None,
        help="Task comment to set immediately after Task.init().",
    )


def _running_under_agent() -> bool:
    return bool(os.environ.get("CLEARML_TASK_ID") or os.environ.get("TRAINS_TASK_ID"))


def early_init(argv: Optional[list] = None, task_name: Optional[str] = None) -> Any:
    """Call BEFORE parse_args() as the very first thing in main()."""
    pre = argparse.ArgumentParser(add_help=False)
    add_clearml_args(pre)
    ns, _ = pre.parse_known_args(argv)

    if not ns.clearml and not _running_under_agent():
        return None

    try:
        from clearml import Task
    except ImportError:
        print("[tracking] ClearML requested but not installed.")
        return None

    try:
        from clearml.backend_interface.task.repo.scriptinfo import ScriptInfo
        ScriptInfo.max_diff_size_bytes = 20_000_000
    except ImportError:
        pass

    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    resolved_name = task_name or ns.clearml_task_name or f"{script_name}_{datetime.now():%Y%m%d_%H%M%S}"
    task = Task.init(
        project_name=ns.clearml_project,
        task_name=resolved_name,
        auto_connect_frameworks={"pytorch": False},
    )
    if os.path.exists("requirements.txt"):
        task.set_packages("./requirements.txt")
    if ns.clearml_tags:
        task.set_tags([t.strip() for t in ns.clearml_tags.split(",") if t.strip()])
    if ns.clearml_comment:
        task.set_comment(ns.clearml_comment)
    return task


def finalize_tracker(task: Any, args: argparse.Namespace) -> Any:
    """Call AFTER parse_args(), with the Task returned by early_init()."""
    if task is None:
        return NullTracker()

    if getattr(args, "clearml_template_only", False):
        print("[tracking] --clearml_template_only set; hyperparameters registered, exiting.")
        task.close()
        sys.exit(0)

    remote_queue = getattr(args, "remote_queue", None)
    if remote_queue and not _running_under_agent():
        print(f"[tracking] Enqueueing to remote queue '{remote_queue}' and exiting local process.")
        task.execute_remotely(queue_name=remote_queue, exit_process=True)

    return ClearMLTracker(task)


def _find_in_dataset(local_root: str, basename: str) -> Optional[str]:
    root_candidate = os.path.join(local_root, basename)
    if os.path.exists(root_candidate):
        return root_candidate
    for dirpath, _dirs, files in os.walk(local_root):
        if basename in files:
            return os.path.join(dirpath, basename)
    return None


def resolve_clearml_dataset(
    args: argparse.Namespace,
    path_fields: Iterable[str] = ("train_file", "test_file", "folds_file"),
) -> None:
    """If --clearml_dataset is set, rewrite path_fields to point at the dataset's local cache."""
    dataset_name = getattr(args, "clearml_dataset", None)
    if not dataset_name:
        return

    from clearml import Dataset

    try:
        local_root = Dataset.get(dataset_name=dataset_name).get_local_copy()
        for field in path_fields:
            current = getattr(args, field, None)
            if not current:
                continue
            basename = os.path.basename(current)
            candidate = _find_in_dataset(local_root, basename)
            if candidate is not None:
                setattr(args, field, candidate)
                print(f"[tracking] Resolved {field} -> {candidate}")
            else:
                print(f"[tracking] Warning: {basename} not found in dataset {dataset_name}")
    except Exception as e:
        print(f"[tracking] Error resolving clearml dataset {dataset_name}: {e}")


def fetch_task_artifact(task_id: str, artifact_name: str) -> str:
    """Return local path to a named artifact uploaded by a previous ClearML task."""
    from clearml import Task
    task = Task.get_task(task_id=task_id)
    return task.artifacts[artifact_name].get_local_copy()


def write_metrics_json(
    output_dir: str,
    results: dict[str, float],
    split: str = "val",
    level: str = "document",
    task: Optional[Any] = None,
) -> str:
    """Write metrics.json conforming to the clearml-labkit registry contract."""
    campaign_id = condition = seed = None
    if task is not None and hasattr(task, "get_tags"):
        for tag in task.get_tags() or []:
            if tag.startswith("campaign:"):
                campaign_id = tag.split(":", 1)[1]
            elif tag.startswith("condition:"):
                condition = tag.split(":", 1)[1]
            elif tag.startswith("seed:"):
                seed = tag.split(":", 1)[1]

    payload = {
        "campaign_id": campaign_id,
        "condition": condition,
        "seed": seed,
        "level": level,
        "split": split,
        "f1": results.get("mean_f1", results.get("f1")),
        "qatar_f1": results.get("Qatar Related"),
        "geography_f1": results.get("Geography"),
        "politics_f1": results.get("Politics & Conflict"),
        "health_f1": results.get("Health & Wellbeing"),
        "science_f1": results.get("Science"),
        "sports_f1": results.get("Sports"),
    }
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "metrics.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)

    if task is not None and hasattr(task, "upload_artifact"):
        try:
            task.upload_artifact("metrics_json", artifact_object=path)
        except Exception:
            pass

    return path
