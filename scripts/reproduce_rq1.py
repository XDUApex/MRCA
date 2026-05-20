#!/usr/bin/env python3
"""Run the canonical MRCA RQ1 reproduction plan.

The plan is intentionally data-driven. Dataset-specific execution profiles live
in configs/final_rq1_mrca.json so the public artifact has one stable entrypoint
instead of a collection of historical tuning commands.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "final_rq1_mrca.json"
DEFAULT_MANIFEST = ROOT / "result" / "final_submission" / "final_reproduction_manifest.json"


def load_plan(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def split_csv(value: Optional[str]) -> Optional[set]:
    if not value:
        return None
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def select_runs(plan: Dict, datasets: Optional[set], modalities: Optional[set], stage: str) -> List[Dict]:
    selected = []
    for run in plan["runs"]:
        if datasets and run["dataset"].lower() not in datasets:
            continue
        if modalities and run["modality"].lower() not in modalities:
            continue
        if stage != "all" and run["stage"] != stage:
            continue
        selected.append(run)
    return selected


def command_to_text(command: List[str], env_delta: Dict[str, str]) -> str:
    prefix = " ".join(f"{key}={value}" for key, value in sorted(env_delta.items()))
    body = " ".join(command)
    return f"{prefix} {body}".strip()


def stage1_command(plan: Dict, run: Dict) -> List[str]:
    return [
        sys.executable,
        "-m",
        plan["stage1_profile"]["entrypoint"],
        "--dataset",
        run["dataset"],
        "--modality",
        run["stage1_modality"],
        "--experiment",
        plan.get("experiment", "rq1"),
        "--variant",
        run.get("variant", "base"),
    ]


def merged_profile(plan: Dict, run: Dict) -> Dict:
    profile = dict(plan["stage2_profiles"][run["profile"]])
    for key in ("top_percentage", "window_minutes", "max_lag", "scoring", "fusion_alpha", "fusion_beta"):
        if key in run:
            profile[key] = run[key]
    env = dict(profile.get("env", {}))
    env.update(run.get("env", {}))
    profile["env"] = env
    return profile


def stage2_command(plan: Dict, run: Dict, profile: Dict) -> List[str]:
    command = [
        sys.executable,
        "-m",
        profile["entrypoint"],
        "--dataset",
        run["dataset"],
        "--modality-combo",
        run["modality_combo"],
        "--experiment",
        plan.get("experiment", "rq1"),
        "--variant",
        run.get("variant", "base"),
    ]
    if profile.get("granger_only"):
        command.append("--granger-only")
        if profile.get("legacy"):
            command.append("--legacy")
        command.extend([
            "--window-minutes",
            str(profile.get("window_minutes", 10)),
            "--max-lag",
            str(profile.get("max_lag", 2)),
        ])
        return command

    command.extend([
        "--top-percentage",
        str(profile.get("top_percentage", 0.8)),
        "--fusion-alpha",
        str(profile.get("fusion_alpha", 0.5)),
        "--fusion-beta",
        str(profile.get("fusion_beta", 0.5)),
        "--window-minutes",
        str(profile.get("window_minutes", 10)),
        "--max-lag",
        str(profile.get("max_lag", 2)),
        "--scoring",
        profile.get("scoring", "min"),
        "--method",
        profile.get("method", "granger"),
    ])
    return command


def anomaly_root(dataset: str, variant: str) -> Path:
    name = dataset if variant == "base" else f"{dataset}-{variant}"
    return ROOT / "anomaly_detection" / "rq1_anomaly_score" / name


def required_stage1_modality(run: Dict) -> Optional[str]:
    combo = run.get("modality_combo")
    if combo == "all+metric":
        return "all"
    if combo == "log+metric":
        return "log"
    if combo == "trace+metric":
        return "trace"
    return run.get("stage1_modality")


def ensure_stage1_link(run: Dict, dry_run: bool) -> Optional[str]:
    source_variant = run.get("stage1_source_variant")
    target_variant = run.get("variant", "base")
    if not source_variant or source_variant == target_variant:
        return None

    modality = required_stage1_modality(run)
    if not modality:
        return None

    source = anomaly_root(run["dataset"], source_variant) / modality
    target = anomaly_root(run["dataset"], target_variant) / modality
    relative_source = os.path.relpath(source, target.parent)

    if target.is_symlink() and os.readlink(target) == relative_source:
        return f"stage1 link exists: {target.relative_to(ROOT)} -> {relative_source}"
    if target.exists() and not target.is_symlink():
        return f"stage1 link skipped, target exists: {target.relative_to(ROOT)}"

    if dry_run:
        return f"would link {target.relative_to(ROOT)} -> {relative_source}"

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        target.unlink()
    target.symlink_to(relative_source)
    return f"created stage1 link: {target.relative_to(ROOT)} -> {relative_source}"


def result_exists(run: Dict) -> bool:
    path = run.get("result_path")
    if not path:
        return False
    return (ROOT / path).exists()


def stage1_exists(run: Dict) -> bool:
    folder = anomaly_root(run["dataset"], run.get("variant", "base")) / run["stage1_modality"]
    return folder.exists() and any(folder.rglob("ranked_services_*.csv"))


def parse_pr_results(path: Path) -> Dict:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    parsed = {}
    cases = None
    for match in re.finditer(r"PR@(\d+):\s+([0-9.]+)%\s+\((\d+)/(\d+)\)", text):
        k, pct, hit, total = match.groups()
        hit_count = int(hit)
        total_count = int(total)
        parsed[f"hr_at_{k}"] = hit_count / total_count if total_count else float(pct) / 100.0
        parsed[f"hits_at_{k}"] = hit_count
        cases = total_count
    if cases is not None:
        parsed["cases"] = cases
    return parsed


def run_command(command: List[str], env_delta: Dict[str, str], dry_run: bool) -> int:
    if dry_run:
        return 0
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.update(env_delta)
    completed = subprocess.run(command, cwd=str(ROOT), env=env, check=False)
    return completed.returncode


def build_record(run: Dict, command: List[str], env_delta: Dict[str, str], status: str, note: str) -> Dict:
    record = {
        "dataset": run["dataset"],
        "modality": run["modality"],
        "stage": run["stage"],
        "command": command_to_text(command, env_delta),
        "env": env_delta,
        "status": status,
        "note": note,
        "expected": run.get("expected", {}),
    }
    if "result_path" in run:
        record["result_path"] = run["result_path"]
        record["observed"] = parse_pr_results(ROOT / run["result_path"])
    return record


def write_manifest(path: Path, config_path: Path, records: List[Dict], dry_run: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "config": str(config_path.relative_to(ROOT)),
        "records": records,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def print_plan(records: Iterable[Dict]) -> None:
    for record in records:
        print(f"[{record['status']}] {record['dataset']} {record['modality']} {record['stage']}")
        if record["env"]:
            print(f"  env: {record['env']}")
        print(f"  cmd: {record['command']}")
        if record["note"]:
            print(f"  note: {record['note']}")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the canonical MRCA RQ1 reproduction plan")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--datasets", help="Comma-separated dataset list, e.g. ob,tt,gaia,aiops")
    parser.add_argument("--modalities", help="Comma-separated modality list, e.g. L,T,TL,ML,TM,TML")
    parser.add_argument("--stage", choices=["stage1", "stage2", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without executing them")
    parser.add_argument("--skip-existing", action="store_true", help="Skip stage2 runs whose pr_results.txt already exists")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    plan = load_plan(args.config)
    datasets = split_csv(args.datasets)
    modalities = split_csv(args.modalities)
    selected = select_runs(plan, datasets, modalities, args.stage)

    records = []
    failed = False
    for run in selected:
        env_delta: Dict[str, str] = {}
        note_parts: List[str] = []
        if run["stage"] == "stage1":
            command = stage1_command(plan, run)
            if args.skip_existing and stage1_exists(run):
                records.append(build_record(run, command, env_delta, "skipped", "stage1 ranked service files already exist"))
                continue
        else:
            profile = merged_profile(plan, run)
            env_delta = profile.get("env", {})
            link_note = ensure_stage1_link(run, args.dry_run)
            if link_note:
                note_parts.append(link_note)
            command = stage2_command(plan, run, profile)
            if args.skip_existing and result_exists(run):
                records.append(build_record(run, command, env_delta, "skipped", "; ".join(note_parts)))
                continue

        print(f"[RUN] {run['dataset']} {run['modality']} {run['stage']}")
        if env_delta:
            print(f"      env: {env_delta}")
        print(f"      {' '.join(command)}")
        returncode = run_command(command, env_delta, args.dry_run)
        if returncode == 0:
            status = "planned" if args.dry_run else "completed"
        else:
            status = f"failed:{returncode}"
            failed = True
        records.append(build_record(run, command, env_delta, status, "; ".join(note_parts)))
        if failed:
            break

    write_manifest(args.manifest, args.config, records, args.dry_run)
    print(f"[INFO] Manifest written to {display_path(args.manifest)}")
    if args.dry_run:
        print_plan(records)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
