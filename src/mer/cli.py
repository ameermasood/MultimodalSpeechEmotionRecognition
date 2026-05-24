"""Command-line entry point for project workflows."""

from __future__ import annotations

import argparse
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class Workflow:
    """A script-backed workflow exposed through the project CLI."""

    script: str
    help: str


WORKFLOWS: dict[str, Workflow] = {
    "zero-shot-esd": Workflow(
        "scripts/evaluation/evaluate_zero_shot_esd.py",
        "Run zero-shot Voxtral evaluation on ESD.",
    ),
    "zero-shot-iemocap": Workflow(
        "scripts/evaluation/evaluate_zero_shot_iemocap.py",
        "Run zero-shot Voxtral evaluation on IEMOCAP.",
    ),
    "train-lora-esd": Workflow(
        "scripts/training/train_voxtral_lora_esd.py",
        "Fine-tune an audio-only LoRA adapter on ESD.",
    ),
    "train-dora-transcript-esd": Workflow(
        "scripts/training/train_voxtral_dora_transcript_esd.py",
        "Fine-tune a DoRA adapter on ESD with transcript input.",
    ),
    "evaluate-esd": Workflow(
        "scripts/evaluation/evaluate_esd_adapters.py",
        "Evaluate available PEFT adapters on ESD.",
    ),
    "evaluate-esd-dora": Workflow(
        "scripts/evaluation/evaluate_esd_dora_transcript.py",
        "Evaluate DoRA transcript adapters on ESD.",
    ),
    "evaluate-iemocap": Workflow(
        "scripts/evaluation/evaluate_iemocap_adapters.py",
        "Evaluate ESD-trained adapters on IEMOCAP.",
    ),
    "evaluate-iemocap-dora": Workflow(
        "scripts/evaluation/evaluate_iemocap_dora_transcript.py",
        "Evaluate DoRA transcript adapters on IEMOCAP.",
    ),
}


ALIASES = {
    "zs-esd": "zero-shot-esd",
    "zs-iemocap": "zero-shot-iemocap",
    "train-lora": "train-lora-esd",
    "train-dora": "train-dora-transcript-esd",
    "eval-esd": "evaluate-esd",
    "eval-esd-dora": "evaluate-esd-dora",
    "eval-iemocap": "evaluate-iemocap",
    "eval-iemocap-dora": "evaluate-iemocap-dora",
}


def project_root() -> Path:
    """Return the repository root for an editable or source checkout."""

    return Path(__file__).resolve().parents[2]


def run_script(script: str, args: Sequence[str]) -> None:
    """Run a repository script as if it were called directly."""

    script_path = project_root() / script
    if not script_path.is_file():
        raise SystemExit(f"Workflow script not found: {script_path}")

    previous_argv = sys.argv[:]
    try:
        sys.argv = [str(script_path), *args]
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = previous_argv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mer",
        description="Run multimodal speech emotion recognition workflows.",
    )
    parser.add_argument(
        "workflow",
        nargs="?",
        choices=sorted([*WORKFLOWS.keys(), *ALIASES.keys()]),
        help="Workflow to run. Use 'mer list' to show descriptions.",
    )
    parser.add_argument(
        "workflow_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the selected workflow.",
    )
    return parser


def print_workflows() -> None:
    print("Available workflows:\n")
    for name, workflow in WORKFLOWS.items():
        print(f"  {name:<28} {workflow.help}")
    print("\nShort aliases:\n")
    for alias, target in ALIASES.items():
        print(f"  {alias:<28} {target}")
    print("\nExamples:\n")
    print("  mer zero-shot-esd --help")
    print("  mer train-lora-esd --help")
    print("  mer evaluate-iemocap --help")


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"list", "workflows"}:
        print_workflows()
        return

    parser = build_parser()
    ns = parser.parse_args(argv)
    workflow_name = ALIASES.get(ns.workflow, ns.workflow)
    workflow = WORKFLOWS[workflow_name]
    run_script(workflow.script, ns.workflow_args)


if __name__ == "__main__":
    main()
