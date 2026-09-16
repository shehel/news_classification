"""Registers the challenge dataset and folds into ClearML Dataset registry."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clearml import Dataset


def main():
    parser = argparse.ArgumentParser(description="Register challenge dataset to ClearML")
    parser.add_argument("--dataset_name", type=str, default="news_google_challenge")
    parser.add_argument("--dataset_project", type=str, default="news_google_challenge/datasets")
    parser.add_argument("--dataset_version", type=str, default="1.0.0")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    files_to_add = ["Train.csv", "Test.csv", "SampleSubmission.csv", "folds.csv"]

    for f in files_to_add:
        if not (repo_root / f).exists():
            print(f"Error: Missing required file {f}")
            sys.exit(1)

    print(f"Creating ClearML Dataset: project='{args.dataset_project}', name='{args.dataset_name}', version='{args.dataset_version}'")
    dataset = Dataset.create(
        dataset_name=args.dataset_name,
        dataset_project=args.dataset_project,
        dataset_version=args.dataset_version,
    )

    for f in files_to_add:
        print(f"Adding file: {f}")
        dataset.add_files(str(repo_root / f))

    print("Uploading dataset...")
    dataset.upload()
    print("Finalizing dataset...")
    dataset.finalize()
    print(f"Dataset successfully registered! ID: {dataset.id}")


if __name__ == "__main__":
    main()
