"""Upload the completed compact BEAM dataset package to Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_FOLDER = Path("outputs/compact_1000/hub")
DEFAULT_REPO_ID = "vm2825/test_beam"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=DEFAULT_FOLDER)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--commit-message",
        default="Upload BEAM-compatible synthetic dataset",
    )
    parser.add_argument("--confirm-upload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.confirm_upload:
        raise SystemExit("Re-run with --confirm-upload to write to Hugging Face Hub")
    if not args.folder.is_dir():
        raise SystemExit(f"Missing dataset package directory: {args.folder}")
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("Missing HF_TOKEN environment variable")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "Install huggingface_hub before uploading: "
            "python -m pip install huggingface_hub"
        ) from exc

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    commit = api.upload_folder(
        folder_path=args.folder,
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=args.commit_message,
    )
    print(f"https://huggingface.co/datasets/{args.repo_id}")
    print(commit)


if __name__ == "__main__":
    main()
