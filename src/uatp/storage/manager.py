"""Storage management: local, Google Drive, Hugging Face Hub."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class StorageLocation:
    kind: str  # local | gdrive | hf
    path: str
    exists: bool


class StorageManager:
    def __init__(self, root: str = "./runs"):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    def local_path(self, *parts) -> str:
        p = os.path.join(self.root, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True) if os.path.dirname(p) else None
        return p

    def list_runs(self) -> List[str]:
        if not os.path.isdir(self.root):
            return []
        return sorted(d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d)))

    def ensure_dir(self, path: str) -> str:
        os.makedirs(path, exist_ok=True)
        return path

    def copy_tree(self, src: str, dst: str) -> None:
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    # Google Drive helpers
    @staticmethod
    def gdrive_root() -> Optional[str]:
        candidates = [
            "/content/drive/MyDrive",
            os.path.expanduser("~/GoogleDrive"),
            os.path.expanduser("~/Drive"),
        ]
        for c in candidates:
            if os.path.isdir(c):
                return c
        return None

    @staticmethod
    def mount_colab_drive() -> Optional[str]:
        try:
            from google.colab import drive  # type: ignore
            drive.mount("/content/drive")
            return "/content/drive/MyDrive"
        except Exception:
            return None

    # Hugging Face Hub helpers
    def hf_upload_folder(self, folder: str, repo_id: str, path_in_repo: str = "", token: Optional[str] = None, private: bool = True):
        try:
            from huggingface_hub import HfApi
        except Exception as e:
            raise RuntimeError("huggingface_hub not installed") from e
        api = HfApi(token=token or os.environ.get("HF_TOKEN"))
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        api.upload_folder(folder_path=folder, repo_id=repo_id, path_in_repo=path_in_repo, commit_message="Upload checkpoint")
        return repo_id

    def hf_download(self, repo_id: str, local_dir: str, path_in_repo: Optional[str] = None, token: Optional[str] = None):
        try:
            from huggingface_hub import snapshot_download
        except Exception as e:
            raise RuntimeError("huggingface_hub not installed") from e
        return snapshot_download(repo_id=repo_id, local_dir=local_dir, allow_patterns=(f"{path_in_repo}/*" if path_in_repo else None), token=token or os.environ.get("HF_TOKEN"))
