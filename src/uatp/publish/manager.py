"""Publishing manager for Hugging Face Hub."""
from __future__ import annotations

import os
from typing import Optional

from ..storage.manager import StorageManager


class PublishManager:
    def __init__(self):
        self.storage = StorageManager()

    def publish(self, folder: str, repo_id: str, token: Optional[str] = None, private: bool = True, commit_message: str = "Publish model") -> str:
        if not token and not os.environ.get("HF_TOKEN"):
            raise RuntimeError("HF_TOKEN not set; cannot publish")
        return self.storage.hf_upload_folder(folder, repo_id, token=token, private=private)
