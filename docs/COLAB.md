# Colab Workflow

1. Mount Google Drive.
2. Choose durable directory under MyDrive.
3. Train with storage.root on local disk and sync checkpoints to Drive.
4. On new runtime, scan durable dir for latest valid checkpoint and resume.

Helpers in `uatp.colab.workflow`. Does not claim to bypass Colab time limits.