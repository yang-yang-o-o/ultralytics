# projects/

Personal / local pipelines that sit beside the Ultralytics library (not part of the installable `ultralytics` package).

| Project | Role |
|---------|------|
| [`object_6d_pose_annotation/`](object_6d_pose_annotation/) | Phone orbit video → SfM → metric scale → browser 6D box → YOLO6D / seg6d labels |

Each project keeps its own `pyproject.toml` and `.venv` when needed. Heavy assets (`data/`, `outputs/`, `third_party/*`, weights) are gitignored; train/predict entrypoints live under repo-root `examples/`.
