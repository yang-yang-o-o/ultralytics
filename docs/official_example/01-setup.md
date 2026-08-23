# 01 · 环境与仓库设置

> **换机复现请先读 [00-new-machine.md](00-new-machine.md)。**  
> 不要从 `origin/main` 重新切分支，否则会丢掉本实验已提交的 Depth TTA 与文档。

## 目标状态

| 项 | 值 |
|---|---|
| 远程 | `https://github.com/yang-yang-o-o/ultralytics` |
| 基线分支 | `origin/main` @ `e13eb540` |
| 工作分支 | `official_example`（从 main 切出） |
| Python | 3.12（uv venv） |
| 安装 | `uv pip install -e ".[dev]"` |
| GPU | NVIDIA GeForce RTX 3080 Ti（约 12GB） |
| 系统补丁 | `git`、`libgl1`、`aria2`、`unzip` |

> 注意：该 fork 默认跟踪分支常为 **dev**。若只 `git clone` 不指定，可能落在 dev 上。本实验明确基于 **main**。

## 推荐步骤

```bash
# 1) 克隆并进入
git clone https://github.com/yang-yang-o-o/ultralytics.git
cd ultralytics

# 2) 确保有 main，并切出实验分支
git fetch origin main
git checkout -B official_example origin/main
git log -1 --oneline   # 期望接近 e13eb540（上游可能前进）

# 3) 系统依赖（Debian/Ubuntu 示例）
apt-get update
apt-get install -y git libgl1 aria2 unzip

# 4) uv 环境
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# 5) 验证 CLI
.venv/bin/yolo checks
```

## 权重

首次 `yolo val ...` 会从 Ultralytics assets 拉取。本实验使用的 nano 权重：

- `yolo26n.pt`
- `yolo26n-seg.pt`
- `yolo26n-pose.pt`
- `yolo26n-sem.pt`
- `yolo26n-cls.pt`
- `yolo26n-depth.pt`
- `yolo26n-obb.pt`

可预先放在仓库根目录，或让 CLI 自动下载。

## 本地代码改动（Depth）

相对 `origin/main`，本分支额外修改了 3 个文件以支持 Depth `augment=True` 表头协议近似（详见 [07-code-changes.md](07-code-changes.md)）：

- `ultralytics/utils/metrics.py` — `align="log_ls"`
- `ultralytics/nn/tasks.py` — `DepthModel._predict_augment`
- `ultralytics/models/yolo/depth/val.py` — `augment=True` 时启用 log_ls

**未改动这些文件时**：仍可复现 Depth **单尺度 δ1=0.783**；无法跑本地 TTA+log-LS 近似。

## 磁盘

完整评测建议预留 **≥40GB** 可用空间（COCO + Cityscapes val + ImageNet val + NYU + DOTA + 解压峰值）。  
本机实测高峰后约 **23G used / 54G avail**（已删部分 zip）。
