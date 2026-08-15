# 03 · 评测命令与结果

除非注明，均在仓库根 `/root/ultralytics`，使用 `.venv/bin/yolo`，`device=0`。  
完整输出追加在 `runs/official_val/eval.log`。

## 命令清单

```bash
# Detect（e2e，默认）
yolo detect val model=yolo26n.pt data=coco.yaml device=0 imgsz=640

# Detect（对齐表上 40.9 列：one2many + NMS）
yolo detect val model=yolo26n.pt data=coco.yaml device=0 imgsz=640 end2end=False

# Segment
yolo segment val model=yolo26n-seg.pt data=coco.yaml device=0 imgsz=640

# Pose（图像需可用；必要时软链 coco 图）
yolo pose val model=yolo26n-pose.pt data=coco-pose.yaml device=0 imgsz=640

# Depth 单尺度（文档可复现 → δ1≈0.783）
yolo depth val model=yolo26n-depth.pt data=nyu-depth.yaml device=0 imgsz=768

# Depth 表头近似（需本地代码改动，见 04 / 07）
yolo depth val model=yolo26n-depth.pt data=nyu-depth.yaml device=0 imgsz=768 augment=True

# OBB val（不可与官方 test 52.4 直接比）
yolo obb val model=yolo26n-obb.pt data=DOTAv1.yaml device=0 imgsz=1024 split=val

# Semantic
yolo semantic val model=yolo26n-sem.pt data=cityscapes.yaml device=0 imgsz=2048

# Classify（需 train→val 软链）
yolo classify val model=yolo26n-cls.pt data=/root/datasets/imagenet device=0 imgsz=224
```

小集 smoke：

```bash
yolo benchmark model=yolo26n.pt data=coco8.yaml device=0 format=-
# 或仓库内各任务官方 small yaml
```

## 本机实测 vs 官方（yolo26n）

以 **COCO/官方评测器主指标** 为准（Detect/Seg/Pose 看 faster-coco / COCO AP；其余看 Ultralytics 表）。

### 已对齐

| 任务 | 本地关键数 | 官方参考 |
|---|---|---|
| Detect e2e | COCO AP **0.400**（Ultralytics mAP50-95 0.395） | e2e **40.1** |
| Detect `end2end=False` | COCO AP **0.408**（Ultralytics 0.402） | **40.9** |
| Segment e2e | box AP **0.398** / mask AP **0.339** | e2e **39.6** / **33.9** |
| Pose e2e | pose AP **0.572** | e2e **57.2** |
| Depth 单尺度 | δ1 **0.7834**，abs_rel 0.156，rmse 0.570 | δ1 **0.783** |
| Semantic | mIoU **0.783**，PixAcc 0.958 | mIoU **78.3** |
| Classify | top1 **0.714**，top5 **0.901** | **71.4** / **90.1** |

### 不可直接对齐

| 任务 | 本地 | 官方表 | 说明 |
|---|---|---|---|
| Depth 表头 | TTA+log_ls 最好 ~**0.839** | **0.882** | 开源无完整协议；见 [04-depth-tta.md](04-depth-tta.md) |
| OBB | val mAP50-95 **0.439** | test 多尺度 **52.4** | 无 test GT；协议不同 |

### OBB 补充

- `augment=True`：模型报不支持，退回单尺度（约 0.437）。
- `split=test save_json=True`：可出 `predictions_txt/`，但 merge 依赖切片命名（`__x___y`），当前全图包会 IndexError；且无 GT 仍无法本地打分。
- 正式对齐路径：多尺度切片 → 推理 → [DOTA Evaluation Server](https://captain-whu.github.io/DOTA/evaluation.html)。

## 如何读官方表

| 列 | 含义 |
|---|---|
| Detect/Seg/Pose 的 **(e2e)** | YOLO26 默认 end2end 头 |
| Detect 无 e2e 的 mAP | `end2end=False`（one2many+NMS） |
| Depth 表头 δ1 | TTA + log-LS（未开源） |
| Depth 文档脚注 | 单尺度 + median → 0.783(n) |
| OBB mAP<sup>test</sup> | test + 多尺度 + 评测服 |

## 日志与产物

| 路径 | 内容 |
|---|---|
| `runs/official_val/eval.log` | 全部 val 文本日志 |
| `runs/detect/...` `runs/segment/...` 等 | 各次 run 目录、曲线图、predictions.json |
| `benchmarks.log` | 小集 benchmark |
| `docs/official_example/` | 本文档集 |
