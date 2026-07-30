# 当前最优配置（维护文档）

> **用途**：记录 beer-seg6d 当前最优可复现配置。后续实验若在关键指标上**明确更好**，再更新本文件（并在「变更记录」追加一行）。  
> **初始化**：`yolo26s-seg-6dpose-official-vocbg-ep100`（2026-07-19）  
> **日志/权重**：只在 `runs/`，不进 `docs/.../logs/`

## 判定标准（更新本文件时）

相对当前最优，至少满足其一再替换：

1. val **mean_proj** 明显下降（建议 ≥0.2px），且 Acc@5px / ADD@0.1d 不回退；或  
2. Acc@5px / ADD@0.1d 提升且 mean_proj 不显著变差；或  
3. 同精度下明显更快/更稳（需在变更记录说明 trade-off）。

对比集：`beer-seg6d` val（30 张，与 YOLO6D `test.txt` 同源）。

---

## 当前最优：`yolo26s-seg-6dpose-official-vocbg-ep100`

| 项 | 值 |
|----|----|
| Run 目录 | `runs/segment/beer-seg/yolo26s-seg-6dpose-official-vocbg-ep100/` |
| 权重 | `weights/best.pt` |
| 预训练 | 官方 `yolo26s-seg.pt` |
| 数据 | `beer-seg6d.yaml`（`yolo_seg6d/`；检测框 = **seg 多边形 AABB**，kpt = YOLO6D 9 点） |
| epochs | 100 |
| imgsz / batch / workers | 960 / **8** / **8**（batch=10 长训会 OOM） |
| mask_ratio / mosaic / close_mosaic | 2 / 0.5 / 20 |
| fliplr / flipud | 0（9 点顺序非左右对称） |
| 课程学习 | `pose_warmup_epochs=15`（前 15 ep pose=0），之后 `pose=24` |
| VOC 随机背景 | `bg_replace=1.0`，`bg_dir=VOCdevkit/VOC2012/JPEGImages`，`bg_mask_dir=data/beer/mask` |
| patience | 60（fitness 含 Pose6D） |
| amp | True |

### Val 指标（best.pt）

| 指标 | 值 |
|------|-----|
| mean_proj | **≈3.58 px** |
| Acc@5px | **96.7%** |
| ADD@0.1d | **100%** |
| box / seg mAP50-95 | **0.995** |

可视化：`predict-val-native-retina-{best,last}/`

### 一键复现（训练）

```bash
cd /root/ultralytics
.venv/bin/python examples/beer_seg6d_train.py \
  --weights /root/ultralytics/yolo26s-seg.pt \
  --epochs 100 \
  --name yolo26s-seg-6dpose-official-vocbg-ep100 \
  --pose 24 --pose-warmup-epochs 15 \
  --close-mosaic 20 --patience 60 \
  --batch 8 --workers 8 --device 0 \
  --bg-replace 1.0 \
  --bg-dir /root/YOLO6D/VOCdevkit/VOC2012/JPEGImages \
  --bg-mask-dir /root/YOLO6D/data/beer/mask
```

### 一键复现（测试 + native 可视化）

```bash
.venv/bin/python examples/beer_seg6d_val.py \
  --run runs/segment/beer-seg/yolo26s-seg-6dpose-official-vocbg-ep100 --weights best
.venv/bin/python examples/beer_seg6d_predict_native.py \
  --run runs/segment/beer-seg/yolo26s-seg-6dpose-official-vocbg-ep100 --weights best
```

---

## 变更记录

| 日期 | Run | 摘要 |
|------|-----|------|
| 2026-07-19 | `yolo26s-seg-6dpose-official-vocbg-ep100` | 初版最优：官方 seg 预训练 + VOC bg + 15ep 课程 + pose=24 + 100ep；mean_proj≈3.58 |
