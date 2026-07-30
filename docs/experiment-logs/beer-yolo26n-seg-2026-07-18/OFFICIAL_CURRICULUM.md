# Official 预训练 + 课程学习：`yolo26s-seg-6dpose-official`

> 更新：2026-07-19  
> 训练产物与日志：`runs/segment/beer-seg/<run>/`（`train.log` / `val-*.log` 在此，**不**放 `docs/.../logs/`）

## 设定（official 40ep）

| 项 | 值 |
|----|----|
| 预训练 | 官方 `yolo26s-seg.pt` |
| epochs | 40 |
| 课程 | 前 15 ep：`kpt_loss=0`（只训 det+seg）；之后 `pose=24` 三损失联合 |
| 其它 | imgsz=960, mask_ratio=2, mosaic=0.5, fliplr=0 |

## 结论（official 40ep）

- 课程学习正常：前 15 ep `kpt_loss=0`，之后 pose=24 联合下降；ep40 `train_kpt≈0.39`。
- 可视化：`predict-val-native-retina-{best,last}/`。
- **best.pt val**：mean_proj **≈3.61px**，Acc@5px **83.3%**，ADD@0.1d **100%**，seg mAP50-95 **0.995**。
- 对比 **v2**（beer seg 域内预训练、100 ep、mean_proj≈**1.25px**）：官方权重 + 40 ep 课程仍偏弱，主因是总 epoch 更少、起步无 beer 域适配。

## 背景过拟合与 VOC 随机背景

beer 训练图前景在高度相似背景上反复出现，易把背景当捷径。已接入 YOLO6D 式 `RandomBackground`（`bg_replace` / `bg_dir` / `bg_mask_dir`）。

### `yolo26s-seg-6dpose-official-vocbg`（40ep，workers=4, batch=8）

- **best.pt val**：mean_proj **≈4.95px**，Acc@5px **70%**，ADD@0.1d **96.7%**，seg mAP50-95 **0.995**，box mAP50-95 **0.321**（高于 official 的 0.285）。
- 相对无 VOC official（proj≈3.61）：同域 val 上 pose 略差、box 略好，符合「换背景加大难度」。

### `yolo26s-seg-6dpose-official-vocbg-ep100`（已完成）

- 同 vocbg + **100 ep**；dataloader：**workers=8, batch=8**（`batch=10` 曾在 ep2 OOM）；`close_mosaic=20`, `patience=60`；seg AABB 框。
- **best.pt val**：mean_proj **≈3.58px**，Acc@5px **96.7%**，ADD@0.1d **100%**，box/seg mAP50-95 **0.995**。
- 对比 vocbg 40ep（proj≈4.95 / Acc@5px 70%）：加长训练明显改善；对比无 VOC official 40ep（proj≈3.61）：pose 相当且 Acc@5px 更高，同时带 VOC 抗背景捷径。
- 可视化：`predict-val-native-retina-{best,last}/`。
- **当前最优配置维护**：见 [`BEST_CONFIG.md`](./BEST_CONFIG.md)（以此 run 为初值）。

## `val_batch*_labels` 大框（已修）

- **原因**：pose 标签 `cx cy w h` 曾用 **9 点 AABB**（外扩），`val_batch*_labels.jpg` 画的是这份 GT。
- **修复**：`beer_seg6d_prepare.py` 改为 **seg 多边形 AABB**；已重生成 `yolo_seg6d/labels/` 并清 cache。

## ep1 提速测（VOC bg 开启，imgsz=960）

| 配置 | 墙钟 | 训练环 | GPU |
|------|------|--------|-----|
| workers=4, batch=8（旧） | 142s | 1:43 | ~8.6G |
| workers=8, batch=8 | **122s**（约快 14%） | 1:29 | ~8.6G |
| workers=8 + cache=ram | 130s | 1:29 | ~8.6G |
| workers=8, batch=10 | **109s**（约快 23%） | 1:15 | ~10.5G / 12G |

- **长训采用**：`workers=8 + batch=8`（`batch=10` 在 ep100 的 ep2 mosaic 峰值 OOM；`cache=ram` 对 VOC 随机读盘几乎无收益）。

## 对外对比

- BOP 小规模数据集选型：见 [`BOP_DATASETS_SURVEY.md`](./BOP_DATASETS_SURVEY.md)（优先 IC-BIN / TUD-L，再 LM-O）。
