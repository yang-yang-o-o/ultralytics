# 新实验：yolo26s-seg-6dpose（seg6d）

> 状态：**正式训练已完成**；原分辨率可视化已出  
> 更新：2026-07-18 23:40（UTC+8）

---

## 训练结果摘要

| 项 | 值 |
|----|-----|
| 跑目录 | `runs/segment/beer-seg/yolo26s-seg-6dpose/` |
| 预训练 | s960 seg `best.pt`（844/928） |
| 计划 | 100 ep / imgsz=960 / batch=8 |
| 实际 | **EarlyStop @55**，patience=40；**best=ep15**（按 mAP fitness） |
| Mask mAP50 / mAP50-95 | ≈0.995 / 0.995 |
| Box mAP50 / mAP50-95 | ≈0.995 / ~0.30（letterbox 尺度下 box 偏松，与此前类似） |

### Pose6D（val=30，对齐 YOLO6D test：ply+内参）

| 权重 | mean_corner | mean_proj | Acc@5px | mean_ADD | Acc@0.1d |
|------|-------------|-----------|---------|----------|----------|
| **last.pt (ep55)** | **4.02 px** | **2.85 px** | **100%** | **0.0043** | **100%** |
| best.pt (ep15) | 11.82 px | 8.24 px | 6.7% | 0.0167 | 63.3% |

**注意**：EarlyStopping 按 box/mask mAP 选 best，**未纳入 Pose6D**。点误差在 ep15 之后仍明显下降 → 看 6D 请优先用 **`last.pt`**（或后续把 fitness 改成含 proj/ADD）。

日志归档：

```text
docs/experiment-logs/beer-yolo26n-seg-2026-07-18/logs/
  beer_seg6d_train.log
  beer_seg6d_pose6d_metrics.txt
  yolo26s-seg-6dpose-results.csv
```

---

## 原分辨率可视化

脚本：`examples/beer_seg6d_predict_native.py`  
配置：imgsz=1312，`retina_masks=True`，输出 **964×1292**  
内容：mask 半透明 + box + 预测 9 点线框（绿/青）+ PnP 重投影（品红）

```text
runs/segment/beer-seg/yolo26s-seg-6dpose/
  predict-val-native-retina-last/   # 30 张 ★
  predict-val-native-retina-best/   # 30 张
```

复现：

```bash
cd /root/ultralytics
.venv/bin/python examples/beer_seg6d_predict_native.py --weights last
.venv/bin/python examples/beer_seg6d_predict_native.py --weights best
```

---

## v2 正式训练（Pose6D fitness，已完成）

| 项 | 值 |
|----|-----|
| 目录 | `runs/segment/beer-seg/yolo26s-seg-6dpose-v2/` |
| epochs | **100 跑满**（patience=60，未早停） |
| best Pose6D | mean_proj **1.25 px**，Acc@5px **100%**，ADD@0.1d **100%**，mean_ADD **0.0021** |
| last Pose6D | mean_proj **1.33 px**（与 best 接近 → fitness 修复生效） |
| Mask mAP50 | 0.995 |

对比 YOLO6D `model100.weights`（mean_proj **2.42**）：v2 best **明显更好**。

原分辨率可视化（964×1292）：

```text
.../yolo26s-seg-6dpose-v2/predict-val-native-retina-best/   # ★ 推荐
.../yolo26s-seg-6dpose-v2/predict-val-native-retina-last/
```

---

## 与 YOLO6D 原权重对比（同 30 张 test）

| | `backup/beer/model100.weights` | seg6d `last.pt` |
|--|--------------------------------|-----------------|
| Acc@5px | 100% | 100% |
| ADD@0.1d | 100% | 100% |
| mean proj (px) | **2.42** | ~2.85 |
| mean corner (px) | **3.46** | ~4.02 |
| mean ADD | 0.00452 | ~0.0044 |

门槛指标打平；平均误差 YOLO6D 略优约 0.4–0.5px。我们额外有分割。

---

## 已确认决策（回顾）

多实例 / 可切换预训练 / YOLO6D MSE loss / proj+ADD / 自定义双标签 / beer.ply 三维点。
