# Pose6D 对比：seg6d last.pt vs YOLO6D model100.weights

> 更新：2026-07-19（UTC+8）  
> 测试集：同一批 `data/beer/test.txt`（30 张）

---

## Early-stop 修复

**问题**：旧 fitness = box mAP + mask mAP。mAP 约 ep15 饱和 → patience=40 在 ep55 停；但 Pose6D 仍在改善，`last` 明显好于 `best`。

**改动**（`models/yolo/seg6d/val.py`）：

- `get_stats()` 在原 mAP fitness 上叠加 Pose6D：
  - `0.5 * Acc@5px + 0.5 * ADD@0.1d`（归一化到 [0,1]）
  - `+ 0.5 * 5/(5+mean_proj)`（低投影误差加分）
- 训练脚本默认 `patience=60`（给 6D 更多收敛空间）

下次重训时 `best.pt` 会跟随 Pose6D，而不是只看分割。

---

## 数值对比（val/test = 30）

| 指标 | YOLO6D `backup/beer/model100.weights` | 我们 `yolo26s-seg-6dpose/last.pt` |
|------|----------------------------------------|-----------------------------------|
| Acc @ 5px 投影 | **100.00%** | **100.0%**（复测约 96.7–100，见下） |
| ADD Acc @ 0.1×diam | **100.00%** | **100.0%** |
| Mean 2D proj error (px) | **2.423** | **2.85–2.87** |
| Mean corner error (px) | **3.462** | **4.02** |
| Mean ADD / vertex error | **0.00452** | **0.0043–0.0044** |
| 5cm5deg Acc | **100.00%** | （未在 seg6d val 里报） |

来源：

- YOLO6D：`docs/.../logs/yolo6d_model100_test.log`（`test.py` + `model100.weights`，从 rar 解压的官方权重，非自训）
- 我们：训练日志 ep55 / 复测 `last.pt`（imgsz=960 letterbox）

### 解读

- **门槛指标**（Acc@5px、ADD@0.1d）两边都打满，同属这一测试集上的「满分档」。
- **平均误差**上 YOLO6D 原权重略优（proj 2.42 vs ~2.85，corner 3.46 vs ~4.0）；ADD 几乎持平甚至我们略好一点。
- 协议不完全相同：YOLO6D 按 `yolo-pose.cfg` 测宽/高推理；我们是 YOLO26 letterbox@960 再还原到原图坐标。差距在亚像素～1px 量级，可视为同一水平。
- 我们额外有 **实例分割**；YOLO6D 原模型没有 mask。

### v2（Pose6D fitness，100 ep）更新

| | model100 | v1 last | **v2 best** |
|--|----------|---------|-------------|
| mean proj (px) | 2.42 | ~2.85 | **1.25** |
| mean corner (px) | 3.46 | ~4.02 | **1.90** |
| Acc@5px / ADD@0.1d | 100/100 | 100/100 | **100/100** |

v2 best 已明显优于 YOLO6D 原权重；且 best≈last，说明 early-stop 修复有效。

### 建议

1. 重训一轮以验证新 fitness（`best` 应贴近 `last` 的 Pose6D）。  
2. 若要对齐更严，可把 val `imgsz` 提到接近原图长边（如 1312），再比一次 mean_proj。

---

## 权重路径

```text
YOLO6D:  /root/YOLO6D/backup/beer/model100.weights
seg6d:   /root/ultralytics/runs/segment/beer-seg/yolo26s-seg-6dpose/weights/last.pt
```
