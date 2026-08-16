# 07 · 本地代码改动说明

相对 `origin/main`（基线约 `e13eb540`），本分支 **`official_example` 已直接包含** Depth 表头协议近似的 **3 个文件**改动，**无需再应用 patch**。

```
ultralytics/utils/metrics.py                 | DepthMetrics + align="log_ls"
ultralytics/nn/tasks.py                      | DepthModel._predict_augment
ultralytics/models/yolo/depth/val.py         | augment=True → log_ls
```

量级约：+57 / −9 行。

## 行为契约

| `yolo depth val` 参数 | 行为 |
|---|---|
| 默认 / `augment=False` | 与 upstream 一致：单尺度 + median（δ1≈0.783） |
| `augment=True` | 多尺度+翻转 TTA + log_ls（本机 ≈0.83–0.84） |

训练路径不受影响（`self.training` 时仍用 median）。

## 新环境用法

```bash
git clone -b official_example git@github.com:yang-yang-o-o/ultralytics.git
cd ultralytics
# Depth TTA 已在分支内，直接：
yolo depth val model=yolo26n-depth.pt data=nyu-depth.yaml device=0 imgsz=768 augment=True
```

## 实现要点（便于审查）

1. **log_ls**：对有效像素做 `lstsq([log(pred), 1], log(gt))`，再指数还原。  
2. **TTA**：尺度默认 `(0.75, 1.0, 1.25)`，每尺度含水平翻转；输出插值回单尺度特征图分辨率后算术平均。  
3. **stride**：Depth 头非 Detect 时 `stride` 默认 32，用于 `scale_img` 对齐。

更完整消融表见 [04-depth-tta.md](04-depth-tta.md)。

## 与 upstream 合并建议

若向上游贡献：建议做成 **可选协议**（显式 `align=log_ls` / 文档化尺度），并注明「近似官方表头，非 bit-exact」。在官方内部脚本公开前，不宜宣称可复现 0.882。
