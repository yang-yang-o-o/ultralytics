# 07 · 本地代码改动说明

相对 `origin/main`（本实验基线 `e13eb540`），仅 **3 个文件** 有业务改动，服务于 Depth 表头协议近似。

```
ultralytics/utils/metrics.py                 | DepthMetrics + align="log_ls"
ultralytics/nn/tasks.py                      | DepthModel._predict_augment
ultralytics/models/yolo/depth/val.py         | augment=True → log_ls
```

`git diff --stat` 量级约：+57 / −9 行。

## 行为契约

| `yolo depth val` 参数 | 行为 |
|---|---|
| 默认 / `augment=False` | 与 upstream 一致：单尺度 + median（δ1≈0.783） |
| `augment=True` | 多尺度+翻转 TTA + log_ls（本机 ≈0.83–0.84） |

训练路径不受影响（`self.training` 时仍用 median）。

## 导出 / 应用补丁

本目录已包含可应用补丁：[`patches/depth-tta-logls.patch`](patches/depth-tta-logls.patch)（相对 `origin/main` 三文件 diff）。

干净 main 上：

```bash
git apply docs/official_example/patches/depth-tta-logls.patch
# 或：git am / cherry-pick 含该改动的 commit
```

若工作树已改动、需重新导出：

```bash
git diff ultralytics/utils/metrics.py \
         ultralytics/nn/tasks.py \
         ultralytics/models/yolo/depth/val.py \
  > docs/official_example/patches/depth-tta-logls.patch
```

## 实现要点（便于审查）

1. **log_ls**：对有效像素做 `lstsq([log(pred), 1], log(gt))`，再指数还原。  
2. **TTA**：尺度默认 `(0.75, 1.0, 1.25)`，每尺度含水平翻转；输出插值回单尺度特征图分辨率后算术平均。  
3. **stride**：Depth 头非 Detect 时 `stride` 默认 32，用于 `scale_img` 对齐。

更完整消融表见 [04-depth-tta.md](04-depth-tta.md)。

## 与 upstream 合并建议

若向上游贡献：建议做成 **可选协议**（显式 `align=log_ls` / 文档化尺度），并注明「近似官方表头，非 bit-exact」。在官方内部脚本公开前，不宜宣称可复现 0.882。
