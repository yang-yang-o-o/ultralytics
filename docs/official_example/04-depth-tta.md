# 04 · Depth TTA + log-LS 本地实现

## 背景

官方 Depth 表头（yolo26n-depth δ1 **0.882**）写明：

> multi-scale + horizontal-flip TTA，再经 **log-least-squares** 对齐后算指标。

文档同时给出**可复现**单尺度命令 → δ1 **0.783**（median 对齐）。  
开源 `DepthValidator` 原先**只有**单尺度 + `align="median"|"none"`，**没有** TTA / log-LS。

本实验在仓库内补了一套近似实现，并用 NYU Eigen（654）消融。

## 代码位置

| 文件 | 改动 |
|---|---|
| `ultralytics/utils/metrics.py` | `DepthMetrics(align="log_ls")`：逐图拟合 `log(gt)≈a·log(pred)+b`，再 `exp(a·log d+b)` |
| `ultralytics/nn/tasks.py` | `DepthModel._predict_augment`：尺度 × 翻转后双线性对齐到单尺度输出分辨率再平均 |
| `ultralytics/models/yolo/depth/val.py` | `augment=True` 时自动 `align="log_ls"` 并打日志 |

默认 TTA 尺度：`(0.75, 1.0, 1.25)` × `{原图, 水平翻转}` → 6 次前向平均。

## 用法

```bash
# 单尺度（upstream 行为，对齐 0.783）
yolo depth val model=yolo26n-depth.pt data=nyu-depth.yaml device=0 imgsz=768

# 表头近似（需上述改动）
yolo depth val model=yolo26n-depth.pt data=nyu-depth.yaml device=0 imgsz=768 augment=True
```

## 消融结果（yolo26n-depth，NYU 654）

| 协议 | δ1 | abs_rel | rmse |
|---|---:|---:|---:|
| 官方表头 | **0.882** | 0.109 | 0.414 |
| 单尺度 + median（文档） | 0.783 | — | — |
| 本地单尺度 + median | 0.783 | 0.156 | 0.570 |
| 本地单尺度 + log_ls | 0.828 | 0.135 | 0.504 |
| 本地 flip-only + log_ls | 0.832 | 0.133 | 0.498 |
| 本地 TTA[0.75–1.25]+flip+log_ls | 0.837 | 0.131 | 0.490 |
| 本地宽 TTA[0.5–1.5]+flip+log_ls | **0.839** | 0.129 | 0.482 |
| + max_depth=10 / log 空间平均 | ≈0.835–0.839 | — | — |

## 解读

1. **log_ls** 贡献最大（约 +4.5 点）。
2. **TTA** 再贡献约 +1 点。
3. 换更宽尺度、`max_depth=10`、log 平均，几乎不动针。
4. 与官方 **0.882** 仍差约 **4.3 点** → 官方 internal eval 细节未公开（精确尺度集、裁剪/letterbox、平均方式等）。在现有公开信息下**无法消除**该差距。

## 参考（业界同类，非 YOLO26 即用）

- ZoeDepth：水平翻转 TTA  
- Depth Anything 社区：least-squares scale/shift（常在 disparity 域）  
- Ultralytics 文档公式：log 仿射 `exp(a·log d + b)`（与 calibrate 模块一致）

ETH3D / iBims / Make3D 文档提到 “dedicated evaluation script”，**未随开源仓库发布**。
