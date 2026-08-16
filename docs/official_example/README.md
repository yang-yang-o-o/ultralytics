# Official Example：YOLO26 官方指标复现实验

> 记录于 2026-08-15，环境：Linux + RTX 3080 Ti + CUDA，仓库分支 `official_example`（基于 `origin/main` @ `e13eb540`）。  
> 目标：在本机用 **yolo26n\*** 权重，按官方文档协议复现各任务验证指标，并与发布表对齐。

## 文档地图（总分）

| 文档 | 内容 |
|---|---|
| [README.md](README.md)（本页） | 起因、结论总表、阅读顺序 |
| [01-setup.md](01-setup.md) | 拉仓、分支、uv 环境、系统依赖 |
| [02-datasets.md](02-datasets.md) | 各任务验证集下载、镜像、目录布局、踩坑 |
| [03-evaluation.md](03-evaluation.md) | 评测命令、指标对照、日志路径 |
| [04-depth-tta.md](04-depth-tta.md) | Depth TTA + log-LS 本地实现与消融 |
| [05-known-gaps.md](05-known-gaps.md) | 未对齐项、不可复现原因 |
| [06-reproduce.md](06-reproduce.md) | 新环境端到端复现清单 |
| [07-code-changes.md](07-code-changes.md) | 相对 upstream main 的本地代码改动（已合入本分支） |
| [08-results-video.md](08-results-video.md) | 定性 MP4 生成流程与脚本 |

## 起因

1. 从 `https://github.com/yang-yang-o-o/ultralytics` 拉 **main**（默认远程常为 `dev`，需显式切 main）。
2. 新建分支 **`official_example`**，用类似官方的 **uv** 开发环境。
3. 跑 YOLO26 各任务基准（检测 / 实例分割 / 语义分割 / 深度 / 分类 / 姿态 / OBB），并与官方发布指标对比。
4. Cityscapes / ImageNet 官方源受限时，改用 Hugging Face 等公开镜像下载 **val-only**。

## 经过（摘要）

1. **环境**：`uv venv` + `uv pip install -e ".[dev]"`，补 `git` / `libgl1` / `aria2` / `unzip`。
2. **小集 smoke**：`benchmark format=-` → `benchmarks.log`。
3. **公开 val 评测**：COCO / NYU / DOTAv1 val / Cityscapes val / ImageNet val。
4. **对齐补洞**：Detect 补跑 `end2end=False`；Depth 本地实现 TTA+log-LS；OBB 确认 test GT 不可得。
5. **文档化**：本目录，供新环境完整复现。

## 最终结论（总表）

模型均为 **yolo26n\***，指标为本地实测 vs 官方文档表（同协议才可对齐）。

| 任务 | 协议要点 | 本地 | 官方 | 对齐？ |
|---|---|---|---|---|
| Detect | COCO val，默认 e2e | AP **40.0** | e2e **40.1** | ✅ |
| Detect | COCO val，`end2end=False` | AP **40.8** | **40.9** | ✅ |
| Segment | COCO val e2e | box **39.8** / mask **33.9** | **39.6** / **33.9** | ✅ |
| Pose | COCO-pose val e2e | pose AP **57.2** | **57.2** | ✅ |
| Semantic | Cityscapes val，`imgsz=2048` | mIoU **78.3** | **78.3** | ✅ |
| Classify | ImageNet val，`imgsz=224` | top1 **71.4** / top5 **90.1** | **71.4** / **90.1** | ✅ |
| Depth | 单尺度 + median（文档可复现） | δ1 **0.783** | **0.783** | ✅ |
| Depth | 表头 TTA + log-LS | 本地近似最好 **~0.839** | **0.882** | ⚠️ 差 ~4.3 点 |
| OBB | DOTAv1 **val** 单尺度 | mAP50-95 **43.9** | （无公开 val 数） | — |
| OBB | DOTAv1 **test** 多尺度 | 无 GT，无法本地算分 | **52.4** | ❌ 需评测服 |

**一句话**：凡官方给出「开源可复现命令」的任务，本地均可对齐；Depth 表头与 OBB test 依赖未开源协议 / 未公开 GT，本地无法完全对齐。

## 定性总览视频

各任务对齐指标后的效果串成一段回顾片（标题总表 → Detect / Seg / Pose / Semantic / Classify / Depth / OBB）：

- 成品示例：[`yolo26n_official_val_results.mp4`](yolo26n_official_val_results.mp4)
- **如何重新生成**：[08-results-video.md](08-results-video.md)
- 脚本：[`scripts/render_clean_batches.py`](scripts/render_clean_batches.py)、[`scripts/make_results_video.py`](scripts/make_results_video.py)

Detect / Segment / OBB：框上无文字 label，左侧色块图例与框颜色对应。

## 关键路径

```
仓库根：/root/ultralytics          # 分支 official_example
数据根：/root/datasets
评测日志：runs/official_val/eval.log
下载日志：runs/official_val/download*.log
小集：benchmarks.log
权重：仓库根目录 yolo26n*.pt（首次 val 会自动下载）
```

## 建议阅读顺序

新环境复现 → 先读 [06-reproduce.md](06-reproduce.md)，出片看 [08-results-video.md](08-results-video.md)，细节回查 01–05、07。  
只关心结论 → 本页总表 + [05-known-gaps.md](05-known-gaps.md)。
