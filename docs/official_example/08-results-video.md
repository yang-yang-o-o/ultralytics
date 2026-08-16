# 08 · 定性结果视频（MP4）生成流程

在完成 [06-reproduce.md](06-reproduce.md) 全量 val 并对齐指标后，可用本流程生成回顾片，直观查看各任务效果。

参考成品（本分支可带一份）：[`yolo26n_official_val_results.mp4`](yolo26n_official_val_results.mp4)

## 脚本位置

| 脚本 | 作用 |
|---|---|
| [`scripts/render_clean_batches.py`](scripts/render_clean_batches.py) | 重跑 Detect / Segment / OBB 的前几批 val 图：**框/mask 只保留颜色、无文字 label**，并写出 `*.classes.json` 供图例 |
| [`scripts/make_results_video.py`](scripts/make_results_video.py) | 拼标题表、GT/Pred、左侧色块图例 → 输出 MP4 |

依赖：仓库已 `uv pip install -e ".[dev]"`，系统有 `ffmpeg`（可选，用于转 H.264；没有则用 OpenCV `mp4v`）。

## 前置条件

1. 分支：`official_example`（已含 Depth TTA 三文件改动，**无需再打 patch**）
2. 数据与权重就绪（见 02 / 06）
3. 至少跑过各任务 `yolo * val`，且 `plots=True`（默认），以便有 `val_batch*_labels.jpg` / `*_pred.jpg`

建议把带图例的 Detect/Seg/OBB 输出落到固定目录（脚本默认）：

```
runs/official_val/results_video/clean/
  det_coco_e2e/
  det_coco_o2m/
  seg_coco_e2e/
  obb_val/
```

其它任务继续用日常 val 目录，例如：

```
runs/semantic/.../sem_cityscapes/
runs/classify/.../cls_imagenet/
runs/depth/.../depth_single/          # 单尺度
runs/depth/.../depth_tta_logls/       # augment=True
```

路径随你 `project=` / `name=` 可能多一层 task 目录；`make_results_video.py` 里对 clean 子目录用 `**/name` 查找。

## 一步步命令

在**仓库根**执行：

```bash
source .venv/bin/activate

# 1)（推荐）生成无文字 label、带 classes.json 的 Detect/Seg/OBB 图
python docs/official_example/scripts/render_clean_batches.py

# 2) 若尚无 Depth single / Semantic / Classify / Pose 的 val_batch 图，按 03 补跑对应 val
#    Depth 表头近似：
#    yolo depth val model=yolo26n-depth.pt data=nyu-depth.yaml device=0 imgsz=768 \
#      augment=True project=runs/depth/runs/official_val name=depth_tta_logls

# 3) 拼视频
python docs/official_example/scripts/make_results_video.py

# 4)（可选）转 H.264，便于播放器兼容
ffmpeg -y -i docs/official_example/yolo26n_official_val_results.mp4 \
  -c:v libx264 -pix_fmt yuv420p -crf 20 -movflags +faststart \
  /tmp/yolo26n_official_val_results_h264.mp4
mv /tmp/yolo26n_official_val_results_h264.mp4 docs/official_example/yolo26n_official_val_results.mp4
```

输出：`docs/official_example/yolo26n_official_val_results.mp4`  
中间产物：`runs/official_val/results_video/`（frames / preds / clean / preview）

## 画面约定

| 任务 | 布局 | 说明 |
|---|---|---|
| Detect / Segment / OBB | 左 Legend + 上 GT / 下 Pred | 无框上文字；颜色 = 类别 |
| Pose | 按源图宽高比自动左右或上下 | COCO GT 关键点与预测成对 |
| Semantic | 宽 mosaic → 上下 | 用 val 自带 labels/pred |
| Classify / Depth | 方图 → 左右 | ImageNet 保留完整 mosaic；Depth 1920² 裁 2×2 再左右 |
| 开场 | Summary 表格 | Task / Protocol / Local / Official / Align |

## 新环境最短路径

```bash
git clone -b official_example git@github.com:yang-yang-o-o/ultralytics.git
cd ultralytics
# …按 01–06 装环境、下数据、跑全量 val…
python docs/official_example/scripts/render_clean_batches.py
python docs/official_example/scripts/make_results_video.py
```

数据集目录默认读 Ultralytics `SETTINGS["datasets_dir"]`（本机曾为 `/root/datasets`）。
