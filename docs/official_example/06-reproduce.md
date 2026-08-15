# 06 · 新环境复现清单

按顺序执行。默认路径可改，但需全局一致。

## A. 代码与环境

```bash
git clone https://github.com/yang-yang-o-o/ultralytics.git
cd ultralytics
git fetch origin main
git checkout -B official_example origin/main

# 若需要 Depth TTA 近似：应用与本实验相同的 3 文件改动
# （见 07-code-changes.md，或从保留该 diff 的分支/补丁打入）

apt-get install -y git libgl1 aria2 unzip   # 按需
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

确认：`yolo checks`，GPU 可见。

## B. 数据

目标根目录示例：`/root/datasets`（或改 Ultralytics settings）。

1. **COCO val2017** + instances → Detect/Seg  
2. **COCO-pose** 标签 + 图像软链到 coco val  
3. **nyu-depth**（yaml 自动下或 assets zip）  
4. **DOTAv1**（yaml/assets；接受 test 无 GT）  
5. **Cityscapes val-only**（HF `danjacobellis/cityscapes_source`）→ organize 成 `images/val` + `masks/val`  
6. **ImageNet val.zip**（HF `mlx-vision/imagenet-1k`，校验大小与非稀疏）→ `val/<synset>/` + `train → val` 软链  

细节与踩坑：[02-datasets.md](02-datasets.md)。

## C. 评测（期望对齐的命令）

```bash
yolo detect val  model=yolo26n.pt      data=coco.yaml        device=0 imgsz=640
yolo detect val  model=yolo26n.pt      data=coco.yaml        device=0 imgsz=640 end2end=False
yolo segment val model=yolo26n-seg.pt  data=coco.yaml        device=0 imgsz=640
yolo pose val    model=yolo26n-pose.pt data=coco-pose.yaml   device=0 imgsz=640
yolo depth val   model=yolo26n-depth.pt data=nyu-depth.yaml  device=0 imgsz=768
yolo semantic val model=yolo26n-sem.pt data=cityscapes.yaml  device=0 imgsz=2048
yolo classify val model=yolo26n-cls.pt data=/path/to/imagenet device=0 imgsz=224
yolo obb val     model=yolo26n-obb.pt  data=DOTAv1.yaml      device=0 imgsz=1024 split=val
```

可选 Depth 表头近似：

```bash
yolo depth val model=yolo26n-depth.pt data=nyu-depth.yaml device=0 imgsz=768 augment=True
# 期望约 0.83–0.84，不是 0.882
```

## D. 验收标准（yolo26n）

| 检查项 | 通过条件 |
|---|---|
| Detect e2e | COCO AP ≈ **0.400±0.003** |
| Detect o2m | COCO AP ≈ **0.408±0.003** |
| Seg e2e | box≈**0.398**，mask≈**0.339** |
| Pose e2e | pose AP ≈ **0.572** |
| Depth 单尺度 | δ1 ≈ **0.783** |
| Semantic | mIoU ≈ **0.783** |
| Classify | top1≈**0.714**，top5≈**0.901** |
| OBB val | 有合理 mAP（本机 ~0.44）；**不**要求等于 52.4 |
| Depth TTA | 若启用补丁，δ1 升至 ~0.84；**不**要求 0.882 |

容差来自浮点 / 驱动 / 依赖版本；数量级与官方一致即可。

## E. 建议落盘

```bash
mkdir -p runs/official_val
# 将各次 yolo 输出 tee 到 runs/official_val/eval.log
```

复现完成后，用 [03-evaluation.md](03-evaluation.md) 总表核对，用 [05-known-gaps.md](05-known-gaps.md) 解释无法对齐项。

## F. 最小磁盘策略

- 只留 val；Cityscapes/ImageNet 解压后删 zip  
- Pose 不复制图，只软链  
- 不下载 ADE20K / KITTI 等旁路集（除非要扩实验）
