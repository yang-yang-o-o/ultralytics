# IC-BIN seg6d：进度与结论（2026-07-19 / 07-20）

> 目的：在 beer 最优配置之外，打通 **BOP 数据 → seg6d 训练 → bop_toolkit 出分**。  
> beer 最优仍见 [`BEST_CONFIG.md`](./BEST_CONFIG.md)（本文件不替代 beer 最优）。

## 当前结论（摘要）

1. **管线已通**：IC-BIN → YOLO seg6d 标签 → 训练 → BOP CSV → `bop_toolkit` lite 评测（MSSD+MSPD）。
2. **渲染视角球 train 不够**：仅用 HF `train/`（非 PBR）短训 → BOP AR = **0**。
3. **PBR 有效**：5 场景 / 5k 图 / 8ep → BOP AR ≈ **0.064**（MSPD ≈ **0.128**，MSSD = 0）。分已脱离 0。
4. **半数据（25/50 场景）训练中断待续**：计划 40ep，已完成 **8/40**，已停；下次 `--resume`。
5. beer 上的 Acc@5px / ADD 仍作诊断；对外对比以 **BOP AR** 为准。

## 数据与磁盘

| 项 | 路径 / 说明 |
|----|-------------|
| BOP 根 | `/root/YOLO6D/datasets/bop/icbin` |
| 精简包 | base + models + train + test_bop19（已下） |
| PBR zip | `icbin_train_pbr.zip` ≈ **22GB**（已下） |
| 已解压 PBR | `train_pbr/`：**25** 场景（000000–000024），无 depth，≈ **6.9GB**，RGB ≈ **24975** |
| YOLO 布局 | `yolo_seg6d/`（当前 train=PBR25，val=test 150） |
| 数据集 yaml | `ultralytics/cfg/datasets/icbin-seg6d.yaml`（`mesh_scale=1.0`，双物体 meshes） |
| 剩余 PBR | 场景 25–49 仍在 zip，未解压（盘约紧） |

## 脚本

| 脚本 | 作用 |
|------|------|
| `examples/icbin_seg6d_prepare.py` | BOP → seg6d（`--train-split train_pbr` / `--fresh-train` / `--max-scenes`） |
| `examples/icbin_seg6d_viz.py` | GT 可视化（标签 vs BOP 重投影） |
| `examples/icbin_seg6d_train.py` | 训练（默认 smoke；可加长） |
| `examples/icbin_seg6d_export_bop.py` | 推理 → BOP19 CSV（R,t mm） |
| `examples/icbin_seg6d_eval_bop.py` | 封装评测；默认 **lite**（MSSD+MSPD）；`--full` 含 VSD |
| `examples/eval_bop19_pose_lite.py` | lite 评测入口（跳过 VSD） |

本地 toolkit：`/root/bop_toolkit`（已改 `sys.executable`，避免子进程找不到 venv）。

## 实验表

### A. 渲染 train smoke（失败对照）

| 项 | 值 |
|----|----|
| Run | `runs/segment/icbin-seg/icbin-seg6d-smoke/` |
| 数据 | HF `train/` 渲染视角球 ~4754 + val test 150 |
| 训练 | 5ep，imgsz=640，batch=16，pose_warmup=2 |
| BOP AR (lite) | **0.0** / MSPD 0 / MSSD 0 |
| 结论 | 分布与真实堆叠 test 差太大，不能当主训 |

### B. PBR 5k smoke（首次出分）

| 项 | 值 |
|----|----|
| Run | `runs/segment/icbin-seg/icbin-seg6d-pbr5k-smoke/` |
| 数据 | PBR 场景 0–4，**5000** 图 / val 150 |
| 训练 | 8ep，warmup=2 |
| Val（ep8） | box mAP50≈0.38，mask mAP50≈0.22；Pose Acc@5px 仍 0，mean_proj≈32px |
| 导出 | ~2694 条 pose / test |
| **BOP AR** | **0.064**（MSPD **0.128**，MSSD **0.0**） |
| 分数文件 | `.../bop_eval/seg6d-pbr5k-smoke_icbin-test/scores_bop19.json` |

GT oracle（管线自检）：AR ≈ **0.917**（`gt-oracle_icbin-test`）。

### C. PBR 25k / 半数据（进行中 · 已暂停）

| 项 | 值 |
|----|----|
| Run | `runs/segment/icbin-seg/icbin-seg6d-pbr25k-ep40/` |
| 数据 | PBR **25/50** 场景，**24975** 图 |
| 计划 | 40ep，pose_warmup=10，close_mosaic=10，patience=25 |
| **已完成** | **epoch 1–8**（第 9 轮途中用户中断） |
| 权重 | `weights/last.pt`（`ckpt_epoch=7` → resume 从第 9 轮） |
| Val @ep8 | box mAP50≈**0.41**，mask mAP50≈**0.27**；Pose Acc@5px 仍 0（warmup 未结束） |
| BOP | **尚未**对 ep8 权重评测（resume 完成后再评） |

#### Resume 命令

```bash
cd /root/ultralytics
.venv/bin/python examples/icbin_seg6d_train.py \
  --epochs 40 \
  --name icbin-seg6d-pbr25k-ep40 \
  --batch 16 --imgsz 640 --workers 8 \
  --pose-warmup-epochs 10 --close-mosaic 10 --patience 25 \
  --resume
```

训完后：

```bash
.venv/bin/python examples/icbin_seg6d_eval_bop.py \
  --weights runs/segment/icbin-seg/icbin-seg6d-pbr25k-ep40/weights/best.pt \
  --result-csv runs/segment/icbin-seg/icbin-seg6d-pbr25k-ep40/bop_results/seg6d-pbr25k_icbin-test.csv \
  --results-path runs/segment/icbin-seg/icbin-seg6d-pbr25k-ep40/bop_results \
  --eval-path runs/segment/icbin-seg/icbin-seg6d-pbr25k-ep40/bop_eval \
  --conf 0.1
```

## 实现注意（踩坑）

- 关键点 AABB 常略出画幅 → 标签需 **clip 到 [0,1]**，否则 Ultralytics cache 标 corrupt。
- PBR RGB 为 **.jpg**；链接保留原后缀。
- BOP 单位 **mm**；`mesh_scale=1.0`（beer 用 0.001）。
- `Pose6DEval` 已支持 `meshes`/`diams` 多类 + `mesh_scale`。
- 默认评测用 **lite**（MSSD+MSPD）；完整 VSD 需 `--full`（更慢、要 depth/renderer）。

## 下一步（建议）

1. **Resume** `pbr25k-ep40` 到 40ep（或至少过完 pose warmup=10）→ BOP 评测，对比 pbr5k 的 0.064。  
2. 若 AR 明显上升，再解压剩余 25 场景做全量 PBR（注意磁盘；可先删 zip 或迁走）。  
3. 中长期：LM-O 对齐论文主表；TUD-L 作「有真实训图」对照。

## 变更记录

| 日期 | 内容 |
|------|------|
| 2026-07-19 | 下 IC-BIN 精简包；prepare/viz；渲染 smoke AR=0 |
| 2026-07-19 | 下 PBR；5 场景短训 AR≈0.064 |
| 2026-07-19→20 | 扩 25 场景开 40ep；ep8 暂停，待 resume |
