# IC-BIN seg6d 进度与结论

> 更新日期：2026-07-20  
> 关联：beer 最优仍见 [`BEST_CONFIG.md`](./BEST_CONFIG.md)；BOP 选型见 [`BOP_DATASETS_SURVEY.md`](./BOP_DATASETS_SURVEY.md)

## 当前结论（先读）

1. **管线已打通**：BOP IC-BIN → seg6d 标签 → YOLO26s-seg6d 训练 → PnP 导出 CSV → `bop_toolkit` lite 评测（MSSD+MSPD）。
2. **渲染视角球 train 不够**：仅用 HF `train/`（单物体渲染）短训，BOP AR=**0**。
3. **PBR subsample 有效**：5 场景 / 5k 图 / 8 epoch → BOP AR **0.064**（MSPD **0.128**，MSSD 仍 0）——分数已脱离 0。
4. **半数据长训进行中（已暂停）**：25/50 场景、~25k 图、计划 40 epoch；已完成 **ep1–8**，用户要求停下，下次 `--resume`。
5. beer 线与 IC-BIN 线并行；**beer 最优配置未因 IC-BIN 实验改动**。

## 数据落地

| 项 | 路径 / 说明 |
|----|-------------|
| BOP 根 | `/root/YOLO6D/datasets/bop/icbin` |
| 已下 zip | `base` / `models` / `train` / `test_bop19` + **`icbin_train_pbr.zip` (~22GB)** |
| PBR 解压 | `train_pbr/`：**25 场景**（000000–000024），跳过 depth，约 6.9GB；共 ~24975 rgb |
| YOLO 布局 | `yolo_seg6d/`（images + labels + labels_seg） |
| yaml | `ultralytics/cfg/datasets/icbin-seg6d.yaml`（`mesh_scale=1.0` mm；2 类） |

未解压：PBR 后 25 场景（000025–000049）。全量解压磁盘偏紧（盘总 ~79G）。

## 脚本

| 脚本 | 作用 |
|------|------|
| `examples/icbin_seg6d_prepare.py` | BOP→seg6d（支持 `--train-split train_pbr`、`--max-scenes`、`--fresh-train`；kpt clip 到 [0,1]） |
| `examples/icbin_seg6d_viz.py` | GT 可视化（标签 vs BOP 重投影） |
| `examples/icbin_seg6d_train.py` | 训练（默认 smoke；支持 `--resume`） |
| `examples/icbin_seg6d_export_bop.py` | 推理 + PnP → BOP CSV |
| `examples/icbin_seg6d_eval_bop.py` | 封装评测；默认 **lite**（MSSD+MSPD）；`--full` 含 VSD |
| `examples/eval_bop19_pose_lite.py` | lite 评测入口 |

`bop_toolkit`：`/root/bop_toolkit`（已改子进程用 `sys.executable`，避免系统 python 无 numpy）。

## 实验对照

### A. `icbin-seg6d-smoke`（渲染 train）

- 数据：HF `train/` 4754 张（视角球）+ test 150  
- 训练：5 ep，imgsz=640，batch=16  
- 权重：`runs/segment/icbin-seg/icbin-seg6d-smoke/weights/best.pt`  
- **BOP lite AR = 0.0**（MSSD/MSPD 皆 0）  
- GT-oracle 自检 AR ≈ **0.917**（说明评测管线正常）

### B. `icbin-seg6d-pbr5k-smoke`（PBR 5 场景）

- 数据：train_pbr 000000–000004，**5000** 张 / ~85k 实例  
- 训练：8 ep，pose warmup=2  
- 权重：`runs/segment/icbin-seg/icbin-seg6d-pbr5k-smoke/weights/best.pt`  
- val（ep8）：box mAP50≈0.38，mask mAP50≈0.22；Pose Acc@5px 仍 0，mean_proj≈32px  
- **BOP lite**：

| 指标 | 值 |
|------|-----|
| **bop19_average_recall** | **0.064** |
| MSPD AR | **0.128** |
| MSSD AR | 0.0 |
| 导出 poses | ~2694 条 |

### C. `icbin-seg6d-pbr25k-ep40`（半数据，进行中 / 已暂停）

- 数据：train_pbr **25 场景**，**24975** 张 / ~424k 实例；val 仍 test 150  
- 计划：40 ep，pose warmup=10，close_mosaic=10，patience=25  
- **状态（2026-07-19 夜）**：用户在约 ep9 要求停止；**已完整写入 results 的为 ep1–8**  
- `last.pt`：`ckpt_epoch=7`（0-based）→ resume 从 **第 9 轮**继续  
- 目录：`runs/segment/icbin-seg/icbin-seg6d-pbr25k-ep40/`  
- ep8 快照：box mAP50≈**0.41**，mask mAP50≈**0.27**（优于 pbr5k-smoke；Pose Acc@5px 仍 0）  
- **尚未跑** 本 run 的 BOP 评测（等 resume 完成或用户指定用当前 last/best 先评）

#### 续训命令

```bash
cd /root/ultralytics
.venv/bin/python examples/icbin_seg6d_train.py \
  --epochs 40 \
  --name icbin-seg6d-pbr25k-ep40 \
  --batch 16 --imgsz 640 --workers 8 \
  --pose-warmup-epochs 10 --close-mosaic 10 --patience 25 \
  --resume
```

#### 训完后评测

```bash
.venv/bin/python examples/icbin_seg6d_eval_bop.py \
  --weights runs/segment/icbin-seg/icbin-seg6d-pbr25k-ep40/weights/best.pt \
  --result-csv runs/segment/icbin-seg/icbin-seg6d-pbr25k-ep40/bop_results/seg6d-pbr25k_icbin-test.csv \
  --results-path runs/segment/icbin-seg/icbin-seg6d-pbr25k-ep40/bop_results \
  --eval-path runs/segment/icbin-seg/icbin-seg6d-pbr25k-ep40/bop_eval \
  --conf 0.1
```

## 技术备注

- IC-BIN 单位 **mm**；`Pose6DEval` 已支持 `mesh_scale` 与按类 `meshes`/`diams`。  
- PBR 图为 `.jpg`；prepare 按真实后缀 symlink。  
- 多实例 + 遮挡：检测/分割先起来，ADD/MSSD 更难；短期看 **BOP AR / MSPD** 比 Acc@5px 更合适。  
- 磁盘：zip 22G + pbr25 解压 6.9G + runs；全量 50 场景前需再腾空间。

## 下一步（建议）

1. `--resume` 跑完 `pbr25k-ep40` → BOP 对比 pbr5k。  
2. 若 AR 明显提升，再解压剩余 25 场景做全量 PBR 训。  
3. 需要官方完整表时加 `--full`（VSD）；日常迭代用 lite。  
4. 可选：TUD-L / LM-O。
