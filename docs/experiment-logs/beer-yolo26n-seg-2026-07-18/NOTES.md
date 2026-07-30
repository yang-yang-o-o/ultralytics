## 2026-07-26

- 用 beer [`BEST_CONFIG.md`](./BEST_CONFIG.md) 在 `projects/object_6d_pose_annotation/outputs/run1/yolo6d_full` 训 seg6d：
  - prepare：`examples/yolo6d_full_seg6d_prepare.py`（train 987 / val 110）
  - run：`runs/segment/yolo6d-full-seg/yolo26s-seg6d-bestcfg/`（100ep）
  - Val：mAP50≈0.995；Pose Acc@5px≈45–50%，mean_proj≈5.7px，ADD@0.1d≈100%
  - 可视化：`predict-val-native-retina-{best,last}/` 与 `predict-val-imgsz960-best/`（box+mask+9pts+PnP）

## 2026-07-20

- IC-BIN 进度专题：[`ICBIN_PROGRESS.md`](./ICBIN_PROGRESS.md)
- 结论摘要：
  - 渲染 train → BOP AR=**0**；PBR 5k/8ep → AR=**0.064**（MSPD 0.128）
  - 半数据 25 场景 / ~25k 图训到 **ep8** 后暂停，可 `--resume` 续 `icbin-seg6d-pbr25k-ep40`
  - beer 最优仍为 `BEST_CONFIG.md`（未改）
- 数据：`/root/YOLO6D/datasets/bop/icbin`（含 `train_pbr` 25 场景 + zip）

## 2026-07-20

- IC-BIN 进度专文：[`ICBIN_SEG6D_PROGRESS.md`](./ICBIN_SEG6D_PROGRESS.md)
- **结论**：渲染 train → BOP AR=0；PBR 5k/8ep → lite **AR≈0.064**（MSPD≈0.128）；管线已通。
- **进度**：PBR 25 场景（半数据）`icbin-seg6d-pbr25k-ep40` 训到 **ep8** 已停，`last.pt` 可 `--resume`。
- beer 最优未变：仍见 [`BEST_CONFIG.md`](./BEST_CONFIG.md)。

## 2026-07-20

- IC-BIN 进度专文：[`ICBIN_SEG6D_PROGRESS.md`](./ICBIN_SEG6D_PROGRESS.md)
- **结论**：PBR 短训已出分（AR≈0.064）；渲染 train→AR=0；半数据 25 场景训到 ep8 已停，待 `--resume`。
- beer 最优未变：仍见 [`BEST_CONFIG.md`](./BEST_CONFIG.md)。

## 2026-07-19

- 训练/测试日志只放 `runs/segment/beer-seg/<name>/`；docs 只留结论。
- 最优维护：[`BEST_CONFIG.md`](./BEST_CONFIG.md) ← 初值 `official-vocbg-ep100`（proj≈3.58, Acc@5px 96.7%）。
- BOP 小集调研：[`BOP_DATASETS_SURVEY.md`](./BOP_DATASETS_SURVEY.md)（优先 **IC-BIN / TUD-L**，再 **LM-O**）。
- 历程摘要见 `OFFICIAL_CURRICULUM.md`。

## 2026-07-18 21:00

- 新实验提案 `yolo26s-seg-6dpose`：seg + YOLO6D 九定点 head + PnP。
- YOLO6D 原理已核实；待用户确认问题后再实现。
- 文档：`YOLO26S_SEG_6DPOSE_PROPOSAL.md`

# NOTES（短记追加）

按时间倒序追加即可。

## 2026-07-18 20:51

- 确认 `predict-val-native-retina` 由 `yolo26s-seg-imgsz960-mr2/weights/best.pt` 推理。
- 已移动到：`runs/segment/beer-seg/yolo26s-seg-imgsz960-mr2/predict-val-native-retina/`（30 张仍在）。
- `beer_seg_predict_native.py` / `beer_seg_test.py` 改为 `project=该 run 目录`，后续预测默认归入对应实验。
- 标注提升现有结论已在 SESSION §5.11 + `LABEL_QUALITY.md` 固化（区域 IoU 高、边界还有一点、原始 mask 可能是天花板、非主瓶颈）。

## 2026-07-18 20:34

- 用户确认原分辨率 + retina 观感「好很多」。
- 新增 `LABEL_QUALITY.md` 与标注统计结论。

## 2026-07-18 20:11

- 原分辨率 retina 推理完成。

## 2026-07-18 18:40

- 提质全链路落地并训练。
