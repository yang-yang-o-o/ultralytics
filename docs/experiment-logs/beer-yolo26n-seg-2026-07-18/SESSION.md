# SESSION：beer → YOLO26n-seg 分割实验复盘

> 续聊入口：先读本文件 + 同目录 `meta.json`。  
> 更新策略：**当前会话内，助手每次回答完毕后更新本文档。**

- **实验 ID**：`beer-yolo26n-seg-2026-07-18`
- **创建时间**：2026-07-18（UTC+8）
- **最近更新**：2026-07-19 00:10（UTC+8）— 修 Pose6D fitness/early-stop；对比 YOLO6D model100

---

## 1. 目标

1. 在 `/root/ultralytics` 下用 **uv** 建立虚拟环境并安装依赖  
2. 跑通 **YOLO26n-seg** 在 beer 数据上的分割训练与测试  
3. 数据源：`/root/YOLO6D/data/beer`（原 YOLO6D 位姿数据，含 mask）  
4. 可新增 txt / json / yaml / seg 标注以适配 Ultralytics 训练格式  

---

## 2. 环境

| 项 | 值 |
|----|-----|
| 仓库 | `/root/ultralytics`（editable 安装 ultralytics 8.4.100） |
| venv | `/root/ultralytics/.venv`（`uv venv --python 3.12`） |
| Python | 3.12.11 |
| Torch | 2.11.0+cu128，CUDA 可用 |
| GPU | RTX 3080 Ti 12GB |
| 注意 | 曾缺 `libGL`，改用 `opencv-python-headless` 解决 |

常用命令：

```bash
cd /root/ultralytics
source .venv/bin/activate
# 或直接
.venv/bin/python examples/beer_seg_train.py
```

---

## 3. 数据适配

### 3.1 原始数据（YOLO6D）

- `JPEGImages/`：1500 张 PNG，约 1292×964  
- `mask/`：二值 mask（0/255）  
- `labels/`：YOLO6D 6D 位姿标注（**不是** YOLO seg 格式）  
- `train.txt`：1500 条；`test.txt`：30 条（且全部包含在 train 中）  

### 3.2 新增 / 生成文件

| 路径 | 说明 |
|------|------|
| `YOLO6D/data/beer/labels_seg/*.txt` | 由 mask 转多边形，YOLO seg 格式 |
| `YOLO6D/data/beer/yolo_seg/{images,labels}/{train,val}/` | 符号链接布局，train=1470 / val=30（test 从 train 剔除） |
| `YOLO6D/data/beer/train_seg.txt` / `val_seg.txt` | 绝对路径列表 |
| `YOLO6D/data/beer/beer-seg.yaml` | 数据集 yaml（副本） |
| `YOLO6D/data/beer/beer-seg.json` | 数据集元信息 |
| `ultralytics/cfg/datasets/beer-seg.yaml` | 训练用 data 配置 |
| `examples/beer_seg_prepare.py` | mask→polygon + 布局脚本 |
| `examples/beer_seg_train.py` | 训练脚本 |
| `examples/beer_seg_test.py` | 验证 + 预测脚本 |

### 3.3 标注转换要点

- 方法：`cv2.findContours` + `approxPolyDP(epsilon=0.001 * peri)`  
- 抽查 mask↔polygon IoU ≈ **0.993–0.994**（标注保真度较高）  
- 类别：单类 `0 beer`  

---

## 4. 训练配置（当前基线）

来自 `examples/beer_seg_train.py`：

```text
model:   yolo26n-seg.pt
data:    beer-seg.yaml
epochs:  30
imgsz:   640
batch:   16
device:  0
project: beer-seg
name:    yolo26n-seg
amp:     True
```

输出目录：`runs/segment/beer-seg/yolo26n-seg/`  
权重：`.../weights/best.pt`、`last.pt`

---

## 5. 过程大事记

### 5.1 首次环境与数据准备

- 创建 uv venv，安装 torch(cu128) + `pip install -e .`  
- 转换 1500 张 mask → `labels_seg`，建立 `yolo_seg` 布局  

### 5.2 多次重叠训练事故（非代码 bug）

- 现象：多个 `beer_seg_train.py` 同时跑，抢显存 OOM；`results.csv` 出现交错 epoch  
- 原因：OOM 后重试 + 前台会话中断但进程未退出 + 再次启动，操作层重叠  
- 处理：`pkill -f beer_seg_train.py`，改为 **nohup 单进程后台**  
- 结论：**训练代码本身无问题**  

### 5.3 用户删除 runs 后干净重跑

- 用户删除 `runs/`  
- 后台重跑：`nohup .venv/bin/python examples/beer_seg_train.py > /tmp/beer_seg_train.log 2>&1 &`  
- 进程单实例确认 OK，约 30 epoch / ~0.18h 完成  

### 5.4 指标（重跑完成后）

| 指标 | 值 |
|------|-----|
| box mAP50-95 | 0.995 |
| seg mAP50-95 | 0.995 |
| seg mAP50 | 0.995 |
| val 集 | 30 images / 30 instances |

预测输出：`runs/segment/beer-seg/predict/`（若已测）

### 5.5 分割“观感质量偏低”讨论（摘要）

用户反馈：mAP 已很高，但分割边缘视觉上仍糙。

**判断：** 不是没学会，而是 **分辨率 + 小模型 + mask 监督粒度** 的上限；标注 IoU≈0.994 不是主因。

**详细学习笔记（原理 / 对策 / 实验表 / 下一跳配置草稿）：**  
→ 见同目录 **[QUALITY_IMPROVEMENT.md](./QUALITY_IMPROVEMENT.md)**（应写细写全，复盘以该文档为准）

**性价比执行顺序（摘要）：**  
更大 imgsz + `retina_masks` → 换 s/m 模型 → `mask_ratio=2` → 加长训练并减弱 mosaic → 加密多边形 → 用 mask IoU / boundary F1 验收  

助手曾提议按该顺序改 `beer_seg_train.py`（如 `yolo26s-seg` + `imgsz=960` + `mask_ratio=2`）并后台重跑；**已于 2026-07-18 18:40 落地并开跑**（见 §5.8）。

### 5.6 建立实验记录目录

- 新建 `docs/experiment-logs/`  
- 本实验目录：`docs/experiment-logs/beer-yolo26n-seg-2026-07-18/`  
- 约定：本会话内每次助手回答后更新 `SESSION.md` / `meta.json`  

### 5.7 提质文档写细

用户要求：有学习价值的信息在实验记录里记完整细致。  
已新增/加厚 `QUALITY_IMPROVEMENT.md`，并约定后续同类分析优先落入该文档而非只写一行摘要。

### 5.8 提质流水线落地并后台重训（2026-07-18 18:40）

用户确认按性价比顺序全部加上。已执行：

| 步骤 | 落地内容 |
|------|----------|
| 加密多边形 | `epsilon_ratio=0.0005` 重生成 `labels_seg`；样例点数约 80→**264–278**，mask IoU 约 **0.9945–0.997** |
| 换 s 模型 | `yolo26s-seg.pt`（~11.4M params，37.1 GFLOPs） |
| 更大 imgsz | `imgsz=960`，`batch=8`（实测 GPU ~8.3GB） |
| mask 监督加细 | `mask_ratio=2` |
| 加长 + 弱 mosaic | `epochs=100`，`close_mosaic=20`，`mosaic=0.5`，`scale=0.3`，`degrees=0`，`patience=40` |
| retina_masks | 写入 `beer_seg_test.py`（`val`/`predict` 均 `retina_masks=True`，`imgsz=960`）；训练阶段该标志主要用于推理 |

**脚本变更：**

- `examples/beer_seg_prepare.py`：默认 epsilon `0.001` → `0.0005`  
- `examples/beer_seg_train.py`：改为上述 s960 配置，run name=`yolo26s-seg-imgsz960-mr2`  
- `examples/beer_seg_test.py`：指向新权重目录 + retina 预测  

**运行结果（已结束）：**

- 计划 100 epoch，实际记录到 **55** epoch（early stop / patience）  
- 用户观察：**约第 15 epoch 已收敛**（box mAP50-95 在 ep15 达 0.995；seg mAP 从 ep1 起已 0.995）  
- 怀疑：**数据过干净 / 单类场景简单**，导致指标很快饱和，区分度不足  

### 5.9 测试集原分辨率 + retina_masks 推理（2026-07-18 20:11）

- 确认为 **`yolo26s-seg-imgsz960-mr2`** 的 `best.pt` 所跑（日志：`weights: .../yolo26s-seg-imgsz960-mr2/weights/best.pt`）  
- 脚本：`examples/beer_seg_predict_native.py`  
- 数据：`yolo_seg/images/val`（30 张）；**`retina_masks=True`**；`imgsz=1312`；输出 **964×1292**  
- **目录归属（2026-07-18 20:51 已整理）：**  
  `runs/segment/beer-seg/yolo26s-seg-imgsz960-mr2/predict-val-native-retina/`  
  （从原 `runs/segment/beer-seg/predict-val-native-retina/` 移入该训练 run，便于与 `yolo26n-seg` 等实验区分）

### 5.10 用户反馈：原分辨率分割观感明显更好（2026-07-18 20:34）

用户确认该目录结果 **看起来好很多**。  
观感提升主要来自 **更大有效分辨率 + retina_masks（+ s 模型）**；mAP 早已饱和。

### 5.11 标注提升 — 现有结论（固化，2026-07-18 20:34 / 20:51 再确认）

全量 1500 张、`epsilon=0.0005`：

| 项 | 结论 |
|----|------|
| mask IoU | mean≈**0.9925**，min≈**0.986** → 区域层已经很高 |
| 边界带重合（近似） | mean≈**0.58** → 折线逼近仍有 **边界级** 误差 |
| 再加密（如 epsilon=0.0002） | 单图 IoU 可→≈1.0，点数大增；对 mAP 几乎无意义 |
| 真正天花板 | 可能在 **原始 `mask/*.png` ↔ RGB 是否像素级对齐**，而非多边形转换 |
| 是否主瓶颈 | **否**。当前观感与收敛速度问题，主因是分辨率/模型/数据过干净，不是标注 |

详文：→ **[LABEL_QUALITY.md](./LABEL_QUALITY.md)**

---


### 5.12 新实验提案：yolo26s-seg-6dpose（2026-07-18 21:00）

用户提议：在与 `yolo26s-seg-imgsz960-mr2` **相同训练配置**下，给 `yolo26s-seg` 增加 YOLO6D 风格 **9 点 2D 关键点 head**，实验名 `yolo26s-seg-6dpose`；输出 det + seg + 9 点，再用 PnP 解 6D。

YOLO6D 原理已核对：**描述基本正确**（网络回归归一化 9 点，中心+8角；PnP 在测试阶段）。  
详文与待确认问题：→ **[YOLO26S_SEG_6DPOSE_PROPOSAL.md](./YOLO26S_SEG_6DPOSE_PROPOSAL.md)**

状态：**尚未实现，等待用户答复疑问后再改代码。**

## 6. 当前状态

- [x] uv 环境就绪  
- [x] beer → YOLO seg 数据适配完成  
- [x] YOLO26n-seg 基线训练跑通  
- [x] 观感提质分析文档化（`QUALITY_IMPROVEMENT.md`）  
- [x] 提质训练完成（~ep15 收敛）  
- [x] 原分辨率 + retina_masks 推理；**用户确认观感明显更好**  
- [x] 标注提升现有结论已写入实验记录（`LABEL_QUALITY.md` + 本节）  
- [x] predict 结果已归入对应训练 run 目录  
- [ ] 可选：抽查 RGB↔原 mask；或 epsilon=0.0002 对照  
- [ ] pred vs 原 mask 的 IoU / boundary 评估脚本  
- [ ] 针对“数据过干净”做更难 val / 外推测试  
- [x] 实验记录目录已建立  

---

## 7. 关键路径速查

```text
# 基线
runs/segment/beer-seg/yolo26n-seg/

# 提质分割
runs/segment/beer-seg/yolo26s-seg-imgsz960-mr2/
  weights/best.pt
  predict-val-native-retina/

# seg6d（检测+分割+9点）
runs/segment/beer-seg/yolo26s-seg-6dpose/
  weights/{best,last}.pt
  results.csv
  predict-val-native-retina-last/   # ★ 推荐看这个（Pose6D 更好）
  predict-val-native-retina-best/

docs/experiment-logs/beer-yolo26n-seg-2026-07-18/
  SESSION.md / meta.json / NOTES.md
  QUALITY_IMPROVEMENT.md / LABEL_QUALITY.md
  YOLO26S_SEG_6DPOSE_PROPOSAL.md
  POSE6D_VS_YOLO6D.md
  logs/
    beer_seg6d_train.log
    beer_seg6d_pose6d_metrics.txt
    yolo26s-seg-6dpose-results.csv
    beer_seg6d_predict_{last,best}.log
```

---

## 8. 会话更新日志

| 时间 (UTC+8) | 摘要 |
|--------------|------|
| 2026-07-18 16:09+ | 用户要求：uv 环境 + YOLO26n-seg 训练测试 + beer 数据适配 |
| 2026-07-18 ~16:30–17:25 | 数据转换、环境安装、重叠训练事故、清理、重跑、测试 |
| 2026-07-18 17:28 | 说明：非代码问题，是重复启动；训练/测试已通 |
| 2026-07-18 17:35 | 用户删 runs，后台干净重训 |
| 2026-07-18 17:49 | 用户问提质方法；给出 imgsz/模型/mask_ratio 等建议 |
| 2026-07-18 17:56 | 用户要求建立 `docs` 下实验记录，并约定会话内每次回答后更新 |
| 2026-07-18 18:00 | 用户要求把提质分析等有学习价值的内容记完整；新增 `QUALITY_IMPROVEMENT.md` |
| 2026-07-18 18:40 | 用户确认落地提质全链路；加密标注 + s960 训练后台开跑 |
| 2026-07-18 20:11 | 训练结束（~ep15 收敛）；val 原分辨率 + retina_masks 推理输出 |
| 2026-07-18 20:34 | 用户确认观感明显更好；补充标注质量评估 `LABEL_QUALITY.md` |
| 2026-07-18 20:51 | 确认预测属 s960 实验；结果目录移入该 run；标注结论再固化 |
| 2026-07-18 21:00 | 提出 yolo26s-seg-6dpose；核对 YOLO6D；列出疑问待确认 |
| 2026-07-18 22:30 | 用户确认全部决策；落地 task=`seg6d`（Head/Loss/Dataset/Train/Val/PnP）；smoke 1ep 通过 |
| 2026-07-18 23:40 | 正式训提前停于 ep55（best=ep15 by mAP）；归档 log；原分辨率 val 可视化 last/best |
| 2026-07-19 00:10 | 修 early-stop fitness 含 Pose6D；对比 model100.weights，见 POSE6D_VS_YOLO6D.md |
| 2026-07-18 23:40 | 正式训结束（55ep early stop，best@15）；log→`logs/`；原分辨率 pred（last/best）可视化 |
