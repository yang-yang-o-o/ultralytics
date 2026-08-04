# 复现手册：yolo6d_full seg6d（训练 / val / 推理）

> **用途**：在**全新机器 / 新对话**中，按本文把 `yolo6d-full-seg6d` 的环境、数据、100ep 训练、Pose6D val、图片/视频推理完整跑通。  
> **会话来源**：2026-08-02 在本机从零搭建并成功跑通的流程（仓库 `dev` @ `45540e71`）。  
> **人工只准备两份压缩包**；其余步骤均可由助手按本文执行。  
> **相关文档**：beer 最优配置见同目录 [`BEST_CONFIG.md`](./BEST_CONFIG.md)（beer 数据，非本手册主线）；本手册主线是 **object 绕拍 → `yolo6d_full`**。

---

## 0. 开场必检（新对话第一件事）

**助手在动手前必须先检查下列两个文件是否存在：**

| 文件 | 期望路径 | 大小量级 | 作用 |
|------|----------|----------|------|
| 带走备份 | `/root/ultralytics_backup_takeaway.zip` | ~2.5GB | 预训练权重、`yolo6d_full` 标签/rgb/mask、历史 runs、测试视频等 |
| VOC 背景 | `/root/VOCdevkit.rar` | ~1.9GB | 训练时 `bg_replace` 随机背景（VOC2012 JPEGImages） |

```bash
ls -lh /root/ultralytics_backup_takeaway.zip /root/VOCdevkit.rar
```

### 若任一文件缺失

**立刻停止后续步骤，向用户索要**，话术示例：

> 复现 `yolo6d_full` seg6d 需要你手动准备的两份数据：  
> 1. `/root/ultralytics_backup_takeaway.zip`  
> 2. `/root/VOCdevkit.rar`  
> 当前环境里找不到它们。请放到 `/root/` 后再让我继续。

不要尝试从网上下载替代 VOC 或自行重跑 SfM/标注来“凑”备份——本手册假定这两包是用户提供的权威输入。

---

## 1. 目标与成功标准

### 1.1 目标

1. 克隆 `yang-yang-o-o/ultralytics` 的 **`dev`** 分支并初始化 submodule  
2. 解压两份用户包，落到脚本硬编码路径  
3. 建 `.venv`，装 CUDA PyTorch + editable ultralytics + `trimesh`  
4. 必要时打上 `Seg6DLoss` 返回 dict 的补丁（见 §6）  
5. `prepare` → 100ep 训练 → Pose6D val → val 图可视化 → 两段测试视频推理  

### 1.2 成功标准（与 2026-08-02 实测对齐）

| 阶段 | 标准 |
|------|------|
| 环境 | `torch.cuda.is_available()==True`，可 `from ultralytics import YOLO` |
| 数据 | `yolo_seg6d`：train **987** / val **110**；VOC JPEGImages **17125** |
| 训练 | 100ep 完成，产出 `weights/best.pt`、`results.csv` |
| val（best） | Box/Mask mAP50 ≈ **0.995**；Pose6D **ADD@0.1d = 100%**；mean_proj ≈ **5.7–6.0 px**；Acc@5px ≈ **49–51%** |
| 推理 | val 可视化 110 张；两段 mp4 有检出帧 |

> 说明：本物体的 Acc@5px（约 50%）低于 beer 最优（约 96.7%），属数据集/标定差异，**不是复现失败**。以 ADD@0.1d、mAP、mean_proj 量级是否接近上表为准。

---

## 2. 硬件 / 系统前提

| 项 | 2026-08-02 实测 | 要求 |
|----|-----------------|------|
| GPU | RTX 3080 Ti 12GB | 建议 ≥10GB；`imgsz=960 batch=8` 约占用 ~9GB |
| OS | Ubuntu + CUDA driver | 需 `nvidia-smi` 可用 |
| Python | 3.12（miniconda `py312` 或等价） | ≥3.10 通常可；本文按 3.12 |
| 磁盘 | 解压后约需 **≥15GB** 空闲（zip+rar+venv+runs） | — |

缺省系统工具时安装（本次会话曾缺 `git/unzip/unrar/libGL`）：

```bash
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git unzip unrar p7zip-full file libgl1 libglib2.0-0 curl
```

国内加速（强烈建议，本次官方 PyTorch 源极慢）：

```bash
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
# 可写入 ~/.bashrc 持久化
grep -q UV_INDEX_URL ~/.bashrc || echo 'export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple' >> ~/.bashrc
```

安装 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

---

## 3. 克隆代码

```bash
cd /root
# 若已存在旧目录，确认是否要覆盖；干净复现建议移走
# mv ultralytics ultralytics.bak.$(date +%Y%m%d) 2>/dev/null || true

git clone -b dev --single-branch https://github.com/yang-yang-o-o/ultralytics.git ultralytics
cd /root/ultralytics
git submodule update --init --recursive
```

钉扎（本次成功会话）：

| 组件 | 值 |
|------|-----|
| 分支 | `dev` |
| ultralytics commit | `45540e71`（`Add YOLO26 seg6d ...`） |
| submodule | `projects/object_6d_pose_annotation` → `https://github.com/yang-yang-o-o/object_6d_pose_annotation.git` @ `10a97bd` |

```bash
git rev-parse --short HEAD          # 期望 45540e71（若更新需自行对照）
git -C projects/object_6d_pose_annotation rev-parse --short HEAD
```

---

## 4. 解压用户数据包并落到硬编码路径

脚本与 yaml **写死了绝对路径**，必须落到下表，不要随意改名。

### 4.1 解压

```bash
mkdir -p /root/_extract
unzip -q /root/ultralytics_backup_takeaway.zip -d /root/_extract
unrar x -o+ /root/VOCdevkit.rar /root/_extract/
```

备份 zip 顶层结构（`ultralytics_backup_takeaway/`）：

```
README.txt
weights/                 # yolo26n.pt, yolo26n-seg.pt, yolo26s-seg.pt
object_6d_pose_annotation/
  data/                  # 绕拍视频与抽帧
  outputs/run1/...       # 含 yolo6d_full/{rgb,mask,labels,yolo_seg6d,...}
runs/                    # 历史训练/预测（可选恢复；含测试视频）
```

### 4.2 放置到期望路径

```bash
# VOC → 训练脚本 DEFAULT_BG_DIR
mkdir -p /root/YOLO6D
rm -rf /root/YOLO6D/VOCdevkit
mv /root/_extract/VOCdevkit /root/YOLO6D/

# 子项目 data/outputs（覆盖 submodule 里空的 gitignore 目录）
PROJ=/root/ultralytics/projects/object_6d_pose_annotation
rm -rf "$PROJ/data" "$PROJ/outputs"
mv /root/_extract/ultralytics_backup_takeaway/object_6d_pose_annotation/data "$PROJ/"
mv /root/_extract/ultralytics_backup_takeaway/object_6d_pose_annotation/outputs "$PROJ/"

# 预训练权重到仓库根（train 默认 /root/ultralytics/yolo26s-seg.pt）
cp -a /root/_extract/ultralytics_backup_takeaway/weights/*.pt /root/ultralytics/

# 历史 runs（含测试视频 VID_*.mp4；可选但推理视频步骤需要）
rm -rf /root/ultralytics/runs
mv /root/_extract/ultralytics_backup_takeaway/runs /root/ultralytics/

# 清理临时解压（可选）
rm -rf /root/_extract
```

### 4.3 路径核对清单

| 路径 | 期望 |
|------|------|
| `/root/YOLO6D/VOCdevkit/VOC2012/JPEGImages/` | ≥17000 张 jpg |
| `/root/ultralytics/yolo26s-seg.pt` | 存在（~23MB） |
| `.../outputs/run1/yolo6d_full/{rgb,mask,labels}/` | 各约 1097 文件 |
| `.../outputs/run1/yolo6d_full/{train,test}.txt` | 存在 |
| `ultralytics/cfg/datasets/yolo6d-full-seg6d.yaml` | `path`/`mesh`/`fx` 等指向上述 `yolo6d_full` |

```bash
ls /root/YOLO6D/VOCdevkit/VOC2012/JPEGImages | wc -l   # → 17125
ls /root/ultralytics/yolo26s-seg.pt
ls /root/ultralytics/projects/object_6d_pose_annotation/outputs/run1/yolo6d_full/rgb | wc -l
```

数据集 yaml（勿改路径除非同步改脚本）：

- `/root/ultralytics/ultralytics/cfg/datasets/yolo6d-full-seg6d.yaml`  
  - `path: .../yolo6d_full/yolo_seg6d`  
  - `mesh: .../yolo6d_full/object.ply`  
  - 内参 1600×900，`kpt_shape: [9, 2]`

---

## 5. Python 环境

```bash
cd /root/ultralytics
export PATH="$HOME/.local/bin:$PATH"
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 使用本机 Python 3.12（示例为 miniconda py312）
uv venv .venv --python "$(command -v python3)"

# PyTorch（清华源；本次得到 2.13.0+cu130，以实际解析为准）
uv pip install --python .venv torch torchvision

# 本仓库 editable + 依赖
uv pip install --python .venv -e .

# Pose6D / PnP 评测需要
uv pip install --python .venv trimesh
```

若 `import cv2` 报 `libGL.so.1`：

```bash
apt-get install -y -qq libgl1 libglib2.0-0
# 或：uv pip install --python .venv opencv-python-headless
```

自检：

```bash
cd /root/ultralytics
.venv/bin/python - <<'PY'
import torch, cv2, trimesh
from ultralytics import YOLO
assert torch.cuda.is_available(), "CUDA 不可用"
print("torch", torch.__version__, torch.cuda.get_device_name(0))
print("ultralytics OK")
PY
```

---

## 6. 必打补丁：`Seg6DLoss` 必须返回 dict

### 6.1 现象

用较新的 torch（本次 `2.13.0+cu130`）训练时，第一个 batch 可能报错：

```text
RuntimeError: values expected sparse tensor layout but got Strided
```

栈在 `ultralytics/engine/trainer.py` 的 `*self.tloss.values()`。  
原因：trainer 期望 `loss_items` 为 **dict**，而旧版 `Seg6DLoss.loss` 返回了 **Tensor**；`Tensor.values()` 在新 torch 上是 sparse API。

### 6.2 修复（若 `dev` 尚未合入则必须本地改）

编辑 `ultralytics/utils/loss.py` 中 `Seg6DLoss`：

1. `__init__` 增加：

```python
self.loss_names = ("box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss", "kpt_loss")
```

2. `loss()` 末尾由：

```python
return loss * batch_size, loss.detach()
```

改为：

```python
return loss * batch_size, dict(zip(self.loss_names, loss.detach()))
```

3. 返回类型注解改为 `tuple[torch.Tensor, dict[str, torch.Tensor]]`。

检查是否已修好：

```bash
rg -n "class Seg6DLoss|loss_names|return loss \* batch_size" ultralytics/utils/loss.py | head -40
# 应看到 loss_names 六元组，且 return ... dict(zip(self.loss_names, ...))
```

> **注意（2026-08-02）**：该修复在当时会话里是**工作区未提交修改**。新环境若 `git status` 干净且仍返回 Tensor，必须按本节再打一次补丁，否则 100ep 训不起来。

---

## 7. 准备 `yolo_seg6d` 布局

zip 里的 `yolo_seg6d/images/{train,val}` 多为**指向 `rgb/` 的符号链接**。解压后链接常失效（images 目录空），**必须重跑 prepare**：

```bash
cd /root/ultralytics
.venv/bin/python examples/yolo6d_full_seg6d_prepare.py
```

期望输出类似：

```text
root=.../yolo6d_full train=987 val=110
Wrote .../yolo_seg6d  images=1097
```

核对：

```bash
ls projects/object_6d_pose_annotation/outputs/run1/yolo6d_full/yolo_seg6d/images/train | wc -l  # 987
ls projects/object_6d_pose_annotation/outputs/run1/yolo6d_full/yolo_seg6d/images/val | wc -l    # 110
```

---

## 8. 冒烟训练（可选，1 epoch）

确认数据 / 补丁 / GPU 后再开 100ep：

```bash
cd /root/ultralytics
.venv/bin/python examples/yolo6d_full_seg6d_train.py \
  --weights /root/ultralytics/yolo26s-seg.pt \
  --epochs 1 --imgsz 960 --batch 8 --workers 4 --device 0 \
  --name smoke-ep1 \
  --pose 24 --pose-warmup-epochs 0 \
  --close-mosaic 0 --patience 1 \
  --bg-replace 1.0 \
  --bg-dir /root/YOLO6D/VOCdevkit/VOC2012/JPEGImages \
  --bg-mask-dir /root/ultralytics/projects/object_6d_pose_annotation/outputs/run1/yolo6d_full/mask
```

通过标准：能扫完 1 个 epoch、写出 `runs/segment/yolo6d-full-seg/smoke-ep1/weights/best.pt`。  
val 阶段 `plot_images` 线程里偶发 `ValueError: x1 must be greater than or equal to x0`（1ep 随机权重坏框）**可忽略**，只要主进程 `Training finished`。

---

## 9. 正式 100 epoch 训练

配置对齐 beer BEST_CONFIG，入口脚本：`examples/yolo6d_full_seg6d_train.py`。

| 超参 | 值 |
|------|-----|
| model | `yolo26s-seg6d.yaml` |
| pretrained | `yolo26s-seg.pt` |
| data | `yolo6d-full-seg6d.yaml` |
| epochs / imgsz / batch / workers | 100 / 960 / **8** / **8** |
| pose / pose_warmup_epochs | 24 / **15**（前 15ep pose=0） |
| close_mosaic / patience | 20 / 60 |
| mosaic / scale / fliplr/flipud | 0.5 / 0.3 / **0**（9 点顺序不对称） |
| bg_replace | 1.0 + VOC JPEGImages + `yolo6d_full/mask` |
| amp | True |

```bash
cd /root/ultralytics

# 若已有同名 run，先备份以免 exist_ok 覆盖
if [ -d runs/segment/yolo6d-full-seg/yolo26s-seg6d-bestcfg ]; then
  mv runs/segment/yolo6d-full-seg/yolo26s-seg6d-bestcfg \
     runs/segment/yolo6d-full-seg/yolo26s-seg6d-bestcfg-prev-$(date +%Y%m%d%H%M)
fi

nohup .venv/bin/python examples/yolo6d_full_seg6d_train.py \
  --weights /root/ultralytics/yolo26s-seg.pt \
  --epochs 100 --imgsz 960 --batch 8 --workers 8 --device 0 \
  --name yolo26s-seg6d-bestcfg \
  --pose 24 --pose-warmup-epochs 15 \
  --close-mosaic 20 --patience 60 \
  --bg-replace 1.0 \
  --bg-dir /root/YOLO6D/VOCdevkit/VOC2012/JPEGImages \
  --bg-mask-dir /root/ultralytics/projects/object_6d_pose_annotation/outputs/run1/yolo6d_full/mask \
  > /tmp/yolo6d_full_ep100.log 2>&1 &

echo "pid=$!"
# 进度：tail -f /tmp/yolo6d_full_ep100.log
# 或：tail -f runs/segment/yolo6d-full-seg/yolo26s-seg6d-bestcfg/train.log
```

训练中应看到：

```text
RandomBackground: 17125 files from .../VOC2012/JPEGImages (p=1.0)
Seg6D pose curriculum: epochs 0..14 det+seg only (pose=0); from epoch 15 enable pose=24.0
Starting training for 100 epochs...
```

产物目录：

```text
runs/segment/yolo6d-full-seg/yolo26s-seg6d-bestcfg/
  weights/{best,last}.pt
  results.csv
  train.log
  ...
```

3080 Ti 上 100ep 约 **1–2 小时量级**（视 IO/负载；本次会话约从开训到完成约 1.5h 内）。

---

## 10. Val（含 Pose6D）

```bash
cd /root/ultralytics
.venv/bin/python - <<'PY'
from ultralytics import YOLO
m = YOLO("runs/segment/yolo6d-full-seg/yolo26s-seg6d-bestcfg/weights/best.pt")
metrics = m.val(
    data="yolo6d-full-seg6d.yaml",
    imgsz=960, batch=8, device=0, workers=8, plots=True,
)
print("BOX/MASK", {k: v for k, v in metrics.results_dict.items() if k.startswith("metrics/")})
print("Pose6D", getattr(metrics, "pose6d", None))
PY
```

若提示 `PnP metrics disabled: No module named 'trimesh'` → `uv pip install --python .venv trimesh` 后重跑。

**2026-08-02 实测（best.pt）**：

| 指标 | 约值 |
|------|------|
| precision/recall (B/M) | ~1.0 / ~0.98–0.99 |
| mAP50 (B/M) | ~0.995 |
| mAP50-95 (B/M) | ~0.983 / ~0.976 |
| mean_proj / mean_corner | ~5.77 / ~6.02 px |
| Acc@5px | ~49.1% |
| ADD@0.1d | **100%** |

`results.csv` 中按 `metrics/pose6d_mean_proj` 最小：约 **ep99**，proj≈**5.745**，Acc@5px≈**50.9%**。

---

## 11. Val 集可视化（native + retina mask + PnP）

```bash
cd /root/ultralytics
.venv/bin/python examples/yolo6d_full_seg6d_predict.py \
  --run runs/segment/yolo6d-full-seg/yolo26s-seg6d-bestcfg \
  --weights best
```

输出：

```text
runs/segment/yolo6d-full-seg/yolo26s-seg6d-bestcfg/predict-val-native-retina-best/
  full_XXXXXX.jpg × 110
  README.txt
```

图例：绿框=检测；橙=mask；绿青=预测 9 点；品红=PnP 重投影。

---

## 12. 视频推理

测试视频在备份的 `runs/.../yolo26s-seg6d-bestcfg/`（或 `*-prev`）下：

- `VID_20260726_225848.mp4`
- `VID_20260726_230213.mp4`

若正式训练新建了空 run 目录，把视频链/拷到当前 run：

```bash
cd /root/ultralytics
RUN=runs/segment/yolo6d-full-seg/yolo26s-seg6d-bestcfg
# 按实际备份目录调整 PREV
PREV=$(ls -d runs/segment/yolo6d-full-seg/yolo26s-seg6d-bestcfg-prev* 2>/dev/null | head -1)
# 或直接从解压备份里找 VID_*.mp4
ln -sf "$(pwd)/$PREV/VID_20260726_225848.mp4" "$RUN/" 2>/dev/null || true
ln -sf "$(pwd)/$PREV/VID_20260726_230213.mp4" "$RUN/" 2>/dev/null || true

.venv/bin/python examples/yolo6d_full_seg6d_predict_video.py \
  --run "$RUN" --weights best
```

也可显式传路径：

```bash
.venv/bin/python examples/yolo6d_full_seg6d_predict_video.py \
  --run "$RUN" --weights best \
  --videos /path/to/VID_20260726_225848.mp4 /path/to/VID_20260726_230213.mp4
```

输出示例：

```text
.../predict-video-imgsz960-best/
  VID_20260726_225848_pred.mp4   # 本次：160 帧，约 120 帧有检出
  VID_20260726_230213_pred.mp4   # 本次：280 帧，约 235 帧有检出
```

> 视频内参由标定分辨率 1600×900 按帧尺寸缩放，手机原片 K 为近似。

---

## 13. 关键路径速查

| 角色 | 路径 |
|------|------|
| 仓库 | `/root/ultralytics` |
| venv | `/root/ultralytics/.venv` |
| 用户 zip | `/root/ultralytics_backup_takeaway.zip` |
| 用户 rar | `/root/VOCdevkit.rar` |
| VOC | `/root/YOLO6D/VOCdevkit/VOC2012/JPEGImages` |
| 6D 数据根 | `.../projects/object_6d_pose_annotation/outputs/run1/yolo6d_full` |
| data yaml | `ultralytics/cfg/datasets/yolo6d-full-seg6d.yaml` |
| 训练脚本 | `examples/yolo6d_full_seg6d_train.py` |
| prepare | `examples/yolo6d_full_seg6d_prepare.py` |
| 图推理 | `examples/yolo6d_full_seg6d_predict.py` |
| 视频推理 | `examples/yolo6d_full_seg6d_predict_video.py` |
| 正式 run | `runs/segment/yolo6d-full-seg/yolo26s-seg6d-bestcfg/` |

---

## 14. 故障排除（本次真实踩坑）

| 现象 | 处理 |
|------|------|
| 缺 `git` / `unzip` / `unrar` | §2 `apt-get install` |
| PyTorch 官方 whl 极慢 | `UV_INDEX_URL=清华源`，不要死磕 `download.pytorch.org` |
| `libGL.so.1` | 装 `libgl1` 或用 `opencv-python-headless` |
| `tloss.values` / sparse layout | §6 `Seg6DLoss` → dict |
| `yolo_seg6d/images` 为空 | §7 重跑 prepare（symlink） |
| `--bg-dir not found` | 确认 VOC 在 `/root/YOLO6D/VOCdevkit/...` |
| PnP 指标全无 / warning trimesh | `uv pip install trimesh` |
| val 绘图线程 `x1 >= x0` | 冒烟阶段可忽略；不影响权重保存 |
| OOM | 降 `batch`（8→4）或 `imgsz`；BEST 用 batch=8 |
| 视频脚本 `assert vp.exists()` | 把 `VID_*.mp4` 放到 `--run` 目录或 `--videos` 显式传入 |
| 两压缩包缺失 | **停下来向用户索要**（§0），禁止瞎编替代数据 |

---

## 15. 给新对话助手的最短执行清单

按顺序勾选：

1. [ ] `ls` 检查两压缩包；缺失 → **只提醒用户提供，不往下做**  
2. [ ] 装系统依赖 + `uv` + 设置 `UV_INDEX_URL`  
3. [ ] `git clone -b dev` + `submodule update --init --recursive`  
4. [ ] 解压并 mv/cp 到 §4 路径；核对 VOC 张数与 `yolo26s-seg.pt`  
5. [ ] `uv venv` + torch/torchvision + `pip install -e .` + `trimesh`  
6. [ ] 检查/打上 `Seg6DLoss` dict 补丁  
7. [ ] `yolo6d_full_seg6d_prepare.py` → train=987 val=110  
8. [ ] （可选）1ep smoke  
9. [ ] 100ep `yolo6d_full_seg6d_train.py`（后台 + 日志）  
10. [ ] `best.pt` val（看 Pose6D）  
11. [ ] `yolo6d_full_seg6d_predict.py`  
12. [ ] `yolo6d_full_seg6d_predict_video.py`  

---

## 16. 变更记录

| 日期 | 内容 |
|------|------|
| 2026-08-02 | 初版：基于当日从零搭建会话；两用户包 + 清华源 + Seg6DLoss dict 补丁 + 100ep/val/视频实测指标 |
| 2026-08-04 | 写入本实验日志目录，供新环境/新对话复现 |

---

## 17. 一键命令块（环境已具备、包已就位时）

```bash
export PATH="$HOME/.local/bin:$PATH"
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
cd /root/ultralytics

# 若尚未 prepare
.venv/bin/python examples/yolo6d_full_seg6d_prepare.py

# 训练
.venv/bin/python examples/yolo6d_full_seg6d_train.py \
  --weights /root/ultralytics/yolo26s-seg.pt \
  --epochs 100 --batch 8 --workers 8 --device 0 \
  --name yolo26s-seg6d-bestcfg \
  --pose 24 --pose-warmup-epochs 15 \
  --close-mosaic 20 --patience 60 \
  --bg-replace 1.0 \
  --bg-dir /root/YOLO6D/VOCdevkit/VOC2012/JPEGImages \
  --bg-mask-dir /root/ultralytics/projects/object_6d_pose_annotation/outputs/run1/yolo6d_full/mask

# val + 图 + 视频
.venv/bin/python -c "from ultralytics import YOLO; m=YOLO('runs/segment/yolo6d-full-seg/yolo26s-seg6d-bestcfg/weights/best.pt'); print(m.val(data='yolo6d-full-seg6d.yaml', imgsz=960, batch=8, device=0))"
.venv/bin/python examples/yolo6d_full_seg6d_predict.py --run runs/segment/yolo6d-full-seg/yolo26s-seg6d-bestcfg --weights best
.venv/bin/python examples/yolo6d_full_seg6d_predict_video.py --run runs/segment/yolo6d-full-seg/yolo26s-seg6d-bestcfg --weights best
```
