# 00 · 新机器从 `origin/official_example` 复现

本页是**删掉原实验机之后**的入口：只依赖 GitHub 上的 `official_example` 分支，在新 Linux + NVIDIA GPU 上重建**代码、环境、val 数据、指标结论**，并可再生成定性 MP4。

原实验机上的 `/root/datasets`、`.venv`、`runs/`、`eval.log` **不在 Git 里**，必须按下面重做。仓库里已有一份示例视频，不重跑也能看效果。

细节与踩坑仍以 [02-datasets.md](02-datasets.md)、[03-evaluation.md](03-evaluation.md)、[08-results-video.md](08-results-video.md) 为准。

## 0. 能复现 / 不能原样拷走

| 内容 | 是否在 `origin/official_example` | 新机器怎么做 |
|---|---|---|
| 代码（含 Depth TTA 三文件） | 是 | `git clone -b official_example` |
| 复现文档与出片脚本 | 是 | `docs/official_example/` |
| 示例 MP4 | 是 | 直接打开 `yolo26n_official_val_results.mp4` |
| `yolo26n*.pt` 权重 | 否（gitignore） | 首次 `yolo val` 自动下载 |
| COCO / NYU / DOTA / Cityscapes / ImageNet | 否 | 本页第 3 节重下 **val-only** |
| `runs/`、评测日志 | 否 | 本页第 4 节重跑 |
| 原机 SSH 私钥、HF token | 否 | 新机重新配 GitHub / HF |

**不要**按 [01-setup.md](01-setup.md) 从 `origin/main` 再切分支：那样会丢掉本实验提交。必须直接使用 **`official_example`**。

期望提交（之后可能还有更新）：至少包含 `Ship Depth TTA on official_example...`（`dcd65645`）。

```bash
git log -1 --oneline   # 应在 official_example 上，且含 docs/official_example/
```

## 1. 代码

新机器需要：能访问 GitHub；推送用 SSH 则先添加公钥。只拉代码可用 HTTPS。

```bash
# SSH（已配密钥）
git clone -b official_example git@github.com:yang-yang-o-o/ultralytics.git
cd ultralytics

# 或 HTTPS
# git clone -b official_example https://github.com/yang-yang-o-o/ultralytics.git
# cd ultralytics

git status -sb
# 期望：## official_example...origin/official_example
```

Depth `augment=True` 已在分支内，**无需 patch**。

## 2. 环境

需要：Debian/Ubuntu 类系统、一块 CUDA GPU（原机为 RTX 3080 Ti 12GB；更小显存可把 batch 调小，指标应仍接近）、磁盘建议 **≥40GB**（ImageNet/Cityscapes 解压峰值）。

```bash
apt-get update
apt-get install -y git libgl1 aria2 unzip ffmpeg python3-venv curl

# 安装 uv（若没有）：https://docs.astral.sh/uv/getting-started/installation/
# 例：curl -LsSf https://astral.sh/uv/install.sh | sh

uv python install 3.12
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"

.venv/bin/yolo checks          # GPU 应可见
.venv/bin/yolo settings datasets_dir=/data/datasets   # 改成新机数据根
```

下文用 **`$DS`** 表示 `datasets_dir`（原机是 `/root/datasets`）。

```bash
export DS=$(.venv/bin/python -c "from ultralytics.utils import SETTINGS; print(SETTINGS['datasets_dir'])")
mkdir -p "$DS" runs/official_val
```

权重会落到仓库根或 `weights_dir`，无需预先拷贝。

## 3. 数据（val-only，对齐官方验证即可）

**不要**直接 `yolo detect val data=coco.yaml` 来触发下载：官方 yaml 会拉 **train2017（约 19GB）**。下面按任务只准备 val。

### 3.1 COCO（Detect / Segment）

```bash
source .venv/bin/activate
python - <<PY
from pathlib import Path
from ultralytics.utils import SETTINGS
from ultralytics.utils.downloads import download
from ultralytics.utils import ASSETS_URL

ds = Path(SETTINGS["datasets_dir"])
# 标签（含 segments，Detect/Seg 共用）
download([f"{ASSETS_URL}/coco2017labels-segments.zip"], dir=ds)
# 只下 val 图（约 1GB），不要 train2017
download(["http://images.cocodataset.org/zips/val2017.zip"], dir=ds / "coco/images")
PY
# labels zip 若留下空 coco/images/val2017，unzip 会被跳过：
empty="$DS/coco/images/val2017"
if [ -d "$empty" ] && [ -z "$(ls -A "$empty" 2>/dev/null)" ]; then
  rmdir "$empty"
  unzip -n "$DS/coco/images/val2017.zip" -d "$DS/coco/images"
fi

验收：`$DS/coco/images/val2017` 约 **5000** 张 jpg；`$DS/coco/annotations/instances_val2017.json` 存在。

### 3.2 COCO-pose（Pose）

```bash
python - <<PY
from pathlib import Path
from ultralytics.utils import SETTINGS
from ultralytics.utils.downloads import download
from ultralytics.utils import ASSETS_URL

ds = Path(SETTINGS["datasets_dir"])
download([f"{ASSETS_URL}/coco2017labels-pose.zip"], dir=ds)
PY

# 图像与 COCO val 共用，禁止再下 19GB train
POSE="$DS/coco-pose/images"
mkdir -p "$POSE"
rmdir "$POSE/val2017" 2>/dev/null || true   # 空目录会导致 ln 失败
ln -sfn "$DS/coco/images/val2017" "$POSE/val2017"
```

验收：`$DS/coco-pose/annotations/person_keypoints_val2017.json`；`val2017.txt` 约 **2346** 行；软链指向 coco val。

### 3.3 NYU Depth / DOTAv1

二者 yaml 带 Ultralytics assets URL，**第 4 节第一次 val 会自动下载**（约 1.5GB + 2GB），无需单独脚本。

验收（val 触发下载后）：NYU `images/val` 与 `depth/val` 各 **654**；DOTA `images/val` 与 `labels/val` 各 **458**。test 无标签是正常的。

### 3.4 Cityscapes val（Semantic）

官方站要账号。用 HF 镜像 **只解 val**：

- 仓库：`danjacobellis/cityscapes_source`
- 文件：`leftImg8bit_trainvaltest.zip`、`gtFine_trainvaltest.zip`

```bash
CS="$DS/cityscapes"
mkdir -p "$CS"
# 需能访问 Hugging Face（可 huggingface-cli login）
uv pip install huggingface_hub
huggingface-cli download danjacobellis/cityscapes_source \
  leftImg8bit_trainvaltest.zip gtFine_trainvaltest.zip --local-dir "$CS"
unzip -n "$CS/leftImg8bit_trainvaltest.zip" "leftImg8bit/val/*" -d "$CS"
unzip -n "$CS/gtFine_trainvaltest.zip" "gtFine/val/*" -d "$CS"

python - <<'PY'
from pathlib import Path
from shutil import copy2
from ultralytics.utils import SETTINGS

cityscapes_dir = Path(SETTINGS["datasets_dir"]) / "cityscapes"
leftimg8bit_dir = cityscapes_dir / "leftImg8bit"
gtfine_dir = cityscapes_dir / "gtFine"
for split in ("val",):  # 仅 val；train/test 可空
    src_image_dir = leftimg8bit_dir / split
    dst_image_dir = cityscapes_dir / "images" / split
    dst_mask_dir = cityscapes_dir / "masks" / split
    dst_image_dir.mkdir(parents=True, exist_ok=True)
    dst_mask_dir.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(src_image_dir.rglob("*_leftImg8bit.png")):
        relative_path = image_path.relative_to(src_image_dir)
        mask_path = gtfine_dir / split / relative_path.parent / image_path.name.replace(
            "_leftImg8bit.png", "_gtFine_labelIds.png"
        )
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        image_name = image_path.name.replace("_leftImg8bit", "")
        mask_name = mask_path.name.replace("_gtFine_labelIds", "")
        copy2(image_path, dst_image_dir / image_name)
        copy2(mask_path, dst_mask_dir / mask_name)
    print(split, "images", len(list(dst_image_dir.glob("*.png"))))
PY

mkdir -p "$CS/images/train" "$CS/masks/train"
# 省盘：确认 images/val、masks/val 各 500 后可删 zip 与 leftImg8bit/ gtFine/
```

逻辑与 `ultralytics/cfg/datasets/cityscapes.yaml` 的 `download:` 一致，只是 split 仅 `val`。

### 3.5 ImageNet-1k val（Classify）

- HF：`mlx-vision/imagenet-1k` 的 `val.zip`，逻辑大小必须是 **6692493311**
- 禁止 aria2 对着 HF `resolve` 二次跳转（易 403）；先 HEAD 拿签名 CDN
- 禁止对**稀疏空洞** zip 续传（详见 [02-datasets.md](02-datasets.md)）

```bash
IM="$DS/imagenet"
mkdir -p "$IM"
python - <<PY
from pathlib import Path
from urllib.request import Request, urlopen
from ultralytics.utils import SETTINGS

im = Path(SETTINGS["datasets_dir"]) / "imagenet"
src = "https://huggingface.co/datasets/mlx-vision/imagenet-1k/resolve/main/val.zip"
req = Request(src, method="HEAD")
with urlopen(req, timeout=60) as r:
    url = r.geturl()
print(url)
(im / "signed_url.txt").write_text(url)
PY
aria2c --allow-overwrite=true -x16 -s16 -k4M --file-allocation=falloc \
  -d "$IM" -o val.zip "$(cat "$IM/signed_url.txt")"

python - <<PY
import os
st = os.stat("$IM/val.zip")
assert st.st_size == 6692493311, st.st_size
assert st.st_blocks * 512 >= st.st_size * 0.95, "sparse zip, re-download"
print("zip ok")
PY

unzip -n "$IM/val.zip" "val/*" -d "$IM"
ln -sfn val "$IM/train"    # classify val 要求存在 train/
```

验收：`val/` 下 **1000** 个 synset 目录、约 **50000** 张图。

## 4. 评测（对齐结论）

在仓库根、已 `source .venv/bin/activate`。`project/name` 是为了出片脚本能找到 `val_batch_*`（脚本会在 `runs/**/<name>` 下查找）。

日志用 `tee` 追加即可，**不要**用 `exec > >(tee ...)`，否则后续所有终端输出都会进 log。

```bash
mkdir -p runs/official_val

yolo detect val  model=yolo26n.pt       data=coco.yaml        device=0 imgsz=640 \
  project=runs/official_val name=det_coco_e2e 2>&1 | tee -a runs/official_val/eval.log
yolo detect val  model=yolo26n.pt       data=coco.yaml        device=0 imgsz=640 end2end=False \
  project=runs/official_val name=det_coco_o2m 2>&1 | tee -a runs/official_val/eval.log
yolo segment val model=yolo26n-seg.pt   data=coco.yaml        device=0 imgsz=640 \
  project=runs/official_val name=seg_coco_e2e 2>&1 | tee -a runs/official_val/eval.log
yolo pose val    model=yolo26n-pose.pt  data=coco-pose.yaml   device=0 imgsz=640 \
  project=runs/official_val name=pose_e2e 2>&1 | tee -a runs/official_val/eval.log
yolo depth val   model=yolo26n-depth.pt data=nyu-depth.yaml   device=0 imgsz=768 \
  project=runs/official_val name=depth_single 2>&1 | tee -a runs/official_val/eval.log
yolo depth val   model=yolo26n-depth.pt data=nyu-depth.yaml   device=0 imgsz=768 augment=True \
  project=runs/official_val name=depth_tta_logls 2>&1 | tee -a runs/official_val/eval.log
yolo semantic val model=yolo26n-sem.pt  data=cityscapes.yaml  device=0 imgsz=2048 \
  project=runs/official_val name=sem_cityscapes 2>&1 | tee -a runs/official_val/eval.log
yolo classify val model=yolo26n-cls.pt  data="$DS/imagenet"   device=0 imgsz=224 \
  project=runs/official_val name=cls_imagenet 2>&1 | tee -a runs/official_val/eval.log
yolo obb val     model=yolo26n-obb.pt   data=DOTAv1.yaml      device=0 imgsz=1024 split=val \
  project=runs/official_val name=obb_val 2>&1 | tee -a runs/official_val/eval.log
```

### 验收（yolo26n，容差约 ±0.003）

| 检查 | 通过 |
|---|---|
| Detect e2e | COCO AP ≈ **0.400**（官方 40.1） |
| Detect o2m | COCO AP ≈ **0.408**（官方 40.9） |
| Seg e2e | box ≈ **0.398**，mask ≈ **0.339** |
| Pose e2e | pose AP ≈ **0.572** |
| Depth 单尺度 | δ1 ≈ **0.783** |
| Depth `augment=True` | δ1 ≈ **0.83–0.84**（官方表头 0.882 **对不齐**） |
| Semantic | mIoU ≈ **0.783** |
| Classify | top1 ≈ **0.714**，top5 ≈ **0.901** |
| OBB val | mAP50-95 ≈ **0.44**（官方 52.4 是 test 多尺度，**不要**对齐） |

读数方式见 [03-evaluation.md](03-evaluation.md)。缺口说明见 [05-known-gaps.md](05-known-gaps.md)。

## 5. 定性 MP4

```bash
python docs/official_example/scripts/render_clean_batches.py
python docs/official_example/scripts/make_results_video.py
# 可选 H.264：见 08-results-video.md
```

成品：`docs/official_example/yolo26n_official_val_results.mp4`。  
`render_clean_batches.py` 会**再跑一遍** Detect/Seg/OBB val，生成无文字框 + `*.classes.json` 图例（第 4 节那几次 val 只负责指标）。Semantic / Classify / Depth 画面来自第 4 节的 `name=` 目录。

## 6. 建议检查清单

```bash
# 代码
git rev-parse --abbrev-ref HEAD   # official_example

# 数据
test -d "$DS/coco/images/val2017"
test -L "$DS/coco-pose/images/val2017"
test -d "$DS/nyu-depth/images/val"
test -d "$DS/DOTAv1/images/val"
test -d "$DS/cityscapes/images/val" && test -d "$DS/cityscapes/masks/val"
test -L "$DS/imagenet/train"

# 环境
.venv/bin/yolo checks
```

原实验机可以删了：结论以本页第 4 节表 + README 总表为准；环境与数据按第 1–3 节重建即可。
