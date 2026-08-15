# 02 · 数据集下载与整理

默认数据根：`/root/datasets`（Ultralytics `settings` 中 `datasets_dir`）。  
原则：**优先 val-only**，够对齐官方验证指标即可。

## 一览

| 任务 | 数据集 | 本地路径 | 规模（本实验） | 来源 |
|---|---|---|---|---|
| Detect / Seg | COCO val2017 | `coco/` | 5000 图 | 官方 / Ultralytics 脚本 |
| Pose | COCO-pose | `coco-pose/` | 2346 有人标签；图软链 coco | 官方标签 + 共享图 |
| Depth | NYU Eigen test | `nyu-depth/` | 654 | Ultralytics `nyu-depth.zip` |
| OBB | DOTAv1 | `DOTAv1/` | val 458 / test 937（无标签） | Ultralytics `DOTAv1.zip` |
| Semantic | Cityscapes val | `cityscapes/` | 500 图+mask | HF `danjacobellis/cityscapes_source` |
| Classify | ImageNet-1k val | `imagenet/` | 1000 类 / 50000 图 | HF `mlx-vision/imagenet-1k` `val.zip` |

小集（smoke）：`coco8*`、`dota8`、`depth8`、`imagenet10`、`cityscapes8` — 由官方 yaml/`benchmark` 自动拉。

## COCO（Detect / Segment）

按 `coco.yaml` 惯例下载 val2017 图与 instances 标注。注意：

- 标签 zip 可能留下空 `val2017/`，导致 unzip 被跳过 → **删空目录再解压**。
- Pose 的 `coco-pose/images/val2017` 若为空目录，`ln -sfn` 会失败 → 删空目录后链到 `coco/images/val2017`。

日志：`runs/official_val/download.log`。

## COCO-Pose

- 标注：pose 官方包。
- 图像：与 COCO val2017 共享（软链），避免重复占盘。
- 本机有效带人标签约 **2346**（与官方 pose val 子集一致）。

## NYU Depth

```bash
# 通常由 data=nyu-depth.yaml 触发自动下载
# 或手动：ultralytics assets nyu-depth.zip → /root/datasets/nyu-depth
```

布局：`images/val` + `depth/val`（`.npy`，路径把 `/images/` 换成 `/depth/`）。

## DOTAv1（OBB）

Ultralytics 发布的 `DOTAv1.zip`：

- `images/train|val|test`，`labels/train|val`
- **test 无 labels**（官方政策）
- 当前包多为原图尺寸，**不是** `rates=[0.5,1.0,1.5]` 多尺度切片；与官方 test 多尺度表不完全同构

官方多尺度切片参考：`ultralytics.data.split_dota.split_trainval / split_test`。

## Cityscapes（Semantic）

官方站点需账号。本实验镜像：

- HF：`danjacobellis/cityscapes_source`
  - `leftImg8bit_trainvaltest.zip` ≈ 11592327197 B
  - `gtFine_trainvaltest.zip` ≈ 252567705 B

**仅解压 val**，再按 `cityscapes.yaml` 的 download 脚本逻辑整理为：

```
cityscapes/
  images/val/*.png
  masks/val/*.png     # 来自 *_gtFine_labelIds.png，文件名去掉 _gtFine_labelIds
  images/train/       # 可空（仅 val）
  masks/train/
```

完成后可删原始 zip 与 `leftImg8bit/`、`gtFine/` 中间目录以省盘。

命令示例：

```bash
CS=/root/datasets/cityscapes
unzip -n "$CS/leftImg8bit_trainvaltest.zip" "leftImg8bit/val/*" -d "$CS"
unzip -n "$CS/gtFine_trainvaltest.zip" "gtFine/val/*" -d "$CS"
# 然后跑 yaml 中的 organize 逻辑（或见会话中的 Python copy 脚本）
```

日志：`runs/official_val/download_cityscapes.log`、`prepare_datasets.log`。

## ImageNet-1k val（Classify）

- HF：`mlx-vision/imagenet-1k` 的 `val.zip`，期望大小 **6692493311**
- 布局已是 `val/<synset>/...JPEG`

### 关键踩坑：稀疏损坏 zip

用 aria2 **续传半成品**时，可能出现：

- `stat` 逻辑大小 = 期望值
- 但 `st_blocks*512` 远小于逻辑大小（大量空洞）
- `unzip -t` 报错 / 解压中途失败

**正确做法**：删掉坏文件，**完整重下**，并校验物理大小：

```bash
# falloc 预分配可避免空洞假完成
aria2c --allow-overwrite=true -x16 -s16 -k4M --file-allocation=falloc \
  -d /root/datasets/imagenet -o val.zip "<HF_SIGNED_CDN_URL>"

python3 - <<'PY'
import os
st=os.stat('/root/datasets/imagenet/val.zip')
assert st.st_size == 6692493311
assert st.st_blocks*512 >= st.st_size * 0.95
print('ok', st.st_size, st.st_blocks*512)
PY
```

获取签名 CDN URL：对  
`https://huggingface.co/datasets/mlx-vision/imagenet-1k/resolve/main/val.zip`  
做 HEAD，读最终 `geturl()`。  
**不要**把 resolve 链接直接丢给 aria2 二次跳转（易 403）；用签名后的 CDN URL。

解压：

```bash
unzip -n /root/datasets/imagenet/val.zip "val/*" -d /root/datasets/imagenet
# Ultralytics classify 检查要求存在 train/：
ln -sfn val /root/datasets/imagenet/train
```

期望：`val` 下 **1000** 类、**50000** 张图。

日志：`runs/official_val/download_imagenet.log`。

## 下载加速建议

| 场景 | 建议 |
|---|---|
| HF 大文件 | 签名 CDN + `aria2c -x16` |
| 断点续传 | 确认非稀疏后再 `-c` |
| 仅验证 | 只解压 val，删 zip |

## 相关日志目录

```
runs/official_val/
  download.log
  download_cityscapes.log
  download_imagenet.log
  prepare_datasets.log
  eval.log
```
