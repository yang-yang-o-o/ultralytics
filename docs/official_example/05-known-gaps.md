# 05 · 已知缺口与未尽细节

## 已关闭（可认为对齐）

- Detect e2e / Detect `end2end=False`
- Segment、Pose、Semantic、Classify
- Depth **单尺度**文档脚注（0.783）

## 仍然开放

### 1. Depth 表头 δ1：0.839 vs 0.882

| 问题 | 说明 |
|---|---|
| 缺什么 | 官方完整 TTA+log-LS eval script |
| 本地做到哪 | 已实现近似；消融卡在 ~0.839 |
| 能否自行补齐 | **现有公开信息下不能** |
| 后续 | 等官方开源；或从内部脚本/issue 拿到精确超参后再验 |

### 2. OBB 官方 52.4（test 多尺度）

| 问题 | 说明 |
|---|---|
| 缺什么 | DOTA **test GT**（官方永不公开） |
| 本地 | val 单尺度 ≈43.9；test 只能出预测 |
| 正路 | 多尺度切片 → 提交 [DOTA Evaluation Server](https://captain-whu.github.io/DOTA/evaluation.html) 或邮件 `dotawebsite3@gmail.com` |
| 数据形态 | Ultralytics `DOTAv1.zip` 多为全图，非 `0.5/1.0/1.5` 切片；merge 脚本依赖 `__x___y` 命名 |

**明确不做**：寻找/使用非官方泄漏的 test GT。

### 3. 工程边角（复现时注意）

| 项 | 现象 | 处理 |
|---|---|---|
| COCO 空 `val2017/` | unzip 跳过 | 删空目录再解压 |
| coco-pose 空 images | 软链失败 | 删空目录后链到 coco |
| ImageNet 稀疏 zip | 大小对但内容空 | 完整重下 + 校验 `st_blocks` |
| HF aria2 403 | resolve URL 二次跳转丢 query | 先 HEAD 取签名 CDN |
| Classify 无 train | `FileNotFoundError` | `ln -sfn val train` |
| `pgrep` 误杀脚本 | 等待下载时匹配到自身 | 用更精确的进程匹配 |
| Depth/OBB `augment` | OBB 不支持；Depth TTA 已在本分支 | 见 04、07 |

### 4. 未跑 / 未深挖

- 更大模型（s/m/l/x）全量复现  
- OBB 正式多尺度切片 + 评测服提交拿回 52.4  
- Depth 与官方逐超参对齐（需内部脚本）  
- ADE20K semantic、KITTI depth 等旁路基准  

## 结论口径（对外）

> 在官方文档给出的**开源可复现协议**下，YOLO26n 各任务验证指标与发布值一致。  
> Depth 表头与 OBB test 多尺度属于**协议/数据不可本地完备复现**项：前者差约 4.3 δ1，后者需官方评测服。
