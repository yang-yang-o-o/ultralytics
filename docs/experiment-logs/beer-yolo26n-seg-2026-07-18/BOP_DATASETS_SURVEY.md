# BOP Challenge 数据集调研（小规模 / 快速训测优先）

> 日期：2026-07-19  
> 目的：选可较快跑通的 BOP 数据，把我们的 seg6d（RGB + 9 控制点 + PnP）与论文/榜单结果对比。  
> 官方入口：[BOP Datasets](https://bop.felk.cvut.cz/datasets/) · [Challenge](https://bop.felk.cvut.cz/challenges/) · HF：`bop-benchmark/<name>`

## 和我们方法的对齐方式

- 我们当前是 **RGB → 检测/分割 + 9×2D 点 → PnP → ADD / 投影误差**，接近 BOP 的 **model-based RGB** 路径。
- 正式对比应报 BOP 官方指标（**AR / MSSD / MSPD / VSD** 等，经 [bop_toolkit](https://github.com/thodan/bop_toolkit)），不能只报 beer 上的 Acc@5px。
- 多数 Core 集提供 **50K PBR** 训图；「小」主要指 **物体数、官方 used test 子集、是否有真实训图**，而非没有 PBR。快速迭代可 **先 subsample PBR（如 5k–10k）** 或 **按物体子集** 训。

## 规模速查（BOP Challenge 2024 表，Classic）

| 数据集 | 物体数 | 官方 used test 图 | used test 实例 | 真实训图 | PBR | 在 Core 榜？ | 体感「小」程度 |
|--------|--------|-------------------|----------------|----------|-----|-------------|---------------|
| **IC-BIN** | **2** | **150** | 1786 | 无 | 50K | 是 | ★★★★★ 最小物体数 |
| **TUD-L** | **3** | **600** | 600 | **~38K** | 50K | 是 | ★★★★ 物体少+有真实训图 |
| **LM-O** | 8 | **200** | 1445 | 无（用 LM 的 PBR） | 50K | 是 | ★★★★ 测试子集小、论文最爱报 |
| HOPEv1 | 28 | 188 | ~2.9K | 无 | 50K | Extra | ★★★ 测图少但物体多 |
| HB | 33 | 300 | 1630 | 有 val | 50K | 是 | ★★★ 官方测子集适中 |
| ITODD | 28 | 721 | 3041 | 仅 val 有 GT | 50K | 是 | ★★ 工业灰图；test GT 不公开 |
| T-LESS | 30 | 1000 | 6423 | ~37K | 50K | 是 | ★★ 经典但偏大 |
| YCB-V | 21 | 900 | 4123 | ~113K | 50K | 是 | ★ 体量大 |
| LM | 15 | 3000 | 3000 | 无 | 50K | Extra | 单目标经典；总测图多 |
| HOT3D / HANDAL / HOPEv2 | 多 | 大 | 大 | 视集合 | 50K | H3 | 不适合「快速」 |

数据来源：BOP Challenge 2024 论文 Table 1（[arXiv](https://arxiv.org/html/2504.02812v2)）。

## 推荐优先级（快速训测 → 再扩大）

### 1. IC-BIN（首选试跑）

- **为什么**：Core 里物体最少（2）；used test 仅 150 张；榜单几乎每篇都报 IC-BIN AR，方便对齐。
- **场景**：bin-picking、重遮挡、多实例（与我们 beer 单实例不完全同分布，但利于测「遮挡+多目标」）。
- **快速做法**：下 `icbin` base+models+`test_bop19`+PBR；PBR **先抽 5k–10k** 或只训 2 类全量但短 epoch；用 `test_targets_bop19.json` 评测。
- **注意**：无真实训图，依赖 PBR；多实例标签要和我们 dual-label（pose+seg）管线对齐。

### 2. TUD-L（首选「有真实训图」的小集）

- **为什么**：仅 **3** 个物体；有 **大规模真实训图**（~38K），可少依赖 PBR；used test 600、单实例为主，和 YOLO6D/beer 设定更接近。
- **场景**：光照变化明显（Light）。
- **快速做法**：可只用 real train 子集（例如每物体几千张）+ 短程课程学习，再与榜单 TUD-L AR 对比。

### 3. LM-O（论文对比刚需，中等成本）

- **为什么**：遮挡版 Linemod，**200** 张 used test；几乎所有 6D 论文主表都有 LM-O。
- **场景**：杂乱+遮挡；8 类。
- **快速做法**：与 LM 共用 PBR；可先 **单物体**（如 ape）做 smoke（经典 SingleshotPose/YOLO6D 路线），再扩到 8 类。
- **注意**：正式分数必须按 BOP 协议（全部 used targets），单物体结果只能作内部对照。

### 4. LM 单物体 smoke（工程最快，非正式榜）

- 每物体约一千级相关图（历史 RGB 流程）；适合验证「BOP 格式 → 我们 seg6d 标签 → PnP/ADD」流水线。
- **不能**单独当作 BOP 挑战分数。

## 不太建议作为「第一枪」的

| 集 | 原因 |
|----|------|
| YCB-V / T-LESS | Core 标杆但训测都大，适合流水线成熟后 |
| ITODD / HB | 工业/家用；ITODD test GT 不公开（只能交服务器） |
| HOT3D / HANDAL | 体量大、任务偏 egocentric / unseen onboarding |
| BOP-Industrial 2025 | 多视角工业，超出当前单目 RGB seg6d 范围 |

## 下载与评测（备忘）

```bash
pip install -U "huggingface_hub[cli]"
export DATASET_NAME=icbin   # 或 tudl / lmo / lm
huggingface-cli download bop-benchmark/$DATASET_NAME \
  --local-dir ./$DATASET_NAME/ --repo-type=dataset
# 评测：bop_toolkit + test_targets_bop19.json（或 bop24）
```

HF 命名常见：`lm`, `lmo`, `tless`, `ycbv`, `tudl`, `icbin`, `hb`, `itodd`, `hope`。

## 和 beer 实验的关系（建议路线）

1. **保持** `BEST_CONFIG.md` 在 beer 上迭代。  
2. **并行** 用 **IC-BIN 或 TUD-L** 打通 BOP 数据转换 + `bop_toolkit` 出分（可先 subsample）。  
3. 稳定后上 **LM-O** 报可与论文直接对比的数字。  
4. 指标映射：我们 beer 的 Acc@5px / ADD@0.1d ≈ 诊断用；对外统一报 BOP AR（及 MSSD/MSPD）。

## IC-BIN 落地进度（2026-07-20）

详见 **[`ICBIN_SEG6D_PROGRESS.md`](./ICBIN_SEG6D_PROGRESS.md)**。摘要：

| 阶段 | 结果 |
|------|------|
| 下载 | `base+models+train+test_bop19` + 全量 `train_pbr.zip`（≈22GB） |
| 转换 | `examples/icbin_seg6d_*.py`；`icbin-seg6d.yaml` |
| 渲染 train smoke | BOP lite AR = **0** |
| PBR 5 场景 / 8ep | BOP lite AR = **0.064**（MSPD 0.128）← 首个非零分 |
| PBR 25 场景 / 40ep | 训到 **ep8** 暂停，待 `--resume`；尚未出 BOP 分 |

推荐优先级不变：IC-BIN（进行中）→ TUD-L → LM-O。

## 参考

- https://bop.felk.cvut.cz/datasets/  
- https://bop.felk.cvut.cz/challenges/  
- BOP Challenge 2024：https://arxiv.org/html/2504.02812v2  
