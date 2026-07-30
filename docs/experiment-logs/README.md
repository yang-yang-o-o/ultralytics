# Experiment Logs

本目录存放 Ultralytics 相关实验复盘记录，便于后续查阅与会话续聊。

## 约定

- 每个实验一个子目录：`docs/experiment-logs/<topic>-<YYYY-MM-DD>/`
- 核心文件：
  - `meta.json`：主题、时间、状态、关键路径（机器可读）
  - `SESSION.md`：完整会话复盘（人读 + 续聊主入口）
  - `NOTES.md`：后续追加的决策/结果短记（可选）
  - 专题文档（如 `QUALITY_IMPROVEMENT.md`）：**有学习价值的分析写细写全**，SESSION 只摘要并链接
- **同一会话内**：助手每次回答完毕后，应更新对应实验的 `SESSION.md`（并视需要更新 `meta.json` / 专题文档）
- **写作标准**：原理、原因、对策、实验表、术语与性价比排序尽量落在专题文档，避免只留一行口号式摘要

## 实验索引

| 目录 | 主题 | 日期 | 状态 |
|------|------|------|------|
| [beer-yolo26n-seg-2026-07-18](./beer-yolo26n-seg-2026-07-18/) | beer YOLO26 seg→seg6d；最优 `BEST_CONFIG.md`；BOP/IC-BIN 见 `BOP_DATASETS_SURVEY.md` + `ICBIN_SEG6D_PROGRESS.md` | 2026-07-18 | beer 最优 vocbg-ep100；IC-BIN PBR25 训到 ep8 待 resume |
