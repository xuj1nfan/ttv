# CAIL2018 过滤测试集 500 案 A/B 实验（2026-10-18）

## 冻结设置

本次使用已准备的 `data/cail2018/test.jsonl` 全部 500 案。文件 SHA256 为 `712200c6f83fb271cee4874e4c8532ff39ec8de2b6cf979a3d8b311bc1dd1f4e`，与 CAIL manifest 一致。数据仍是 CAIL2018 `exercise_contest` 的过滤子集：单被告、单罪名、罪名在训练集建立的 202 类词表中、事实长度不超过既定 400-token 限制并通过原答案线索过滤；不是完整官方测试集。

逐案运行顺序为：生成一次可观察事实分析与命题 checkpoint；A 不做验证，直接续写；随后 B 对同一命题运行 XLM-R-large-XNLI，保存 Support / Conflict / Insufficient 全部概率，按固定阈值 Conflict `0.80`、Insufficient `0.85` 决定是否反馈。两组复用相同的 checkpoint、最终提示、模型、候选罪名列表、逐案 seed 和官方罪名字符串 JSON Schema enum。没有训练或修改模型，没有调整提示、阈值或策略。金标准只用于离线评分。

## 主要结果

| 指标 | A：Qwen 无反馈 | B：Qwen + NLI 反馈 |
|---|---:|---:|
| 最终准确率 | 292/500（58.4%） | 294/500（58.8%） |
| 有效官方罪名输出 | 500/500 | 500/500 |
| NLI 调用 | 0 | 500 |
| 实际反馈 | 0 | 79 |
| 每组 tokens（含一次共享 checkpoint） | 2,031,553 | 2,035,118 |
| 每组累计延迟（含一次共享 checkpoint） | 1,002.15 秒 | 1,017.11 秒 |

实际物理调用共 1,500 次 Qwen（500 次共享 checkpoint、500 次 A 续写、500 次 B 续写）及 500 次 NLI。去重后的 Qwen usage 为 3,064,316 tokens（3,015,515 prompt、48,801 completion）。

配对正确性变化：两组都正确 290 案、两组都错误 204 案、A 错而 B 对 4 案、A 对而 B 错 2 案。准确率净差 **+0.4 个百分点**。1,000 次配对 bootstrap 的 95% 区间为 `[-0.6, +1.4]` 个百分点，包含零，不能据此确认稳定提升。

## 反馈和有害干预

NLI 预测为 Support 387、Insufficient 107、Conflict 6。阈值触发 79 次：Insufficient 77 次，Conflict 2 次。完整类别概率、阈值判定和原始逐案数据保存在 B 组 JSONL。

在 79 次真实反馈触发的案件中：

- 1 案由错变对（`cail-test-28307`）；
- 2 案由对变错（`cail-test-6099`、`cail-test-22431`），即按最终罪名正确性定义的 **2 次有害干预**；
- 1 案改变标签但仍错误；
- 其余 75 案最终罪名不变。

所以本轮有干预反馈的同时，也观察到纠错与误伤，净准确率收益只有 2/500。反馈发生 79 次不等于纠错 79 次。

## 解读限制

对 421 个未触发反馈的案件，A/B 最终请求消息逐字一致、seed 一致，但仍有 5 案输出不同；其中 3 案由错变对、2 案标签变化但仍错。这显示服务端在相同 seed 下仍有非确定性。因此全部 A/B 配对差异不能简单归因于反馈；在触发反馈的子集中观察到的 1 次纠错和 2 次误伤也应结合这一运行随机性谨慎解释。测试集已用于本次评估，不应继续用于调参或选择策略。

**Verification Quality** 尚未评估：没有针对这些 500 条局部命题的独立人工事实标签，NLI 预测分布不是验证准确率。**Feedback Utilization** 也未单独标注：最终标签变化不说明 Qwen 是否正确理解了命题级反馈；本轮仅能报告最终罪名的配对变化。**Oracle C** 未运行，因为没有人工确认的局部事实标签；不能把模型判断当作 Oracle。

## 文件

- 逐案 checkpoint、完整消息、概率、反馈和预测：`results/forked-intervention-test500-20261018/ab/`
- 配对统计：`results/forked-intervention-test500-20261018/ab/summary.json`
- 运行配置、模型元数据及数据哈希：`results/forked-intervention-test500-20261018/ab/run.json`
- 协议与实现：[`run_forked_intervention.py`](../experiments/run_forked_intervention.py)
