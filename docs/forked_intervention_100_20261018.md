# A/B 过程反馈扩大实验：完整 100 案开发集（2026-10-18）

## 实验范围

针对 30 案试跑中仅触发 1 次反馈、难以判断反馈效果的问题，本轮使用原有 `data/cail2018/dev.jsonl` 的全部 100 案，按同一协议重新运行 A/B。没有复用先前 30 案的生成结果，也没有抽取、查看或使用 500 案测试集。30 案和本轮 100 案均来自开发集；100 案曾用于既有 Qwen/NLI 开发诊断，因此本轮是扩大后的开发集实验，不是独立盲测。本轮 A 的 58/100 也不能直接与旧三阶段协议下的 70/100 比较，旧实验与当前单 checkpoint 提示流程不同；A/B 的因果对照只在本轮内部成立。

每案只生成一次可观察的初步事实分析和原子命题，并保存消息历史。A 从共享 checkpoint 直接续写最终罪名，不执行验证；之后 B 对同一命题运行现有 XLM-R-large-XNLI，完整保存 Support / Conflict / Insufficient 概率，并使用固定阈值（Conflict 0.80、Insufficient 0.85）决定是否反馈。Qwen 两组使用同一案件、候选词表、最终提示、seed、解码参数和官方罪名 JSON Schema enum；仅反馈消息可不同。显式消息前缀分叉不代表恢复模型隐藏状态。

100 案的 Oracle 人工标注模板在 [`oracle_forked_intervention_100_20261018.jsonl`](../annotations/oracle_forked_intervention_100_20261018.jsonl)，目前全部是 pending，未包含罪名金标准。完成人工审阅后，使用 `run-oracle` 从已保存的 A/B checkpoint 直接续跑 C；该命令校验注释覆盖和模型参数，不重新生成前缀：

```bash
/data/xujinfan/nli_experiment/.venv/bin/python experiments/run_forked_intervention.py run-oracle \
  --run-dir results/forked-intervention-100-20261018/ab \
  --oracle-annotations annotations/oracle_forked_intervention_100_20261018.jsonl
```

## 结果

| 指标 | A：Qwen 无验证反馈 | B：Qwen + XLM-R 反馈 |
|---|---:|---:|
| 最终准确率 | 58/100（58%） | 59/100（59%） |
| 有效官方罪名输出 | 100/100 | 100/100 |
| 反馈 | 0 | 14 |
| 每组 Qwen 调用数（分组统计，含共享 checkpoint） | 200 | 200 |
| NLI 调用 | 0 | 100 |
| 每组 Qwen tokens（含一次共享 checkpoint） | 405,053 | 405,674 |
| 每组累计延迟（含一次共享 checkpoint） | 193.00 秒 | 195.88 秒 |

实际物理调用为 300 次 Qwen（100 次共享 checkpoint、100 次 A 续写、100 次 B 续写）和 100 次 NLI。去重后的 Qwen 用量是 610,892 tokens（601,422 prompt，9,470 completion）。

NLI 预测分布为 Support 78、Conflict 0、Insufficient 22；14 条 Insufficient 达到 0.85 阈值并触发反馈，另外 8 条低于阈值。Conflict 为 0，因此本轮没有由 Conflict 触发的反馈。

配对结果：58 案两组都正确，41 案两组都错误，1 案由错变对，0 案由对变错。准确率净变化为 **+1 个百分点**。1,000 次配对 bootstrap 的 95% 区间为 `[0, 0.03]`；只有一个不一致样本，证据仍不足以支持稳定提升的结论。

14 次反馈后，最终罪名有 2 案发生变化：

- `cail-dev-5070`：A 从预测“走私毒品”错判，B 在 Insufficient 反馈后改为金标准“非法经营”，这是本轮唯一一次由错变对。
- `cail-dev-10561`：A 预测“集资诈骗”，B 在 Insufficient 反馈后改为“合同诈骗”；两者都不符合金标准“非法吸收公众存款”，因此不是纠错。

没有出现 A 正确而 B 错误的案件，故按最终罪名准确性定义的有害干预为 **0/14**。有 12 次反馈没有改变最终罪名。反馈触发数不是成功纠错数；本轮只有 1 次成功纠错。

## 分开解读

- **Verification Quality**：没有对这 100 条新生成的命题完成独立人工事实标注。Support/Insufficient 的预测分布不能代替验证准确率；不能报告 NLI 的准确率、冲突召回率或校准情况。未触发 Conflict 也不证明它正确处理了冲突。
- **Intervention Effect**：A/B 配对观察到 1 次由错变对、0 次由对变错，净增 1/100。结果方向偏正，但样本中只有 1 个不一致对，区间包含零，仍属不确定。
- **Feedback Utilization**：14 次反馈后只有 2 个最终罪名变化，其中 1 个正确、1 个仍错、12 个不变。不能仅凭这些变化证明模型正确理解了局部事实反馈；本协议没有要求反馈后再提交可审计的事实修订。若要评价利用质量，应由盲审者对命题变化、原文证据和最终判断逐案标注。

该扩大实验重复使用了此前已经用于开发诊断的 100 案，结果可能受既往模型和人工审阅暴露影响。它回答的是当前开发样本上的配对效果，不是 500 案测试表现，也不应被描述成独立确认性实验。阈值仍是预设操作阈值，不代表 NLI 概率已校准；没有训练新模型、训练策略或改变 NLI 权重。

## 结果文件

- 完整 trace、概率、反馈决策、消息、调用用量和预测：`results/forked-intervention-100-20261018/ab/`
- 汇总：`results/forked-intervention-100-20261018/ab/summary.json`
- 运行配置及数据哈希：`results/forked-intervention-100-20261018/ab/run.json`
- Oracle 标注模板：[`oracle_forked_intervention_100_20261018.jsonl`](../annotations/oracle_forked_intervention_100_20261018.jsonl)
- 实验实现：[`run_forked_intervention.py`](../experiments/run_forked_intervention.py)
- 30 案先导报告：[`forked_intervention_20261018.md`](forked_intervention_20261018.md)
