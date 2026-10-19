# 共享显式 checkpoint 的 A/B/C 过程反馈实验（2026-10-18）

## 数据与样本

实验复用服务器上已经冻结的 `data/cail2018/dev.jsonl`，没有重跑过滤流程，也没有读取 500 案测试集。数据 manifest 表明，这 100 案来自 CAIL2018 `exercise_contest/data_valid.json`：罪名词表只由训练集建立，共 202 类；保留单被告、单罪名、标签合法案件，使用原有 XLM-R tokenizer 过滤事实长度（不超过 400 tokens）、排除既有答案线索规则，并按无空白文本 SHA256 去重。100 案文件 SHA256 为 `f3a93e6407d99d01d905c1536933411f9cc654d8f9163d1f0b0d44e145344027`（见数据自身的 manifest 校验值）。

从这 100 案以 seed `20261018` 进行可复现的分层随机抽样：按事实 token 长度四分位分层、层内随机抽样，按层大小比例分配 30 个名额。抽样没有使用 gold charge、模型预测或纠错难度。实际案件、原始数据行号、金标准标签、事实长度、文件哈希、过滤条件及既有审阅重叠保存在 `results/forked-intervention-30-20261018/sampling_manifest.json`，所选案件保存在同目录 `cases.jsonl`。

这批案件与旧的 NLI 命题分析有重叠：与 147 条 XLM-R 自然命题导出重叠 30 案；与 55 条自然命题审阅集重叠 17 案；与两份非金标草稿分别重叠 9 案和 12 案。它们仍是开发集上的重复诊断材料，不能称为独立测试集。两份草稿明确不是独立人工金标。

## 实验实现

脚本：[`run_forked_intervention.py`](../experiments/run_forked_intervention.py)。每案只调用一次 Qwen 生成明确可观察的事实摘要和一条原子核查命题。程序保存这段完整消息历史，然后从完全相同的显式对话前缀分叉：A 直接要求最终罪名；B 对相同命题运行 XLM-R，再按固定策略决定是否发送反馈；C 在同一前缀和命题上使用已人工确认的事实状态。Chat API 不恢复模型内部隐藏状态，本实验只声称复用了可见消息和显式分析文本。

A/B/C 的最终调用使用同一 Qwen 服务、案件原文、202 类官方罪名列表、相同最终提示、相同逐案 seed、相同采样参数，并以官方字符串 JSON Schema enum 输出。A/B 之间唯一的提示差异是 B 可能多出的一条反馈消息。checkpoint、实际调用参数、seed、完整消息、原始响应、概率、token usage、调用耗时和最终结果保存在 `results/forked-intervention-30-20261018/ab-v2/`。对 29 个未触发反馈的案件，A/B 最终请求消息逐字相同，所有案件 A/B 使用相同最终 seed。A 先从 checkpoint 直接续写，之后才进入 B 的 NLI 验证分支。

B 保留每案 Support / Conflict / Insufficient 三类完整 softmax 概率，包括未触发反馈的样本。操作阈值固定为 Conflict `0.80`、Insufficient `0.85`；它们是预先设定的操作阈值，不表示 XLM-R 已校准。Support 不干预。Conflict 只提示回到原文核查，不伪造反例；Insufficient 明确表示证据不足，不能推导为命题为假。XLM-R 使用服务器已有权重，没有训练或修改。

人工 Oracle 模板已生成于 [`oracle_forked_intervention_30_20261018.jsonl`](../annotations/oracle_forked_intervention_30_20261018.jsonl)。模板只含案件原文和待核查命题，不含罪名金标准或模型判断。Oracle 导入要求 `gold_label`、原文证据、`annotation_status="confirmed"`、`annotator_id` 和 `annotation_time`；C 组要求 30 案全部有经人工确认的标注，否则拒绝启动，避免用部分案件伪装成成对对照。目前没有符合条件的 Oracle 标注，故 C 组未运行。完成标注后可运行：

```bash
/data/xujinfan/nli_experiment/.venv/bin/python experiments/run_forked_intervention.py run-oracle \
  --run-dir results/forked-intervention-30-20261018/ab-v2 \
  --oracle-annotations annotations/oracle_forked_intervention_30_20261018.jsonl
```

## A/B 初步结果

| 指标 | A：无验证反馈 | B：NLI 策略反馈 |
|---|---:|---:|
| 准确率 | 17/30（56.7%） | 17/30（56.7%） |
| 有效 Schema 输出 | 30/30 | 30/30 |
| NLI 检查 | 0 | 30 |
| NLI 预测分布 | — | Support 28，Conflict 0，Insufficient 2 |
| 实际反馈 | 0 | 1（Insufficient） |
| Qwen token（每组计入一次共享 checkpoint） | 121,079 | 121,123 |
| Qwen 调用（每组视角，含共享 checkpoint） | 60 | 60 |
| NLI 调用 | 0 | 30 |
| 累计延迟（每组计入共享 checkpoint） | 58.68 秒 | 59.76 秒 |

真实物理调用共 90 次 Qwen（30 次共享 checkpoint、30 次 A 续写、30 次 B 续写）和 30 次 NLI。两组各自的 token 数都计入同一份共享 checkpoint；按实际物理调用去重后的 Qwen usage 为 182,466 tokens（179,608 prompt、2,858 completion）。

### 对研究问题的回答

1. **在本次 30 案开发样本中，NLI 反馈没有提高最终准确率**：A、B 都是 17/30，配对准确性变化为 0；配对 bootstrap 95% 区间为 `[0, 0]`。样本量小且来自已有开发集，不能推断总体没有收益。
2. **观察到 0 次有害干预**，即没有“A 正确、且 B 触发反馈后变错”的案件；也观察到 0 次由错变对。仅 1 案触发反馈，该案最终标签从错误的“集资诈骗”变为错误的“合同诈骗”，仍未纠正。另 1 案虽然 NLI 判 Insufficient（0.7006）但低于阈值，因此未反馈。此处“有害干预”按最终罪名正确性变化定义；不能把所有标签变化归为伤害或纠错。
3. **人工 Oracle 的潜在收益尚无结果**：模板与校验逻辑已就绪，但当前没有人工标注。不能让 LLM 代填 Oracle，也不能根据这轮 NLI 预测冒充人工事实标签。待 30 条独立人工命题核查完成后，C 可与 A/B 配对运行。即使运行，C 仅估计理想事实判断反馈的潜在收益，不代表最优行动策略。

## 三类结果必须分开解释

- **Verification Quality**：当前 30 条还没有独立人工事实标签，所以不能报告 NLI accuracy、conflict precision/recall、校准误差或宣称验证器可靠。28/0/2 只是模型预测类别分布。
- **Intervention Effect**：当前可报告 A/B 最终罪名的配对变化；本轮准确性净变化、纠错数和有害干预数均为 0。
- **Feedback Utilization**：本轮只有一条反馈，最终预测改了但仍错误。系统没有在反馈后要求模型再输出可审计的事实修订，因此无法判断它是否正确理解或恰当利用了反馈。反馈送达不等同于成功利用，更不等同于成功纠错。

这 30 案与旧版 100 案 70% 结果使用的推理阶段协议不同：本轮只在事实分析与命题提取后设一个 checkpoint，再进行最终分类。因此 A 的 56.7% 是本轮协议下的配对控制基线，不应与 70/100 直接作模型优劣比较。整个实验未使用测试集，不能作为正式测试集成绩。
