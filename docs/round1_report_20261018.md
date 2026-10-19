# 法律判决预测与 NLI 决策模块首轮实验汇报

**日期：** 2026 年 10 月 18 日  
**数据：** CAIL2018 `exercise_contest` 过滤开发子集  
**生成模型：** 本地 Qwen3-8B（OpenAI 兼容服务）  
**NLI 模型：** XLM-R-large-XNLI、Lawformer-NLI v1、Lawformer-NLI v2

## 摘要

本轮完成了法律罪名预测框架的真实模型试跑、最终输出格式对照，以及 NLI 决策模型的初步比较。数据使用 CAIL2018 筛出的 100 个开发案件、202 个候选罪名；另准备了 500 个测试案件，但测试集没有用于本轮模型选择或调参。

Qwen3-8B 使用 JSON Schema 将最终罪名限制为候选表中的官方标签后，开发集准确率从普通 JSON 自由输出的 **66/100 提高到 70/100**，无效标签从 12 个降到 0。要求模型输出罪名编号的方案准确率为 46/100，因此没有采用。

NLI 比较中，XLM-R 在 55 条自然命题上的助手草稿一致数为 39/55，Lawformer-NLI v1 为 14/55，v2 为 9/55（另有 2 条超长跳过）。这 55 条标签不是独立人工金标准。XLM-R 还在 100 案开发集的 147 条去重自然命题上完成独立重跑，与原 trace 中的预测类别 147/147 一致。现有结果支持继续把 XLM-R 作为候选验证器进行人工核查，但尚不足以启用 NLI 自动反馈或声称其具有可靠的冲突识别能力。

## 实验设置

CAIL2018 原始验证数据经过筛选，保留单被告、单罪名且罪名在训练标签词表中的案件，并按 NLI tokenizer 长度和答案线索规则过滤。开发子集抽样 100 案，训练标签共 202 类；本次 100 案中出现 56 个金标准罪名类别。该子集不是 CAIL2018 完整官方测试集。

Qwen3-8B 通过本地 chat-completions 服务运行。各轮协议对照使用同一开发集和候选标签词表。NLI 以案件事实为 premise、模型生成的命题为 hypothesis，输出 `support`、`conflict`、`insufficient` 三类。NLI 推理使用 GPU 1、本地权重、`max_length=512`、温度 1.0。v1 和 v2 使用各自随模型保存的 tokenizer，不互换 tokenizer。

## Qwen3-8B 罪名预测

### 最终输出协议对照

| 最终输出协议 | 准确率 | 无效标签 | 运行错误 | Macro-F1（本次出现的 56 类） |
| --- | ---: | ---: | ---: | ---: |
| 普通 JSON，自由输出罪名文字 | 66/100 | 12 | 0 | 0.547 |
| 从 202 个候选罪名中输出整数 ID | 46/100 | 0 | 0 | — |
| JSON Schema enum 限制为官方罪名字符串 | **70/100** | **0** | 0 | **0.582** |

JSON Schema 方案相对普通 JSON 自由输出有 4 例由错变对、0 例由对变错。原先 88 个合法输出保持相同标签；12 个无效输出中，4 个改为正确标签，8 个仍选错罪名。模型报告的总 token 数分别为 627,693 和 627,710。

编号方案虽然消除了无效标签，但相对普通 JSON 有 29 例由对变错、9 例由错变对，准确率降至 46/100。典型问题是前序分析给出正确罪名，最终返回的编号却对应其他罪名。

### NLI shadow 观察

在旧的自由文本输出协议下，对 100 案分别运行 baseline 和 shadow。两组准确率均为 66/100，Macro-F1（56 个出现类别）分别为 0.547 和 0.550；两组正确/错误案件集合相同，但有 4 案输出内容不同。shadow 只记录 NLI 判断、不将反馈发给 Qwen，因此该组不检验 NLI 是否能提升罪名预测。

200 次 NLI 检查中，XLM-R 判为 `support` 153 次、`insufficient` 46 次、`conflict` 1 次。按案件与命题去重后有 147 条不同命题，53 次检查是重复命题。以上是预测类别分布，不是 NLI 准确率。

## NLI 模型对照

### 自然命题与助手草稿标签

从开发集 Qwen 命题中选取 55 条已有暂定标签的自然命题，三个模型使用同一输入。结果是模型与助手草稿标签的一致数：

| 模型 | 一致数 | 主要预测类别分布 | 超长跳过 |
| --- | ---: | --- | ---: |
| XLM-R-large-XNLI | **39/55** | support 41，insufficient 14，conflict 0 | 0 |
| Lawformer-NLI v1 | 14/55 | support 12，insufficient 38，conflict 5 | 0 |
| Lawformer-NLI v2 | 9/55 | support 9，insufficient 43，conflict 1 | 2 |

这 55 个标签是助手起草，未经过独立人工复核；起草前也接触过少数 NLI 输出，不能作为正式准确率或严格盲标结果。模型间的一致数只能作为挑选待复核样本的线索。

### 受控命题诊断

受控组包含 6 个案件，每案各构造 1 条 support、conflict、insufficient 命题。另有 4 条加入更明确否定措辞的冲突变体。结果如下：

| 诊断样本 | XLM-R-large-XNLI | Lawformer-NLI v1 | Lawformer-NLI v2 |
| --- | ---: | ---: | ---: |
| 18 条受控命题总一致数 | **12/18** | 6/18 | 9/18 |
| 受控组的 6 条明确冲突 | **2/6** | 1/6 | **2/6** |
| 4 条强化否定冲突变体 | **1/4** | 0/4 | 0/4 |

受控样例数量小且由人工编写，不能代表真实分布。三个模型都没有稳定识别冲突，尤其对强化否定变体的表现不足。当前证据不支持任何一个模型的 conflict 判断直接触发 Qwen 自动反馈。

### 扩展 XLM-R 开发集复测

从 100 案 shadow trace 中去重得到 147 条自然命题，并用当前 XLM-R 配置单独重跑。147 条全部完成评估，预测类别为 support 112、insufficient 34、conflict 1；与旧 trace 中相同命题的类别预测 **147/147 一致**。这验证了预测类别的可复现性，但这些命题仍没有人工金标签，不能据此计算 NLI 准确率、精确率或召回率。

## 主要发现与限制

1. **输出约束解决了标签格式问题。** JSON Schema enum 将 100 案无效标签从 12 个降至 0，并将准确率提高 4 个百分点；剩余错误仍主要涉及罪名判断本身。
2. **XLM-R 是目前较值得继续人工核查的 NLI 候选。** 在 55 条暂定自然命题和 18 条受控命题上的结果优于两个 Lawformer 版本，但自然命题标签不是独立金标准，不能据此宣称模型准确率更高。
3. **冲突识别还没有得到验证。** 147 条自然命题中 XLM-R 只判出 1 条 conflict，受控诊断中三个模型对明确冲突和强化否定冲突的召回都偏低，无法估计真实 conflict 召回或反馈精度。
4. **Lawformer v1 的中文分词值得关注。** 检查发现其保存的 tokenizer 会把中文转成类似 ByteLevel 符号 token；这可能影响结果，但本次实验不能确认因果关系。v2 tokenizer 能正常输出中文字符 token，然而 v2 的本轮指标仍未显示出优势。
5. **XLM-R 重跑有 tokenizer 警告。** Transformers 提示 tokenizer 正则兼容问题；重跑类别与已有 trace 完全一致。正式实验前应固定并记录 Transformers/tokenizers 版本，确认 tokenizer 配置后再冻结运行环境。
6. **实验集范围有限。** 所有 Qwen 准确率来自 100 案过滤开发子集；未使用 500 案测试集。测试集尚未用于本轮选择或调参，后续应在开发阶段完成人工标注、模型选择和协议冻结后，再进行一次性测试。

## 后续工作

已导出 [147 条 XLM-R 自然命题盲审表](../annotations/nli_xlm_roberta_dev_full_blind147_20261018.jsonl)，表内不含模型预测。逐条填写 `gold_label`（support、conflict 或 insufficient）及 `evidence` 后，可对原 shadow trace 计算混淆矩阵、覆盖率、NLL、Brier score 和 conflict 阈值精度。还应补充由人工编写并复核的类别平衡冲突样例，再决定是否运行反馈组。完成开发集评估并冻结配置后，才使用 500 案测试子集报告最终预测结果。

## 结果文件

- Qwen 自由输出与 shadow：[`dev-full-20261018`](../results/dev-full-20261018/)
- Qwen 编号输出协议：[`dev-id-baseline-20261018`](../results/dev-id-baseline-20261018/)
- Qwen JSON Schema 输出协议：[`dev-schema-baseline-20261018`](../results/dev-schema-baseline-20261018/)
- Lawformer v1 与 XLM-R 同输入对照：[`nli-lawformer-v1-20261018`](../results/nli-lawformer-v1-20261018/)、[`nli-xlm-roberta-20261018`](../results/nli-xlm-roberta-20261018/)
- Lawformer v2 同输入对照：[`nli-lawformer-v2-20261018`](../results/nli-lawformer-v2-20261018/)
- XLM-R 147 条扩展复测：[`claims147`](../results/nli-xlm-roberta-dev100-20261018/claims147/)
