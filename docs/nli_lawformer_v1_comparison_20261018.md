# 旧版 Lawformer-NLI 与 XLM-R 的同输入对照（2026-10-18）

按“之前的 NLI”为服务器上的旧版
`/data/xujinfan/nli_experiment/models/finetuned/Lawformer-NLI` 理解，
与项目当前使用的 `xlm-roberta-large-xnli` 比较；Lawformer-NLI-v2 的后续同输入结果见本文末尾。
两者均用原文作 premise、命题作 hypothesis，三类标签显式映射，`max_length=512`、
温度 1.0、本地权重、GPU 1。55 条自然命题来自此前随机抽取的两批开发集样本，
标签为助手草稿，**不是独立人工金标准**。受控样例是手写诊断，不代表真实分布。

| 样本 | 旧版 Lawformer-NLI | XLM-R-large-XNLI |
| --- | ---: | ---: |
| 6 条基础接口样例 | 5/6 | 5/6 |
| 55 条自然命题与助手草稿一致 | **14/55** | **39/55** |
| 18 条受控事实命题 | **6/18** | **12/18** |
| 受控命题中的明确冲突 | **1/6** | **2/6** |
| 4 条强化否定措辞的冲突 | **0/4** | **1/4** |

两模型的 55 条自然命题输入文件 SHA256 均为
`7b3ecb1b1cc062d62988c1d6c8f4ad950313ff7da5d970e58d40b94d04ba3a32`，
案件文件 SHA256 均为
`f3a93e6407d99d01d905c1536933411f9cc654d8f9163d1f0b0d44e145344027`；
受控样例的输入哈希也已逐组核对一致。上述三组均没有超长跳过。
旧版 Lawformer 在 55 条上输出 support 12、insufficient 38、conflict 5；
XLM-R 分别为 41、14、0。旧版的平均最高类别概率约 0.431，XLM-R 约 0.933；
这只是分数分布，不等于概率校准质量。旧版在自然命题中仅有的一个暂定 conflict
也没有识别出来。当前证据不支持用旧版 Lawformer-NLI 的判断触发反馈。

旧版权重加载时，Transformers 提示配置中的 `bos_token_id=21128`、
`eos_token_id=21129` 超过模型 `vocab_size=21128`；本次实际成对编码使用了词表内的
`[CLS]`/`[SEP]` ID，77 条比较输入均无越界 token，最长成对输入 448 tokens。
进一步检查发现，旧版**随权重保存的 tokenizer** 对中文会输出类似 `«`、`º` 的
ByteLevel 符号 token；这 77 条的最大 token ID 仅为 192。显式加载该 tokenizer 的
fast 实现也得到相同结果，故并非适配器选择慢版造成。只读取了 v2 的 tokenizer
进行格式检查，它能正常输出中文字符 token。旧版的分词特征
很可能影响效果，但仅凭本次对照不能确定其因果贡献；为忠实测试旧版，未给它替换 tokenizer。
旧版测试配置在 [`nli_lawformer_v1.json`](../configs/nli_lawformer_v1.json)，
旧版逐条结果在 [`nli-lawformer-v1-20261018`](../results/nli-lawformer-v1-20261018/)，
XLM-R 同输入结果在 [`nli-xlm-roberta-20261018`](../results/nli-xlm-roberta-20261018/)。
它们均保存了模型信息、输入路径和哈希、三分类概率。

## 运行 Lawformer-NLI-v2

已备好独立配置 [`nli_lawformer_v2.json`](../configs/nli_lawformer_v2.json)，
指向本地 v2 权重。它不会修改项目默认 NLI 或任何模型文件。你可以在项目根目录运行：

```bash
/data/xujinfan/nli_experiment/.venv/bin/python experiments/smoke_nli.py \
  --config configs/nli_lawformer_v2.json \
  --probes results/nli-comparison-20261018/natural55_probes.jsonl \
  --cases data/cail2018/dev.jsonl \
  --output-dir results/nli-lawformer-v2-20261018/natural55

/data/xujinfan/nli_experiment/.venv/bin/python experiments/smoke_nli.py \
  --config configs/nli_lawformer_v2.json \
  --probes annotations/nli_contrast_probes_20261018.jsonl \
  --cases data/cail2018/dev.jsonl \
  --output-dir results/nli-lawformer-v2-20261018/contrast18

/data/xujinfan/nli_experiment/.venv/bin/python experiments/smoke_nli.py \
  --config configs/nli_lawformer_v2.json \
  --probes annotations/nli_contrast_variants_20261018.jsonl \
  --cases data/cail2018/dev.jsonl \
  --output-dir results/nli-lawformer-v2-20261018/contrast_variants4
```

每个输出目录必须尚不存在。先看每组的 `results.json`，特别是 conflict 召回和
高置信错误；不要将自然命题的助手草稿一致率当作正式准确率，测试集仍保持未用。

## Lawformer-NLI-v2 同输入复测（2026-10-18）

以上三组已在 `cuda:1` 完成推理。自然命题、受控命题和冲突变体的 probe 与案件文件
SHA256 均和 v1、XLM-R 结果一致，确保三模型接收相同输入。

| 样本 | Lawformer-NLI-v1 | XLM-R-large-XNLI | Lawformer-NLI-v2 |
| --- | ---: | ---: | ---: |
| 55 条自然命题与助手草稿一致 | 14/55 | **39/55** | 9/55（另 2 条跳过） |
| 18 条受控事实命题 | 6/18 | **12/18** | 9/18 |
| 受控命题中的明确冲突 | 1/6 | **2/6** | **2/6** |
| 4 条强化否定措辞的冲突 | 0/4 | **1/4** | 0/4 |

v2 在 55 条自然命题上输出 43 条 `insufficient`、9 条 `support`、1 条 `conflict`，
另有 2 条因输入超长记为 `unverifiable`。18 条受控样例中，它输出 12 条
`insufficient`、4 条 `support`、2 条 `conflict`。v2 未识别 4 条强化否定冲突，
自然命题上的草稿一致数也低于 XLM-R。现有结果不支持启用 v2 的自动 conflict 反馈；
自然命题标签仍是助手草稿，不能作为正式准确率。下一步应由人独立复核自然命题，重点
检查实体、数字、否定及支持/信息不足的边界，再决定是否保留 XLM-R 或尝试其他决策模型。

已准备 [55 条自然命题盲审表](../annotations/nli_blind_review_natural55_20261018.jsonl)。
逐条填写 `gold_label`（`support`、`conflict` 或 `insufficient`）和 `evidence`；
`probe_id`、`case_id`、`premise`、`hypothesis` 请保持原样。文件不含任何模型预测或草稿标签。

v2 逐条输出位于 [`nli-lawformer-v2-20261018`](../results/nli-lawformer-v2-20261018/)。

## 扩大 XLM-R 自然命题样本

开发集已有一轮 100 案 shadow 运行，包含 200 次 XLM-R 检查；按案件和命题去重后为
147 条自然命题（53 次是重复命题检查）。从 trace 导出后又用当前 XLM-R 配置独立重跑
这 147 条；新旧逐条预测 **147/147 一致**。新一轮结果在
[`nli-xlm-roberta-dev100-20261018/claims147`](../results/nli-xlm-roberta-dev100-20261018/claims147/)，
输入 probe SHA256 为 `439f2d09ac9ca40ee669104224c057cbda440814e0877128cefb2dccb51d80ab`，
案件文件 SHA256 与其他模型相同。147 条均可评估；新一轮输出 112 条 `support`、
34 条 `insufficient`、1 条 `conflict`。原始 200 次检查（含重复）输出 153 条 `support`、
46 条 `insufficient`、1 条 `conflict`，没有超长跳过。样本由 Qwen 生成且类别严重偏斜，因此只能作为开发集
自然命题诊断；单独这批数据不足以评估 conflict 召回或代表 NLI 的通用准确率。

已从 trace 导出 [147 条去重盲审表](../annotations/nli_xlm_roberta_dev_full_blind147_20261018.jsonl)，
包含事实、命题及重复检查 ID，不包含 NLI 输出。逐条填写 `gold_label`、`evidence` 后，
可用 `experiments/nli_diagnostics.py score` 直接对原 trace 评分。想评估冲突识别时，
还需另加一组由人工编写并复核、类别平衡的事实命题；不能靠这批自然样本中唯一一条
模型判为 conflict 的记录估计召回率。
