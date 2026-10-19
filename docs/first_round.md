# 首轮真实模型实验

本入口支持本地 Qwen chat-completions 服务、三分类 NLI、CAIL 数据准备和成对评测。
现有 `experiments/run_process_verification.py` 继续作为纯标准库 mock 示例。
本轮可用于汇报的汇总见 [首轮实验汇报](round1_report_20261018.md)。
已完成的真实模型开发集试跑及发现见 [2026-10-18 试跑报告](dev_pilot_20261018.md)。
最终罪名输出协议的开发集对照见 [输出协议报告](charge_output_protocol_20261018.md)。
NLI 命题首批核查草稿见 [标注报告](nli_annotation_draft_20261018.md)。
旧版 Lawformer-NLI 的同输入复测及 v2 命令见 [NLI 模型对照](nli_lawformer_v1_comparison_20261018.md)。

## 1. 环境和服务约定

从项目根目录运行：

```bash
source ~/envs/research/bin/activate
```

Qwen 服务部署在你自己的 serving 环境。客户端只需要标准库，默认连接：

```text
http://127.0.0.1:8000/v1
served model name: qwen3-8b
```

当前默认模型名取自服务器上 `/v1/models` 的实际返回值。若部署发生变化，修改
`configs/first_round.json` 的 `generator.base_url` 和 `generator.model` 以匹配实际名称。
连接本机回环地址时客户端直接访问服务，不经过环境代理。如服务需要密钥，在 `LOCAL_LLM_API_KEY` 环境变量中提供；
配置文件只保存变量名。框架不会保存请求头或密钥。

默认每案三次调用，每次最多输出 512 tokens；请求包含 `seed`，前两阶段使用 JSON object 格式，
最终阶段使用 JSON Schema 候选标签约束。请求还包含
`chat_template_kwargs.enable_thinking=false`。如果服务不支持 JSON mode，可在开发阶段
把 `json_mode` 改为 false；仍须输出合法 JSON。最终阶段仍要求服务支持 JSON Schema。
没有隐式重试、JSON 修复或第二次抽取调用，
截断/格式错误会作为失败保留，不能通过重跑测试集挑选输出。

主模型先用单张 A100 40GB，BF16、较小并发、8192 左右的服务上下文上限起步。
服务上下文必须容纳候选罪名、原文和三轮消息。固定模型 revision、serving 版本和解码参数，
在 `configs/first_round.json` 的 `deployment` 中记录实际部署信息。
同 seed 仅控制请求配置，不保证 GPU 推理跨环境完全一致。

服务器当前已有 `/data/xujinfan/nli_experiment/.venv`，其中包含 PyTorch 和 Transformers；
默认配置已指向该项目现有 XLM-R-large-XNLI 权重，并设置本地加载。要复用这个环境，
以它的 Python 执行以下实验脚本即可：

```bash
/data/xujinfan/nli_experiment/.venv/bin/python experiments/run_first_round.py --help
```

若改用其他环境，真实 NLI 需要额外依赖。在用于 NLI 的环境安装匹配 GPU 节点 CUDA/驱动的 PyTorch，之后安装：

```bash
python -m pip install -r requirements-nli.txt
```

NLI 默认使用服务器现有的 `xlm-roberta-large-xnli` 权重，配置中的 `device` 为 `cuda:1`。
这是**当前 NLI 进程的逻辑设备编号**：若该进程只通过 `CUDA_VISIBLE_DEVICES=1` 暴露了
第二张卡，配置应改为 `cuda:0`。也可设为 `cpu` 检查小样本。
Qwen 服务和 NLI 进程各自设置设备；客户端不会改变服务的 GPU 分配。

准备数据时，`--nli-tokenizer` 也应指向配置中的同一份本地模型目录，
并启用 `--local-files-only`。正式运行应固定同一模型 revision。

## 2. Qwen 部署好后先跑两个合成样例

无需 NLI 依赖或 CAIL 下载，就能检查服务连通性和三阶段生成：

```bash
python experiments/run_first_round.py \
  --data examples/smoke_cases.jsonl \
  --labels examples/smoke_labels.json \
  --groups baseline \
  --verbose
```

这是人工编写的接口样例，只有两个候选标签，不是 benchmark，分数不代表模型效果。
如果出现错误，检查该次运行目录中 `baseline/predictions.jsonl` 的 `error`，以及
`baseline/traces.jsonl` 的 `run_error`。模型原始输出、finish reason、调用 seed 和
可获得的 usage 会保留。启动时会检查 `/models`，不会因地址或模型名错误跑完整批。

## 3. 准备 CAIL2018-Small

官方说明和下载入口：<https://github.com/thunlp/CAIL>。
本工作区已从官方压缩包提取 `exercise_contest` 中的训练、验证和测试文件，
压缩包位于 `data/raw/CAIL2018_ALL_DATA.zip`。原始划分和压缩包原名保留；
该目录由 `.gitignore` 排除。准备命令可直接使用下列实际路径：

```bash
/data/xujinfan/nli_experiment/.venv/bin/python experiments/prepare_cail.py \
  --train data/raw/final_all_data/exercise_contest/data_train.json \
  --dev data/raw/final_all_data/exercise_contest/data_valid.json \
  --test data/raw/final_all_data/exercise_contest/data_test.json \
  --output-dir data/cail2018 \
  --dev-size 100 \
  --test-size 500 \
  --seed 42 \
  --nli-tokenizer /data/xujinfan/nli_experiment/models/pretrained/xlm-roberta-large-xnli \
  --local-files-only
```

准备脚本执行以下规则：

- 候选罪名来自训练集的全部 accusation 标签，仅去除首尾空白，保留官方标签中的方括号，使用精确匹配。
- dev/test 保留单被告、单罪名且标签在训练词表中的案件。
- 按实际 NLI tokenizer 筛选原文不超过 400 tokens 的案件。
- 默认排除含“本院认为”“判决如下”等潜在答案线索的案件，不改写事实。
  这是保守的启发式筛查，不保证消除所有答案线索，需抽查筛选前后文本。
- 使用去空白文本的 SHA256 去重，依次排除与训练集、全部开发集重叠的测试事实。
- 使用固定种子的 reservoir sampling，从合格案件中随机抽样；不足目标数量时保留实际数量并记录。

工作区已按这些参数生成 `data/cail2018/`。重新抽样时需指定一个全新的输出目录。
输出 `dev.jsonl`、`test.jsonl`、`labels.json`、`manifest.json` 和
`answer_cue_exclusions.jsonl`。manifest 记录原始文件哈希、抽样数量、过滤计数和输出哈希。
线索排除列表给出原文件行号，便于人工检查。输出目录已存在时会拒绝覆盖。
`data/` 默认不进入 Git。

结果属于 **CAIL2018 单被告、单罪名、短事实过滤子集**，不能直接声称为官方完整测试成绩。
去重只覆盖规范化后的完全重复文本，不覆盖语义近似重复。

## 4. 开发集先旁路记录，再决定阈值

```bash
/data/xujinfan/nli_experiment/.venv/bin/python experiments/run_first_round.py \
  --data data/cail2018/dev.jsonl \
  --groups baseline shadow \
  --limit 20 \
  --output-dir results/dev-smoke
```

`--limit` 使用配置中的 seed 从整个准备好的开发集随机选择案件，避免原文件按罪名排列时
只取到前几类。检查 20 案后，用新输出目录去掉 `--limit` 跑完开发集。输出目录始终不可覆盖。
不指定 `--groups` 时默认只运行 `baseline shadow`；反馈组需要明确列出。
各组使用相同候选列表、三个阶段和每案/每阶段 seed：

| 组名 | 验证和反馈 |
| --- | --- |
| `baseline` | 不调用 NLI，不发送反馈 |
| `shadow` | 两个检查点调用 NLI，记录结果但不反馈 |
| `self_check` | 不调用 NLI，每个检查点要求模型自行复核 |
| `nli_conflict` | 仅在选中 conflict 且超过阈值时反馈 |
| `nli_all` | 可选消融：同时允许 conflict 和 insufficient 反馈 |

每案阶段为“关键事实 → 候选罪名比较 → 最终罪名”。前两步输出 `analysis` 和
一个原子事实 `claim`。最终阶段仍输出 `{"charge":"罪名"}`，但请求附带 JSON Schema，
将 `charge` 限制为固定训练词表中的官方原样标签。客户端仍会验证标签是否在词表中，
不做模糊映射、隐藏修复或额外模型调用。原始响应和协议版本保留在运行记录中。
检查点固定在前两步之后。服务端必须支持 chat-completions 的 `json_schema` 响应格式。
反馈会包含被检查的具体命题，并进入**下一次**请求。金标准标签和 meta 不传给模型；
所有组都可见训练集导出的候选罪名列表。

NLI 以原始 fact 为 premise、claim 为 hypothesis，通过配置 `id2label` 映射三类标签，
对全部三个 logits 做 softmax。通用 `LABEL_0` 名称会拒绝加载，需要人工核实后配置
`verifier.label_map`，例如 `{"0":"contradiction","1":"neutral","2":"entailment"}`；
这个例子是 XLM-R-large-XNLI 的映射，不可套用到其他模型。

NLI 的**完整成对输入**超过上限时跳过该检查，记录 `unverifiable`、`abstained=true` 和
token 数，主模型继续。这个标签是控制状态，`{"unverifiable":1}` 不是模型判断概率，
不会触发反馈或进入 NLI 三分类分数。模型看到的事实不会被静默截短。
默认不做检索、窗口聚合或生成式证据解释。

## 5. 导出盲标注、查看 NLI 精度

从开发集 shadow trace 导出命题，文件不含 NLI 预测、干预信息或金标准罪名。
`--unique` 会按同案同命题去重，保留重复检查点 ID；`--limit` 在去重后按固定种子抽样：

```bash
python experiments/nli_diagnostics.py export \
  --trace results/dev-smoke/shadow/traces.jsonl \
  --unique \
  --output results/dev-annotations.jsonl
```

逐行填 `gold_label`，取值只能是 `support`、`conflict`、`insufficient`；
`evidence` 填原文依据，`notes` 可记录标注分歧。不要改动原文、命题、ID。
不完整标注会在统计中显示为未标注，不会猜标签。

“未提到暴力”和“明确未使用暴力”不同：前者对“使用了暴力”通常是信息不足，
后者才是明确冲突。优先标注自然生成的完整随机命题样本，再单独增加数字、否定、
主体混淆等诊断样例。小规模结果和人工标注本身都需要复核。

```bash
python experiments/nli_diagnostics.py score \
  --trace results/dev-smoke/shadow/traces.jsonl \
  --annotations results/dev-annotations.jsonl \
  --output results/dev-nli-metrics.json \
  --thresholds 0.8 0.9 0.95
```

输出三类混淆矩阵、NLI 可评估覆盖率、accuracy、NLL、Brier score，以及不同阈值的
conflict 精度和触发比例。未触发时精度为 null。没有自动选择阈值或拟合校准参数。
`configs/first_round.json` 中的 0.9/0.95 是起始参数，不能解释为已校准的真实可靠性。
在开发集确定规则后冻结配置，不能在测试集继续调阈值或提示词。

## 6. 冻结后的四组实验

```bash
/data/xujinfan/nli_experiment/.venv/bin/python experiments/run_first_round.py \
  --data data/cail2018/test.jsonl \
  --groups baseline shadow self_check nli_conflict \
  --output-dir results/test-round-1
```

每组输出 `traces.jsonl`、`predictions.jsonl`、`metrics.json`；运行根目录包含
`run.json`、固定词表和 `summary.json`。run 快照保存配置、客户端依赖版本、NLI 模型
commit、输入文件哈希和实际案件 ID。Qwen 服务版本和权重 revision 需要自行填入 deployment。

指标包括：

- Accuracy；固定全部词表的 Macro-F1；本次样本实际出现的 gold 类别 Macro-F1；每类支持数。
- 相对于 baseline 的错→对、对→错、双方正确/错误和净 accuracy 差，以及案件配对 bootstrap 区间。
- 接受反馈次数、NLI 实际前向次数、跳过次数、验证输入 tokens 和验证耗时。
- 主模型输入/输出/总 tokens、每案耗时和格式/运行失败数。

非法罪名用精确标签规则判断，计错而不作模糊映射。运行失败也计错，保留在分母中；
任务级错误会在跑完后返回非零退出码。未知 usage 不伪装成 0，汇总完整 token 数为 null，
另保存已报告部分。调用耗时包含冷启动影响；严谨效率比较需另做相同设置的预热测量。
各组输出预算相同，但实际输入/输出 tokens 仍可能不同，必须结合成本解释收益。

可从保存结果重新评测，无需再次调用模型：

```bash
python experiments/evaluate_first_round.py --run-dir results/test-round-1
```

重新评测会拒绝缺失、重复、事实不同或 gold 不一致的配对案件。当前不实现断点续跑；
进程中断后保留已有逐案输出，不能把不完整组当成完整测试，需用新目录重新运行。
人工正确反馈的前缀分叉实验、训练/拟合校准、长文本证据检索和多随机种子汇总留作后续扩展。

## 7. 本地验证

```bash
python -m unittest discover -s tests -v
```

测试使用脚本化 HTTP 响应和 NLI logits 检查完整编排、反馈送达、异常计分、数据隔离和
评测，不下载权重，也不能证明真实模型的 NLI 准确率或纠错效果。
