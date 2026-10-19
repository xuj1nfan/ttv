# 测试时法律过程验证 MVP

Python 3.11，同步执行。核心框架和 mock 示例仅使用标准库。
真实模型入口已支持本地 Qwen chat-completions 服务、三分类 NLI、CAIL 数据准备和成对评测；
NLI 的可选依赖见 `requirements-nli.txt`。首轮运行步骤见 [首轮实验说明](docs/first_round.md)。

流程为：

```text
Generator.step
  -> Scheduler.get_checkpoint
  -> Generator.extract_claim
  -> Verifier.verify(case_text, local_claim)
  -> InterventionPolicy.decide
  -> Generator.feedback (仅在策略选择介入时)
  -> 下一次 Generator.step
```

验证对象是中间步骤中的局部命题。生成器结束后不再验证或发送反馈。

## 运行与迁移

在项目根目录运行（Windows PowerShell 和 Ubuntu bash 命令相同）：

```bash
python experiments/run_process_verification.py
python -m unittest discover -s tests -v
```

Ubuntu 如果仅有 `python3` 命令，将上面的 `python` 替换为 `python3`，并确认版本为
3.11。复制 `framework/`、`experiments/run_process_verification.py`、
`configs/process_verification.json`、`tests/test_framework.py` 和本说明到服务器
repository；保留原有目录结构，合并已有同名目录时注意文件名冲突，无需搬动原有实验。

可从其他工作目录用脚本的绝对路径直接运行。默认配置按脚本位置查找，默认输出目录
相对于项目根目录；显式传入的相对命令行路径相对于当前工作目录。

```bash
python experiments/run_process_verification.py --config configs/process_verification.json
python experiments/run_process_verification.py --trace-path results/traces/my-run.jsonl
```

默认每次生成唯一的 JSONL 文件，并在旁边保存实验配置快照。指定 `--trace-path`
时追加到已有文件，配置快照带唯一后缀以避免覆盖；分析不同实验时建议使用不同文件。
`results/` 和 Python 缓存已列入 `.gitignore`。

## 模块与接口

| 文件 | 职责 |
| --- | --- |
| `framework/schema.py` | Claim、Decision、Intervention、ReasoningState、Checkpoint、TraceRecord、RunResult |
| `framework/verifier.py` | `Verifier.verify(case_text, claim) -> Decision` |
| `framework/mock_verifier.py` | 按命题文本查找固定概率，未知命题默认 insufficient |
| `framework/scheduler.py` | `Scheduler.get_checkpoint(state)` 和固定时点实现 |
| `framework/intervention.py` | `InterventionPolicy.decide(decision)` 和独立阈值策略 |
| `framework/generator.py` | start、step、extract_claim、feedback 抽象接口及 MockGenerator |
| `framework/runner.py` | 串行交互、反馈送达、推理观察、JSONL 保存 |
| `framework/__init__.py` | 公共接口导出 |
| `experiments/run_process_verification.py` | 配置读取与组件组装，无模型路径硬编码 |
| `configs/process_verification.json` | 案件、脚本、mock 概率、检查点、阈值、输出设置 |
| `tests/test_framework.py` | 标准库 unittest 行为测试 |

### 状态与生成器

生成器原地维护 `ReasoningState`。每次 `step` 必须将 `current_step` 加一并追加一个
文本步骤，结束时设置 `finished=True` 和 `answer`。`feedback` 只排队消息，不能推进
步骤；下一次 `step` 使用消息继续分析。适配真实生成器时，应把消息加入后续调用的
上下文，并从当前步骤提取明确的局部命题，而非把整个答案提交给验证器。

MockGenerator 的步骤文本就是命题；接收反馈后按配置在下一步输出预设修订。
`answer_after_feedback` 只要收到过反馈就选择预设答案，**不是根据修订内容推导结论**。
它只证明消息能进入后续步骤，不能说明真实模型的纠错收益。

### Scheduler

默认 20%、50%、80%，步数为 `ceil(fraction * total_steps)`。总长度可通过 scheduler
的 `total_steps` 参数传入，也可通过 `state.metadata["total_steps"]` 提供。
百分比依赖计划长度，不能代表未知长度推理的真实完成百分比。

步数模式不需要总长度，将配置中的 `checkpoints` 替换为：

```json
{"type": "fixed", "step_checkpoints": [2, 5, 8]}
```

同一步的多个百分比检查点合并为一次验证，名称和 `metadata.targets` 保留对应目标。
结束时不再检查；如果实际推理提前结束，尚未到达的检查点不会补跑。自适应、不确定性、
熵或学习型 scheduler 可实现相同接口，所需观测量放入 state.metadata。

### Decision 与 intervention

初始标签为 `support`、`conflict`、`insufficient`，对应 NLI 的 entailment、
contradiction、neutral。`confidence` 明确定义为所选标签的概率；概率必须有限、
在 [0, 1] 内且总和为 1。evidence 可为 None。

标签使用字符串而非封闭枚举，因此新标签无需修改 runner。额外 score、choice、
模型原始标签、校准方法等放在 metadata。本版 Decision 仍要求归一化概率；未来只有
score 的模型需在适配层转换为概率或显式扩展 schema，不能把任意分数伪装成概率。
所有 metadata 必须可 JSON 序列化。

阈值使用 `>=`：选中 conflict 且 confidence >= conflict_threshold 时介入，
选中 insufficient 且 confidence >= insufficient_threshold 时介入。support 和
尚无策略规则的新标签不介入。阈值来自配置，runner 不解释标签或阈值。

## Trace 约定（schema_version = 1.0）

一行一条 JSON，UTF-8。正常运行包含 N 条 `record_type="checkpoint"` 和一条
`record_type="case_result"`。每次 checkpoint 包含：

```text
case_id
checkpoint: {name, step, progress, metadata}
reasoning_before: [截至检查点的完整步骤快照]
claim: {text, source_step, claim_type, metadata}
decision: {label, confidence, probabilities, evidence, metadata}
intervene
feedback
intervention_reason
reasoning_after: [再生成一步之后的完整步骤快照] 或 null
metadata: {checkpoint_id, observed_at_step, ...}
```

`reasoning_after` 同样记录不介入时的自然续写；它不是即时调用 feedback 后的状态，
也不自动代表纠错成功。checkpoint 先进入内存 trace，下一步成功生成后写入 JSONL。
如果生成、验证或反馈等抛出异常，待完成的 checkpoint 仍会写出，缺失续写时保持
null，同时记录 `run_error` 并向调用方重新抛出异常。强制终止进程或断电时，尚未落盘
的最后一个 checkpoint 可能丢失。本版不实现崩溃恢复。

case_result 保存原始案件、所有步骤、最终答案、介入次数和状态 metadata，方便核对
最终结果。用 `record_type` 筛选 checkpoint 行进行概率轨迹和介入次数分析。
checkpoint_id 在单次 case 内有序；跨重复 case 的关联需要结合实验文件路径。

验证器错误率、纠错率和 harmful intervention 需要额外的命题标注或人工评估；
final answer change 需要在相同案件和生成条件下做无反馈对照。当前 trace 提供观察
材料，但不会把“发生反馈”直接标记为“成功纠错”。

## 真实模型与 CAIL 首轮入口

- `framework/local_generator.py`：本地模型 HTTP 客户端、三阶段生成和反馈上下文。
- `framework/nli_verifier.py`：显式标签映射、三类 softmax、过长输入跳过。
- `experiments/prepare_cail.py`：训练词表、官方划分内抽样、去重和短事实筛选。
- `experiments/run_first_round.py`：无反馈、旁路 NLI、自主复核和 NLI 反馈对照。
- `experiments/evaluate_first_round.py`：精确标签评分、失败计错和成对纠错/误伤。
- `experiments/nli_diagnostics.py`：开发集盲标注导出和阈值诊断。

Qwen 部署后，可先不加载 NLI，用两个合成样例检查服务：

```bash
python experiments/run_first_round.py --data examples/smoke_cases.jsonl --labels examples/smoke_labels.json --groups baseline --verbose
```

服务地址和模型名在 `configs/first_round.json` 中配置。合成样例不代表 benchmark 效果。
完整数据准备、GPU 编号、开发集诊断、冻结测试集和结果解释见 [首轮实验说明](docs/first_round.md)。
真实权重、数据集和 GPU 服务需另外准备；当前不实现校准训练、长文本检索和断点恢复。
