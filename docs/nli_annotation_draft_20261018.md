# NLI 命题核查草稿与受控诊断（2026-10-18）

从首次 100 例开发集 shadow trace 的 200 条检查点记录中，按“案件 ID＋命题原文”
去重得到 147 个不同命题；重复检查点 ID 保存在导出记录中。用 seed 42 从 147 个命题
等概率抽取 30 个，先根据案件事实与命题写出暂定判断，再对照 NLI 结果。
这份 trace 使用旧的最终罪名输出协议，但 NLI 检查点位于最终阶段之前；后续比较
新协议各实验组的预测收益时，仍须在新协议下重跑成对实验。
固定的盲样本在 [`annotations_random30_blind.jsonl`](../results/dev-full-20261018/annotations_random30_blind.jsonl)，
独立命题全集在 [`annotations_unique_blind.jsonl`](../results/dev-full-20261018/annotations_unique_blind.jsonl)。

这批判断由助手起草，**不是经过独立人工复核的金标准**。在本次标注前，助手已从早期
诊断接触过该开发集的少数 NLI 输出，因此不能称为严格盲标。逐条判断和依据存于
[`nli_random30_draft_20261018.json`](../annotations/nli_random30_draft_20261018.json)，
按评分工具格式填好的文件在
[`annotations_random30_draft.jsonl`](../results/dev-full-20261018/annotations_random30_draft.jsonl)。

30 个命题中，26 个已给出暂定判断（support 25、insufficient 1），4 个因原文笔误、
省略或命题措辞边界留待复核。在 26 个暂定标签上，NLI 同意 20 个；5 个 support
被判为 insufficient，1 个 insufficient 被判为 support。
这仅是**草稿一致性统计**，样本里没有经确认的 conflict，不能据此估计真实三分类准确率
或设置反馈阈值。详细数值在
[`nli_random30_draft_metrics.json`](../results/dev-full-20261018/nli_random30_draft_metrics.json)。

值得复核的例子：

- `cail-dev-671:1`：原文只给“抽水泵、腊肉”合计 2022 元；命题说全部被盗财物价值
  共计 2022 元。草稿判 insufficient，NLI 判 support（约 0.998）。
- `cail-dev-6491:2`：原文明确记载交警执法时被告人驾车撞伤两名交警；命题概括了该事实。
  草稿判 support，NLI 判 insufficient（约 0.961）。
- `cail-dev-5045:1`：无本人烟草证、借他人证订售卷烟、非法进货金额均见于原文。
  草稿判 support，NLI 判 insufficient（约 0.950）。

另按 NLI 输出定向查看了全量开发集中唯一一个 conflict：`cail-dev-15623:2`。
命题说被告人弃婴离开后“未采取任何救助措施”；原文写弃婴离开，后来女婴由他人发现，
没有明确记录被告人是否采取过救助措施。草稿倾向 insufficient，至少看不出与原文明确冲突，
但 NLI 给 conflict 约 0.999。该例是**按模型判断挑出的诊断案例**，不属于随机 30 条；
需独立复核，不能用作无偏的 conflict 精度估计。当前不宜仅凭这个分数启用冲突反馈。

第一批的 4 条待定命题、6 条分歧及唯一 NLI conflict 均需独立复核。

## 第二批随机命题

从第一批以外的 117 个独立命题中，用 seed 20261018 再随机抽取 30 个。
样本见 [`annotations_random30b_blind.jsonl`](../results/dev-full-20261018/annotations_random30b_blind.jsonl)，
逐条草稿见 [`nli_random30b_draft_20261018.json`](../annotations/nli_random30b_draft_20261018.json)。
29 个有暂定标签，1 个因“有毒有害非食品原料”的指代不明留待复核。
本批有 26 个 support、2 个 insufficient、1 个 conflict；NLI 与草稿一致 19/29。
最清晰的 conflict 是 `cail-dev-15668:1`：原文写骆某持刀伤人，命题把行为写成被告人曹某，
而 NLI 以约 0.997 的概率判为 support。两批合计 60 个独立命题、55 个暂定标签、5 个待复核；
NLI 与草稿一致 39/55。合并统计在
[`nli_random60_draft_metrics.json`](../results/dev-full-20261018/nli_random60_draft_metrics.json)。
其中暂定标签分布为 support 51、insufficient 3、conflict 1；类别严重失衡，不能据此校准阈值。

## 受控事实改写

另从开发集 6 个案件人工构造每案各 1 条 support、conflict、insufficient 命题，
仅调用已有 NLI，不调用 Qwen，也不接触测试集。命题规格在
[`nli_contrast_probes_20261018.jsonl`](../annotations/nli_contrast_probes_20261018.jsonl)，
结果在 [`nli-contrast-20261018`](../results/nli-contrast-20261018/results.json)。
18 条中判对 12 条：support 5/6、conflict 2/6、insufficient 5/6。
将其中 4 条漏判的冲突改成更明确的“不是 A，而是 B”表述后，只判对 1/4，
规格与结果分别在
[`nli_contrast_variants_20261018.jsonl`](../annotations/nli_contrast_variants_20261018.jsonl) 和
[`nli-contrast-variants-20261018`](../results/nli-contrast-variants-20261018/results.json)。
这些是定向构造的小样本，用来暴露错误模式，不能并入自然生成命题的准确率。

当前实验脚本默认只跑 baseline 与 shadow；nli_conflict 仍可显式运行，
但应等独立复核和规则冻结后再用于正式对照。
下一步应优先复核两批共 5 条待定命题、16 条草稿分歧及受控改写中的主体与数量冲突；
之后再标注剩余独立命题，计算经复核的三分类指标。
