VLM Training-Stage Checkpoints for Visual Attention / Gaze-Head Analysis
Purpose
This document lists VLM families for which model checkpoints can be compared acrossdifferent training stages. The main criterion is that the checkpoints should beuseful for studying whether visual attention routing, visual-selection heads, orgaze-like circuits change as training progresses.
The strongest verified case is InternVL3.5, because the authors explicitlyrelease weights after multiple stages of the same training pipeline.
￼
1. InternVL3.5 — strongest 4-stage candidate
Training pipeline
InternVL3.5 releases checkpoints corresponding to:
```textStage 1: CPT    ↓Stage 2: CPT + SFT    ↓Stage 3: CPT + SFT + MPO    ↓Stage 4: CPT + SFT + CascadeRL```
The official model card states that weights are open-sourced after differenttraining stages.
Stage meanings
|Stage|Checkpoint suffix|Training             |Interpretation                               ||-----|-----------------|---------------------|---------------------------------------------||1    |`-Pretrained`    |CPT                  |Multimodal continual pretraining             ||2    |`-Instruct`      |CPT + SFT            |Supervised / instruction tuning              ||3    |`-MPO`           |CPT + SFT + MPO      |Additional multimodal preference optimization||4    |no suffix        |CPT + SFT + CascadeRL|Full CascadeRL stage                         |
For a training-stage study, this is much cleaner than treating three unrelatedmodels as “base / instruct / agentic.”
Recommended model sizes
1B
|Stage    |Hugging Face weights                                      ||---------|----------------------------------------------------------||CPT      |https://huggingface.co/OpenGVLab/InternVL3_5-1B-Pretrained||SFT      |https://huggingface.co/OpenGVLab/InternVL3_5-1B-Instruct  ||MPO      |https://huggingface.co/OpenGVLab/InternVL3_5-1B-MPO       ||CascadeRL|https://huggingface.co/OpenGVLab/InternVL3_5-1B           |
2B
|Stage    |Hugging Face weights                                      ||---------|----------------------------------------------------------||CPT      |https://huggingface.co/OpenGVLab/InternVL3_5-2B-Pretrained||SFT      |https://huggingface.co/OpenGVLab/InternVL3_5-2B-Instruct  ||MPO      |https://huggingface.co/OpenGVLab/InternVL3_5-2B-MPO       ||CascadeRL|https://huggingface.co/OpenGVLab/InternVL3_5-2B           |
4B
|Stage    |Hugging Face weights                                      ||---------|----------------------------------------------------------||CPT      |https://huggingface.co/OpenGVLab/InternVL3_5-4B-Pretrained||SFT      |https://huggingface.co/OpenGVLab/InternVL3_5-4B-Instruct  ||MPO      |https://huggingface.co/OpenGVLab/InternVL3_5-4B-MPO       ||CascadeRL|https://huggingface.co/OpenGVLab/InternVL3_5-4B           |
8B
|Stage    |Hugging Face weights                                      ||---------|----------------------------------------------------------||CPT      |https://huggingface.co/OpenGVLab/InternVL3_5-8B-Pretrained||SFT      |https://huggingface.co/OpenGVLab/InternVL3_5-8B-Instruct  ||MPO      |https://huggingface.co/OpenGVLab/InternVL3_5-8B-MPO       ||CascadeRL|https://huggingface.co/OpenGVLab/InternVL3_5-8B           |
14B
|Stage    |Hugging Face weights                                       ||---------|-----------------------------------------------------------||CPT      |https://huggingface.co/OpenGVLab/InternVL3_5-14B-Pretrained||SFT      |https://huggingface.co/OpenGVLab/InternVL3_5-14B-Instruct  ||MPO      |https://huggingface.co/OpenGVLab/InternVL3_5-14B-MPO       ||CascadeRL|https://huggingface.co/OpenGVLab/InternVL3_5-14B           |
30B-A3B
|Stage    |Hugging Face weights                                           ||---------|---------------------------------------------------------------||CPT      |https://huggingface.co/OpenGVLab/InternVL3_5-30B-A3B-Pretrained||SFT      |https://huggingface.co/OpenGVLab/InternVL3_5-30B-A3B-Instruct  ||MPO      |https://huggingface.co/OpenGVLab/InternVL3_5-30B-A3B-MPO       ||CascadeRL|https://huggingface.co/OpenGVLab/InternVL3_5-30B-A3B           |
38B
|Stage    |Hugging Face weights                                       ||---------|-----------------------------------------------------------||CPT      |https://huggingface.co/OpenGVLab/InternVL3_5-38B-Pretrained||SFT      |https://huggingface.co/OpenGVLab/InternVL3_5-38B-Instruct  ||MPO      |https://huggingface.co/OpenGVLab/InternVL3_5-38B-MPO       ||CascadeRL|https://huggingface.co/OpenGVLab/InternVL3_5-38B           |
241B-A28B
|Stage    |Hugging Face weights                                             ||---------|-----------------------------------------------------------------||CPT      |https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B-Pretrained||SFT      |https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B-Instruct  ||MPO      |https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B-MPO       ||CascadeRL|https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B           |
Best practical choice
For attention-head analysis, InternVL3.5-4B or 8B is likely the mostpractical starting point.
The clean comparison is:
```textInternVL3.5-8B-Pretrained        ↓InternVL3.5-8B-Instruct        ↓InternVL3.5-8B-MPO        ↓InternVL3.5-8B```
This lets you ask four separate questions:
	1.	Which visual heads emerge during multimodal CPT?
	2.	Which heads become specialized after instruction SFT?
	3.	Does MPO strengthen or reorganize visual grounding?
	4.	Does CascadeRL produce additional task/action-oriented visual routing?
￼
2. Qwen-VL / Qwen2-VL / Qwen2.5-VL / Qwen3-VL
Qwen is a very important family for this research because there are manyinstruction-tuned checkpoints and a large ecosystem of agentic derivatives.
However, do not automatically treat an agentic Qwen derivative as a fourthofficial training stage of the original Qwen checkpoint.
A scientifically safer representation is:
```textQwen base / pretrained        ↓Qwen instruction model        ↓Qwen-derived agent model```
The exact lineage must be checked for each agent checkpoint.
Useful official model hubs:
	●	Qwen2-VL: https://huggingface.co/Qwen
	●	Qwen2.5-VL: https://huggingface.co/Qwen
	●	Qwen3-VL: https://huggingface.co/Qwen
For a paper, only call the third checkpoint “agentically tuned” if its modelcard documents additional agentic SFT/RL/tool-use training.
￼
3. LLaVA
Training stages
The original LLaVA training recipe has a clear two-stage structure:
```textStage 1: feature alignment / projector pretraining        ↓Stage 2: visual instruction tuning```
Official repository:
https://github.com/haotian-liu/LLaVA
Model zoo:
https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md
LLaVA is therefore useful for:
```textpre-alignment → instruction tuning```
but it is not a clean four-stage base → SFT → offline RL → online RLfamily.
Agentic LLaVA derivatives exist, but they should be treated as separatederivatives unless their training lineage is explicitly documented.
￼
4. IDEFICS / IDEFICS3
IDEFICS provides a useful base/instruction comparison.
For example:
```textIDEFICS base     ↓IDEFICS-Instruct```
The instruction models are fine-tuned from their corresponding base modelsusing supervised and instruction-tuning data.
Official documentation:
https://huggingface.co/docs/transformers/tasks/idefics
Example weights:
	●	IDEFICS 9B base: https://huggingface.co/HuggingFaceM4/idefics-9b
	●	IDEFICS 9B instruct: https://huggingface.co/HuggingFaceM4/idefics-9b-instruct
	●	IDEFICS 80B instruct: https://huggingface.co/HuggingFaceM4/idefics-80b-instruct
	●	IDEFICS3: https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3
This is useful for a two-stage study, but I would not currently classify itas a verified four-stage agentic progression.
￼
5. PaliGemma / PaliGemma 2
PaliGemma has multiple training/configuration stages and is useful for studyingthe effect of multimodal training and task fine-tuning.
However, its stages are not equivalent to:
```textCPT → instruction SFT → offline RL → online RL```
Therefore it is better suited to a controlled multimodalpretraining/fine-tuning study than to the specific “emergence of visualagentic circuits” question.
Official model family:
https://huggingface.co/google
￼
Summary: what I would actually use
|Family         |Verified same-family stages|Clean 4-stage progression|Agent/RL stage        |Recommendation||---------------|--------------------------:|------------------------:|---------------------:|--------------||**InternVL3.5**|**4**                      |**Yes**                  |**Yes**               |**★★★★★**     ||Qwen-VL lineage|2–3+ depending derivative  |Not consistently         |Yes in derivatives    |**★★★★☆**     ||LLaVA          |2                          |No                       |Derivatives           |**★★★☆☆**     ||IDEFICS        |2                          |No                       |Not cleanly verified  |**★★☆☆☆**     ||PaliGemma      |Multiple                   |No                       |No comparable RL stage|**★★☆☆☆**     |
Recommended experimental setup
For the strongest controlled experiment, use:
```text                         SAME ARCHITECTURE                              │                              ▼             InternVL3.5-8B-Pretrained                              │                              │ CPT                              ▼               InternVL3.5-8B-Instruct                              │                              │ SFT                              ▼                  InternVL3.5-8B-MPO                              │                              │ MPO / offline preference optimization                              ▼                     InternVL3.5-8B                              │                              │ CascadeRL                              ▼                   FINAL AGENTIC MODEL```
Then run the same images, prompts, layers, heads, and causal interventionsat every stage.
This is substantially stronger than comparing unrelated VLM families becausethe model family provides checkpoints from the same training pipeline.
Important terminology
For the paper, I recommend calling the stages:
	●	CPT stage
	●	SFT stage
	●	MPO stage
	●	CascadeRL stage
rather than simply:
	●	Base
	●	Instruction
	●	Agentic-1
	●	Agentic-2
That avoids implying that MPO and CascadeRL are identical types of “agentictraining.”
Primary source
InternVL3.5’s official Hugging Face model card explicitly lists the fourtraining pipelines and the corresponding released weights:
https://huggingface.co/OpenGVLab/InternVL3_5-1B-HF
The same four-stage checkpoint structure is provided across the 1B, 2B, 4B,8B, 14B, 30B-A3B, 38B, and 241B-A28B variants
---

## Verification pass (checked against live HuggingFace listings + AutoConfig)

### InternVL3.5 — CONFIRMED, all 32 checkpoint URLs real
Every one of the 32 listed checkpoints (8 sizes x 4 stages, plus the `-1B-HF` model
card) resolves on HuggingFace — verified via `HfApi.model_info()` for all of them,
no failures. For the 8B tier specifically, also verified via `AutoConfig` (with
`trust_remote_code=True`, since InternVL uses custom modeling code — worth flagging
as a supply-chain consideration before running it) that all four stages
(`-Pretrained`, `-Instruct`, `-MPO`, no-suffix) share **identical architecture**:
`model_type=internvl_chat`, 36 layers x 32 heads. This is a genuinely stronger
same-architecture 4-stage progression than anything available in the Qwen-VL
family (see below).

One correction/addition: the underlying LLM backbone for the 8B tier is
**Qwen3** (`llm_config.architectures = ['Qwen3ForCausalLM']`, `model_type=qwen3`),
paired with an **InternViT-6B** vision encoder (`vision_config.model_type=intern_vit_6b`).
This is architecturally different from this project's Qwen-VL pipeline (different
vision tower, different image-token/patch-grid conventions, custom remote code
rather than a standard `transformers` model class) — integrating it would need the
same kind of adapter work done for LLaVA (`steer_llava_family.ipynb`), not a
drop-in reuse of the existing Qwen-VL-specific modeling code. Still, the fact that
all four stages are confirmed architecturally identical to each other makes this
the strongest verified candidate in this document if that integration work is done.

### Qwen-VL lineage — CONFIRMED, and one addition this project already validated
The document's caution ("do not automatically treat an agentic Qwen derivative as
a fourth official training stage... check lineage per checkpoint") matches this
project's own findings exactly:
- **Qwen2-VL**: `Qwen/Qwen2-VL-7B` (base) and `Qwen/Qwen2-VL-7B-Instruct` are both
  official Qwen releases (confirmed, same architecture, 28 layers x 28 heads). The
  "agentic" stage used throughout this project, `ByteDance-Seed/UI-TARS-7B-DPO`, is
  a **third-party** GUI-agent fine-tune (ByteDance-Seed, not Qwen) — exactly the
  kind of unverified-lineage case the document warns about. It shares the same
  base architecture (confirmed via `AutoConfig` earlier in this project) but is not
  an "official fourth Qwen stage."
- **Qwen3-VL**: this document does not mention it, but this project verified
  directly (via `HfApi.list_models(author="Qwen", search="Qwen3-VL")`) that **no
  official base (pretrained-only) or agentic checkpoint exists** for Qwen3-VL at
  any size. What *does* exist officially, matched at 8B (`Qwen/Qwen3-VL-8B-Instruct`
  vs. `Qwen/Qwen3-VL-8B-Thinking`, confirmed identical architecture: 36 layers x 32
  heads), is a genuine two-stage **Instruct vs. Thinking** (reasoning/CoT-tuned)
  comparison — not a base/instruct/agentic progression, but a real, official,
  architecture-matched two-stage pair worth adding to this document's Qwen section.
  This project ran the full discovery + causal-steering pipeline on this pair
  (`vis_head_across_qwen3vl_stages.ipynb`) and found a qualitatively different
  causal profile between the two stages (see `APPENDIX_GRID_AND_PROMPTS.md` Section
  5.3) — Thinking requires a fundamentally different attention-discovery method
  (measuring attention throughout the generated reasoning trace, not just the final
  pre-generation query) to find heads with any causal effect at all.

### LLaVA — PARTIALLY CONFIRMED
`liuhaotian/llava-pretrain-vicuna-7b-v1.3` (a stage-1 projector-pretraining
checkpoint) resolves on HuggingFace, supporting the document's two-stage claim.
Not verified beyond repo existence (no `AutoConfig`/architecture check performed
for this pair). This project's own LLaVA work (`steer_llava_family.ipynb`) used
`llava-hf/llava-1.5-7b-hf`, an instruction-tuned-only checkpoint, and separately
confirmed (via direct MCQ capability testing) that LLaVA-1.5 and LLaVA-1.6 cannot
do the forced-choice MCQ evaluation format above chance regardless of prompt
phrasing — a practical obstacle for using LLaVA in the causal-intervention
framework used throughout this project, independent of the training-stage
question this document is about.

### IDEFICS — CONFIRMED, all 4 listed URLs real
`HuggingFaceM4/idefics-9b`, `idefics-9b-instruct`, `idefics-80b-instruct`, and
`Idefics3-8B-Llama3` all resolve on HuggingFace. Not verified beyond repo
existence in this pass.

### PaliGemma — CONFIRMED, real "pt" vs. "mix" stage distinction
`google/paligemma-3b-pt-224`, `google/paligemma-3b-mix-224`, and
`google/paligemma2-3b-pt-224` all resolve on HuggingFace — supporting the
document's characterization of PaliGemma as having a genuine but non-agentic
pretrain/fine-tune stage distinction rather than a CPT->SFT->RL->RL progression.

### Bottom line
No claims in this document were found to be incorrect. The star-rating table's
ordering is validated by this pass — InternVL3.5 is confirmed as the strongest
same-architecture multi-stage candidate, with the important caveat that it would
require new architecture-specific integration work (custom remote code, different
vision tower) before this project's existing steering pipeline could use it,
analogous to the LLaVA integration already done. The one gap worth filling in the
Qwen section is Qwen3-VL's official Instruct-vs-Thinking pair, which this project
has already run end-to-end and which produced one of its most interesting findings
(the CoT-trace discovery method).
