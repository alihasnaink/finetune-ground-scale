# ATML PA3: Fine-tuning, Visual Grounding, Test-Time Compute

Code for EE-5102/CS-6304 Programming Assignment 3. Each experiment runs as a Kaggle notebook
that calls the scripts in this repository and produces a downloadable evidence zip.

## Layout

```
task1/                     LLM fine-tuning (Qwen3-0.6B on MetaMathQA)
  configs/common.yaml      shared recipe (data, optimiser, schedule, eval, diagnostics)
  configs/full_ft.yaml     Step 1: Full FT (lr 2e-5, lambda 0)
  configs/l2sp.yaml        Step 2: Full FT + L2-SP (lr 2e-5, lambda 1e-3)
  configs/lora4.yaml, lora16.yaml, oplora.yaml   Steps 3-4: adapters (lr 2e-4)
  adapters.py              LoRA / OPLoRA modules, bases, merging, subspace checks
  common.py                config loading, paths, I/O, environment capture
  data.py                  MetaMathQA selection + tokenisation, batching, probe set, qualitative prompts
  prepare.py               model download, data build, probe set, eval-dataset prefetch
  train.py                 SFT training loop (AMP FP16, FP32 master weights, recovery checkpoints)
  evaluate.py              lm-eval benchmarks (GSM8K / HellaSwag / BoolQ / ARC-Easy) + qualitative generation
  diagnostics.py           rho_k, eta, R(z) on down_proj layers 0/13/27
  import_prior.py          copies results of earlier notebooks (baseline, other runs) into this one
  report.py                tables, figures, annotation sheet, zip
  annotate.py              command-line tool for the manual qualitative annotation
  assets/first_plot_questions.yaml   qualitative prompts (see Attribution)
notebooks/
  task1_full_ft.ipynb      Kaggle notebook for Task 1 Step 1 (+ shared baseline)
  task1_l2sp.ipynb         Kaggle notebook for Task 1 Step 2 (L2-SP)
  task1_adapters.ipynb     Kaggle notebook for Task 1 Steps 3-4 (LoRA-4, LoRA-16, OPLoRA)
task2/                     SigLIP-Qwen VLM and visual grounding
  configs/common.yaml      shared recipe; vlm196.yaml (main), vlm49.yaml (optional 49-token run)
  prepare.py               annotations, split selection + checks, image fetch, SigLIP feature cache
  model.py                 connector, scale calibration, visual-token insertion, losses, greedy decoding
  train.py                 three-stage curriculum, sampler, dev evaluation, resumable checkpoints
  evaluate.py              final test: 4 image conditions, scoring, pair metrics, pair bootstrap
  report.py                learning curves, tables, qualitative pairs, resources, zip
  vqa_norm.py              official VQA answer normalisation + soft accuracy
notebooks/task2_vlm.ipynb  Kaggle notebook for Task 2
task3/                     test-time compute (supplied rollouts + live MATH generation)
  configs/live.yaml        Step 4 settings (model, sampling, prompts, token allowances, seeds)
  configs/offline.yaml     Steps 1-3, 5, 6 settings (budgets, orders, bootstrap, stopping grid)
  common.py                config loading and the seeded dev/eval split
  prepare_splits.py        writes the splits and the live subset to task3/data/
  prepare_mathvista_meta.py  writes the MathVista metadata used for answer normalisation
  extract.py               last \boxed{} answer with balanced braces
  scoring.py               math-verify answer equivalence, answer groups, self-consistency
  live.py                  Step 4 generation on GPU (single / parallel / refine), resumable, per-call logs
  live_eval.py             Step 4 scoring: accuracy, tokens, runtime, truncation, transitions, bootstrap
  mathvista_norm.py        MathVista answer normalisation (official rules)
  offline_data.py          join + verify the supplied rollouts, answer groups, correctness
  engine.py                selection rules (FirstAns, SC, highest score, RWC, VisConf)
  offline.py               Steps 1-2: fixed budgets, all selectors, failure decomposition, bootstrap
  signals.py               Step 3: signal associations, confidence-vision analysis, trajectories
  adaptive.py              Step 5: stopping policies, dev selection, held-out evaluation
  analysis.py              Step 6: compute allocation, figures, cases
  data/                    committed splits, live questions, MathVista metadata
  results/                 machine-readable Task 3 results (start with results/README.md)
notebooks/task3_live.ipynb  Kaggle notebook for Task 3 Step 4
kaggle_outputs/            downloaded Kaggle outputs: logs, metrics, samples and report tables of every run
report/                    LaTeX source of the report; make_tables.py builds its tables and figures
requirements-task1.txt, requirements-task2.txt, requirements-task3.txt
```

## Running Task 1 on Kaggle

### Step 1: Full FT

1. Push this repository to GitHub (the notebook clones `REPO_URL`, branch `main`).
2. Create a Kaggle notebook from `notebooks/task1_full_ft.ipynb`. Set the accelerator to **GPU T4 x2**
   and turn **Internet on**.
3. Optionally set `SMOKE_TEST = True` first. This runs a tiny end-to-end check (8 updates, 16 eval
   examples per task) in a separate output folder.
4. **Save Version → Save & Run All**. When it finishes, download `task1_full_ft_report.zip` from the
   output tab.

With two GPUs, the baseline and Full FT midpoint evaluations run on GPU 1 while training runs on
GPU 0. With one GPU, everything runs sequentially.

**Recovering a failed version.** Attach the failed version's output with **Add Input** and run again.
`task1.import_prior` reuses its finished evaluations. If training had finished, it also reuses the
FP16 exports in `task1_artifacts/exports/`, so training is skipped and only the missing evaluations
run. Each import is used only if its training-data hash matches.

Each script can also be run by hand, for example:

```bash
export T1_OUT_ROOT=$PWD/out/task1 T1_SCRATCH_ROOT=/tmp/atml_task1
python -m task1.prepare     --config task1/configs/full_ft.yaml
python -m task1.train       --config task1/configs/full_ft.yaml
python -m task1.evaluate    --config task1/configs/full_ft.yaml --conditions baseline full_ft_mid full_ft_final
python -m task1.diagnostics --config task1/configs/full_ft.yaml
python -m task1.report      --config task1/configs/full_ft.yaml --zip out/task1_full_ft_report.zip
```

### Step 2: L2-SP (and later runs)

Use `notebooks/task1_l2sp.ipynb` the same way. Before running it, attach the finished **Full FT
notebook's output** with **Add Input**. The notebook's `task1.import_prior` step then copies the
shared baseline and the Full FT results into this run, but only if their training-data hash matches.
That lets the L2-SP report show changes from baseline and the Full FT vs L2-SP comparison. If no
baseline is attached, it is evaluated again on GPU 1 while L2-SP trains on GPU 0. Training always
uses one GPU, so training time and memory are comparable across methods.

### Steps 3-4: LoRA-4, LoRA-16 and OPLoRA

`notebooks/task1_adapters.ipynb` trains all three adapter runs in one notebook, as two lanes in
parallel, one training run per GPU at a time. GPU 0 trains LoRA-16 and then OPLoRA; GPU 1 trains
LoRA-4. Each final model is evaluated straight after its training. Attach the Full FT notebook
output as for Step 2. If no baseline is attached, it is evaluated at the end of the GPU 1 lane.
Expect about 6-7 h on T4 x2. Because two runs share the machine, CPU-side timings may be slightly
affected; GPU memory and per-update times are still measured per run. The lanes can be changed in
the `LANES` cell of the notebook, or in `RUNS["adapters"]` of the generator.

Any config value can be overridden with `--set key.sub=value`, e.g. `--set train.num_updates=8 eval.limit=16`.

## Running Task 2 on Kaggle

`notebooks/task2_vlm.ipynb` runs everything:
data preparation, the 196-token main run on GPU 0, the optional 49-token run on GPU 1 in
parallel, final evaluation and the report (`task2_report.zip`). Use T4 x2 with Internet on.

1. `task2.prepare` downloads the COCO 2014 caption/instance annotations and VQAv2. It then
   selects every split (seed 6304) and downloads only the selected COCO images (about 10k).
   If a COCO 2014 dataset is attached, its `train2014/` and `val2014/` folders are used instead.
   Finally it caches the SigLIP features in FP16 under `task2_artifacts/features/`.
2. `task2.train` runs the three-stage curriculum (2,250 + 1,500 + 750 updates).
3. `task2.evaluate` evaluates both test suites under four image conditions.
4. `task2.report` builds the tables, figures and zip.

If a version fails, attach its output and run again. Cached features and the newest checkpoint
are reused (same data-selection hash). Finished stages and evaluations are not repeated.

## Running Task 3

**Setup (CPU, once).** Put the supplied ATML-PA3-Data in `task3_data/` (git-ignored) and install
`requirements-task3.txt`. The derived files in `task3/data/` are committed:

```bash
python -m task3.prepare_splits                 # split_math.json, split_mathvista.json, live_questions.json
python -m task3.prepare_mathvista_meta --parquet <AI4Math/MathVista testmini parquet>   # mathvista_meta.json
```

Splits: for each dataset, the question IDs are sorted, then permuted with `default_rng(6304)`. The
first 200 are development and the remaining 800 held-out evaluation. The live subset is the first 50
MATH evaluation questions. MathVista metadata (question/answer type, precision, choices) comes from
`AI4Math/MathVista` at revision `2b6ad694`, testmini, metadata columns only.

**Step 4 (GPU, Kaggle).** Run `notebooks/task3_live.ipynb` on T4 x2 with Internet on. No dataset
is needed, because the live questions are committed. `task3.live` runs one lane per GPU (questions
split by index; every strategy of a question on the same GPU). Per question it makes three parallel
calls (1,024 tokens each), one single call (3,072 tokens), and a refinement of the first parallel
response (2,048 tokens). Each call is seeded with `6304 + 1000 * index + offset` and saved
immediately. `task3.live_eval` then scores everything. Outputs are in `kaggle_outputs/task3_live/`.

**Steps 1–3, 5, 6 (CPU, about 15 minutes, mostly math-verify grouping):**

```bash
C=task3/configs/offline.yaml
python -m task3.offline_data --config $C  # join + verify token alignment, answer groups, correctness (cache in outputs/)
python -m task3.offline      --config $C  # Steps 1-2: 10 seeded orders, N = 1..32, all selectors, bootstrap
python -m task3.signals      --config $C  # Step 3: score/length/correctness associations, confidence-vision, trajectories
python -m task3.adaptive     --config $C  # Step 5: dev-selected stopping thresholds, frozen, held-out evaluation
python -m task3.analysis     --config $C  # Step 6: compute allocation, figures, cases, task3/results/README.md
```

Results (machine-readable CSV/Parquet/JSON plus Markdown summaries) are in `task3/results/`; start
with `task3/results/README.md`. Implementation notes:

* **Answers.** MATH answer groups and grading both use math-verify 0.9.0 equivalence (`task3/scoring.py`).
  MathVista uses the official MathVista normalisation with the released metadata, keeping option
  letters for multiple choice (`task3/mathvista_norm.py`). Missing extractions form one NO_ANSWER
  group, which is always incorrect. Answer groups are formed from the candidates alone, never from
  the gold answer.
* **Selection and stopping.** Everything is recomputed from the revealed prefix only: scores, VisConf
  calibration (5th/95th percentiles of the pooled token KL) and ranks.
  * Ties go to the earliest supporting rollout in the current order.
  * RWC uses average ranks.
  * A FirstAns run is charged only for its first rollout.
* **Adaptive thresholds** were selected on the development split and frozen
  (`task3/results/<ds>/adaptive_selected.json`) before the held-out evaluation.

## Task 2 implementation notes

* **Data** (`task2/prepare.py`). Train images come from train2014; dev and test images come from
  val2014. Every dev/test item is image-disjoint from every other dev/test item and from training
  (`data/split_checks.json`).
  * VQAv2 complementary pairs are kept when both images have ≥ 8/10 annotator agreement on
    normalised answers and the consensus answers differ. The training target is the most frequent
    original string in the consensus group, and all ten answers are kept for scoring.
  * Grounding pairs are generated from COCO instances:
    * presence: an image with a non-crowd instance of ≥ 32×32 px, paired with an image with no
      annotation of that category;
    * counting: 1–4 instances, all non-crowd and ≥ 32×32 px, with the two images having
      different counts.
  * Training grounding pools are sampled from the 6,000 caption-training images.
  * One fixed answer normalisation (official VQA processing) is used everywhere.
* **Features**: SigLIP-base `last_hidden_state` (the final LN applied once, verified against
  the encoder output), 196 × 768 in 14×14 row-major order, FP16. The 49-token run averages each
  2×2 block before the connector.
* **Interface** (`task2/model.py`). The visual tokens replace a placeholder at the start of the
  user message, followed by `\n` and the question, inside Qwen's non-thinking chat template.
  * No-image inputs drop only the visual tokens. Positions and masks are rebuilt.
  * The connector output is multiplied by a fixed scale, calibrated once on training data: the
    mean visual-token norm at initialisation is matched to the mean Qwen input-embedding norm of
    training-caption/question tokens (`runs/<run>/calibration.json`).
* **Training** (`task2/train.py`). Each draw gives two examples: a pair, or two captions for
  replay. Stage 1 makes two seeded passes over the 18,000 captions; stages 2–3 use a seeded
  mixture sampler.
  * Every stage gets a fresh AdamW, because the trainable set and learning rates change.
  * The schedule per stage is 5% linear warmup, then cosine decay to 10% of peak.
  * LoRA (r16, α32, dropout 0.05) on q/k/v/o is added at stage 2.
  * The loss is the per-example mean token NLL (target + `<|im_end|>`), then the mean over the
    16 examples.
  * Checkpoints every 250 updates hold everything needed to resume exactly (verified bit-exact,
    at and inside stages). `sanity_checks.json` records:
    * feature and pooled shapes, and the LN check;
    * that the prompt pieces decode exactly to the chat template;
    * trainable parameters by stage, and that the backbone stays frozen;
    * that connector (and LoRA-B) gradients are non-zero;
    * train/dev isolation.
* **Evaluation** (`task2/evaluate.py`): greedy decoding with a KV cache, 16 new tokens, batch
  size 8.
  * Scoring: normalised exact match for grounding, VQA soft accuracy (leave-one-out over 10
    answers) for VQAv2.
  * Also reported: pair accuracy, identical- and empty-answer rates, reference NLL (termination
    excluded), generation time, and what complementary swaps do (keep the original answer,
    follow the partner's, or give another).
  * 95% CIs come from 4,000 resamples of complete pairs, shared across conditions.

## Task 1 implementation notes

* **Data**: the first 40,000 eligible MetaMathQA rows are used in source order. Excluded rows are
  those with an empty query or response; those whose `query` or `original_question` equals a GSM8K
  test question after NFKC, lowercasing and whitespace collapsing; and those whose formatted length
  is above 2,048 tokens. Each example is the Qwen3 chat prompt (thinking disabled) followed by the
  response and `<|im_end|>`. Loss is computed only on the response tokens and `<|im_end|>`.
  `data/data_stats.json` records the exclusion counts, a check against the chat template, and a
  SHA-256 of the token data. That hash is verified again when training resumes.
* **Loss normalisation**: the summed token cross-entropy over all 16 micro-batches is divided by the
  number of response tokens in the whole effective batch of 32. The LM head is applied only at
  response positions, in checkpointed chunks. This saves memory and does not change the loss value.
* **Precision**: FP32 master weights, FP16 autocast and dynamic loss scaling (initial scale 2^16).
  Gradients are unscaled before clipping. Skipped updates are logged.
* **Schedule**: `transformers.get_cosine_schedule_with_warmup` with 38 warmup updates and decay to 0.
  With the standard HF convention, the learning rate of the very first update is 0.
* **Checkpoints**: recovery checkpoints at updates 312, 625, 937 and 1250 hold the model, optimizer,
  scheduler, scaler, RNG state and data position. Only the newest is kept on disk, to fit Kaggle's
  storage. Training resumes from it automatically. FP16 inference exports are written at the
  midpoint (625) and final (1250) updates.
* **Evaluation**: lm-eval-harness 0.4.13, zero-shot, complete splits, FP16, `model.eval()`.
  Batch sizes are 32 for GSM8K and 8 for the likelihood tasks. At 32, the longest BoolQ batch times
  the 152k-word vocabulary of FP16 logits does not fit on a T4.
  * GSM8K uses `gsm8k_cot` with the non-thinking chat template, greedy decoding, a 1,024-token cap and
    the stock stop sequences and filters. A response is flagged as truncated when its re-tokenised
    length is within 4 tokens of the cap. lm-eval removes stop strings from the output, so exact
    stop information is not available.
  * HellaSwag, BoolQ and ARC-Easy use the stock completion prompts without a chat template.
* **Probe set**: 32 GSM8K test and 32 HellaSwag validation documents, chosen with
  `numpy.default_rng(6304)`. Prompts are rendered by lm-eval's own `fewshot_context`, and their
  hashes are checked against the prompts lm-eval logged for the baseline (`diagnostics/probe_check.json`).
* **L2-SP**: the objective is `L_SFT + λ·Ω` with `Ω = ½‖θ−θ0‖²`, summed (not averaged) over all
  unique parameters, including biases and norms, with the tied embedding counted once. θ0 stays fixed.
  The penalty gradient `λ(θ−θ0)` is added to the unscaled FP32 gradients before global-norm clipping,
  which is identical to backpropagating λΩ but keeps the penalty out of FP16 loss scaling. This was
  checked against autograd. AdamW weight decay (0.01, decoupled, towards zero) is still applied as in
  the shared recipe. θ0 is kept on the GPU in BF16, which is bit-exact because Qwen3-0.6B is released
  in BF16 (checked at start-up; otherwise FP32). That costs 1.1 GiB of reference storage, with one
  host-to-device transfer and no transfers per update. Each update logs Ω, λΩ, SFT + λΩ, ‖∇L_SFT‖
  and ‖λ(θ−θ0)‖. Ω in the log is evaluated at the parameters that update's SFT loss used, i.e.
  before its step. `train_summary.json` records the reference bytes, setup time, transfer bytes and
  penalty compute time (`overhead_*`).
* **LoRA / OPLoRA** (`task1/adapters.py`, no PEFT dependency):
  * The backbone is frozen and loaded in FP16. FP32 factors are added to q/k/v/o/gate/up/down
    in all 28 layers: `W = W0 + (α/r)·B·A`.
  * A is Kaiming-uniform, PEFT's default, drawn from a dedicated generator with seed 6304 in a
    fixed module order. B is 0. Dropout 0.05 is applied to the adapter input. There are no
    adapter biases.
  * LoRA-16 and OPLoRA therefore start from identical factors.
  * Weight decay (0.01) applies to A and B, as in the shared recipe.
  * OPLoRA (k = 128): `U_k`, `V_k` are computed once, by SVD of the FP32 starting weights, and
    stay on the GPU. Every forward pass uses the effective factors
    `B~ = B − U_k(U_kᵀB)` and `A~ = A − (A V_k)V_kᵀ`. Dense projection matrices are never formed.
  * Recovery checkpoints hold only the factors, plus the optimizer and other training state.
  * Export: `ΔW = (α/r)·B~·A~` is merged in FP32 into a fresh copy of the starting weights, which
    is then saved in FP16. The exported factors are saved alongside as `adapter_factors.pt`.
  * Each export is checked in two ways:
    * `merge_check_final.json`: merged FP16 logits vs. unmerged logits on 4 training sequences,
      with the adapter's own effect on the logits given for scale.
    * `adapter_subspace_final.*`: ρ_out = ‖U_kᵀΔW‖²/‖ΔW‖² and ρ_in = ‖ΔW V_k‖²/‖ΔW‖² for every
      adapted projection. This numerically verifies the OPLoRA constraints (≈ 0) and shows the
      LoRA values for comparison.
  * `train_summary.json` records the adapter parameter count and size, the basis storage and
    preprocessing time, and the export/merge time (`overhead_*`).
  * Diagnostics and inference use the merged ΔW, which comes from the projected factors for
    OPLoRA.
* **Diagnostics**: FP32, starting-model `down_proj` inputs at the last prompt token. Each FP32 Full FT
  delta is computed as `W_trained − W0` from the master weights during training, not from the FP16
  export. `rho_k` uses k = 128. The output-side chance level k/d_out and the input-side ratio are
  also reported for reference.
* **Outputs on Kaggle**: `/kaggle/working/task1/` holds the small evidence files and is what gets
  zipped. `/kaggle/working/task1_artifacts/` is kept as notebook output but not zipped. It holds
  `exports/<run>/<stage>/`, the FP16 inference models (about 1.2 GB each), so a failure after
  training only costs a re-run of the evaluation. It also holds `deltas/<run>/<stage>.pt`, the FP32
  `down_proj` deltas (about 75 MB), so diagnostics can be recomputed without retraining. Recovery
  checkpoints live in `/tmp` and are not saved. The zip step skips any weights/tensor file and anything over 50 MB.

## Attribution

* `task1/assets/first_plot_questions.yaml` is copied from
  [emergent-misalignment/emergent-misalignment](https://github.com/emergent-misalignment/emergent-misalignment/blob/e40daef5791139a2417e22034ab0a0e72900524b/evaluation/first_plot_questions.yaml)
  (MIT License), commit `e40daef`.
* Benchmarks are run through [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
* Models and datasets: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) and
  [meta-math/MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA).
* Code was written with the help of an LLM coding assistant 