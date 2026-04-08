# Eagle3 Draft Head Training Plan for Nemotron-Cascade-2-30B-A3B

## Overview

Train an Eagle3 speculative decoding draft head for `chankhavu/c2-softcpy-fp8` using SpecForge (https://github.com/sgl-project/specforge). This would be the first Eagle3 head for a hybrid Mamba-Transformer MoE model. Our north star reference is NVIDIA's long-context Eagle3 for gpt-oss-120b (https://huggingface.co/nvidia/gpt-oss-120b-Eagle3-long-context).

## Key Model Details

- **Verifier (FP8)**: `chankhavu/c2-softcpy-fp8`
- **Verifier (BF16)**: `nvidia/Nemotron-Cascade-2-30B-A3B`
- **Architecture**: `NemotronHForCausalLM` (model_type: `nemotron_h`)
- **Hidden size**: 2688
- **Num layers**: 52
- **Layer pattern** (`hybrid_override_pattern`): `MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME`
  - M = Mamba-2, E = MoE, * = GQA Attention
  - Attention layers at indices: 5, 12, 19, 26, 33, 42
  - Final layer (51) is MoE
- **Vocab size**: 131072
- **MoE config**: 128 routed experts + 1 shared expert, top-6 activation

### CRITICAL: Mamba SSM State Precision

**The Mamba SSM cache/state MUST be kept at Float32 during forward passes.** Using BFloat16 for SSM state will catastrophically degrade math and reasoning performance. Ensure any forward pass configuration sets `mamba_ssm_cache_dtype=float32`. This applies to hidden state extraction, online training, and inference.

## Eagle3 Hidden State Layer Selection

Attach to layers **2, 26, 48**:
- **Layer 2** (Mamba): low-level features, early in the stack (~4% depth)
- **Layer 26** (Attention/GQA): mid-level features, 4th attention layer at midpoint (~50% depth)
- **Layer 48** (Mamba): high-level features, deep but not fully bottlenecked (~92% depth)

Rationale: follows NVIDIA's long-context Eagle3 proportional spacing (they used 1/17/32 out of 36 layers for gpt-oss-120b). Layer 26 is an attention layer providing global context mixing at the midpoint. Layer 48 is deep but avoids the information bottleneck of the very final layer, retaining richer multi-step prediction signal.

## Hardware

- 2x RTX PRO 6000 Blackwell GPUs
- Verifier FP8 weights: ~32GB, fits on 2 GPUs with TP=2

## Disk Space Constraint

**Only ~70GB of free disk space is available** — barely enough for BF16 weights and the Eagle3 draft head weights. This rules out offline training (which would need ~6.5 TB for 400M tokens of hidden states). All training MUST use **online mode**.

**If disk space runs out at any point**: delete `/workspace/models` (the vLLM model cache folder) and retry:

```bash
rm -rf /workspace/models
```

This folder can be rebuilt by re-downloading models. After clearing, retry the failed operation.

## Training Strategy: ONLINE Mode

Online mode keeps the frozen target model resident in GPU memory and generates hidden states on the fly during training — no disk storage of hidden states needed.

All data is from https://huggingface.co/datasets/chankhavu/c2_eagle3_train:
- `cascade2_sft_train.jsonl` — slightly off-policy SFT data
- `c2_traces_train.jsonl` — reasoning traces at T=1.0, top_p=0.95
- `c2_traces_cot_train.jsonl` — chain-of-thought reasoning traces at T=1.0, top_p=0.95

### Experiment 1: 2-Stage Training (recommended first attempt)

#### Stage 1: Off-Policy Pretraining
- Purpose: teach the draft head the verifier's hidden-state-to-token mapping
- Data: `cascade2_sft_train.jsonl`
- Higher learning rate (1e-4)
- General distribution, larger volume

#### Stage 2: On-Policy Fine-tuning
- Purpose: specialize the draft head for actual inference distribution
- Data: `c2_traces_train.jsonl` + `c2_traces_cot_train.jsonl` (merged and shuffled)
- Lower learning rate (5e-5) to avoid overwriting Stage 1 knowledge
- This is the final training phase — the draft head should last see this distribution

### Experiment 2: Single-Stage Mixed Training

Concatenate ALL three data files, shuffle the combined dataset, and train in one pass:

```bash
cat cascade2_sft_train.jsonl c2_traces_train.jsonl c2_traces_cot_train.jsonl > all_data_merged.jsonl
shuf all_data_merged.jsonl > all_data_shuffled.jsonl
```

- Data: `all_data_shuffled.jsonl`
- Single training run, no resume logic needed
- Learning rate: 1e-4 (or sweep between 5e-5 and 1e-4)
- Simpler pipeline, avoids the resume/weight-loading complexity of Experiment 1
- Tradeoff: the draft head sees SFT and reasoning data simultaneously rather than specializing last on the inference distribution. May result in slightly lower acceptance rates on math reasoning, but avoids any risk of catastrophic forgetting during Stage 2.

Run Experiment 1 first. If resume training proves difficult to implement in SpecForge, fall back to Experiment 2.

### Experiment 3: On-Policy Only

Skip the SFT data entirely. Train only on the inference-distribution data:

```bash
cat c2_traces_train.jsonl c2_traces_cot_train.jsonl > c2_traces_merged.jsonl
shuf c2_traces_merged.jsonl > c2_traces_shuffled.jsonl
```

- Data: `c2_traces_shuffled.jsonl`
- Single training run, no resume logic needed
- Learning rate: 1e-4
- The draft head only ever sees data from the actual inference distribution
- Tradeoff: less total training data (no SFT), but zero distribution mismatch. If the on-policy traces are large enough, this may produce the highest acceptance rates on math reasoning since every training sample matches the deployment setting.

## Step-by-Step Execution Plan

### Step 0: Install SpecForge

```bash
git clone https://github.com/sgl-project/specforge.git
cd specforge
pip install -e .
```

### Step 1: Add Nemotron-H Support to SpecForge

SpecForge does not natively support `NemotronHForCausalLM`. You need to:

1. **Add a target model class** in `specforge/modeling/target/`:
   - Create `nemotron_h.py` implementing `DistributedTargetModel`
   - The model uses `trust_remote_code=True` (custom modeling code on HF)
   - Apply `ColumnParallelLinear` and `RowParallelLinear` for TP support
   - Key: the model's forward pass must expose hidden states at specified layer indices (2, 26, 48)
   - The hidden states are just the residual stream tensors after each layer — layer type (Mamba/MoE/Attention) doesn't matter
   - **CRITICAL**: ensure `mamba_ssm_cache_dtype=float32` is set when loading the model. Do NOT allow it to default to bfloat16 or float16.

2. **Register the model** in `specforge/modeling/auto.py`:
   ```python
   class AutoDistributedTargetModel(AutoModelForCausalLMBase):
       _model_mapping = {
           ...existing entries...
           NemotronHConfig: [NemotronHForCausalLM],
       }
   ```

3. **Create a draft model config** (e.g., `configs/nemotron-nano-eagle3.json`):
   ```json
   {
     "hidden_size": 2688,
     "vocab_size": 131072,
     "draft_vocab_size": 32000,
     "num_layers": 1,
     "eagle_aux_hidden_state_layer_ids": [2, 26, 48],
     "max_position_embeddings": 262144
   }
   ```
   Note: adjust `draft_vocab_size` based on token frequency analysis. 32K is a reasonable starting point.

4. **Add a chat template** in `specforge/data/template.py`:
   ```python
   TEMPLATE_REGISTRY.register(
       name="nemotron-nano-v3",
       template=ChatTemplate(
           assistant_header="<|im_start|>assistant\n",
           user_header="<|im_start|>user\n",
           system_prompt="<|im_start|>system\n<|im_end|>\n",
           end_of_turn_token="<|im_end|>\n",
       ),
   )
   ```
   Verify this against the actual tokenizer chat template from the HF repo.

### Step 2: Prepare Training Data

#### Stage 1 data (off-policy):
- Dataset: `cascade2_sft_train.jsonl` from https://huggingface.co/datasets/chankhavu/c2_eagle3_train
- This is slightly off-policy SFT data for learning the verifier's general distribution
- No hidden state pre-computation needed (online mode generates on the fly)

#### Stage 2 data (on-policy rollouts):
- Datasets from https://huggingface.co/datasets/chankhavu/c2_eagle3_train:
  - `c2_traces_train.jsonl` — reasoning traces at T=1.0, top_p=0.95
  - `c2_traces_cot_train.jsonl` — chain-of-thought reasoning traces at T=1.0, top_p=0.95
- Multiple rollouts per prompt to capture distribution branching
- No hidden state pre-computation needed (online mode generates on the fly)

### Step 3: Build Vocabulary Mapping

The token frequency file is generated during the first online training run or can be built separately. Then:

```bash
python scripts/build_vocab_mapping.py \
    --token-freq-path ./token_freq.pt \
    --draft-vocab-size 32000 \
    --target-model-path chankhavu/c2-softcpy-fp8 \
    --output-path ./vocab_mapping/
```

### Step 4: Train Draft Head — Stage 1 (Off-Policy, Online Mode)

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_eagle3_online.py \
    --target-model-path chankhavu/c2-softcpy-fp8 \
    --draft-model-config ./configs/nemotron-nano-eagle3.json \
    --train-data-path chankhavu/c2_eagle3_train/cascade2_sft_train.jsonl \
    --output-dir ./checkpoints/nemotron-nano-eagle3-stage1 \
    --num-epochs 5 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template nemotron-nano-v3 \
    --trust-remote-code \
    --tp-size 2 \
    --mem-frac 0.85 \
    --save-interval 5000 \
    --log-interval 100
```

**CRITICAL**: Verify that the target model is loaded with `mamba_ssm_cache_dtype=float32`. If SpecForge does not pass this through, patch the model loading code to enforce it.

### Step 5: Train Draft Head — Stage 2 (On-Policy Fine-tune, Online Mode)

**Before running Stage 2, check two things in SpecForge:**

#### 5a. Resume training support
Run `python scripts/train_eagle3_online.py --help` and look for a `--resume-from-checkpoint` or `--load-draft-weights` flag. **It is unclear whether SpecForge supports resume training natively.**

- **If resume IS supported**: use the flag to load the Stage 1 checkpoint (e.g., `--resume-from-checkpoint ./checkpoints/nemotron-nano-eagle3-stage1/best`)
- **If resume is NOT supported**: modify the training script to manually load the Stage 1 draft head weights before training begins. The draft model is a small `Eagle3DraftModel` — find where it is initialized in the training script and add:
  ```python
  # After draft_model is created but before training loop
  stage1_state = torch.load("./checkpoints/nemotron-nano-eagle3-stage1/best/model.safetensors")  # or .pt
  draft_model.load_state_dict(stage1_state, strict=False)
  ```
  Inspect the checkpoint directory from Stage 1 to find the exact filename and format of saved weights.

#### 5b. Data shuffling
Check whether the SpecForge dataloader shuffles data between epochs. Look in `specforge/data/` or the training script for `shuffle=True` on the DataLoader or a `RandomSampler`.

- **If shuffling IS built in**: no action needed
- **If shuffling is NOT built in**: merge and shuffle the Stage 2 files before training:
  ```bash
  cat c2_traces_train.jsonl c2_traces_cot_train.jsonl > c2_traces_merged.jsonl
  shuf c2_traces_merged.jsonl > c2_traces_shuffled.jsonl
  ```
  Use `c2_traces_shuffled.jsonl` as the training data path.

**Even if SpecForge does shuffle**, still merge the two files with shuffling so that traces and CoT data are interleaved rather than presented sequentially:
```bash
cat c2_traces_train.jsonl c2_traces_cot_train.jsonl > c2_traces_merged.jsonl
shuf c2_traces_merged.jsonl > c2_traces_shuffled.jsonl
```

#### 5c. Run Stage 2 training

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_eagle3_online.py \
    --target-model-path chankhavu/c2-softcpy-fp8 \
    --draft-model-config ./configs/nemotron-nano-eagle3.json \
    --train-data-path c2_traces_shuffled.jsonl \
    --output-dir ./checkpoints/nemotron-nano-eagle3-stage2 \
    --num-epochs 3 \
    --batch-size 1 \
    --learning-rate 5e-5 \
    --max-length 8192 \
    --chat-template nemotron-nano-v3 \
    --trust-remote-code \
    --tp-size 2 \
    --mem-frac 0.85 \
    --save-interval 2000 \
    --log-interval 100
    # ADD --resume-from-checkpoint ./checkpoints/nemotron-nano-eagle3-stage1/best IF supported
    # OTHERWISE apply the manual weight loading patch from Step 5a above
```

Note: lower learning rate (5e-5 vs 1e-4) to preserve Stage 1 knowledge while specializing.

### Step 5-ALT: Experiment 2 — Single-Stage Mixed Training (fallback)

If resume training proves difficult to implement in SpecForge, use this simpler approach instead of Steps 4+5.

```bash
# Merge and shuffle all data
cat cascade2_sft_train.jsonl c2_traces_train.jsonl c2_traces_cot_train.jsonl > all_data_merged.jsonl
shuf all_data_merged.jsonl > all_data_shuffled.jsonl
```

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_eagle3_online.py \
    --target-model-path chankhavu/c2-softcpy-fp8 \
    --draft-model-config ./configs/nemotron-nano-eagle3.json \
    --train-data-path all_data_shuffled.jsonl \
    --output-dir ./checkpoints/nemotron-nano-eagle3-mixed \
    --num-epochs 5 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 8192 \
    --chat-template nemotron-nano-v3 \
    --trust-remote-code \
    --tp-size 2 \
    --mem-frac 0.85 \
    --save-interval 5000 \
    --log-interval 100
```

**CRITICAL**: Verify that the target model is loaded with `mamba_ssm_cache_dtype=float32`.

### Step 5-ALT-2: Experiment 3 — On-Policy Only

Skip SFT data entirely. Uses the same shuffled traces file from Step 5b.

```bash
# If not already done:
cat c2_traces_train.jsonl c2_traces_cot_train.jsonl > c2_traces_merged.jsonl
shuf c2_traces_merged.jsonl > c2_traces_shuffled.jsonl
```

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_eagle3_online.py \
    --target-model-path chankhavu/c2-softcpy-fp8 \
    --draft-model-config ./configs/nemotron-nano-eagle3.json \
    --train-data-path c2_traces_shuffled.jsonl \
    --output-dir ./checkpoints/nemotron-nano-eagle3-onpolicy \
    --num-epochs 5 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 8192 \
    --chat-template nemotron-nano-v3 \
    --trust-remote-code \
    --tp-size 2 \
    --mem-frac 0.85 \
    --save-interval 5000 \
    --log-interval 100
```

**CRITICAL**: Verify that the target model is loaded with `mamba_ssm_cache_dtype=float32`.

### Step 6: Integrate with vLLM / SGLang for Inference

This requires implementing the `SupportsEagle3` interface on the `NemotronHForCausalLM` model class in vLLM:

1. In `vllm/model_executor/models/nemotron_nas.py` (or equivalent), add:
   - `set_aux_hidden_state_layers(layer_ids)` method
   - `get_eagle3_aux_hidden_state_layers()` method returning `[2, 26, 48]` as default
   - Hook into the forward pass to extract and cache hidden states at those layers

2. Create a `config.json` for the trained draft model with `speculators_config`:
   ```json
   {
     "speculators_config": {
       "speculator_type": "eagle3",
       "verifier_model": "chankhavu/c2-softcpy-fp8",
       "num_speculative_tokens": 3,
       "eagle_aux_hidden_state_layer_ids": [2, 26, 48]
     }
   }
   ```

3. Serve with vLLM (use the checkpoint from whichever experiment was run — `nemotron-nano-eagle3-stage2` for Experiment 1, `nemotron-nano-eagle3-mixed` for Experiment 2, `nemotron-nano-eagle3-onpolicy` for Experiment 3):
   ```bash
   vllm serve chankhavu/c2-softcpy-fp8 \
       --tensor-parallel-size 2 \
       --trust-remote-code \
       --mamba-ssm-cache-dtype float32 \
       --speculative-config '{
         "model": "./checkpoints/nemotron-nano-eagle3-stage2",
         "num_speculative_tokens": 3,
         "method": "eagle3",
         "draft_tensor_parallel_size": 1
       }' \
       --max-num-seqs 8 \
       --max-model-len 262144
   ```

   Or with SGLang:
   ```bash
   python -m sglang.launch_server \
       --model chankhavu/c2-softcpy-fp8 \
       --speculative-algorithm EAGLE3 \
       --speculative-draft-model-path ./checkpoints/nemotron-nano-eagle3-stage2 \
       --speculative-num-steps 3 \
       --speculative-eagle-topk 4 \
       --speculative-num-draft-tokens 16 \
       --trust-remote-code \
       --tp 2
   ```

## Key Risks and Unknowns

1. **SpecForge Nemotron-H support**: Nobody has done this before. The target model integration is the biggest engineering lift. The model uses `trust_remote_code` with custom modeling files on HuggingFace, which complicates TP integration.

2. **Hidden state extraction from hybrid layers**: Architecturally sound (residual stream is uniform at hidden_size=2688 regardless of layer type), but untested. The draft head may need more training data to learn the different representation patterns from Mamba vs Attention vs MoE layers.

3. **vLLM Eagle3 interface**: The `SupportsEagle3` interface needs to be added to the NemotronH model class for inference. This is a separate PR/patch needed on top of vLLM.

4. **Acceptance rates on math reasoning**: Eagle3 trained on general chat achieves ~3x speedup. Domain-specialized training (Stage 2 on math rollouts) should improve acceptance rates for the specific workload, but the hybrid architecture is uncharted territory.

5. **Mamba SSM precision**: If any part of the pipeline silently downcasts SSM state to bf16/fp16, the hidden states fed to the draft head will be corrupted. Verify precision at every step.

## Reference Projects

- SpecForge: https://github.com/sgl-project/specforge
- vLLM Speculators: https://github.com/vllm-project/speculators
- NVIDIA gpt-oss-120b Eagle3 long-context: https://huggingface.co/nvidia/gpt-oss-120b-Eagle3-long-context — our north star reference for layer selection and long-context Eagle3
- RedHatAI gpt-oss-120b speculator: https://huggingface.co/RedHatAI/gpt-oss-120b-speculator.eagle3 — reference for Speculators-trained Eagle3
- Eagle3 paper: https://arxiv.org/abs/2503.01840