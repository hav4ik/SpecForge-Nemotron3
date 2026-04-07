import json
import logging
import os
import re
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, Shard, distribute_tensor
from transformers import AutoConfig, PretrainedConfig

logger = logging.getLogger(__name__)


@contextmanager
def rank_0_priority():
    rank = dist.get_rank()

    if rank == 0:
        yield
        dist.barrier()
    else:
        dist.barrier()
        yield


@contextmanager
def default_torch_dtype(dtype: torch.dtype):
    current_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(current_dtype)


def padding(tensor, left=True):
    # Two implementations:
    #   * out-of-place (the original): allocates a new tensor via torch.cat,
    #     so peak memory is 2x the input. Required for tensors that may be
    #     tracked by autograd (loss_mask, position_mask, input_ids etc.) -
    #     mutating them in place would break a saved-for-backward version.
    #   * in-place chunked: overwrites the input via a small scratch buffer.
    #     Used for the verifier's huge [seq, full_vocab] logits tensor at
    #     long context (~16 GiB at L=65536, vocab=131072), where the cat
    #     variant pushed us past the 96 GB GPU ceiling.
    # Heuristic: only go in-place when (a) the tensor is large enough that
    # out-of-place would be a real memory hazard, AND (b) it's a floating
    # point tensor (the integer masks are tiny anyway and can be on the
    # autograd graph). The threshold is high enough that none of the masks
    # / input_ids ever take the in-place path.
    is_big_float = (
        tensor.dim() >= 2
        and tensor.is_floating_point()
        and tensor.element_size() * tensor.numel() >= 1024 * 1024 * 1024  # 1 GiB
    )
    if not is_big_float:
        # Slice along dim=1 (seq), preserving any trailing dims (e.g.
        # position_mask is [batch, seq, 1]).
        zeropadding = torch.zeros_like(tensor[:, -1:])
        if left:
            return torch.cat((zeropadding, tensor[:, :-1]), dim=1)
        return torch.cat((tensor[:, 1:], zeropadding), dim=1)

    # In-place chunked shift along dim=1.
    with torch.no_grad():
        N = tensor.shape[1]
        if N <= 1:
            tensor.zero_()
            return tensor

        bytes_per_col = max(1, tensor[:, :1].element_size() * tensor[:, :1].numel())
        SCRATCH_BYTES = 2 * 1024 * 1024 * 1024
        chunk_cols = max(1, min(N, SCRATCH_BYTES // bytes_per_col))

        if left:
            # Shift right by 1: out[k] = tensor[k-1] for k>=1, out[0] = 0.
            end = N
            while end > 1:
                start = max(end - chunk_cols, 1)
                chunk = tensor[:, start - 1 : end - 1].clone()
                tensor[:, start:end].copy_(chunk)
                end = start
            tensor[:, 0].zero_()
        else:
            # Shift left by 1: out[k] = tensor[k+1] for k<N-1, out[N-1] = 0.
            start = 0
            while start < N - 1:
                end = min(start + chunk_cols, N - 1)
                chunk = tensor[:, start + 1 : end + 1].clone()
                tensor[:, start:end].copy_(chunk)
                start = end
            tensor[:, -1].zero_()
    return tensor


def load_config_from_file(config_path: str):
    with open(config_path, "r") as f:
        config = json.load(f)

    return PretrainedConfig.from_dict(config)


def print_with_rank(message):
    if dist.is_available() and dist.is_initialized():
        logger.info(f"rank {dist.get_rank()}: {message}")
    else:
        logger.info(f"non-distributed: {message}")


def print_args_with_dots(args):
    if dist.get_rank() == 0:
        args_dict = vars(args)
        max_key_length = max(len(key) for key in args_dict.keys())
        total_width = 50

        print("\n -----------【args】-----------")
        for key, value in args_dict.items():
            key_str = f"{key:<{max_key_length}}"
            value_str = str(value)
            dot_count = total_width - len(key_str) - len(value_str)
            dot_fill = "·" * dot_count
            print(f"{key_str} {dot_fill} {value_str}")


def print_on_rank0(message):
    if dist.get_rank() == 0:
        logger.info(message)


def get_last_checkpoint(folder, prefix="epoch"):
    """
    Get the latest checkpoint directory along with its epoch and step information.

    Args:
        folder: The folder path containing checkpoints.
        prefix: The prefix for checkpoint directories, default is "epoch".

    Returns:
        tuple: (checkpoint_path, epoch, step)
               - Returns (None, None, None) if no checkpoint is found.
               - step is 0 if not present in the directory name.
    """
    content = os.listdir(folder)
    # Match: epoch_X or epoch_X_step_Y
    _re_checkpoint = re.compile(rf"^{re.escape(prefix)}_(\d+)(?:_step_(\d+))?$")

    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None
        and os.path.isdir(os.path.join(folder, path))
    ]

    if len(checkpoints) == 0:
        return None, (0, 0)

    # Sort key: (epoch, step), step=0 when not present
    def sort_key(x):
        match = _re_checkpoint.search(x)
        epoch = int(match.group(1))
        step = int(match.group(2)) if match.group(2) else 0
        return (epoch, step)

    last_checkpoint = max(checkpoints, key=sort_key)
    match = _re_checkpoint.search(last_checkpoint)
    epoch = int(match.group(1))
    step = int(match.group(2)) if match.group(2) else 0

    return os.path.join(folder, last_checkpoint), (epoch, step)


def generate_draft_model_config(
    target_model_path: str, template_config_path: str = None, cache_dir: str = None
):
    """
    Auto-generate draft model config based on target model parameters aligned with template config

    Args:
        target_model_path (str): Path to the target model
        template_config_path (str, optional): Template config file path, defaults to llama3-8B-eagle3.json
        cache_dir (str, optional): Cache directory

    Returns:
        dict: Generated draft model config dictionary
    """
    # Get target model config
    target_config = AutoConfig.from_pretrained(target_model_path, cache_dir=cache_dir)

    # If no template specified, use default llama3-8B-eagle3.json
    if template_config_path is None:
        # Use the script execution directory as base
        import sys

        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        project_root = os.path.dirname(script_dir)  # Go up one level from scripts/
        template_config_path = os.path.join(
            project_root, "configs", "llama3-8B-eagle3.json"
        )

    # Read template config
    with open(template_config_path, "r") as f:
        draft_config = json.load(f)

    # Adjust architecture config based on target model type
    if hasattr(target_config, "model_type"):
        # Default to llama architecture
        draft_config["model_type"] = "llama"

    # Align key parameters
    param_mappings = {
        "vocab_size": "vocab_size",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_key_value_heads": "num_key_value_heads",
        "intermediate_size": "intermediate_size",
        "max_position_embeddings": "max_position_embeddings",
        "rms_norm_eps": "rms_norm_eps",
        "hidden_act": "hidden_act",
        "bos_token_id": "bos_token_id",
        "eos_token_id": "eos_token_id",
        "torch_dtype": "torch_dtype",
    }

    # Copy parameters from target model to draft config
    for target_param, draft_param in param_mappings.items():
        if hasattr(target_config, target_param):
            value = getattr(target_config, target_param)
            # Special handling for torch_dtype to make it JSON serializable
            if target_param == "torch_dtype" and isinstance(value, torch.dtype):
                value = str(value).replace("torch.", "")
            draft_config[draft_param] = value

    # Special handling for some parameters
    # Ensure num_hidden_layers is always 1 (EAGLE3 feature)
    draft_config["num_hidden_layers"] = 1

    # Keep some fixed draft model specific parameters
    draft_config["tie_word_embeddings"] = False
    draft_config["use_cache"] = True

    # If template doesn't have draft_vocab_size, set default
    if "draft_vocab_size" not in draft_config:
        draft_config["draft_vocab_size"] = 32000  # Default value

    return draft_config


def save_draft_model_config(config_dict: dict, output_path: str):
    """
    Save draft model config to file

    Args:
        config_dict (dict): Config dictionary
        output_path (str): Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    print(f"Draft model config saved to: {output_path}")


def create_draft_config_from_target(
    target_model_path: str,
    output_dir: str = None,
    template_config_path: str = None,
    cache_dir: str = None,
):
    """
    Convenient function to create draft model config file from target model

    Args:
        target_model_path (str): Target model path
        output_dir (str, optional): Output directory, defaults to configs folder in current directory
        template_config_path (str, optional): Template config path
        cache_dir (str, optional): Cache directory

    Returns:
        str: Generated config file path
    """
    # Generate config
    rank = dist.get_rank()

    if rank == 0:
        print_with_rank(
            "No draft model config provided, auto-generating from target model..."
        )
        config_dict = generate_draft_model_config(
            target_model_path, template_config_path, cache_dir
        )
    dist.barrier()

    # Determine output path
    if output_dir is None:
        # Use the script execution directory as base
        import sys

        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        project_root = os.path.dirname(script_dir)  # Go up one level from scripts/
        output_dir = os.path.join(project_root, "configs")

    # Extract model name from model path
    model_name = target_model_path.split("/")[-1].lower()
    output_filename = f"{model_name}-eagle3-auto.json"
    output_path = os.path.join(output_dir, output_filename)

    # Save config
    if rank == 0:
        save_draft_model_config(config_dict, output_path)
        print_with_rank(f"Auto-generated draft model config saved to: {output_path}")
    dist.barrier()

    return output_path


def get_full_optimizer_state(optimizer_state_dict: dict):
    """
    Convert optimizer state dict with DTensor to full tensors for saving

    Args:
        optimizer_state_dict (dict): Optimizer state dict possibly containing DTensors
    Returns:
        dict: Optimizer state dict with full tensors
    """
    full_optimizer_state_dict = {
        k: v for k, v in optimizer_state_dict.items() if k != "state"
    }
    if "state" in optimizer_state_dict:
        full_optimizer_state_dict["state"] = {
            param_id: {
                state_key: (
                    state_tensor.full_tensor()
                    if isinstance(state_tensor, torch.distributed.tensor.DTensor)
                    else state_tensor
                )
                for state_key, state_tensor in param_state.items()
            }
            for param_id, param_state in optimizer_state_dict["state"].items()
        }
    return full_optimizer_state_dict


def shard_optimizer_state_with_dtensor(bf16_optimizer, device_mesh):
    """
    Shards the optimizer state tensors of a BF16Optimizer instance using DTensor.

    Args:
        bf16_optimizer (BF16Optimizer): An instance of BF16Optimizer, which contains
            the actual optimizer (e.g., torch.optim.Adam) as its `.optimizer` attribute.
    """

    optim = bf16_optimizer.optimizer

    for group in optim.param_groups:
        for p in group["params"]:
            if not isinstance(p, DTensor):
                continue

            state = optim.state.get(p, None)
            if state is None:
                continue

            mesh = device_mesh
            placements = (Shard(dim=0),)

            for k, v in list(state.items()):
                if k == "step":
                    continue

                if isinstance(v, DTensor):
                    continue

                if not isinstance(v, torch.Tensor):
                    continue

                state[k] = distribute_tensor(
                    v.to(p.device), device_mesh=mesh, placements=placements
                )


def safe_conversations_generator(file_path):
    """
    Generator that:
    1. Extracts the 'conversations' field.
    2. Preserves all original fields within each message.
    3. [Key step] Converts all list/dict-type field values to strings to resolve mixed-type conflicts (e.g., for Arrow compatibility).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                raw_convs = row.get("conversations", [])

                # 1. Ensure 'conversations' is a list
                if not isinstance(raw_convs, list):
                    # If it's None or some unexpected type, treat as empty or skip
                    if raw_convs is None:
                        raw_convs = []
                    else:
                        # Edge case: 'conversations' is a plain string or non-iterable—skip this line
                        logger.warning(
                            f"Line {i + 1}: 'conversations' is not a list. Please check!"
                        )
                        continue

                cleaned_convs = []
                for msg in raw_convs:
                    # 2. Ensure each item in the list is a dictionary
                    if not isinstance(msg, dict):
                        # Skip if an element is not a dict (e.g., malformed like ["user", "hi"])
                        continue

                    # 3. [Core logic] Iterate over all fields in the message (role, content, tools, etc.)
                    new_msg = {}
                    for k, v in msg.items():
                        # If the value is a list or dict, serialize it to a JSON string
                        # This ensures Arrow treats the column as string type instead of list/struct
                        if isinstance(v, (list, dict)):
                            new_msg[k] = json.dumps(v, ensure_ascii=False)
                        else:
                            # Keep primitive types (str, int, float, bool, None) unchanged
                            new_msg[k] = v

                    cleaned_convs.append(new_msg)

                # 3b. Normalize the message struct schema across the whole
                # conversation. PyArrow infers a strict struct type from the
                # first batch of rows; if a later row contains a message with
                # extra fields (e.g. tool_calls / tool_call_id / name on
                # SFT tool-use turns) the cast will fail. Ensure every message
                # carries the same set of keys, defaulting absent ones to "".
                _ALL_MSG_KEYS = (
                    "role",
                    "content",
                    "name",
                    "tool_call_id",
                    "tool_calls",
                )
                for msg in cleaned_convs:
                    for k in _ALL_MSG_KEYS:
                        if k not in msg:
                            # Empty string keeps the pyarrow inferred dtype as
                            # `string` for every message column. None would let
                            # the first batch lock the column to `null`, then
                            # later batches with real strings fail to cast.
                            msg[k] = ""

                # Build result with conversations
                result = {"conversations": cleaned_convs}

                # Preserve 'tools' field if present
                if "tools" in row:
                    tools = row["tools"]
                    if tools is not None:
                        # If tools is a JSON string, parse it first
                        if isinstance(tools, str):
                            try:
                                tools = json.loads(tools)
                            except json.JSONDecodeError:
                                logger.warning(
                                    f"Line {i + 1}: 'tools' is a string but not valid JSON, keeping as-is"
                                )
                                result["tools"] = tools
                                yield result
                                continue

                        # Serialize tools to JSON string for Arrow compatibility
                        # (same treatment as list/dict fields in conversations)
                        if isinstance(tools, (list, dict)):
                            result["tools"] = json.dumps(tools, ensure_ascii=False)
                        else:
                            # Primitive type, keep as-is
                            result["tools"] = tools
                    else:
                        result["tools"] = []

                yield result

            except Exception as e:
                logger.warning(f"Skipping line {i + 1}: {e}")
                continue
