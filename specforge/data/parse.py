import json
import re
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import torch
from transformers import PreTrainedTokenizer

from .template import ChatTemplate

__all__ = ["GeneralParser", "HarmonyParser", "ThinkingParser"]


class Parser(ABC):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        chat_template: ChatTemplate,
    ):
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.standard_keys = {"role", "content", "tool_calls"}

    @abstractmethod
    def parse(
        self, conversation: "Conversation", max_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parse the conversation into a list of tensors.

        Args:
            conversation: The conversation to parse.

        Returns:
            A list of tensors: [input_ids, loss_mask]
        """

    def _sanitize_message(self, message: dict) -> dict:
        """
        Clean up individual messages, handling the following issues:
        1. `tool_calls` is a string → Parse as a list
        2. `tool_calls[].function.arguments` is a string → Parse as a dictionary
        3. Non-standard fields (extra, etc.) in `tool_calls[]` → Remove
        """
        cleaned = {k: v for k, v in message.items() if k in self.standard_keys}

        # ===== handle tool_calls =====
        if "tool_calls" in cleaned:
            tool_calls = cleaned["tool_calls"]

            # safe_conversations_generator inserts an empty-string placeholder
            # for the tool_calls field on non-tool messages so that pyarrow
            # locks every message column to the same `string` dtype. Treat
            # that as "no tool calls" silently.
            if tool_calls == "" or tool_calls is None:
                cleaned.pop("tool_calls", None)
                return cleaned

            # tool_calls is a string → Parsing
            if isinstance(tool_calls, str):
                try:
                    tool_calls = json.loads(tool_calls)
                except json.JSONDecodeError:
                    warnings.warn(
                        f"Failed to parse tool_calls JSON string, removing tool_calls"
                    )
                    cleaned.pop("tool_calls", None)
                    return cleaned

            # Clean each tool_call
            if isinstance(tool_calls, list):
                sanitized_tool_calls = []

                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue

                    # Only retain the standard fields: id, type, function
                    clean_tc = {
                        "id": tc.get("id", ""),
                        "type": tc.get("type", "function"),
                    }

                    # handle function
                    func = tc.get("function", {})
                    if isinstance(func, dict):
                        clean_func = {
                            "name": func.get("name", ""),
                        }

                        arguments = func.get("arguments", {})
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except json.JSONDecodeError:
                                warnings.warn(
                                    f"Failed to parse arguments for tool '{clean_func['name']}': "
                                    f"{arguments[:100]}..."
                                )
                                arguments = {}

                        clean_func["arguments"] = arguments
                        clean_tc["function"] = clean_func

                    sanitized_tool_calls.append(clean_tc)

                cleaned["tool_calls"] = sanitized_tool_calls

        return cleaned


_harmony_encoding = None


class GeneralParser(Parser):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        chat_template: ChatTemplate,
    ):
        super().__init__(tokenizer, chat_template)
        self.system_prompt = chat_template.system_prompt
        self.user_message_separator = f"{chat_template.end_of_turn_token}"
        self.assistant_message_separator = f"{chat_template.assistant_header}"
        self.set_assistant_pattern(chat_template)

    def apply_chat_template(self, messages, tool, **kwargs) -> str:
        conversation = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tool,
            **kwargs,
        )
        return conversation

    def set_assistant_pattern(self, chat_template: ChatTemplate):
        if chat_template.assistant_pattern_type == "longcat":
            self.assistant_pattern = (
                re.escape(self.assistant_message_separator)
                + r"([\s\S]*?(?:"
                + re.escape("[Round ")
                + r"\d+"
                + re.escape("] USER:")
                + "|$))"
            )
        else:
            self.assistant_pattern = (
                re.escape(self.assistant_message_separator)
                + r"([\s\S]*?(?:"
                + re.escape(self.chat_template.end_of_turn_token)
                + "|$))"
            )

    def parse(
        self,
        conversation: "Conversation",
        max_length: int,
        preformatted: bool = False,
        train_only_last_turn: bool = False,
        tool: List[Dict] = [],
        **kwargs,
    ) -> Dict[str, List[torch.Tensor]]:
        if not preformatted:
            messages = []

            if conversation[0]["role"] == "system":
                warnings.warn(
                    f"The first message is from system, we will use the system prompt from the data and ignore the system prompt from the template"
                )
                messages.append(
                    {"role": "system", "content": conversation[0]["content"]}
                )
                conversation = conversation[1:]
            else:
                if self.system_prompt:
                    messages.append({"role": "system", "content": self.system_prompt})

            for j, sentence in enumerate(conversation):
                role = sentence["role"]
                if j == 0:
                    if role != "user":
                        warnings.warn(
                            f"Conversation must start with a 'user' role, but found '{role}'. Conversation truncated."
                        )
                        break
                else:
                    prev_role = conversation[j - 1]["role"]
                    if role == "tool" and prev_role not in ["assistant", "tool"]:
                        warnings.warn(
                            f"A 'tool' message must follow an 'assistant' or 'tool' message, but was preceded by '{prev_role}'. Conversation truncated."
                        )
                        break
                    if role == "assistant" and prev_role not in ["user", "tool"]:
                        warnings.warn(
                            f"An 'assistant' message must follow a 'user' or 'tool' message, but was preceded by '{prev_role}'. Conversation truncated."
                        )
                        break
                sentence = self._sanitize_message(sentence)
                messages.append(sentence)
            try:
                conversation = self.apply_chat_template(messages, tool=tool, **kwargs)
            except (ValueError, TypeError):
                # Fallback rendering for tokenizers without built-in chat_template
                warnings.warn(
                    "Tokenizer does not have a chat_template, using fallback rendering."
                )
                parts = []
                bos_token = getattr(self.tokenizer, "bos_token", None)
                user_header = self.chat_template.user_header or ""
                assistant_header = self.chat_template.assistant_header or ""
                end_of_turn = self.chat_template.end_of_turn_token or ""

                # Add BOS token at the start
                if bos_token:
                    parts.append(bos_token)

                for msg in messages:
                    if msg["role"] == "system":
                        parts.append(msg["content"])
                    elif msg["role"] == "user":
                        parts.append(f"{user_header}{msg['content']}")
                    elif msg["role"] == "assistant":
                        parts.append(f"{assistant_header}{msg['content']}{end_of_turn}")
                conversation = "".join(parts)

        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

        # Single tokenize call with offset mapping. For fast tokenizers this
        # is dramatically cheaper than re-encoding conversation prefixes for
        # every assistant turn (the previous implementation was O(num_turns
        # * seq_len), which dominates wall-clock for long reasoning traces
        # at max_length=4096+).
        try:
            encoding = self.tokenizer(
                conversation,
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
            offsets = encoding["offset_mapping"][0].tolist()
            fast_path = True
        except (TypeError, NotImplementedError, ValueError):
            # Slow tokenizer fallback: no offset mapping available.
            encoding = self.tokenizer(
                conversation,
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            offsets = None
            fast_path = False

        input_ids = encoding.input_ids[0]
        loss_mask = torch.zeros(len(input_ids), dtype=torch.long)

        # ---- helper: map (char_start, char_end) -> token (start, end) ----
        # Token i covers chars [offsets[i][0], offsets[i][1]). The first token
        # whose end > char_start is our start; the first token whose start
        # >= char_end is our end. Binary search both via bisect.
        import bisect

        if fast_path:
            token_starts = [o[0] for o in offsets]
            token_ends = [o[1] for o in offsets]

            def char_span_to_token_span(c_start: int, c_end: int):
                # start: first token whose end > c_start
                start = bisect.bisect_right(token_ends, c_start)
                # end: first token whose start >= c_end
                end = bisect.bisect_left(token_starts, c_end)
                start = min(start, len(input_ids))
                end = min(end, len(input_ids))
                return start, end
        else:
            # Slow-path fallback: re-tokenize prefixes (the original
            # implementation). Only used for non-fast tokenizers.
            def char_span_to_token_span(c_start: int, c_end: int):
                prefix_ids = self.tokenizer.encode(
                    conversation[:c_start],
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_length,
                )
                full_ids = self.tokenizer.encode(
                    conversation[:c_end],
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_length,
                )
                return (
                    min(len(prefix_ids), len(input_ids)),
                    min(len(full_ids), len(input_ids)),
                )

        matches = list(re.finditer(self.assistant_pattern, conversation, re.DOTALL))
        if train_only_last_turn and matches:
            matches = [matches[-1]]  # Only keep the last match

        for match in matches:
            content_start_char = match.start(1)
            content_end_char = match.end(1)
            actual_start, actual_end = char_span_to_token_span(
                content_start_char, content_end_char
            )
            if actual_start < actual_end:
                loss_mask[actual_start:actual_end] = 1

        # Zero out loss_mask for ignore_tokens
        ignore_tokens = self.chat_template.ignore_token
        if ignore_tokens:
            for token_str in ignore_tokens:
                start = 0
                while True:
                    idx = conversation.find(token_str, start)
                    if idx == -1:
                        break
                    ignore_start_char = idx
                    ignore_end_char = idx + len(token_str)
                    s, e = char_span_to_token_span(ignore_start_char, ignore_end_char)
                    if s < e:
                        loss_mask[s:e] = 0
                    start = ignore_end_char

        return input_ids, loss_mask


class HarmonyParser(Parser):
    def __init__(self, tokenizer: PreTrainedTokenizer, chat_template: ChatTemplate):
        super().__init__(tokenizer, chat_template)
        self.reasoning_levels = ["low", "medium", "high"]
        self.default_reasoning_level = "low"

    def build_single_turn_prompt(
        self,
        prompt_text: str,
        role: str,
        content: str,
    ) -> str:
        """Embed user message into the required prompt template."""
        if role == "system":
            prompt_text = f"<|start|>system<|message|>{content}<|end|>"
        elif role == "assistant_reasoning_effort":
            prompt_text = f"<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2025-06-28\n\nReasoning: {content.lower()}\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
        elif role == "user":
            prompt_text += f"<|start|>user<|message|>{content}<|end|>"
        elif role == "assistant_analysis":
            prompt_text += (
                f"<|start|>assistant<|channel|>analysis<|message|>{content}<|end|>"
            )
        elif role == "assistant_commentary":
            prompt_text += (
                f"<|start|>assistant<|channel|>commentary<|message|>{content}<|end|>"
            )
        elif role == "assistant_final":
            prompt_text += (
                f"<|start|>assistant<|channel|>final<|message|>{content}<|end|>"
            )
        else:
            raise ValueError(f"Unknown role: {role}")
        return prompt_text

    def parse(
        self,
        conversation: "Conversation",
        max_length: int,
        preformatted: bool = False,
        train_only_last_turn: bool = False,
        tool: List[Dict] = [],
    ) -> List[torch.Tensor]:
        # conversation = process_harmony_conversations(conversation)
        if not preformatted:
            prompt_text = ""
            for j, message in enumerate(conversation):
                if j == 0 and (
                    message["role"] != "system"
                    or message["role"] != "assistant_reasoning_effort"
                ):
                    prompt_text = self.build_single_turn_prompt(
                        prompt_text,
                        "assistant_reasoning_effort",
                        self.default_reasoning_level,
                    )
                prompt_text = self.build_single_turn_prompt(
                    prompt_text, message["role"], message["content"]
                )
            conversation = prompt_text

        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

        encoding = self.tokenizer(
            conversation,
            return_offsets_mapping=True,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoding.input_ids[0]
        offsets = encoding.offset_mapping[0]
        loss_mask = torch.zeros(len(input_ids), dtype=torch.long)

        # Find spans of assistant responses using regex
        # We match `<|start|>assistant` and only extract the content following it.
        # This continues until `<|start|>user<|message|>` appears, or until the end of the string.
        pattern = re.compile(
            r"<\|start\|>assistant([\s\S]*?)(?=<\|start\|>user<\|message\|>|$)"
        )

        # Find all matching segments
        matches = list(pattern.finditer(conversation))
        if train_only_last_turn and matches:
            matches = [matches[-1]]  # Only keep the last match

        for match in matches:
            # match.start(0) is the start index of the full match (including `<|start|>assistant`)
            # match.start(1) is the start index of the first capture group (excluding `<|start|>assistant`)
            # match.end(1) is the end index of the content
            start_char = match.start(1)
            end_char = match.end(1)

            # Map character indices to token indices
            for idx, (ts, te) in enumerate(offsets):
                # Set mask to 1 only if the token's character range falls entirely within the "content area"
                if ts >= start_char and te <= end_char:
                    loss_mask[idx] = 1

        return input_ids, loss_mask


class ThinkingParser(GeneralParser):
    """
    Parser for thinking/reasoning models.

    This parser processes the entire conversation (not just the last turn).
    It handles reasoning_content and tool_calls in assistant messages.
    The loss mask covers from assistant_header to end_of_turn_token (inclusive).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        chat_template: ChatTemplate,
    ):
        super().__init__(tokenizer, chat_template)
        self.standard_keys = {"role", "content", "tool_calls", "reasoning_content"}

    def apply_chat_template(self, messages, tool, **kwargs) -> str:
        """Apply chat template to all messages, handling reasoning_content and tool_calls."""
        conversation = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
            tools=tool,
            **kwargs,
        )
        return conversation

    def parse(
        self,
        conversation: "Conversation",
        max_length: int,
        preformatted: bool = False,
        train_only_last_turn: bool = False,
        tool: List[Dict] = [],
        **kwargs,
    ) -> Dict[str, List[torch.Tensor]]:
        """Parse conversation, processing all assistant turns for loss mask."""
        if self.chat_template.enable_thinking:
            kwargs["enable_thinking"] = True
        return super().parse(
            conversation, max_length, preformatted, train_only_last_turn, tool, **kwargs
        )
