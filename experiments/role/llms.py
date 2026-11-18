"""Delivery-role arms for the relabelling experiment (arms A, B and C reproduce the first-wave arms unchanged).

All arms talk to one OpenAI-compatible vLLM server. Tool-output content is the raw string from AgentDojo's
tool-output formatter (`tool_result_to_str`, YAML) or the raw error string; it is the same string in every arm, and
only its role (and, where the model's template forces it, its JSON encoding) changes.

A   native calls (tools=)  tool output in role `tool`                              (wave 1)
Au  native calls (tools=)  tool output in role `user`, same rendered bytes as A      (wave 2, new)
B   text calls             tool output in role `tool`                              (wave 1)
C   text calls             tool output in role `user`; Llama: json.dumps(raw) so the rendered bytes equal B's
                           (Llama's template renders tool content through `tojson`)  (wave 1)
Cr  text calls             tool output in role `user`, RAW string (realistic ReAct observation) (wave 2, new)
Di  text calls             tool output in role `input`, RAW string (Meta-SecAlign's convention for undefended
                           local models; Llama-3.3 renders an unknown role as a plain `input` header)  (wave 2, new)

Decoding: temperature 0 by default (wave-1 setting). Seeded mode (wave-2 seed cells): temperature T, top_p 1.0,
top_k 0 (disabled), and a per-request `seed` = episode base seed + call index; the runner sets the episode base seed
(a stable hash of seed, arm, suite, user task, injection task) in thread-local state, so arms get independent streams.
"""
import json
import threading
import uuid
from collections.abc import Sequence

import openai
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.llms.local_llm import _make_system_prompt, _parse_model_output
from agentdojo.agent_pipeline.llms.openai_llm import _function_to_openai, _openai_to_tool_call
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatAssistantMessage, ChatMessage, get_text_content_as_str, text_content_block_from_string

_TLS = threading.local()


def raw_log() -> list:
    if not hasattr(_TLS, "log"):
        _TLS.log = []
    return _TLS.log


def reset_raw_log():
    _TLS.log = []


def set_episode_seed(base):
    _TLS.seed_base = base
    _TLS.call_idx = 0


def _next_seed():
    base = getattr(_TLS, "seed_base", None)
    if base is None:
        return None
    s = base + _TLS.call_idx
    _TLS.call_idx += 1
    return s % (2 ** 31 - 1)


def tool_output_string(m) -> str:
    if m.get("error") is not None:
        return m["error"]
    return get_text_content_as_str(m["content"]) if m["content"] is not None else ""


def tool_content_matching_tool_render(raw: str, family: str) -> str:
    """User-role content whose rendered bytes equal the tool-role render (Llama templates apply `tojson`)."""
    if family in ("llama33", "llama31"):
        return json.dumps(raw, ensure_ascii=False)
    return raw


class _Base(BasePipelineElement):
    def __init__(self, client, model, family, max_tokens, extra_body=None, reasoning_effort=None,
                 temperature=0.0):
        self.client = client
        self.model = model
        self.family = family
        self.max_tokens = max_tokens
        self.extra_body = dict(extra_body or {})
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature

    def _create(self, messages, tools=None):
        extra = dict(self.extra_body)
        kw = dict(model=self.model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens)
        seed = _next_seed() if self.temperature > 0 else None
        if self.temperature > 0:
            kw["top_p"] = 1.0
            extra["top_k"] = 0
            kw["seed"] = seed
        kw["extra_body"] = extra
        if tools:
            kw["tools"] = tools
            kw["tool_choice"] = "auto"
        if self.reasoning_effort:
            kw["reasoning_effort"] = self.reasoning_effort
        resp = self.client.chat.completions.create(**kw)
        ch = resp.choices[0]
        msg = ch.message
        raw_log().append({
            "finish_reason": ch.finish_reason,
            "content": msg.content,
            "reasoning": getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None),
            "tool_calls": [tc.model_dump() for tc in (msg.tool_calls or [])],
            "usage": resp.usage.model_dump() if resp.usage else None,
            "n_messages": len(messages),
            "seed": seed,
            "temperature": self.temperature,
        })
        return msg


class NativeLLM(_Base):
    """Arms A (tool_role='tool') and Au (tool_role='user')."""

    def __init__(self, *a, tool_role="tool", **kw):
        super().__init__(*a, **kw)
        assert tool_role in ("tool", "user")
        self.tool_role = tool_role
        self.arm = "A" if tool_role == "tool" else "Au"

    def build_request(self, messages, runtime):
        out = []
        for m in messages:
            role = m["role"]
            if role in ("system", "user"):
                out.append({"role": role, "content": get_text_content_as_str(m["content"])})
            elif role == "assistant":
                d = {"role": "assistant",
                     "content": get_text_content_as_str(m["content"]) if m["content"] is not None else None}
                if m["tool_calls"]:
                    d["tool_calls"] = [{"id": tc.id, "type": "function",
                                        "function": {"name": tc.function, "arguments": json.dumps(tc.args)}}
                                       for tc in m["tool_calls"]]
                out.append(d)
            elif role == "tool":
                raw = tool_output_string(m)
                if self.tool_role == "tool":
                    out.append({"role": "tool", "content": raw, "tool_call_id": m["tool_call_id"],
                                "name": m["tool_call"].function})
                else:
                    out.append({"role": "user", "content": tool_content_matching_tool_render(raw, self.family)})
        tools = [_function_to_openai(t) for t in runtime.functions.values()]
        return out, tools

    def query(self, query: str, runtime: FunctionsRuntime, env: Env = EmptyEnv(), messages: Sequence[ChatMessage] = [],
              extra_args: dict = {}):
        out, tools = self.build_request(messages, runtime)
        msg = self._create(out, tools=tools)
        tool_calls = [_openai_to_tool_call(tc) for tc in msg.tool_calls] if msg.tool_calls else None
        output = ChatAssistantMessage(
            role="assistant",
            content=[text_content_block_from_string(msg.content)] if msg.content is not None else None,
            tool_calls=tool_calls)
        return query, runtime, env, [*messages, output], extra_args


class TextCallLLM(_Base):
    """Arms B ('tool'), C ('user', template-matched bytes), Cr ('user', raw), Di ('input', raw)."""

    MODES = {"B": ("tool", False), "C": ("user", True), "Cr": ("user", False), "Di": ("input", False)}

    def __init__(self, *a, arm="B", **kw):
        super().__init__(*a, **kw)
        assert arm in self.MODES
        self.arm = arm
        self.tool_role, self.match_bytes = self.MODES[arm]

    def build_request(self, messages, runtime):
        out = []
        for m in messages:
            role = m["role"]
            if role == "system":
                out.append({"role": "system",
                            "content": _make_system_prompt(get_text_content_as_str(m["content"]),
                                                           runtime.functions.values())})
            elif role == "user":
                out.append({"role": "user", "content": get_text_content_as_str(m["content"])})
            elif role == "assistant":
                out.append({"role": "assistant",
                            "content": get_text_content_as_str(m["content"]) if m["content"] is not None else ""})
            elif role == "tool":
                raw = tool_output_string(m)
                if self.tool_role == "tool":
                    d = {"role": "tool", "content": raw}
                    if self.family == "gptoss":
                        tcid = "call_" + uuid.uuid5(uuid.NAMESPACE_OID, raw + m["tool_call"].function).hex[:12]
                        d["tool_call_id"] = tcid
                        d["tool_calls"] = [{"id": tcid, "type": "function",
                                            "function": {"name": m["tool_call"].function, "arguments": "{}"}}]
                    out.append(d)
                else:
                    content = tool_content_matching_tool_render(raw, self.family) if self.match_bytes else raw
                    out.append({"role": self.tool_role, "content": content})
        return out, None

    def query(self, query: str, runtime: FunctionsRuntime, env: Env = EmptyEnv(), messages: Sequence[ChatMessage] = [],
              extra_args: dict = {}):
        out, _ = self.build_request(messages, runtime)
        msg = self._create(out, tools=None)
        output = _parse_model_output(msg.content or "")
        return query, runtime, env, [*messages, output], extra_args


ARMS = ("A", "Au", "B", "C", "Cr", "Di")


def make_llm(arm, client, model, family, max_tokens, extra_body=None, reasoning_effort=None, temperature=0.0):
    kw = dict(extra_body=extra_body, reasoning_effort=reasoning_effort, temperature=temperature)
    if arm == "A":
        return NativeLLM(client, model, family, max_tokens, tool_role="tool", **kw)
    if arm == "Au":
        return NativeLLM(client, model, family, max_tokens, tool_role="user", **kw)
    if arm in TextCallLLM.MODES:
        return TextCallLLM(client, model, family, max_tokens, arm=arm, **kw)
    raise ValueError(arm)
