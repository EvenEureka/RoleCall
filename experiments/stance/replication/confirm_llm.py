"""AgentDojo pipeline element: native tool calling through a local vLLM server, with the OpenHands `security_risk`
(+ `summary`) field patched into every tool schema (oh_patch.py). Adapted from the role experiments' `NativeLLM` (arm A, experiments/role/llms.py).

Per LLM call:
- system message = AgentDojo default system message + OpenHands <SECURITY_RISK_ASSESSMENT> section;
- tools = AgentDojo's own OpenAI schemas (agentdojo `_function_to_openai`) + the OpenHands meta-fields;
- earlier assistant tool calls are re-sent with their ORIGINAL arguments (incl. security_risk / summary), as the
  OpenHands event history does; reasoning of earlier turns is not re-sent (AgentDojo native behaviour);
- tool outputs = AgentDojo's tool_result_to_str string (or the error string), role `tool`;
- the returned tool calls are split: `security_risk`/`summary` are popped (Agent._extract_security_risk semantics)
  and recorded; AgentDojo executes the remaining arguments. No gate is applied (every call executes, i.e. an
  auto-approving human); the gate is evaluated counterfactually in the analysis.
Raw per-call records (content, reasoning, tool calls with labels, finish_reason, usage) go to a thread-local log
that the runner flushes per episode, together with the final message list.
"""
import hashlib
import json
import threading
import uuid
from collections.abc import Sequence

import openai
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.llms.openai_llm import _function_to_openai
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime
from agentdojo.types import ChatAssistantMessage, ChatMessage, get_text_content_as_str, text_content_block_from_string

import oh_patch

_TLS = threading.local()

CONFIGS = {
    # sampling = each model's recommended setting; seed per (episode, turn)
    "gptoss": {"served": "gpt-oss-120b", "family": "gptoss", "max_tokens": 4096,
               "sampling": {"temperature": 1.0, "top_p": 1.0}, "reasoning_effort": "medium", "extra": {}},
    "qwen3": {"served": "qwen3-32b", "family": "qwen3", "max_tokens": 2048,
              "sampling": {"temperature": 0.7, "top_p": 0.8}, "reasoning_effort": None,
              "extra": {"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}},
    "qwen3think": {"served": "qwen3-32b", "family": "qwen3", "max_tokens": 6144,
                   "sampling": {"temperature": 0.6, "top_p": 0.95}, "reasoning_effort": None,
                   "extra": {"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}}},
    "llama33": {"served": "llama-3.3-70b", "family": "llama33", "max_tokens": 2048,
                "sampling": {"temperature": 0.6, "top_p": 0.9}, "reasoning_effort": None, "extra": {}},
}


def tls_reset(ep_key: str):
    _TLS.log = []
    _TLS.argmap = {}
    _TLS.ep_key = ep_key
    _TLS.last_msgs = None


def tls_get():
    return getattr(_TLS, "log", []), getattr(_TLS, "last_msgs", None)


def seed_of(*xs) -> int:
    return int(hashlib.md5("|".join(map(str, xs)).encode()).hexdigest()[:8], 16)


def tool_output_string(m) -> str:
    if m.get("error") is not None:
        return m["error"]
    return get_text_content_as_str(m["content"]) if m["content"] is not None else ""


def serialize_messages(msgs) -> list:
    """AgentDojo messages -> plain JSON (tool-call args as executed, i.e. without security_risk/summary)."""
    out = []
    for m in msgs:
        role = m["role"]
        d = {"role": role}
        c = m.get("content")
        d["content"] = get_text_content_as_str(c) if c is not None else None
        if role == "assistant":
            d["tool_calls"] = [{"function": tc.function, "args": dict(tc.args), "id": tc.id}
                               for tc in (m.get("tool_calls") or [])]
        elif role == "tool":
            tc = m.get("tool_call")
            d["tool_call_id"] = m.get("tool_call_id")
            d["function"] = tc.function if tc is not None else None
            d["error"] = m.get("error")
            d["output"] = tool_output_string(m)
        out.append(d)
    return out


class OHNativeLLM(BasePipelineElement):
    def __init__(self, client: openai.OpenAI, config: str, cfg: str = "main"):
        self.client = client
        self.config = config
        self.c = CONFIGS[config]
        self.cfg = cfg
        self.name = f"oh-{config}-{cfg}"

    def build_request(self, messages, runtime):
        argmap = getattr(_TLS, "argmap", {})
        out = []
        for m in messages:
            role = m["role"]
            if role == "system":
                out.append({"role": "system",
                            "content": oh_patch.system_prompt(get_text_content_as_str(m["content"]), self.cfg)})
            elif role == "user":
                out.append({"role": "user", "content": get_text_content_as_str(m["content"])})
            elif role == "assistant":
                d = {"role": "assistant",
                     "content": get_text_content_as_str(m["content"]) if m["content"] is not None else None}
                if m["tool_calls"]:
                    d["tool_calls"] = [{"id": tc.id, "type": "function",
                                        "function": {"name": tc.function,
                                                     "arguments": argmap.get(tc.id, json.dumps(dict(tc.args)))}}
                                       for tc in m["tool_calls"]]
                out.append(d)
            elif role == "tool":
                out.append({"role": "tool", "content": tool_output_string(m), "tool_call_id": m["tool_call_id"],
                            "name": m["tool_call"].function})
        tools = [oh_patch.patch_tool(dict(_function_to_openai(t)), self.cfg) for t in runtime.functions.values()]
        return out, tools

    def query(self, query: str, runtime: FunctionsRuntime, env: Env = EmptyEnv(), messages: Sequence[ChatMessage] = [],
              extra_args: dict = {}):
        out, tools = self.build_request(messages, runtime)
        log = getattr(_TLS, "log", None)
        if log is None:
            tls_reset("nokey")
            log = _TLS.log
        turn = len(log)
        kw = dict(model=self.c["served"], messages=out, max_tokens=self.c["max_tokens"], tools=tools,
                  tool_choice="auto", seed=seed_of(_TLS.ep_key, turn), extra_body=dict(self.c["extra"]),
                  **self.c["sampling"])
        if self.c["reasoning_effort"]:
            kw["reasoning_effort"] = self.c["reasoning_effort"]
        resp = self.client.chat.completions.create(**kw)
        ch = resp.choices[0]
        msg = ch.message
        calls, tcs = [], []
        for tc in (msg.tool_calls or []):
            raw_args = tc.function.arguments or "{}"
            try:
                args = json.loads(raw_args)
                bad = False
            except json.JSONDecodeError:
                args, bad = {}, True
            if not isinstance(args, dict):
                args, bad = {}, True
            stripped, lab, raw, summ = oh_patch.extract_meta(args, self.cfg)
            cid = tc.id or f"call_{uuid.uuid4().hex[:12]}"
            _TLS.argmap[cid] = raw_args
            calls.append({"id": cid, "function": tc.function.name, "args": stripped, "label": lab, "raw_label": raw,
                          "summary": summ, "bad_json": bad, "raw_arguments": raw_args})
            tcs.append(FunctionCall(function=tc.function.name, args=stripped, id=cid))
        rec = {"turn": turn, "finish_reason": ch.finish_reason, "content": msg.content,
               "reasoning": getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None),
               "calls": calls, "usage": resp.usage.model_dump() if resp.usage else None, "n_messages": len(messages)}
        log.append(rec)
        output = ChatAssistantMessage(
            role="assistant",
            content=[text_content_block_from_string(msg.content)] if msg.content is not None else None,
            tool_calls=tcs or None)
        new_msgs = [*messages, output]
        _TLS.last_msgs = new_msgs
        return query, runtime, env, new_msgs, extra_args
