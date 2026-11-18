"""AgentDojo pipeline element for the stance experiment (adapted from replication/confirm_llm.py; pre-registered).

Unchanged from the first rating run (replication/): native tool calling through the job's vLLM server; OpenHands `security_risk` + `summary`
inserted first in every tool schema (oh_patch.py); AgentDojo system message + OpenHands policy section; earlier tool
calls re-sent with their ORIGINAL arguments (labels included); reasoning of earlier turns not re-sent; labels popped
before AgentDojo executes the call; no gate is applied (auto-approving human).

Changes (all pre-registered):
- served model name comes from the environment (ROLECALL_SERVED, unique per job, prefix `rolecall-`);
- cfg "rw" = reviewer-wording actor (oh_patch.POLICY_RW / SR_PROP_RW);
- every LLM-call record carries the AgentDojo attempt index (set by the runner), so ALL attempts are kept;
- Llama-3.3's chat template rejects a history turn with several tool calls. For family llama33 only, an assistant
  message with k > 1 tool calls is sent as k consecutive (assistant call_i, tool result_i) pairs. The calls and
  results are the same; only their grouping in the request changes. The first rating run lost 19 Llama episodes to this.
"""
import hashlib
import json
import os
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
SERVED = os.environ.get("ROLECALL_SERVED", "rolecall-model")

CONFIGS = {
    "gptoss": {"family": "gptoss", "max_tokens": 4096, "sampling": {"temperature": 1.0, "top_p": 1.0},
               "reasoning_effort": "medium", "extra": {}},
    "qwen3": {"family": "qwen3", "max_tokens": 2048, "sampling": {"temperature": 0.7, "top_p": 0.8},
              "reasoning_effort": None, "extra": {"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}},
    "llama33": {"family": "llama33", "max_tokens": 2048, "sampling": {"temperature": 0.6, "top_p": 0.9},
                "reasoning_effort": None, "extra": {}},
}
SEED_SALT = "r04g2"   # salt string of the original runs; kept so that per-request seeds reproduce exactly


def tls_reset(ep_key: str):
    _TLS.log = []
    _TLS.argmap = {}
    _TLS.ep_key = ep_key
    _TLS.attempt = 0
    _TLS.last_msgs = None


def tls_set_attempt(a: int):
    _TLS.attempt = a
    _TLS.last_msgs = None


def tls_get():
    return getattr(_TLS, "log", []), getattr(_TLS, "last_msgs", None)


def seed_of(*xs) -> int:
    return int(hashlib.md5("|".join(map(str, (SEED_SALT,) + xs)).encode()).hexdigest()[:8], 16)


def tool_output_string(m) -> str:
    if m.get("error") is not None:
        return m["error"]
    return get_text_content_as_str(m["content"]) if m["content"] is not None else ""


def serialize_messages(msgs) -> list:
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


def sequentialize(out):
    """Llama only: [asst(c1..ck), tool(c1)..tool(ck)] -> [asst(c1), tool(c1), ..., asst(ck), tool(ck)]."""
    res, i = [], 0
    while i < len(out):
        m = out[i]
        tcs = m.get("tool_calls") if m["role"] == "assistant" else None
        if tcs and len(tcs) > 1:
            tools = {}
            j = i + 1
            while j < len(out) and out[j]["role"] == "tool":
                tools[out[j]["tool_call_id"]] = out[j]
                j += 1
            for n, tc in enumerate(tcs):
                res.append({"role": "assistant", "content": m.get("content") if n == 0 else None, "tool_calls": [tc]})
                if tc["id"] in tools:
                    res.append(tools[tc["id"]])
            i = j
            continue
        res.append(m)
        i += 1
    return res


def build_native(messages, tools_fn, cfg, family, argmap):
    """AgentDojo messages -> OpenAI chat messages exactly as the actor sees them (also used by the raters to rebuild
    the actor's own request). argmap: tool-call id -> raw argument string the model emitted (labels included)."""
    out = []
    for m in messages:
        role = m["role"]
        if role == "system":
            out.append({"role": "system", "content": oh_patch.system_prompt(get_text_content_as_str(m["content"]), cfg)})
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
    if family == "llama33":
        out = sequentialize(out)
    return out


def tool_schemas(runtime, cfg):
    return [oh_patch.patch_tool(dict(_function_to_openai(t)), cfg) for t in runtime.functions.values()]


class OHNativeLLM(BasePipelineElement):
    def __init__(self, client: openai.OpenAI, config: str, cfg: str = "main"):
        self.client = client
        self.config = config
        self.c = CONFIGS[config]
        self.cfg = cfg
        self.name = f"oh-{config}-{cfg}"

    def query(self, query: str, runtime: FunctionsRuntime, env: Env = EmptyEnv(), messages: Sequence[ChatMessage] = [],
              extra_args: dict = {}):
        if getattr(_TLS, "log", None) is None:
            tls_reset("nokey")
        out = build_native(messages, None, self.cfg, self.c["family"], _TLS.argmap)
        tools = tool_schemas(runtime, self.cfg)
        log = _TLS.log
        turn = len(log)
        kw = dict(model=SERVED, messages=out, max_tokens=self.c["max_tokens"], tools=tools, tool_choice="auto",
                  seed=seed_of(_TLS.ep_key, turn), extra_body=dict(self.c["extra"]), **self.c["sampling"])
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
        rec = {"turn": turn, "attempt": _TLS.attempt, "finish_reason": ch.finish_reason, "content": msg.content,
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
