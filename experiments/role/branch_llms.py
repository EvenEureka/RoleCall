"""LLM pipeline elements for the branching experiment.

Identical-history branching
    BranchTextCallLLM  (Llama-3.3)  = wave-2 arm B (text calls, AgentDojo `local` prompt, tool output in role `tool`,
                                      which Llama renders as an `ipython` turn through `tojson`), except that a tool
                                      message carrying m["r3_header"] == "user" is sent exactly as wave-2 arm C
                                      (role `user`, content json.dumps(raw): same rendered bytes, only the header differs).
    BranchNativeLLM    (gpt-oss)    = wave-2 arm A (native calls, role `tool` -> harmony `functions.NAME`), except that a
                                      tool message with r3_header == "user" is sent exactly as wave-2 arm Au (role `user`,
                                      raw content).
    Both subclass the unchanged relabelling classes (llms.py) and only override one message.

(The source-reimplemented framework adapters of the same round were superseded by the installed frameworks,
see installed_agents.py, and are not part of this release.)
"""
import json
import os
import re
import threading
import time
import uuid
from collections.abc import Sequence

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime
from agentdojo.types import ChatAssistantMessage, ChatMessage, get_text_content_as_str, text_content_block_from_string

from llms import (NativeLLM, TextCallLLM, _TLS, make_llm as w2_make_llm, raw_log, reset_raw_log,  # noqa: F401
                     set_episode_seed, tool_content_matching_tool_render, tool_output_string)

# ------------------------------------------------------------------ heartbeat (watchdog support)
HEARTBEAT = {"t": time.time(), "n": 0, "path": None}
_HB_LOCK = threading.Lock()


def heartbeat():
    with _HB_LOCK:
        HEARTBEAT["t"] = time.time()
        HEARTBEAT["n"] += 1
        if HEARTBEAT["path"] and HEARTBEAT["n"] % 10 == 0:
            try:
                with open(HEARTBEAT["path"], "w") as f:
                    f.write(f"{HEARTBEAT['t']:.0f} {HEARTBEAT['n']}\n")
            except OSError:
                pass


def reset_episode_state():
    reset_raw_log()
    _TLS.adapter_calls = 0


def _count_call():
    _TLS.adapter_calls = getattr(_TLS, "adapter_calls", 0) + 1
    return _TLS.adapter_calls


def _log_completion(resp, n_messages, temperature, extra=None):
    ch = resp.choices[0]
    msg = ch.message
    rec = {"finish_reason": ch.finish_reason, "content": msg.content,
           "reasoning": getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None),
           "tool_calls": [tc.model_dump() for tc in (msg.tool_calls or [])],
           "usage": resp.usage.model_dump() if resp.usage else None, "n_messages": n_messages, "seed": None,
           "temperature": temperature}
    if extra:
        rec.update(extra)
    raw_log().append(rec)
    heartbeat()
    return rec


# ------------------------------------------------------------------ (a) branching arms
class BranchTextCallLLM(TextCallLLM):
    def __init__(self, *a, **kw):
        super().__init__(*a, arm="B", **kw)
        self.arm = "Bbr"

    def build_request(self, messages, runtime):
        out, _ = super().build_request(messages, runtime)  # one output message per AgentDojo message, same order
        assert len(out) == len(messages)
        hdr = []
        for idx, m in enumerate(messages):
            if m["role"] == "tool" and m.get("r3_header") == "user":
                out[idx] = {"role": "user", "content": tool_content_matching_tool_render(tool_output_string(m),
                                                                                         self.family)}
                hdr.append(idx)
        _TLS.user_header_idx = hdr
        return out, None

    def _create(self, messages, tools=None):
        msg = super()._create(messages, tools)
        raw_log()[-1]["user_header_idx"] = getattr(_TLS, "user_header_idx", [])
        heartbeat()
        return msg


class BranchNativeLLM(NativeLLM):
    def __init__(self, *a, **kw):
        super().__init__(*a, tool_role="tool", **kw)
        self.arm = "Abr"

    def build_request(self, messages, runtime):
        out, tools = super().build_request(messages, runtime)
        assert len(out) == len(messages)
        hdr = []
        for idx, m in enumerate(messages):
            if m["role"] == "tool" and m.get("r3_header") == "user":
                out[idx] = {"role": "user", "content": tool_content_matching_tool_render(tool_output_string(m),
                                                                                         self.family)}
                hdr.append(idx)
        _TLS.user_header_idx = hdr
        return out, tools

    def _create(self, messages, tools=None):
        msg = super()._create(messages, tools)
        raw_log()[-1]["user_header_idx"] = getattr(_TLS, "user_header_idx", [])
        heartbeat()
        return msg


# ------------------------------------------------------------------ factory
def make_llm(arm, client, model, family, max_tokens, extra_body=None, reasoning_effort=None, temperature=0.0):
    kw = dict(extra_body=extra_body, reasoning_effort=reasoning_effort, temperature=temperature)
    if arm == "Bbr":
        return BranchTextCallLLM(client, model, family, max_tokens, **kw)
    if arm == "Abr":
        return BranchNativeLLM(client, model, family, max_tokens, **kw)
    return w2_make_llm(arm, client, model, family, max_tokens, extra_body, reasoning_effort, temperature)
