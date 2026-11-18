"""Installed-framework agents: the INSTALLED frameworks drive the episode (an earlier round reimplemented them from source).

One AgentDojo pipeline element runs a real framework agent loop:
  NLIu / NLIt  llama-index-core 0.14.25 ReActAgent (workflow), OpenAILike LLM; observation_role USER (default) / TOOL
  NSAu / NSAt  smolagents 1.26.0 ToolCallingAgent, OpenAIModel; TOOL_RESPONSE -> user (default) / custom_role_conversions
               {"tool-call": "assistant", "tool-response": "tool"}
Parsing, retries, argument checking, step limits and the end-of-run behaviour are the frameworks' own code.
AgentDojo only builds the injected environment, executes the functions and scores the episode.

Tools: each AgentDojo function becomes a framework-native tool around a real Python function with keyword-only
parameters named after the function's pydantic fields (so unexpected keyword arguments raise TypeError, as a
hand-written tool would). The function executes the call through AgentDojo's FunctionsRuntime and returns AgentDojo's
`tool_result_to_str` (the same string every delivery arm saw); an AgentDojo error ("ValidationError: ...") is raised as an
exception, which each framework turns into its own error observation. Every executed call is recorded as an AgentDojo
assistant tool call + tool result, so AgentDojo's utility / security checks see the trace.
Every HTTP request to the model server and its raw response are logged per episode (httpx event hooks).
"""
import asyncio
import json
import keyword
import threading
import time

import httpx
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.functions_runtime import FunctionCall
from agentdojo.types import get_text_content_as_str, text_content_block_from_string

ARMS = {"NLIu": ("llamaindex", "user"), "NLIt": ("llamaindex", "tool"),
        "NSAu": ("smolagents", "user"), "NSAt": ("smolagents", "tool")}
_EP = threading.local()   # per-episode diagnostics, read by the runner after each episode
_MISSING = object()


def episode_log():
    return getattr(_EP, "log", None)


def reset_episode_log():
    """called by the runner before every episode (a resumed episode never reaches NativeAgent.query)"""
    _EP.log = None


class Recorder:
    """executes AgentDojo functions for one episode and records them as AgentDojo messages"""

    def __init__(self, runtime, env):
        self.runtime, self.env, self.msgs, self.lock, self.n = runtime, env, [], threading.Lock(), 0

    def call(self, name, kwargs):
        with self.lock:
            self.n += 1
            fc = FunctionCall(function=name, args=dict(kwargs), id=f"native-{self.n}")
            result, err = self.runtime.run_function(self.env, name, kwargs)
            text = tool_result_to_str(result) if err is None else ""
            self.msgs.append({"role": "assistant", "content": None, "tool_calls": [fc]})
            self.msgs.append({"role": "tool", "content": [text_content_block_from_string(text)], "tool_call_id": fc.id,
                              "tool_call": fc, "error": err})
        if err is not None:
            raise RuntimeError(err)
        return text


def py_function(f, rec):
    """real Python function `name(*, field1, field2=<missing>, ...)` that executes the AgentDojo function"""
    fields = f.parameters.model_fields
    for p in fields:
        assert p.isidentifier() and not keyword.iskeyword(p), (f.name, p)
    params = ", ".join(p if fi.is_required() else f"{p}=_M" for p, fi in fields.items())
    collect = ", ".join(f"{p!r}: {p}" for p in fields)
    src = (f"def {f.name}({'*, ' + params if params else ''}):\n"
           f"    _kw = {{k: v for k, v in {{{collect}}}.items() if v is not _M}}\n"
           f"    return _rec.call({f.name!r}, _kw)\n")
    ns = {"_M": _MISSING, "_rec": rec}
    exec(src, ns)  # noqa: S102 - builds a signature from our own pydantic field names
    fn = ns[f.name]
    fn.__doc__ = f.description
    return fn


def sa_inputs(f):
    """smolagents `inputs` from the pydantic JSON schema (identical to the earlier adapter round's spec dump)"""
    sc = f.parameters.model_json_schema()
    req = set(sc.get("required", []))
    out = {}
    for name, p in sc.get("properties", {}).items():
        d = {}
        if "anyOf" in p:
            ts = [t.get("type", "any") for t in p["anyOf"] if t.get("type") != "null"]
            d["type"] = ts[0] if len(ts) == 1 else ts
            items = [t["items"] for t in p["anyOf"] if "items" in t]
            if items:
                d["items"] = items[0]
            nullable = any(t.get("type") == "null" for t in p["anyOf"])
        else:
            d["type"] = p.get("type", "any")
            if "items" in p:
                d["items"] = p["items"]
            if "enum" in p:
                d["enum"] = p["enum"]
            nullable = False
        d["description"] = p.get("description", "")
        if nullable or name not in req:
            d["nullable"] = True
        out[name] = d
    return out


def _hooks(log, is_async):
    def req(r):
        try:
            body = json.loads(r.content.decode()) if r.content else None
        except Exception:
            body = r.content.decode(errors="replace")
        log["wire"].append({"t": time.time(), "url": str(r.url), "request": body})

    if is_async:
        async def areq(r):
            req(r)

        async def aresp(r):
            await r.aread()
            log["wire"][-1]["status"] = r.status_code
            log["wire"][-1]["response"] = r.text
        return {"request": [areq], "response": [aresp]}

    def resp(r):
        r.read()
        log["wire"][-1]["status"] = r.status_code
        log["wire"][-1]["response"] = r.text
    return {"request": [req], "response": [resp]}


async def _run_llamaindex(role, sysmsg, query, functions, rec, base_url, model, max_tokens, log, extra_body=None):
    from llama_index.core.agent.react.formatter import ReActChatFormatter
    from llama_index.core.agent.workflow import ReActAgent
    from llama_index.core.base.llms.types import MessageRole
    from llama_index.core.tools import FunctionTool
    from llama_index.core.workflow.errors import WorkflowRuntimeError
    from llama_index.llms.openai_like import OpenAILike

    async with httpx.AsyncClient(event_hooks=_hooks(log, True), timeout=900) as ah:
        llm = OpenAILike(model=model, api_base=base_url, api_key="EMPTY", is_chat_model=True,
                         is_function_calling_model=False, context_window=32768, temperature=0.0,
                         max_tokens=max_tokens, timeout=900, max_retries=1, async_http_client=ah,
                         additional_kwargs={"extra_body": extra_body} if extra_body else {})
        tools = [FunctionTool.from_defaults(fn=py_function(f, rec), name=f.name, description=f.description,
                                            fn_schema=f.parameters) for f in functions]
        kw = {}
        if role == "tool":
            kw["formatter"] = ReActChatFormatter.from_defaults(observation_role=MessageRole.TOOL)
        agent = ReActAgent(tools=tools, llm=llm, system_prompt=sysmsg, verbose=False, **kw)
        try:
            out = await agent.run(user_msg=query)
            log["termination"] = "final_answer"
            return str(out.response.content or "")
        except WorkflowRuntimeError as e:
            if "Max iterations" in str(e):
                log["termination"] = "max_iterations"
                return ""
            raise


def _run_smolagents(role, sysmsg, query, functions, rec, base_url, model, max_tokens, log, extra_body=None):
    from smolagents import OpenAIModel, Tool, ToolCallingAgent
    from smolagents.monitoring import LogLevel

    def make_tool(f):
        fn = py_function(f, rec)

        def forward(self, **kwargs):
            return fn(**kwargs)
        return type(f"T_{f.name}", (Tool,), dict(name=f.name, description=f.description, inputs=sa_inputs(f),
                                                  output_type="string", forward=forward,
                                                  skip_forward_signature_validation=True))()

    conv = {"tool-call": "assistant", "tool-response": "tool"} if role == "tool" else None
    with httpx.Client(event_hooks=_hooks(log, False), timeout=900) as h:
        m = OpenAIModel(model_id=model, api_base=base_url, api_key="EMPTY",
                        client_kwargs={"http_client": h, "timeout": 900, "max_retries": 1},
                        custom_role_conversions=conv, temperature=0.0, max_tokens=max_tokens,
                        **({"extra_body": extra_body} if extra_body else {}))
        agent = ToolCallingAgent(tools=[make_tool(f) for f in functions], model=m, instructions=sysmsg,
                                 verbosity_level=LogLevel.OFF)
        answer = agent.run(query)
        steps = agent.memory.steps
        log["sa_steps"] = len([s for s in steps if type(s).__name__ == "ActionStep"])
        log["sa_step_errors"] = [str(s.error)[:300] for s in steps if getattr(s, "error", None) is not None]
        log["termination"] = ("final_answer" if any(getattr(s, "is_final_answer", False) for s in steps)
                              else "max_steps_fallback")
        return str(answer) if answer is not None else ""


class NativeAgent(BasePipelineElement):
    def __init__(self, arm, base_url, model, max_tokens=2048, extra_body=None):
        self.arm = arm
        self.extra_body = extra_body
        self.framework, self.role = ARMS[arm]
        self.base_url, self.model, self.max_tokens = base_url, model, max_tokens

    def query(self, query, runtime, env=None, messages=(), extra_args={}):
        assert messages and messages[0]["role"] == "system" and messages[-1]["role"] == "user", [m["role"] for m in messages]
        sysmsg = get_text_content_as_str(messages[0]["content"])
        user = get_text_content_as_str(messages[-1]["content"])
        rec = Recorder(runtime, env)
        log = {"arm": self.arm, "wire": [], "termination": None, "t0": time.time()}
        _EP.log = log
        functions = list(runtime.functions.values())
        args = (self.role, sysmsg, user, functions, rec, self.base_url, self.model, self.max_tokens, log)
        if self.framework == "llamaindex":
            answer = asyncio.run(_run_llamaindex(*args, extra_body=self.extra_body))
        else:
            answer = _run_smolagents(*args, extra_body=self.extra_body)
        log["n_requests"] = len(log["wire"])
        log["n_executed_calls"] = rec.n
        log["answer"] = answer
        final = {"role": "assistant", "content": [text_content_block_from_string(answer)], "tool_calls": None}
        return query, runtime, env, [*messages, *rec.msgs, final], extra_args
