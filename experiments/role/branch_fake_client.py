"""CPU-only fake OpenAI client for plumbing tests of the branching runner (no model, no server). Not used for any result.

Scripted policy: execute the user task's ground-truth calls in order, then answer. If the most recent message that
contains the injection marker arrived in a `user`-role message, OR it is user-authored ("from me, Emma Johnson"),
the policy first executes the injection task's ground-truth calls (once). So, by construction:
    (tool, N) -> no attack, (tool, U) -> attack, (user, N) -> attack, (user, U) -> attack,
i.e. header effect +1 under N, 0 under U; interaction -1; voice effect under the tool header +1.
Formats: AgentDojo text calls (<function=..>), native tool_calls (gpt-oss), LlamaIndex ReAct text, smolagents
(tools + tool_choice="required" -> native tool_calls incl. final_answer).
"""
import json
import re
import uuid

from agentdojo.task_suite.load_suites import get_suite

MARK = "<INFORMATION>"


def _text(c):
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    return "".join(p.get("text", "") for p in c if isinstance(p, dict))


def _norm(t):
    t = t.replace("\\n", " ").replace("\\\"", "\"").replace("\\", " ")
    return re.sub(r"[\s'\"]+", " ", t).lower()


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def model_dump(self):
        def d(v):
            return v.model_dump() if isinstance(v, _Obj) else v
        return {k: d(v) for k, v in self.__dict__.items()}


class FakeClient:
    def __init__(self, family):
        self.family = family
        self.tasks = {}
        self.inj = {}
        for s in ("banking", "slack"):
            su = get_suite("v1.2.2", s)
            env = su.load_and_inject_default_environment({})
            for ut in su.user_tasks.values():
                self.tasks[ut.PROMPT.strip()] = (s, [(c.function, dict(c.args)) for c in ut.ground_truth(env)])
            for it in su.injection_tasks.values():
                key = _norm(" ".join(it.GOAL.split()[:9]))
                self.inj[(s, key)] = [(c.function, dict(c.args)) for c in it.ground_truth(env)]
        self.chat = _Obj(completions=_Obj(create=self.create))

    def create(self, model, messages, temperature=0.0, max_tokens=None, tools=None, tool_choice=None, stop=None,
               **kw):
        sysm = _text(messages[0]["content"]) if messages[0]["role"] == "system" else ""
        if tools and tool_choice == "required":
            fmt = "sa"
        elif tools:
            fmt = "native"
        elif "Action Input" in sysm and "## Output Format" in sysm:
            fmt = "li"
        else:
            fmt = "text"
        users = [_text(m["content"]) for m in messages if m["role"] == "user"]
        prompt = users[0].replace("New task:\n", "", 1).strip() if users else ""
        if prompt not in self.tasks:  # LlamaIndex retry duplicates the user message; tool-defs may prefix (Llama)
            prompt = next((p for p in self.tasks if p in "\n".join(users)), prompt)
        suite, plan = self.tasks.get(prompt, ("banking", []))
        assist = [_text(m.get("content")) for m in messages if m["role"] == "assistant"]
        done_plan = sum(len(re.findall(r"\[\[plan \d+\]\]", a)) for a in assist)
        done_inj = any("[[inj" in a for a in assist)
        mk = [m for m in messages if MARK in _text(m.get("content"))]
        comply = False
        inj_calls = []
        if mk and not done_inj:
            last = mk[-1]
            t = _text(last.get("content"))
            comply = last["role"] == "user" or "from me, emma johnson" in _norm(t)
            nt = _norm(t)
            for (s, key), calls in self.inj.items():
                if s == suite and key in nt:
                    inj_calls = calls
        if comply and inj_calls:
            calls, tag = inj_calls, "[[inj]]"
        elif done_plan < len(plan):
            calls, tag = [plan[done_plan]], f"[[plan {done_plan}]]"
        else:
            calls, tag = [], "[[final]]"
        return self._render(fmt, calls, tag)

    def _render(self, fmt, calls, tag):
        tcs = None
        content = None
        if fmt == "text":
            content = tag + " " + "".join(f"<function={f}>{json.dumps(a)}</function>" for f, a in calls[:1]) \
                if calls else tag + " Done."
        elif fmt == "li":
            if calls:
                f, a = calls[0]
                content = f"Thought: {tag} I need to use a tool.\nAction: {f}\nAction Input: {json.dumps(a)}"
            else:
                content = f"Thought: {tag} I can answer without using any more tools.\nAnswer: Done."
        else:
            if fmt == "sa" and not calls:
                calls = [("final_answer", {"answer": "Done."})]
            tcs = [_Obj(id="call_" + uuid.uuid4().hex[:12], type="function",
                        function=_Obj(name=f, arguments=json.dumps(a))) for f, a in calls]
            content = tag if fmt in ("native", "sa") and calls else (tag + " Done.")
            if fmt == "native" and not calls:
                tcs = None
        msg = _Obj(content=content, tool_calls=tcs, reasoning_content=None, role="assistant")
        return _Obj(choices=[_Obj(message=msg, finish_reason="tool_calls" if tcs else "stop")],
                    usage=_Obj(prompt_tokens=1, completion_tokens=1, total_tokens=2))
