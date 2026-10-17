"""Test doubles shared across test layers."""
from langchain_core.messages import AIMessage


class StubLLM:
    """Returns scripted responses in order. A response that is an Exception is raised.
    The last response repeats once the script runs out. Records every prompt."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages)
        r = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(r, Exception):
            raise r
        return AIMessage(content=r)


class RoutingStubLLM:
    """Answers router prompts and responder prompts from separate scripts."""

    def __init__(self, route_responses, reply_responses):
        self.router = StubLLM(*route_responses)
        self.responder = StubLLM(*reply_responses)

    def invoke(self, messages):
        is_router = "route messages" in messages[0].content
        return (self.router if is_router else self.responder).invoke(messages)


def tool_script_llm(*messages):
    """Chat model that replays scripted AIMessages (which may contain tool_calls) and accepts bind_tools,
    so a LangGraph ReAct agent can run end to end without a network."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    class _ToolScript(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    return _ToolScript(messages=iter(messages))
