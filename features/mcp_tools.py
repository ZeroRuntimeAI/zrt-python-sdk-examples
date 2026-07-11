import zrt
from zrt import Agent, Pipeline, Room, MCPServerStdio, MCPServerHTTP
from zrt.plugins import CartesiaSTT, DeepgramTTS, OpenAILLM, SileroVAD, TurnDetector

from dotenv import load_dotenv
load_dotenv(override=True)

AGENT_ID = "mcp-tools-agent-py09"


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="Assistant",
            agent_id=AGENT_ID,
            instructions=(
                "You are a voice assistant whose tools are auto-discovered from the "
                "connected MCP servers. Use the available tools to answer the user."
            ),
            pipeline=build_pipeline(),
            mcp_servers=[
                # Placeholders; replace with real MCP servers in production.
                MCPServerStdio(command="uvx", args=["mcp-server-time"]),
                MCPServerHTTP(
                    url="https://example.com/mcp-server-weather",
                ),
            ],
        )

    async def on_enter(self) -> None:
        await self.session.say("Hi! I'm connected to a few tools. How can I help?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")


def build_pipeline() -> Pipeline:
    """Return a fresh Pipeline; serve() builds a new agent + pipeline ."""
    return Pipeline(
        stt=CartesiaSTT(model="ink-2"),
        llm=OpenAILLM(model="gpt-5.4-nano-2026-03-17", streaming=True,
                      reasoning_effort="none", verbosity="low"),
        tts=DeepgramTTS(model="aura-2-thalia-en", stream=True),
        vad=SileroVAD(),
        turn_detector=TurnDetector(model="echo-large"),
    )


def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    zrt.invoke(AGENT_ID, room=Room(playground=True))


if __name__ == "__main__":
    zrt.serve(Assistant, on_ready=invoke_agent)
