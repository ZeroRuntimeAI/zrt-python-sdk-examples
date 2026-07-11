import zrt
from zrt import Agent, Pipeline, RealtimeConfig, Room, function_tool
from zrt.plugins import DeepgramSTT, GeminiLiveConfig, GeminiRealtime, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)

AGENT_ID = "hybrid-stt-agent-py11"


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="Assistant",
            agent_id=AGENT_ID,
            instructions=(
                "You are a lively, low-latency voice assistant running on a hybrid pipeline. "
                "Keep replies short and natural. When asked about the weather, call get_weather."
            ),
            pipeline=pipeline,
        )

    async def on_enter(self) -> None:
        await self.session.say("Hi! I'm running in hybrid STT mode. Ask me about the weather somewhere!")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

    @function_tool
    async def get_weather(self, city: str) -> dict:
        """Get the current weather for a city.

        Args:
            city: The city to look up, e.g. "Paris".
        """
        return {"city": city, "conditions": "sunny", "temp_c": 24}


# External ear + native voice: Deepgram transcribes the caller; the realtime model
# answers in its own voice.
pipeline = Pipeline(
    llm=GeminiRealtime(config=GeminiLiveConfig()),
    stt=DeepgramSTT(),
    vad=SileroVAD(),
    realtime_config=RealtimeConfig(mode="hybrid_stt"),
)


def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    zrt.invoke(AGENT_ID, room=Room(playground=True))


if __name__ == "__main__":
    zrt.serve(Assistant, on_ready=invoke_agent)
