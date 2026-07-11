import zrt
from zrt import Agent, Pipeline, Room, function_tool
from zrt.plugins import CartesiaTTS, GeminiLiveConfig, GeminiRealtime, SileroVAD, TurnDetector

from dotenv import load_dotenv
load_dotenv(override=True)

AGENT_ID = "hybrid-tts-agent-py11"


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="Assistant",
            agent_id=AGENT_ID,
            instructions=(
                "You are a lively, low-latency voice assistant running on a hybrid pipeline. "
                "Keep replies short and natural. When asked about the weather, call get_weather."
            ),
            pipeline=build_pipeline(),
        )

    async def on_enter(self) -> None:
        await self.session.say("Hi! I'm running in hybrid TTS mode. Ask me about the weather somewhere!")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

    @function_tool
    async def get_weather(self, city: str) -> dict:
        """Get the current weather for a city.

        Args:
            city: The city to look up, e.g. "Paris".
        """
        # Replace with a real API in production.
        return {"city": city, "conditions": "sunny", "temp_c": 24}


# Native ear + external voice: the realtime model understands audio itself, but a
# dedicated TTS speaks its text.
def build_pipeline() -> Pipeline:
    """Return a fresh Pipeline; serve() builds a new agent + pipeline ."""
    return Pipeline(
        llm=GeminiRealtime(config=GeminiLiveConfig()),
        tts=CartesiaTTS(model="sonic-3.5"),
        vad=SileroVAD(),
        turn_detector=TurnDetector(model="echo_large"),
    )


def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    zrt.invoke(AGENT_ID, room=Room(playground=True))


if __name__ == "__main__":
    zrt.serve(Assistant, on_ready=invoke_agent)
