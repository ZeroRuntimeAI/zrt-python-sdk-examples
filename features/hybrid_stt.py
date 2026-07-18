"""
Hybrid STT: a cascade STT provider feeding a realtime LLM.

Feature:  Run a dedicated STT provider (Deepgram) in front of a realtime model, so you keep
          your own transcript while the model produces the audio response (mode="hybrid_stt").
Pipeline: Deepgram (STT) · Gemini Realtime (LLM, speech-out) · Silero VAD
Env:      ZRT_AUTH_TOKEN, DEEPGRAM_API_KEY, GOOGLE_API_KEY
Run:      uv run features/hybrid_stt.py
"""
import zrt
from zrt import Agent, Pipeline, Room, function_tool
from zrt.plugins import DeepgramSTT, GeminiLiveConfig, GeminiRealtime, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)

AGENT_ID = "hybrid-stt-agent"


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
def build_pipeline() -> Pipeline:
    """Return a fresh Pipeline; serve() builds a new agent + pipeline ."""
    return Pipeline(
        llm=GeminiRealtime(config=GeminiLiveConfig()),
        stt=DeepgramSTT(),
        vad=SileroVAD(),
    )


def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    zrt.invoke(AGENT_ID, room=Room(playground=True))


if __name__ == "__main__":
    zrt.serve(Assistant, on_ready=invoke_agent)
