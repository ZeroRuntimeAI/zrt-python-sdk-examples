"""
01 · Basic cascade: the smallest complete voice agent.

Feature:  STT -> LLM -> TTS cascade with VAD + turn detection, one function tool.
Pipeline: Deepgram (STT) · Google Gemini (LLM) · Cartesia (TTS) · Silero VAD · Namo turn detector
Env:      ZRT_AUTH_TOKEN, DEEPGRAM_API_KEY, GOOGLE_API_KEY, CARTESIA_API_KEY
Run:      uv run features/basic_cascade.py
"""
import zrt
from zrt import Agent, Pipeline, Room, function_tool
from zrt.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD, TurnDetector

from dotenv import load_dotenv
load_dotenv(override=True)

AGENT_ID = "basic-cascade-agent-py01"


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="Assistant",
            agent_id=AGENT_ID,
            instructions=(
                "You are a friendly voice assistant. Keep replies short and natural. "
                "When asked about the weather, call the get_weather tool."
            ),
            pipeline=build_pipeline(),
        )

    async def on_enter(self) -> None:
        await self.session.say("Hi! I'm your assistant. Ask me about the weather in any city.")

    async def on_exit(self) -> None:
        await self.session.say("Thanks for calling. Goodbye!")

    @function_tool
    async def get_weather(self, city: str) -> dict:
        """Get the current weather for a city.

        Args:
            city: Name of the city to look up.
        """
        # Replace with a real weather API call in production.
        return {"city": city, "temperature_c": 28, "condition": "Sunny", "humidity": 55}


def build_pipeline() -> Pipeline:
    """Return a fresh Pipeline; serve() builds a new agent + pipeline ."""
    return Pipeline(
        stt=DeepgramSTT(model="nova-2-conversationalai"),
        llm=GoogleLLM(model="gemini-3-flash-preview", thinking_budget=0),
        tts=CartesiaTTS(model="sonic-3.5"),
        vad=SileroVAD(),
        turn_detector=TurnDetector(model="echo_large"),
    )


def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    zrt.invoke(AGENT_ID, room=Room(playground=True))


if __name__ == "__main__":
    zrt.serve(Assistant, on_ready=invoke_agent)
