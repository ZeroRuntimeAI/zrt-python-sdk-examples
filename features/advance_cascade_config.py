import zrt
from zrt import Agent, InterruptConfig, Pipeline, Room, function_tool
from zrt.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD, TurnDetector

from dotenv import load_dotenv
load_dotenv(override=True)

AGENT_ID = "advance-cascade-config"


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
        return {"city": city, "temperature_c": 28, "condition": "Sunny", "humidity": 55}


def build_pipeline() -> Pipeline:
    """Return a fresh Pipeline; serve() builds a new agent + pipeline ."""
    return Pipeline(
        stt=DeepgramSTT(model="nova-2-conversationalai"),
        llm=GoogleLLM(model="gemini-3-flash-preview", thinking_budget=0),
        tts=CartesiaTTS(model="sonic-3.5"),
        vad=SileroVAD(),
        turn_detector=TurnDetector(model="echo_large"),
        interrupt_config=InterruptConfig(
            mode="HYBRID",
            interrupt_min_duration=0.5,
            interrupt_min_words=2,
            interrupt_min_confidence=0.0,
            false_interrupt_pause_duration=2.0,
            resume_on_false_interrupt=True,
            false_interrupt_pause_duration_ms=2000,
            interrupt_fade_duration=0.0,
            interrupt_fade_duration_ms=400,
        ),
    )


def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    zrt.invoke(AGENT_ID, room=Room(playground=True))


if __name__ == "__main__":
    zrt.serve(Assistant, on_ready=invoke_agent)
