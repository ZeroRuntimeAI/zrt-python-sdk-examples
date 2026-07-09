"""
02 · Function tools — chaining multiple tools in one turn.

Feature:  Three tools the LLM can call and chain (forecast -> clothing) in a single reply.
Pipeline: Google (STT) · OpenAI (LLM) · Deepgram (TTS) · Silero VAD · Namo turn detector
Env:      ZRT_AUTH_TOKEN, GOOGLE_APPLICATION_CREDENTIALS, OPENAI_API_KEY, DEEPGRAM_API_KEY
Run:      uv run features/function_tools.py
"""
import zrt
from zrt import Agent, Pipeline, Room, function_tool
from zrt.plugins import DeepgramTTS, GoogleSTT, OpenAILLM, SileroVAD, TurnDetector

from dotenv import load_dotenv
load_dotenv(override=True)

AGENT_ID = "function-tools-agent-py002"


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="Assistant",
            agent_id=AGENT_ID,
            instructions=(
                "You are a helpful weather assistant. Keep replies short and natural. "
                "When the user asks what to wear, first call get_forecast for the city, "
                "then pass the resulting condition and temperature into recommend_clothing, "
                "and summarize the advice. Use get_weather for simple current conditions."
            ),
            pipeline=pipeline,
        )

    async def on_enter(self) -> None:
        # Greet the caller as soon as the session is live.
        await self.session.say("Hi! Ask me about the weather, a forecast, or what to wear today.")

    async def on_exit(self) -> None:
        await self.session.say("Stay comfortable out there. Goodbye!")

    @function_tool
    async def get_weather(self, city: str) -> dict:
        """Get the current weather for a city.

        Args:
            city: Name of the city to look up.
        """
        # replace with real API in production
        return {"city": city, "temperature_c": 22, "condition": "Cloudy", "humidity": 60}

    @function_tool
    async def get_forecast(self, city: str, days: int) -> dict:
        """Get a multi-day weather forecast for a city.

        Args:
            city: Name of the city to look up.
            days: Number of days to forecast (1-7).
        """
        # replace with real API in production
        forecast = [
            {"day": i + 1, "condition": "Rainy" if i %
                2 else "Sunny", "temp_c": 18 + i}
            for i in range(max(1, min(days, 7)))
        ]
        return {"city": city, "days": days, "forecast": forecast}

    @function_tool
    async def recommend_clothing(self, condition: str, temp_c: int) -> dict:
        """Recommend what to wear given a weather condition and temperature.

        Args:
            condition: Weather condition such as Sunny, Rainy, or Cloudy.
            temp_c: Temperature in degrees Celsius.
        """
        # replace with real API in production
        layers = "light clothing" if temp_c >= 20 else "a warm jacket"
        extras = "an umbrella" if condition.lower() == "rainy" else "sunglasses"
        return {"condition": condition, "temp_c": temp_c, "wear": layers, "bring": extras}


pipeline = Pipeline(
    stt=GoogleSTT(model="chirp_3", location="us", stream=True),
    llm=OpenAILLM(model="gpt-5.4-nano-2026-03-17", streaming=True,
                  reasoning_effort="none", verbosity="low"),
    tts=DeepgramTTS(model="aura-2-thalia-en"),
    vad=SileroVAD(),
    turn_detector=TurnDetector(model="echo_large"),
)


def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    zrt.invoke(AGENT_ID, room=Room(playground=True))


if __name__ == "__main__":
    zrt.serve(Assistant, on_ready=invoke_agent)
