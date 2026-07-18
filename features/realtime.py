"""
Realtime: full speech-to-speech with a single realtime model.

Feature:  Use a native realtime model in the LLM slot for low-latency, end-to-end
          speech-to-speech. No separate STT/TTS/VAD/turn-detector; the model handles
          audio in and audio out directly.
Pipeline: Gemini Realtime (speech-to-speech)
Env:      ZRT_AUTH_TOKEN, GOOGLE_API_KEY
Run:      uv run features/realtime.py
"""
import zrt
from zrt import Agent, Pipeline, Room, function_tool
from zrt.plugins import GeminiLiveConfig, GeminiRealtime

from dotenv import load_dotenv
load_dotenv(override=True)

AGENT_ID = "realtime-agent"


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="Assistant",
            agent_id=AGENT_ID,
            instructions=(
                "You are a lively, low-latency voice assistant. Keep replies short and natural. "
                "When asked for a fun fact, call the get_fun_fact tool and share it."
            ),
            pipeline=build_pipeline(),
        )

    async def on_enter(self) -> None:
        # Greet the caller as soon as the session is live.
        await self.session.say("Hey there! I'm a realtime voice assistant. Ask me for a fun fact!")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

    @function_tool
    async def get_fun_fact(self, topic: str) -> dict:
        """Get a fun fact about a topic.

        Args:
            topic: The subject to get a fun fact about, e.g. space or oceans.
        """
        # replace with real API in production
        return {"topic": topic, "fact": f"Here's something surprising about {topic}: it's more fascinating than most people realize!"}


def build_pipeline() -> Pipeline:
    """Return a fresh Pipeline; serve() builds a new agent + pipeline ."""
    return Pipeline(
        llm=GeminiRealtime(config=GeminiLiveConfig()),
    )


def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    zrt.invoke(AGENT_ID, room=Room(playground=True))


if __name__ == "__main__":
    zrt.serve(Assistant, on_ready=invoke_agent)
