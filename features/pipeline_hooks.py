"""
03 · Pipeline hooks — observing the dataflow without changing it.

Feature:  Register observation-only hooks on pipeline events (user/agent turns + llm output).
          These hooks are for logging/observability; they do NOT mutate the pipeline data.
Pipeline: Cartesia (STT) · Groq (LLM) · Sarvam (TTS) · Silero VAD · Namo turn detector
Env:      ZRT_AUTH_TOKEN, CARTESIA_API_KEY, GROQ_API_KEY, SARVAM_API_KEY
Run:      uv run features/pipeline_hooks.py
"""
from datetime import datetime
from zoneinfo import ZoneInfo


import zrt
from zrt import Agent, Pipeline, Room, function_tool
from zrt.plugins import CartesiaSTT, CartesiaTTS, GroqLLM, SarvamAITTS, SileroVAD, TurnDetector

from dotenv import load_dotenv
load_dotenv(override=True)

AGENT_ID = "pipeline-hooks-agent-py03"


pipeline = Pipeline(
    stt=CartesiaSTT(model="ink-2"),
    llm=GroqLLM(model="llama-3.3-70b-versatile"),
    tts=CartesiaTTS(model="sonic-3.5"),
    vad=SileroVAD(),
    turn_detector=TurnDetector(model="namo", language="en", threshold=0.8),
)
class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="Assistant",
            agent_id=AGENT_ID,
            instructions=(
                "You are a concise voice assistant. Keep replies short and natural. "
                "When asked for the time, call the get_time tool."
            ),
            pipeline=pipeline,
        )

    async def on_enter(self) -> None:
        # Greet the caller as soon as the session is live.
        await self.session.say("Hi! I'm listening. Ask me for the time in any timezone.")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

    @function_tool
    async def get_time(self, timezone: str) -> dict:
        """Get the current time for an IANA timezone.

        Args:
            timezone: IANA timezone name such as America/New_York or Asia/Kolkata.
        """
        # replace with real API in production
        try:
            now = datetime.now(ZoneInfo(timezone))
            return {"timezone": timezone, "time": now.strftime("%H:%M:%S"), "date": now.strftime("%Y-%m-%d")}
        except Exception:
            return {"timezone": timezone, "error": "Unknown timezone"}

@pipeline.on("user_turn_start")
async def on_user_turn_start(transcript: str) -> None:
    print(f"[hook] user_turn_start: {transcript!r}")


@pipeline.on("user_turn_end")
async def on_user_turn_end() -> None:
    print("[hook] user_turn_end")


@pipeline.on("agent_turn_start")
async def on_agent_turn_start() -> None:
    print("[hook] agent_turn_start")


@pipeline.on("agent_turn_end")
async def on_agent_turn_end() -> None:
    print("[hook] agent_turn_end")


@pipeline.on("llm")
async def on_llm(data: dict) -> None:
    print(
        f"[hook] llm: text={data.get('text')!r} interrupted={data.get('interrupted')}")


def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    zrt.invoke(AGENT_ID, room=Room(playground=True))


if __name__ == "__main__":
    zrt.serve(Assistant, on_ready=invoke_agent)
