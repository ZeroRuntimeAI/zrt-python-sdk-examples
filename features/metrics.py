"""
12 · Metrics: observe STT/LLM/TTS/EOU metric hooks.

Feature:  Subscribe to per-stage metric hooks and log them. zrt exposes metric
          HOOKS only; the collector/export pipeline is runtime-side.
Pipeline: Cartesia ink-2 (STT) · Groq llama-3.1-8b (LLM) · Cartesia sonic-3.5 (TTS) · Silero VAD · Namo turn detector
Env:      ZRT_AUTH_TOKEN, CARTESIA_API_KEY, GROQ_API_KEY
Run:      uv run features/metrics.py
"""
import zrt
from zrt import Agent, Pipeline, Room, function_tool
from zrt.plugins import CartesiaSTT, CartesiaTTS, GroqLLM, SileroVAD, TurnDetector

from dotenv import load_dotenv
load_dotenv(override=True)

AGENT_ID = "metrics-agent-py12"


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="Assistant",
            agent_id=AGENT_ID,
            instructions=(
                "You are a voice assistant. Keep replies short. Call ping when asked "
                "to check that you're responsive."
            ),
            pipeline=build_pipeline(),
        )

    async def on_enter(self) -> None:
        await self.session.say("Hi! I'm running with metrics hooks enabled.")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

    @function_tool
    async def ping(self) -> dict:
        """Check that the agent is responsive.

        Args:
            None.
        """
        return {"status": "ok", "pong": True}


def build_pipeline() -> Pipeline:
    """Return a fresh Pipeline (with its metrics hooks); serve() builds a new agent + pipeline ."""
    pipeline = Pipeline(
        stt=CartesiaSTT(model="ink-2"),
        llm=GroqLLM(model="llama-3.3-70b-versatile"),
        tts=CartesiaTTS(model="sonic-3.5"),
        vad=SileroVAD(),
        turn_detector=TurnDetector(model="echo_large"),
    )

    @pipeline.metrics.on("stt")
    async def on_stt_metrics(metrics) -> None:
        print(f"[metrics] stt: {metrics}")

    @pipeline.metrics.on("llm")
    async def on_llm_metrics(metrics) -> None:
        print(f"[metrics] llm: {metrics}")

    @pipeline.metrics.on("tts")
    async def on_tts_metrics(metrics) -> None:
        print(f"[metrics] tts: {metrics}")

    @pipeline.metrics.on("eou")
    async def on_eou_metrics(metrics) -> None:
        print(f"[metrics] eou: {metrics}")

    return pipeline


def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    zrt.invoke(AGENT_ID, room=Room(playground=True))


if __name__ == "__main__":
    zrt.serve(Assistant, on_ready=invoke_agent)
