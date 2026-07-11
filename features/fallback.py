"""
05 · Fallback chains: automatic provider failover for STT, LLM, and TTS.

Feature:  Each slot is a Fallback wrapper over a list of providers. If the primary
          provider errors out or exceeds its latency budget, the pipeline transparently
          fails over to the next provider in the list; no agent code changes required.
Pipeline: Fallback STT (Deepgram -> Google) · Fallback LLM (OpenAI -> Google) ·
          Fallback TTS (Cartesia -> Deepgram) · Silero VAD · Namo turn detector
Env:      ZRT_AUTH_TOKEN,
          DEEPGRAM_API_KEY (STT primary + TTS fallback),
          GOOGLE_APPLICATION_CREDENTIALS (STT fallback), GOOGLE_API_KEY (LLM fallback),
          OPENAI_API_KEY (LLM primary),
          CARTESIA_API_KEY (TTS primary)
Run:      uv run features/fallback.py
"""
import zrt
from zrt import Agent, Pipeline, Room, function_tool, FallbackSTT, FallbackLLM, FallbackTTS
from zrt.plugins import CartesiaTTS, DeepgramSTT, DeepgramTTS, GoogleLLM, OpenAILLM, SarvamAISTT, SileroVAD, TurnDetector, OpenAISTT, OpenAITTS
import asyncio

from dotenv import load_dotenv
load_dotenv(override=True)

AGENT_ID = "fallback-agent-py05"

# Fallback configuration:
# 1. Define a list of providers (in priority order).
# 2. temporary_disable_sec: Time to wait before retrying a failed primary provider.
# 3. permanent_disable_after_attempts: Disable a provider permanently after N failed recovery attempts.
# 4. latency_threshold_ms: Per-component latency budget in ms (STT stt_latency / LLM llm_ttft / TTS ttfb).
#    Off by default; pass a value to enable latency-based fallback.
# 5. consecutive_latency_hits: Switch only after this many consecutive turns above the threshold (default 3).
#    Recovery/cooldown use the same temporary_disable_sec / permanent_disable_after_attempts as the error path.

stt_provider = FallbackSTT(
    [OpenAISTT(), DeepgramSTT(model="nova-2")],
    temporary_disable_sec=30.0,
    permanent_disable_after_attempts=3,
    latency_threshold_ms=350,
    consecutive_latency_hits=3,
)

llm_provider = FallbackLLM(
    [OpenAILLM(model="gpt-4o-mini"),
     GoogleLLM(model="gemini-2.5-flash", thinking_budget=0)],
    temporary_disable_sec=30.0,
    permanent_disable_after_attempts=3,
    latency_threshold_ms=800,
    consecutive_latency_hits=3,
)

tts_provider = FallbackTTS(
    [CartesiaTTS(model="sonic-3.5"),
     DeepgramTTS(model="aura-2-thalia-en", stream=True)],
    temporary_disable_sec=30.0,
    permanent_disable_after_attempts=3,
    latency_threshold_ms=300,
    consecutive_latency_hits=3,
)


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="Assistant",
            agent_id=AGENT_ID,
            instructions=(
                "You are a resilient status-desk voice assistant. Keep replies short. "
                "When asked about a service, call the get_status tool and report its state."
            ),
            pipeline=build_pipeline(),
        )

    async def on_enter(self) -> None:
        # Greet the caller as soon as the session is live.
        await self.session.say("Hi! Ask me about the status of any service.")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

    @function_tool
    async def get_status(self, service: str) -> dict:
        """Get the operational status of a named service.

        Args:
            service: Name of the service to check, e.g. api, database, or billing.
        """
        # replace with real API in production
        return {"service": service, "status": "operational", "latency_ms": 42, "uptime": "94.99%"}


def build_pipeline() -> Pipeline:
    """Return a fresh Pipeline; serve() builds a new agent + pipeline ."""
    return Pipeline(
        stt=stt_provider,
        llm=llm_provider,
        tts=tts_provider,
        vad=SileroVAD(),
        turn_detector=TurnDetector(model="echo_large"),
    )


def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    zrt.invoke(AGENT_ID, room=Room(playground=True))


if __name__ == "__main__":
    zrt.serve(Assistant, on_ready=invoke_agent)
