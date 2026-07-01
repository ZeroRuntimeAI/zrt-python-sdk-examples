"""
05 · Fallback chains — automatic provider failover for STT, LLM, and TTS.

Feature:  Each slot is a Fallback wrapper over a list of providers. If the primary
          provider errors out or exceeds its latency budget, the pipeline transparently
          fails over to the next provider in the list — no agent code changes required.
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
from zrt.plugins import (
    CartesiaTTS,
    DeepgramSTT,
    DeepgramTTS,
    GoogleLLM,
    OpenAILLM,
    SarvamAISTT,
    SileroVAD,
    TurnDetector,
)

from dotenv import load_dotenv
load_dotenv(override=True)

AGENT_ID = "fallback-agent-py05"

pipeline = Pipeline(
    stt=FallbackSTT([SarvamAISTT(model="123"), DeepgramSTT(model="nova-2")]),
    llm=FallbackLLM([OpenAILLM(model="gpt-5.4-nano-2026-03-17", streaming=True, reasoning_effort="none",
                    verbosity="low"), GoogleLLM(model="gemini-2.5-flash", thinking_budget=0)]),
    tts=FallbackTTS([CartesiaTTS(model="sonic-3.5"),
                    DeepgramTTS(model="aura-2-thalia-en", stream=True)]),
    vad=SileroVAD(),
    turn_detector=TurnDetector(model="namo", language="en", threshold=0.8),
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
            pipeline=pipeline,
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
        return {"service": service, "status": "operational", "latency_ms": 42, "uptime": "99.99%"}



def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    zrt.invoke(AGENT_ID, room=Room(playground=True))


if __name__ == "__main__":
    zrt.serve(Assistant, on_ready=invoke_agent)
