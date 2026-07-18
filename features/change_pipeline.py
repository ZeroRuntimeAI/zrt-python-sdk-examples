"""
Change pipeline: switch the whole pipeline (cascade <-> realtime) at runtime.

Feature:  Swap the entire pipeline mid-session, from a cascade (STT/LLM/TTS) to a realtime
          speech-to-speech model and back. SWITCH_AFTER_SECONDS controls the timing.
Pipeline: Cascade: Deepgram (STT) · Google Gemini (LLM) · Cartesia (TTS)  <->  Gemini Realtime (speech-to-speech)
Env:      ZRT_AUTH_TOKEN, DEEPGRAM_API_KEY, GOOGLE_API_KEY, CARTESIA_API_KEY
Run:      uv run features/change_pipeline.py
"""
import asyncio
import logging
import os

import zrt
from zrt import Agent, Pipeline, Room
from zrt.plugins import CartesiaTTS, DeepgramSTT, GeminiLiveConfig, GeminiRealtime, GoogleLLM, SileroVAD, TurnDetector, DeepgramTTS

from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("switch-cascade-realtime")

AGENT_ID = "change-pipeline-agent"
SWITCH_AFTER = float(os.environ.get("SWITCH_AFTER_SECONDS", "10"))


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="Assistant",
            agent_id=AGENT_ID,
            instructions="You are a friendly voice assistant. Chat naturally with the caller.",
            pipeline=build_pipeline(),
        )
        self._switch_task: asyncio.Task | None = None

    async def on_enter(self) -> None:
        logger.info("[assistant] session started in cascade mode")
        await self.session.say(
            f"Hi! I'm on a cascade pipeline. In {int(SWITCH_AFTER)} seconds I'll try to "
            "switch myself to a realtime model; watch the logs."
        )
        self._switch_task = asyncio.create_task(self._try_switch())

    async def on_exit(self) -> None:
        logger.info("[assistant] session ended")
        if self._switch_task is not None:
            self._switch_task.cancel()

    async def _try_switch(self) -> None:
        await asyncio.sleep(SWITCH_AFTER)
        logger.info(
            "[switch] attempting cascade -> hybrid_tts via change_pipeline")
        try:
            await self.session.pipeline.change_pipeline(
                llm=GeminiRealtime(config=GeminiLiveConfig(
                    model="gemini-3.1-flash-live-preview", voice="Puck")),
                tts=DeepgramTTS(),
            )
        except ValueError as e:
            # Shape is fixed at session creation: cascade -> realtime can't be done live.
            logger.info("[switch] rejected as expected: %s", e)
            await self.session.say(
                "As expected, I can't switch to realtime mid-call; the pipeline shape is "
                "fixed when the session starts."
            )
            return
        logger.warning("[switch] switch was accepted (unexpected)")
        await self.session.say("I switched to a realtime model!")


def build_pipeline() -> Pipeline:
    """Return a fresh Pipeline; serve() builds a new agent + pipeline ."""
    return Pipeline(
        stt=DeepgramSTT(language="en"),
        llm=GoogleLLM(model="gemini-2.5-flash"),
        tts=CartesiaTTS(model="sonic-3.5"),
        vad=SileroVAD(threshold=0.4),
        turn_detector=TurnDetector(model="echo-large"),
    )


def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    logger.info(
        "[startup] agent registered; inviting caller into the playground")
    zrt.invoke(AGENT_ID, room=Room(playground=True))


if __name__ == "__main__":
    logger.info("[startup] serving %s", AGENT_ID)
    zrt.serve(Assistant, on_ready=invoke_agent)
