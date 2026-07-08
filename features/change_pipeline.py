import asyncio
import logging
import os

import zrt
from zrt import Agent, Pipeline, Room
from zrt.plugins import CartesiaTTS, DeepgramSTT, GeminiLiveConfig, GeminiRealtime, GoogleLLM, SileroVAD, TurnDetector

from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("switch-cascade-realtime")

AGENT_ID = "switch-cascade-realtime-py"
SWITCH_AFTER = float(os.environ.get("SWITCH_AFTER_SECONDS", "10"))


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="Assistant",
            agent_id=AGENT_ID,
            instructions="You are a friendly voice assistant. Chat naturally with the caller.",
            pipeline=pipeline,
        )
        self._switch_task: asyncio.Task | None = None

    async def on_enter(self) -> None:
        logger.info("[assistant] session started in cascade mode")
        await self.session.say(
            f"Hi! I'm on a cascade pipeline. In {int(SWITCH_AFTER)} seconds I'll try to "
            "switch myself to a realtime model — watch the logs."
        )
        self._switch_task = asyncio.create_task(self._try_switch())

    async def on_exit(self) -> None:
        logger.info("[assistant] session ended")
        if self._switch_task is not None:
            self._switch_task.cancel()

    async def _try_switch(self) -> None:
        await asyncio.sleep(SWITCH_AFTER)
        logger.info("[switch] attempting cascade -> realtime via change_pipeline")
        try:
            await self.session.pipeline.change_pipeline(
                llm=GeminiRealtime(config=GeminiLiveConfig()),
            )
        except ValueError as e:
            # Shape is fixed at session creation: cascade -> realtime can't be done live.
            logger.info("[switch] rejected as expected: %s", e)
            await self.session.say(
                "As expected, I can't switch to realtime mid-call — the pipeline shape is "
                "fixed when the session starts."
            )
            return
        logger.warning("[switch] switch was accepted (unexpected)")
        await self.session.say("I switched to a realtime model!")


pipeline = Pipeline(
    stt=DeepgramSTT(language="en"),
    llm=GoogleLLM(model="gemini-2.5-flash"),
    tts=CartesiaTTS(model="sonic-3.5"),
    vad=SileroVAD(threshold=0.4),
    turn_detector=TurnDetector(language="en", threshold=0.8),
)


def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    logger.info("[startup] agent registered — inviting caller into the playground")
    zrt.invoke(AGENT_ID, room=Room(playground=True))


if __name__ == "__main__":
    logger.info("[startup] serving %s", AGENT_ID)
    zrt.serve(Assistant, on_ready=invoke_agent)
