import logging
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import zrt
from zrt import Agent, Pipeline, Room, function_tool, run_stt, run_tts
from zrt.plugins import CartesiaSTT, CartesiaTTS, GroqLLM, SileroVAD, TurnDetector

from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

AGENT_ID = "pipeline-hooks-agent"


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="Assistant",
            agent_id=AGENT_ID,
            instructions=(
                "You are a concise voice assistant. Keep replies short and natural. "
                "When asked for the time, call the get_time tool."
            ),
            pipeline=build_pipeline(),
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
        try:
            now = datetime.now(ZoneInfo(timezone))
            return {"timezone": timezone, "time": now.strftime("%H:%M:%S"), "date": now.strftime("%Y-%m-%d")}
        except Exception:
            return {"timezone": timezone, "error": "Unknown timezone"}


def build_pipeline() -> Pipeline:
    """Return a fresh Pipeline (with its hooks); serve() builds a new agent + pipeline ."""
    pipeline = Pipeline(
        stt=CartesiaSTT(model="ink-2"),
        llm=GroqLLM(model="llama-3.3-70b-versatile"),
        tts=CartesiaTTS(model="sonic-3.5"),
        vad=SileroVAD(),
        turn_detector=TurnDetector(model="echo_large"),
    )

    @pipeline.on("stt")
    async def stt_hook(audio_stream):
        """Preprocess audio before STT, then normalize the transcript after it."""
        async def audio_phase():
            async for audio in audio_stream:
                if len(audio) < 300:
                    continue
                yield audio

        async for event in run_stt(audio_phase()):
            if event.data and event.data.text:
                text = event.data.text.lower()
                text = re.sub(r"\b(uh|um|like)\b", "", text)

                replacements = {
                    "working hours": "office hours",
                    "timing": "office hours",
                }
                for src, dst in replacements.items():
                    text = re.sub(rf"\b{src}\b", dst, text)

                event.data.text = " ".join(text.split())
                logging.info(f"[STT] {event.data.text}")

            yield event

    @pipeline.on("llm")
    async def llm_text_filter(text_stream):
        """Streaming LLM hook: rewrite text chunks in real time before they reach TTS.

        Swapping a few common words makes the change clearly AUDIBLE in the agent's
        speech (and logged), proving in-flight per-token modification. Each chunk is
        processed and replaced as it streams from the LLM.
        """
        word_swaps = {
            "office": "business",
            "available": "around",
            "anytime": "any time",
            "hello": "hey",
            "hi": "hey",
            "hey": "hello apple",
        }
        async for chunk in text_stream:
            original = chunk
            for src, dst in word_swaps.items():
                chunk = re.sub(rf"(?i)\b{src}\b", dst, chunk)
            if chunk != original:
                logging.info(f"[LLM STREAM] {original!r} -> {chunk!r}")
            yield chunk

    @pipeline.on("tts")
    async def tts_hook(text_stream):
        """Final text shaping just before speech synthesis."""
        async def preprocess_text():
            async for text in text_stream:
                yield text.replace("Hello", "Heyy")

        async for audio in run_tts(preprocess_text()):
            yield audio

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

    return pipeline


def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    zrt.invoke(AGENT_ID, room=Room(playground=True))


if __name__ == "__main__":
    zrt.serve(Assistant, on_ready=invoke_agent)
