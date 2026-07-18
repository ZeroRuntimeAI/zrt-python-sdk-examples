"""
Vision: describe the camera on a PubSub trigger.

Feature:  A client publishes {"message": "capture_frames"} on the CHAT PubSub topic
          (e.g. a "capture" button in your UI). On receipt the agent asks the runtime
          to attach its own latest decoded camera frame(s) to an LLM turn and describe
          them via session.reply(latest_frames=...). No image bytes round-trip through
          the SDK; the runtime grabs the frames from its own buffer.
          (To pull the bytes into the SDK first; to save/inspect them; use
          frames = self.capture_frames(2) then await self.session.reply(prompt, frames=frames).)
Pipeline: Deepgram nova-2 (STT) · Google Gemini (LLM, vision-capable) · Cartesia sonic-3.5 (TTS) · Silero VAD · Namo turn detector
Env:      ZRT_AUTH_TOKEN, DEEPGRAM_API_KEY, GOOGLE_API_KEY, CARTESIA_API_KEY
Requires: serve(vision=True) (so the runtime subscribes + buffers the camera), a
          vision-capable LLM (gemini-2.5-flash), and your CAMERA ON.
Run:      uv run features/vision.py
          (join the printed playground URL with your camera on, then publish
           {"message": "capture_frames"} on the CHAT topic)
"""
import asyncio

import zrt
from zrt import Agent, Pipeline, Room
from zrt.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD, TurnDetector

from dotenv import load_dotenv
load_dotenv(override=True)

AGENT_ID = "vision-agent"

# A UI publishes {"message": "capture_frames"} on this topic (e.g. a capture button).
CHAT_TOPIC = "CHAT"
CAPTURE_MESSAGE = "capture_frames"


class VisionAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="VisionAgent",
            agent_id=AGENT_ID,
            instructions=(
                "You are a helpful voice assistant that can describe what the camera "
                "sees when asked. Keep replies short."
            ),
            pipeline=build_pipeline(),
        )

    async def on_enter(self) -> None:
        await self.session.say(
            "Hi! Publish capture_frames on the CHAT topic and I'll describe what I see."
        )
        # A UI publishes {"message": "capture_frames"} on CHAT (e.g. a capture button).
        await self.session.subscribe_pubsub(CHAT_TOPIC, self._on_chat)

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

    def _on_chat(self, msg: dict) -> None:
        # subscribe_pubsub delivers a dict: {topic, message, sender_id, sender_name, ...}.
        if msg.get("message") == CAPTURE_MESSAGE:
            # The callback is sync; run the async capture+describe on the event loop.
            asyncio.create_task(self._capture_and_describe())

    async def _capture_and_describe(self) -> None:
        # Cancel any in-flight turn first (the greeting or a prior description still
        # generating) so the runtime accepts these frames instead of dropping them -
        # this makes capture_frames a "capture now" trigger.
        await self.session.cancel_generation()
        # latest_frames=N: the RUNTIME attaches its own newest decoded camera frames to the
        # turn; no capture round-trip, no image bytes sent from the SDK. If the buffer is
        # empty (camera off / vision disabled) the runtime warns and skips.
        await self.session.reply(
            "Please analyze this image and describe what you see in detail, in one line.",
            latest_frames=2,
        )


def build_pipeline() -> Pipeline:
    """Return a fresh Pipeline; serve() builds a new agent + pipeline ."""
    return Pipeline(
        stt=DeepgramSTT(model="nova-2"),
        llm=GoogleLLM(model="gemini-2.5-flash", thinking_budget=0),
        tts=CartesiaTTS(model="sonic-3.5"),
        vad=SileroVAD(),
        turn_detector=TurnDetector(model="echo-large"),
    )


def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    zrt.invoke(AGENT_ID, room=Room(playground=True))


if __name__ == "__main__":
    # vision=True tells the worker session to subscribe + decode the participant's camera,
    # so the runtime forwards vision frames (capture_frames / latest_frames have data).
    zrt.serve(VisionAgent, on_ready=invoke_agent, vision=True)
