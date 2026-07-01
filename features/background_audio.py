"""
06 · Background audio — ambient music mixed under the conversation.

Feature:  Play a looping ambient track (office-noise.mp3, bundled next to this file)
          underneath the agent's voice. The user can start/stop it via function tools.
          Needs serve(background_audio=True) so the worker session enables the mixer.
Pipeline: Google (STT) · Google Gemini (LLM) · Cartesia (TTS) · Silero VAD · Namo turn detector
Env:      ZRT_AUTH_TOKEN, GOOGLE_APPLICATION_CREDENTIALS, GOOGLE_API_KEY, CARTESIA_API_KEY
          that shares the host filesystem; use an http(s) URL for a hosted runtime.
Run:      uv run features/background_audio.py
"""
import os

import zrt
from zrt import Agent, Pipeline, Room, function_tool, BackgroundAudioHandlerConfig
from zrt.plugins import CartesiaTTS, GoogleLLM, SarvamAISTT, SileroVAD, TurnDetector

from dotenv import load_dotenv
load_dotenv(override=True)

AGENT_ID = "background-audio-agent-py06"

AUDIO_FILE = "http://cdn.zeroruntime.ai/zrt/bg-audio/bg-noise-1.wav"

pipeline = Pipeline(
    stt=SarvamAISTT(),
    llm=GoogleLLM(model="gemini-3-flash-preview", thinking_budget=0),
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
                "You are a calm, relaxing voice assistant with gentle background music. "
                "Keep replies short. Use the start_music and stop_music tools when the "
                "user asks to control the ambient track."
            ),
            pipeline=pipeline,
        )

    async def on_enter(self) -> None:
        # Greet the caller, then start the ambient track underneath the conversation.
        await self.session.say("Hi! I've got some calm music playing. Ask me to stop or start it anytime.")
        await self.session.play_background_audio(
            BackgroundAudioHandlerConfig(
                file=AUDIO_FILE,
                volume=0.2,
                looping=True,
                mode="mixing",
            )
        )

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

    @function_tool
    async def start_music(self) -> str:
        """Start the looping ambient background music."""
        # replace with real API in production
        await self.session.play_background_audio(
            BackgroundAudioHandlerConfig(
                file=AUDIO_FILE,
                volume=0.2,
                looping=True,
                mode="mixing",
            )
        )
        return "Background music started."

    @function_tool
    async def stop_music(self) -> str:
        """Stop the ambient background music."""
        # replace with real API in production
        await self.session.stop_background_audio()
        return "Background music stopped."


def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    zrt.invoke(AGENT_ID, room=Room(playground=True))


if __name__ == "__main__":
    # background_audio=True enables the worker session's mixer (same worker-session lever as vision).
    zrt.serve(Assistant, on_ready=invoke_agent, background_audio=True)
