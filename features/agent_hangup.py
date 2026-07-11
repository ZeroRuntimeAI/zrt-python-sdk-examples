"""
14 · Agent hangup — let the agent end the call itself.

Feature:  A function tool calls session.hangup() so the agent can gracefully end the
          call once its job is done (e.g. after confirming an order or saying goodbye).
Pipeline: Google (STT) · OpenAI (LLM) · Deepgram (TTS) · Silero VAD · Namo turn detector
Env:      ZRT_AUTH_TOKEN, GOOGLE_APPLICATION_CREDENTIALS, OPENAI_API_KEY, DEEPGRAM_API_KEY
Run:      uv run features/agent_hangup.py
"""
import zrt
from zrt import Agent, Pipeline, Room, function_tool
from zrt.plugins import DeepgramTTS, GoogleSTT, OpenAILLM, SileroVAD, TurnDetector

from dotenv import load_dotenv
load_dotenv(override=True)

AGENT_ID = "hangup-agent-py14"


class Receptionist(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="Receptionist",
            agent_id=AGENT_ID,
            instructions=(
                "You are a brief reception line. Answer the caller's question, then ask if "
                "there's anything else. When the caller says they're done (or says goodbye), "
                "say a short farewell and call end_call to hang up. Do not hang up before saying goodbye."
            ),
            pipeline=pipeline,
        )

    async def on_enter(self) -> None:
        await self.session.say("Reception, how can I help?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

    @function_tool
    async def end_call(self, reason: str) -> dict:
        """End the call once the caller is finished.

        Args:
            reason: Why the call is ending (e.g. "caller said goodbye").
        """
        handle = await self.session.say("Thanks for calling. Goodbye!")
        await handle
        await self.session.hangup(reason=reason)
        return {"ended": True, "reason": reason}


pipeline = Pipeline(
    stt=GoogleSTT(model="latest_long", location="global", stream=True),
    llm=OpenAILLM(model="gpt-5.4-nano-2026-03-17", streaming=True,
                  reasoning_effort="none", verbosity="low"),
    tts=DeepgramTTS(model="aura-2-thalia-en", stream=True),
    vad=SileroVAD(),
    turn_detector=TurnDetector(model="echo_large"),
)


def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    zrt.invoke(AGENT_ID, room=Room(playground=True))


if __name__ == "__main__":
    zrt.serve(Receptionist, on_ready=invoke_agent)
