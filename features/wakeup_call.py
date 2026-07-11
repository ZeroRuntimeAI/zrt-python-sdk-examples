"""
17 · Wake-up call — re-engage a caller who has gone silent.

Feature:  Pipeline(wake_up=<seconds>) arms a silence timer. If the caller stops
          speaking for that long, the runtime calls the agent's on_wake_up() hook, which
          nudges them. wake_up_max_attempts caps how many nudges before the call ends.
Pipeline: Cartesia (STT) · OpenAI (LLM) · SarvamAI (TTS) · Silero VAD · Namo turn detector
Env:      ZRT_AUTH_TOKEN, CARTESIA_API_KEY, OPENAI_API_KEY, SARVAM_API_KEY
Run:      uv run features/wakeup_call.py
"""
import zrt
from zrt import Agent, Pipeline, Room, function_tool
from zrt.plugins import CartesiaSTT, OpenAILLM, SarvamAITTS, SileroVAD, TurnDetector

from dotenv import load_dotenv
load_dotenv(override=True)

AGENT_ID = "wakeup-agent-py17"


class PatientAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="PatientAgent",
            agent_id=AGENT_ID,
            instructions=(
                "You are a patient assistant. Answer questions and help the caller. If they go "
                "quiet, you'll gently check in on them."
            ),
            pipeline=pipeline,
        )
        self._nudges = 0

    async def on_enter(self) -> None:
        await self.session.say("Hi! Take your time — I'm here whenever you're ready.")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

    async def on_wake_up(self) -> None:
        # Called by the runtime when the caller has been silent for `wake_up` seconds.
        self._nudges += 1
        await self.session.say("Are you still there? I'm happy to keep helping.")

    @function_tool
    async def get_help_topics(self) -> dict:
        """List the topics this assistant can help with.

        Args:
            None.
        """
        # Replace with a real catalog in production.
        return {"topics": ["account", "billing", "technical support"]}


pipeline = Pipeline(
    stt=CartesiaSTT(model="ink-2"),
    llm=OpenAILLM(model="gpt-5.4-nano-2026-03-17", streaming=True,
                  reasoning_effort="none", verbosity="low"),
    tts=SarvamAITTS(streaming=True),
    vad=SileroVAD(),
    turn_detector=TurnDetector(model="echo_large"),
    wake_up=10,              # nudge after 10s of caller silence
)


def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    zrt.invoke(AGENT_ID, room=Room(playground=True))


if __name__ == "__main__":
    zrt.serve(PatientAgent, on_ready=invoke_agent)
