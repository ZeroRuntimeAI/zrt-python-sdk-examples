"""
18 · Inference Gateway turn detection — run turn-taking on Zero Runtime's gateway.

Feature:  Instead of a locally-configured detector, TurnDetector(model="echo-large")
          from zrt.inference runs turn detection on the Zero Runtime Inference
          Gateway. Drop it into the pipeline's turn_detector slot exactly like a plugin
          detector — the gateway handles end-of-utterance classification.
Pipeline: Deepgram (STT) · Google Gemini (LLM) · Cartesia (TTS) · Silero VAD · Inference-Gateway turn detector
Env:      ZRT_AUTH_TOKEN, DEEPGRAM_API_KEY, GOOGLE_API_KEY, CARTESIA_API_KEY
Run:      uv run features/inference_gateway.py
"""
import zrt
from zrt import Agent, Pipeline, Room, function_tool
from zrt.inference import DeepgramSTT, CartesiaTTS, GoogleLLM, TurnDetector, SarvamAISTT
from zrt.plugins import SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)

AGENT_ID = "inference-gateway-agent-py18"


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="Assistant",
            agent_id=AGENT_ID,
            instructions=(
                "You are a friendly assistant. Keep replies short. The turn detection for this "
                "agent runs on the Zero Runtime inference gateway."
            ),
            pipeline=pipeline,
        )

    async def on_enter(self) -> None:
        await self.session.say("Hi! Pipeline Providers are handled by the inference gateway. How can I help?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

    @function_tool
    async def get_time_of_day(self) -> dict:
        """Return a friendly time-of-day greeting.

        Args:
            None.
        """
        # Replace with a real clock/timezone lookup in production.
        return {"part_of_day": "afternoon", "greeting": "Good afternoon"}


pipeline = Pipeline(
    stt=SarvamAISTT(),
    llm=GoogleLLM(model="gemini-2.5-flash", thinking_budget=0),
    tts=CartesiaTTS(),
    vad=SileroVAD(),
    # echo-large runs on the inference gateway (gRPC), unlike the local model="namo".
    turn_detector=TurnDetector(model="echo_large"),
)


def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    zrt.invoke(AGENT_ID, room=Room(playground=False))


if __name__ == "__main__":
    zrt.serve(Assistant, on_ready=invoke_agent)
