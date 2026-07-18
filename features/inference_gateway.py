"""
Inference Gateway: run the entire pipeline on Zero Runtime's gateway.

Feature:  Import STT/LLM/TTS and the turn detector from zrt.inference (not zrt.plugins).
          Every provider then runs gateway-hosted over gRPC, and the gateway holds the
          provider credentials, so no per-provider API keys are needed locally; only the
          VAD stays local. Drop them into the pipeline exactly like plugin components.
Pipeline: SarvamAI STT · Google Gemini LLM · Cartesia TTS · echo-large turn detector — all gateway-hosted · Silero VAD (local)
Env:      ZRT_AUTH_TOKEN
Run:      uv run features/inference_gateway.py
"""
import zrt
from zrt import Agent, Pipeline, Room, function_tool
from zrt.inference import CartesiaTTS, GoogleLLM, SarvamAISTT, TurnDetector
from zrt.plugins import SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)

AGENT_ID = "inference-gateway-agent"


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="Assistant",
            agent_id=AGENT_ID,
            instructions=(
                "You are a friendly assistant. Keep replies short. The turn detection for this "
                "agent runs on the Zero Runtime inference gateway."
            ),
            pipeline=build_pipeline(),
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


def build_pipeline() -> Pipeline:
    """Return a fresh Pipeline; serve() builds a new agent + pipeline ."""
    return Pipeline(
        stt=SarvamAISTT(),
        llm=GoogleLLM(model="gemini-2.5-flash", thinking_budget=0),
        tts=CartesiaTTS(),
        vad=SileroVAD(),
        # echo-large runs on the inference gateway (gRPC), unlike the local model="namo".
        turn_detector=TurnDetector(model="echo-large"),
    )


def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    zrt.invoke(AGENT_ID, room=Room(playground=False))


if __name__ == "__main__":
    zrt.serve(Assistant, on_ready=invoke_agent)
