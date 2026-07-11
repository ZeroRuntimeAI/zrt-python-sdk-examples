"""
19 · Chat context: manage the conversation window and read the history back.

Feature:  Two chat-context tools the SDK exposes:
            - ContextWindow on the Pipeline: the runtime keeps the prompt within a token
              budget (max_tokens) while always keeping the most recent turns
              (keep_recent_turns), so long conversations don't blow up cost/latency.
            - session.get_context_history(last_n=...): read the live conversation back
              (await it for a fresh copy from the runtime). Here a `recap` tool uses it so
              the agent can summarize what's been discussed.
Pipeline: Deepgram nova-2 (STT) · OpenAI gpt-5.4-nano (LLM) · Cartesia sonic-3.5 (TTS) · Silero VAD · Namo turn detector
Env:      ZRT_AUTH_TOKEN, DEEPGRAM_API_KEY, OPENAI_API_KEY, CARTESIA_API_KEY
Run:      uv run features/chat_context.py
"""
import zrt
from zrt import Agent, Pipeline, Room, function_tool, ContextWindow
from zrt.plugins import CartesiaTTS, DeepgramSTT, OpenAILLM, SileroVAD, TurnDetector

from dotenv import load_dotenv
load_dotenv(override=True)

AGENT_ID = "chat-context-agent-py19"


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="Assistant",
            agent_id=AGENT_ID,
            instructions=(
                "You are a helpful assistant. Keep replies short. When the caller asks what "
                "you've talked about, call recap_conversation and summarize it for them."
            ),
            pipeline=build_pipeline(),
        )

    async def on_enter(self) -> None:
        await self.session.say("Hi! Chat with me, then ask me to recap our conversation anytime.")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

    @function_tool
    async def recap_conversation(self) -> dict:
        """Recap what the user and assistant have discussed so far.

        Args:
            None.
        """
        # Awaiting fetches a fresh copy from the runtime; each item is a
        # dict {role, content, ...}. last_n caps how far back we look.
        history = await self.session.get_context_history(last_n=8)
        turns = [{"role": m.get("role"), "content": m.get("content")}
                 for m in history]
        return {"turn_count": len(turns), "recent_turns": turns}


def build_pipeline() -> Pipeline:
    """Return a fresh Pipeline; serve() builds a new agent + pipeline ."""
    return Pipeline(
        stt=DeepgramSTT(model="nova-2"),
        llm=OpenAILLM(model="gpt-5.4-nano-2026-03-17", streaming=True,
                      reasoning_effort="none", verbosity="low"),
        tts=CartesiaTTS(model="sonic-3.5"),
        vad=SileroVAD(),
        turn_detector=TurnDetector(model="echo-large"),
        # Keep the prompt within a token budget; always retain the last few turns.
        context_window=ContextWindow(max_tokens=800, keep_recent_turns=4),
    )


def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    zrt.invoke(AGENT_ID, room=Room(playground=True))


if __name__ == "__main__":
    zrt.serve(Assistant, on_ready=invoke_agent)
