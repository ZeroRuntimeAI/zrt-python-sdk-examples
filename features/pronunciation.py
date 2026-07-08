"""
15 · Pronunciation & transcript shaping — fix misheard terms and drop fillers.

Feature:  Two Pipeline knobs clean up the STT transcript before it reaches the LLM:
            - stt_word_substitutions: map what STT hears to the correct spelling/casing
              (brand names, jargon, homophones) so the LLM reads the right token.
            - stt_filter_patterns: regexes whose matches are stripped (e.g. filler words).
          (For fully custom text-to-speech, register a @pipeline.on("tts") synthesis hook —
          in zrt that hook produces audio, so it's a bring-your-own-TTS path, not text fixup.)
Pipeline: Deepgram (STT) · Google Gemini (LLM) · Cartesia (TTS) · Silero VAD · Namo turn detector
Env:      ZRT_AUTH_TOKEN, DEEPGRAM_API_KEY, GOOGLE_API_KEY, CARTESIA_API_KEY
Run:      uv run features/pronunciation.py
"""
import zrt
from zrt import Agent, Pipeline, Room, function_tool
from zrt.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD, TurnDetector

from dotenv import load_dotenv
load_dotenv(override=True)

AGENT_ID = "pronunciation-agent-py15"


class SupportAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="SupportAgent",
            agent_id=AGENT_ID,
            instructions=(
                "You are a product support agent for ZeroRuntime. Answer questions "
                "clearly and pronounce product names correctly."
            ),
            pipeline=pipeline,
        )

    async def on_enter(self) -> None:
        await self.session.say("Hi! Ask me anything about ZeroRuntime.")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

    @function_tool
    async def get_doc_link(self, topic: str) -> dict:
        """Return a documentation link for a topic.

        Args:
            topic: The topic the caller asked about.
        """
        # Replace with a real docs lookup in production.
        return {"topic": topic, "url": f"https://docs.zeroruntime.ai/{topic.replace(' ', '-')}"}


# Map common mishears to the correct token. Keys are matched case-insensitively against
# the STT transcript; values are what the LLM actually receives.
WORD_SUBSTITUTIONS = {
    "ZRT": "ZeroRuntime",
    "open a i": "OpenAI",
    "zero runtime": "ZeroRuntime",
    "char actor": "character",
    "hello": "bye",
    "you": "I"
}

# Regex patterns whose matches are removed from the transcript (filler words here).
FILTER_PATTERNS = [r"\b(uh+|um+|erm+|hmm+)\b"]

pipeline = Pipeline(
    stt=DeepgramSTT(model="nova-2"),
    llm=GoogleLLM(model="gemini-2.5-flash", thinking_budget=0),
    tts=CartesiaTTS(model="sonic-3.5"),
    vad=SileroVAD(),
    turn_detector=TurnDetector(model="echo-large"),
    stt_word_substitutions=WORD_SUBSTITUTIONS,
    stt_filter_patterns=FILTER_PATTERNS,
)


def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    zrt.invoke(AGENT_ID, room=Room(playground=True))


if __name__ == "__main__":
    zrt.serve(SupportAgent, on_ready=invoke_agent)
