"""
13 · Multilingual — detect and reply in the user's language.

Feature:  Deepgram multi-language STT + language-mirroring instructions. A
          user_turn_start hook logs each utterance.
Pipeline: SarvamAI (STT) · Google Gemini (LLM) · SarvamAI (TTS) · Silero VAD · Namo turn detector
Env:      ZRT_AUTH_TOKEN, SARVAM_API_KEY, GOOGLE_API_KEY
Run:      uv run features/multilingual.py
"""
import zrt
from zrt import Agent, Pipeline, Room, function_tool
from zrt.plugins import GoogleLLM, SarvamAISTT, SarvamAITTS, SileroVAD, TurnDetector

from dotenv import load_dotenv
load_dotenv(override=True)

AGENT_ID = "multilingual-agent-py13"


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="Assistant",
            agent_id=AGENT_ID,
            instructions=(
                "You are a multilingual voice assistant. Detect the language the user "
                "speaks and ALWAYS reply in that same language. Use translate_phrase "
                "when the user asks for a translation."
            ),
            pipeline=pipeline,
        )

    async def on_enter(self) -> None:
        await self.session.say("Hi! Speak to me in any language and I'll reply in it.")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

    @function_tool
    async def translate_phrase(self, text: str, target_language: str) -> dict:
        """Translate a phrase into a target language.

        Args:
            text: The phrase to translate.
            target_language: The language to translate the phrase into.
        """
        # Replace with a real translation API call in production.
        return {"source_text": text, "target_language": target_language, "translation": text}


pipeline = Pipeline(
    stt=SarvamAISTT(model="saaras:v3", language="unknown"),
    llm=GoogleLLM(model="gemini-2.5-flash", thinking_budget=0),
    tts=SarvamAITTS(),
    vad=SileroVAD(),
    turn_detector=TurnDetector(model="echo_large"),
)


@pipeline.on("user_turn_start")
async def on_user_turn_start(transcript: str) -> None:
    print(f"[multilingual] utterance: {transcript}")


def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    zrt.invoke(AGENT_ID, room=Room(playground=True))


if __name__ == "__main__":
    zrt.serve(Assistant, on_ready=invoke_agent)
