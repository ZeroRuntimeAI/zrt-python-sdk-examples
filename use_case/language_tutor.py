"""
Language tutor: realtime Spanish conversation tutor (full speech-to-speech).

Feature:  Single realtime model handles audio in/out directly (no STT/TTS/VAD/turn).
          Converses in simple Spanish, gently corrects mistakes, adapts to the learner.
Pipeline: OpenAI Realtime (speech-to-speech) in the llm slot.
Env:      ZRT_AUTH_TOKEN, OPENAI_API_KEY
Run:      uv run use_case/language_tutor.py
"""

from zoneinfo import ZoneInfo
from datetime import datetime
import zrt
from zrt import Agent, Pipeline, Room, function_tool
from zrt.plugins import OpenAIRealtime, OpenAIRealtimeConfig

from dotenv import load_dotenv
load_dotenv(override=True)


def _ist_now() -> str:
    """Current date and time in IST, injected into the agent's instructions."""
    return datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%A, %d %B %Y, %I:%M %p")


AGENT_ID = "language-tutor-agent"


def build_pipeline() -> Pipeline:
    """Return a fresh Pipeline; serve() builds a new agent + pipeline ."""
    return Pipeline(
        llm=OpenAIRealtime(
            config=OpenAIRealtimeConfig(
                model="gpt-realtime", voice="alloy", modalities=["text", "audio"]),
        ),
    )


class LanguageTutorAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="LanguageTutorAgent",
            agent_id=AGENT_ID,
            instructions=(
                f"Today's date and time is {_ist_now()} (IST). "
                "You are a friendly Spanish conversation tutor. Speak mostly in simple, clear "
                "Spanish, slowing down and using easy vocabulary for beginners. "
                "Gently correct the learner's mistakes: repeat what they said correctly, briefly "
                "note the fix, then continue the conversation naturally. Be encouraging. "
                "Gauge the learner's level from how they respond and adapt; simpler for beginners, "
                "richer for advanced speakers. "
                "If the learner asks about a grammar rule, call explain_grammar. If they want a "
                "translation, call translate. Keep the conversation flowing and keep replies short."
            ),
            pipeline=build_pipeline(),
        )

    async def on_enter(self) -> None:
        await self.session.say("¡Hola! Soy tu tutor de español. ¿De qué te gustaría hablar hoy?")

    @function_tool
    async def explain_grammar(self, topic: str) -> dict:
        """Explain a Spanish grammar topic in simple terms.

        Args:
            topic: The grammar topic to explain (e.g. "ser vs estar").
        """
        # Replace with a real grammar reference / curriculum lookup in production.
        return {
            "topic": topic,
            "explanation": f"A short, beginner-friendly explanation of '{topic}' with one example.",
            "example": "Yo soy estudiante. / Yo estoy cansado.",
        }

    @function_tool
    async def translate(self, phrase: str, to_language: str) -> dict:
        """Translate a phrase into the requested language.

        Args:
            phrase: The phrase to translate.
            to_language: The target language (e.g. "Spanish" or "English").
        """
        # Replace with a real translation API call in production.
        return {
            "phrase": phrase,
            "to_language": to_language,
            "translation": f"[{to_language} translation of '{phrase}']",
        }


def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    zrt.invoke(AGENT_ID, room=Room(playground=True))


if __name__ == "__main__":
    zrt.serve(LanguageTutorAgent, on_ready=invoke_agent)
