from __future__ import annotations

import logging
import os

import zrt
from zrt import Agent, Pipeline, Room, function_tool
from zrt.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SarvamAISTT, SarvamAITTS, SileroVAD, TurnDetector

from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("component-swap")

AGENT_ID = "component-swap-agent-py"

LANGUAGES = {
    "en": {
        "label": "English",
        "stt": lambda: DeepgramSTT(language="en"),
        "tts": lambda: CartesiaTTS(language="en"),
    },
    "hi": {
        "label": "Hindi",
        "stt": lambda: SarvamAISTT(language="hi"),
        "tts": lambda: SarvamAITTS(),
    },
}


class Assistant(Agent):
    def __init__(self):
        super().__init__(
            name="Assistant",
            agent_id=AGENT_ID,
            instructions=(
                "You are a friendly multilingual assistant. You start in English. "
                "When the caller asks to talk in another language, call switch_language "
                "with its two-letter code (en or hi), then continue in that language."
            ),
            pipeline=pipeline,
        )
        self.current_language = "en"

    async def on_enter(self) -> None:
        logger.info("[assistant] session started — current language: %s",
                    LANGUAGES[self.current_language]["label"])
        self.session.on("runtime_warning", lambda p: logger.warning(
            "[runtime] %s: %s", p.get("code"), p.get("message")))
        await self.session.say(
            "Hi! I can switch languages mid-call — just ask me to talk in Hindi."
        )

    async def on_exit(self) -> None:
        logger.info("[assistant] session ended")

    @function_tool
    async def switch_language(self, language: str) -> dict:
        """Switch the conversation to a different language.

        Args:
            language: Two-letter language code to switch to — "en" or "hi".
        """
        lang = language.strip().lower()
        logger.info("[switch_language] requested=%r current=%r", lang, self.current_language)

        spec = LANGUAGES.get(lang)
        if spec is None:
            logger.warning("[switch_language] rejected — unsupported language %r", language)
            return {"ok": False, "error": f"unsupported language {language!r}; use en or hi"}

        if lang == self.current_language:
            logger.info("[switch_language] no-op — already speaking %s", spec["label"])
            return {"ok": True, "note": f"already speaking {spec['label']}"}

        new_stt, new_tts = spec["stt"](), spec["tts"]()
        logger.info("[switch_language] swapping stt=%s tts=%s -> %s",
                    type(new_stt).__name__, type(new_tts).__name__, spec["label"])

        try:
            await self.session.pipeline.change_component(stt=new_stt, tts=new_tts)
        except ValueError as e:
            logger.error("[switch_language] swap rejected: %s", e)
            return {"ok": False, "error": str(e)}

        self.current_language = lang
        logger.info("[switch_language] done — now speaking %s", spec["label"])
        return {"ok": True, "language": spec["label"]}


pipeline = Pipeline(
    stt=LANGUAGES["en"]["stt"](),
    llm=GoogleLLM(model="gemini-2.5-flash"),
    tts=LANGUAGES["en"]["tts"](),
    vad=SileroVAD(threshold=0.4),
    turn_detector=TurnDetector(model="echo-large"),
)

def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    logger.info("[startup] agent registered — inviting caller into the playground")
    zrt.invoke(AGENT_ID, room=Room(playground=True))


if __name__ == "__main__":
    logger.info("[startup] serving %s (languages: %s)", AGENT_ID, ", ".join(LANGUAGES))
    zrt.serve(Assistant, on_ready=invoke_agent)
