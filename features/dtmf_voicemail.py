import zrt
from zrt import Agent, Pipeline, Room, function_tool, DTMFHandler, VoiceMailDetector
from zrt.plugins import DeepgramTTS, GoogleLLM, TurnDetector, SarvamAISTT, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)

AGENT_ID = "dtmf_and_voice_mail_example"


def to_agent(digit: str) -> None:
    """DTMF callback: caller pressed 0 to reach a human agent."""
    print(f"[dtmf] digit {digit} pressed -> routing to a human agent")


def pin_ok(sequence: str) -> None:
    """DTMF callback: caller entered the correct PIN sequence."""
    print(f"[dtmf] PIN sequence {sequence} accepted")


def on_voicemail(info: dict) -> None:
    """VoiceMailDetector callback: answering machine detected."""
    print(f"[voicemail] detected: {info}")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="Assistant",
            agent_id=AGENT_ID,
            instructions=(
                "You are a phone assistant. Ask the caller to press 0 for an agent or "
                "enter their 4-digit PIN. Use check_account to verify their PIN."
            ),
            pipeline=pipeline,
        )

    async def on_enter(self) -> None:
        await self.session.say("Press 0 for an agent, or enter your 4-digit PIN.")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

    @function_tool
    async def check_account(self, pin: str) -> dict:
        """Verify a caller's account using their PIN.

        Args:
            pin: The 4-digit PIN the caller entered.
        """
        # Replace with a real account-verification call in production.
        return {"pin": pin, "verified": True, "account_id": "ACC-7781"}


dtmf_handler = DTMFHandler()
dtmf_handler.on_digit("0", to_agent)
dtmf_handler.on_sequence("1234", pin_ok)

voice_mail_detector = VoiceMailDetector(
    llm=GoogleLLM(model="gemini-2.5-flash", thinking_budget=0),
    callback=on_voicemail,
    auto_hangup=True,
)

pipeline = Pipeline(
    stt=SarvamAISTT(),
    llm=GoogleLLM(model="gemini-2.5-flash", thinking_budget=0),
    tts=DeepgramTTS(model="aura-2-andromeda-en", stream=True),
    vad=SileroVAD(),
    turn_detector=TurnDetector(model="echo-large"),
    dtmf_handler=dtmf_handler,
    voice_mail_detector=voice_mail_detector,
)


def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    zrt.invoke(AGENT_ID, room=Room(playground=True))


if __name__ == "__main__":
    zrt.serve(Assistant, on_ready=invoke_agent)
