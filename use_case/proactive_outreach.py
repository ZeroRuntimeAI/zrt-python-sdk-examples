"""
Proactive outreach: outbound subscription-renewal call with re-engagement.

Feature:  The agent opens the call proactively, explains an upcoming renewal, and offers a
          retention discount before accepting a cancellation. A wake_up nudge re-engages a
          silent customer so the call doesn't stall.
Pipeline: Deepgram (STT) · OpenAI (LLM) · Sarvam AI (TTS) · Silero VAD · Namo turn detector
Env:      ZRT_AUTH_TOKEN, DEEPGRAM_API_KEY, OPENAI_API_KEY, SARVAM_API_KEY
Run:      uv run use_case/proactive_outreach.py
"""

from zoneinfo import ZoneInfo
from datetime import datetime
import zrt
from zrt import Agent, Pipeline, Room, function_tool
from zrt.plugins import DeepgramSTT, OpenAILLM, SarvamAITTS, SileroVAD, TurnDetector

from dotenv import load_dotenv
load_dotenv(override=True)


def _ist_now() -> str:
    """Current date and time in IST, injected into the agent's instructions."""
    return datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%A, %d %B %Y, %I:%M %p")


AGENT_ID = "proactive-outreach-agent"


def build_pipeline() -> Pipeline:
    """Return a fresh Pipeline; serve() builds a new agent + pipeline ."""
    return Pipeline(
        # If the customer goes quiet for 8s, proactively re-engage them.
        wake_up=8,
        stt=DeepgramSTT(model="nova-2"),
        llm=OpenAILLM(model="gpt-5.4-nano-2026-03-17", streaming=True,
                      reasoning_effort="none", verbosity="low"),
        tts=SarvamAITTS(streaming=True),
        vad=SileroVAD(),
        turn_detector=TurnDetector(model="echo-large"),
    )


class OutreachAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="OutreachAgent",
            agent_id=AGENT_ID,
            instructions=(
                f"Today's date and time is {_ist_now()} (IST). "
                "You are a warm, helpful retention specialist making an outbound courtesy call "
                "about a customer's upcoming subscription renewal. "
                "Open by greeting the customer and explaining their plan renews soon. "
                "If they want to keep the subscription, call confirm_renewal. "
                "If they hesitate or want to cancel, do NOT cancel immediately; first acknowledge "
                "their concern and offer a retention discount by calling offer_discount. "
                "Only if they still want to cancel after the offer should you call process_cancellation "
                "with their reason. Be respectful of their time and keep replies short and friendly."
            ),
            pipeline=build_pipeline(),
        )

    async def on_enter(self) -> None:
        # Proactively open the outbound call the moment the session is live.
        await self.session.reply(
            instructions="Greet the customer by name and explain their subscription renews soon.",
        )

    async def on_exit(self) -> None:
        await self.session.say("Thanks so much for your time today. Take care!")

    @function_tool
    async def confirm_renewal(self, customer_id: str) -> dict:
        """Confirm the customer's subscription renewal.

        Args:
            customer_id: The customer's account identifier.
        """
        # Replace with a real billing / subscription API call in production.
        return {
            "status": "renewed",
            "customer_id": customer_id,
            "next_billing_date": "2026-07-25",
            "plan": "Pro Annual",
        }

    @function_tool
    async def offer_discount(self, customer_id: str, percent: int) -> dict:
        """Offer a retention discount to the customer.

        Args:
            customer_id: The customer's account identifier.
            percent: The discount percentage to offer (e.g. 20).
        """
        # Replace with a real promotions / pricing engine call in production.
        return {
            "status": "discount_offered",
            "customer_id": customer_id,
            "percent": percent,
            "offer_id": "RETAIN-3390",
        }

    @function_tool
    async def process_cancellation(self, customer_id: str, reason: str) -> dict:
        """Cancel the customer's subscription and record the reason.

        Args:
            customer_id: The customer's account identifier.
            reason: The customer's stated reason for cancelling.
        """
        # Replace with a real cancellation API call in production.
        return {
            "status": "cancelled",
            "customer_id": customer_id,
            "reason": reason,
            "effective_date": "2026-07-25",
        }


def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    zrt.invoke(AGENT_ID, room=Room(playground=True))


if __name__ == "__main__":
    zrt.serve(OutreachAgent, on_ready=invoke_agent)
