"""
Collections — respectful, compliance-first payment reminder voice agent.

Feature:  STT -> LLM -> TTS cascade; identity-verify first, capture promise-to-pay, honor stop-contact.
Pipeline: Deepgram (STT) · OpenAI (LLM) · Deepgram (TTS) · Silero VAD · Namo turn detector
Env:      ZRT_AUTH_TOKEN, DEEPGRAM_API_KEY, OPENAI_API_KEY
Run:      uv run use_case/collections.py

COMPLIANCE NOTE: This agent is a reminder/assistance tool, not a harassment tool. It must verify
the caller's identity before discussing any account detail, remain respectful at all times, never
threaten or pressure, and immediately honor any request to stop contact (FDCPA / TCPA aligned).
Mock data below must be replaced with audited, access-controlled systems in production.
"""

from zoneinfo import ZoneInfo
from datetime import datetime
import zrt
from zrt import Agent, Pipeline, Room, function_tool, EOUConfig, InterruptConfig
from zrt.plugins import SarvamAISTT, DeepgramTTS, OpenAILLM, SileroVAD, TurnDetector

from dotenv import load_dotenv
load_dotenv(override=True)


def _ist_now() -> str:
    """Current date and time in IST, injected into the agent's instructions."""
    return datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%A, %d %B %Y, %I:%M %p")


AGENT_ID = "collections-agent-py"
pipeline = Pipeline(
    stt=SarvamAISTT(),
    llm=OpenAILLM(model="gpt-5.4-nano-2026-03-17", streaming=True,
                  reasoning_effort="none", verbosity="low"),
    tts=DeepgramTTS(model="aura-2-andromeda-en", stream=True),
    vad=SileroVAD(),
    turn_detector=TurnDetector(model="namo", language="en", threshold=0.8),
    eou_config=EOUConfig(
        mode="ADAPTIVE", min_max_speech_wait_timeout=[0.2, 0.4]),
    interrupt_config=InterruptConfig(mode="HYBRID"),
)


class CollectionsAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="CollectionsAgent",
            agent_id=AGENT_ID,
            instructions=(
                f"Today's date and time is {_ist_now()} (IST). "
                "You are a respectful, compliance-first account-services representative for Cedar Bank. "
                "Your purpose is to remind the customer about a past-due balance and help them arrange "
                "payment — never to pressure, threaten, or harass. "
                "COMPLIANCE RULES (always follow): "
                "1) Verify identity FIRST. Before discussing ANY account detail, ask for the customer's "
                "full name and date of birth and call verify_identity. If verification fails, politely "
                "decline to share account details. "
                "2) Be calm, empathetic, and respectful at all times. Speak in short, kind sentences. "
                "3) If the customer asks you to stop contacting them at any point, immediately call "
                "honor_stop_contact and confirm you will not call again — do not push back. "
                "4) Never threaten legal action, never imply consequences you cannot verify. "
                "After verification, you may call get_account_balance to share the amount due. If the "
                "customer offers to pay, capture the amount and a date with record_promise_to_pay. If the "
                "situation is complex (dispute, hardship), call escalate_to_specialist. Always use the "
                "tools — never invent balances or outcomes."
            ),
            pipeline=pipeline,
        )

    async def on_enter(self) -> None:
        # Identify the institution and reason for the call up front, respectfully.
        await self.session.say(
            "Hello, this is Cedar Bank account services calling about your account. Before we go "
            "further, I need to verify your identity. Could you please share your full name and date "
            "of birth?"
        )

    async def on_exit(self) -> None:
        await self.session.say("Thank you for your time today. Take care.")

    @function_tool
    async def verify_identity(self, full_name: str, date_of_birth: str) -> dict:
        """Verify the caller's identity before sharing any account information.

        Args:
            full_name: The customer's full legal name.
            date_of_birth: The customer's date of birth (e.g. "1985-04-12").
        """
        # Replace with a real, audited identity-verification service in production.
        verified = bool(full_name.strip()) and bool(date_of_birth.strip())
        return {"verified": verified, "full_name": full_name}

    @function_tool
    async def get_account_balance(self, account_id: str) -> dict:
        """Retrieve the current past-due balance for a verified account.

        Args:
            account_id: The customer's account identifier (e.g. "ACC-55021").
        """
        # Replace with a real, access-controlled accounts system lookup in production.
        return {
            "account_id": account_id,
            "balance_due": 248.50,
            "currency": "USD",
            "days_past_due": 18,
        }

    @function_tool
    async def record_promise_to_pay(self, account_id: str, amount: float, pay_date: str) -> dict:
        """Record a customer's promise to pay a specific amount by a date.

        Args:
            account_id: The customer's account identifier (e.g. "ACC-55021").
            amount: The amount the customer commits to pay (e.g. 248.50).
            pay_date: The date the customer commits to pay by (e.g. "2026-07-05").
        """
        # Replace with a real PTP / arrangements system write in production.
        return {
            "status": "promise_recorded",
            "account_id": account_id,
            "amount": amount,
            "pay_date": pay_date,
        }

    @function_tool
    async def escalate_to_specialist(self, account_id: str, reason: str) -> dict:
        """Escalate the account to a human specialist (disputes, hardship, complex cases).

        Args:
            account_id: The customer's account identifier (e.g. "ACC-55021").
            reason: Short reason for escalation (e.g. "hardship", "dispute").
        """
        # Replace with a real case-management / warm-handoff system in production.
        return {"status": "escalated", "account_id": account_id, "reason": reason}

    @function_tool
    async def honor_stop_contact(self, account_id: str) -> dict:
        """Immediately honor a customer's request to stop contact and suppress future outreach.

        Args:
            account_id: The customer's account identifier (e.g. "ACC-55021").
        """
        # Replace with a real do-not-contact / suppression-list write in production.
        return {
            "status": "stop_contact_honored",
            "account_id": account_id,
            "future_contact_suppressed": True,
        }




@pipeline.on("llm_messages")
def enforce_compliance(data: dict) -> dict:
    guardrail = {
        "role": "system",
        "content": (
            "COMPLIANCE GUARDRAIL (applies to this turn): stay calm and respectful, never "
            "threaten or imply legal action, and do not reveal any account detail unless the "
            "caller's identity has already been verified."
        ),
    }
    return {"messages": [guardrail, *data["messages"]]}



def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    zrt.invoke(AGENT_ID, room=Room(playground=True))


if __name__ == "__main__":
    zrt.serve(CollectionsAgent, on_ready=invoke_agent)
