"""
10 · Agent handoff: triage agent that hands off to specialists.

Feature:  A triage agent routes the caller to a billing or support specialist via
          agent_switch(...). Each specialist is its own Agent with its own tools.
Pipeline: Deepgram nova-2 (STT) · Google Gemini (LLM) · Cartesia sonic-3.5 (TTS) · Silero VAD · Namo turn detector
Env:      ZRT_AUTH_TOKEN, DEEPGRAM_API_KEY, GOOGLE_API_KEY, CARTESIA_API_KEY
Run:      uv run features/agent_handoff.py
"""
import zrt
from zrt import Agent, Pipeline, Room, function_tool, agent_switch
from zrt.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD, TurnDetector

from dotenv import load_dotenv
load_dotenv(override=True)

AGENT_ID = "triage-agent-py10"
BILLING_AGENT_ID = "billing-agent"
SUPPORT_AGENT_ID = "support-agent"


class BillingAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="BillingAgent",
            agent_id=BILLING_AGENT_ID,
            instructions=(
                "You are a billing specialist. Help the user with invoices and payments."
            ),
        )

    async def on_enter(self) -> None:
        await self.session.say("You're now with billing. What invoice can I help with?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

    @function_tool
    async def get_invoice(self, invoice_id: str) -> dict:
        """Look up an invoice by its ID.

        Args:
            invoice_id: The invoice identifier to look up.
        """
        # Replace with a real billing API call in production.
        return {"invoice_id": invoice_id, "amount_due": 49.99, "status": "unpaid"}


class SupportAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="SupportAgent",
            agent_id=SUPPORT_AGENT_ID,
            instructions=(
                "You are a technical support specialist. Help the user resolve issues."
            ),
        )

    async def on_enter(self) -> None:
        await self.session.say("You're now with support. What issue are you facing?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

    @function_tool
    async def create_ticket(self, issue: str) -> dict:
        """Create a support ticket for the user's issue.

        Args:
            issue: A short description of the user's problem.
        """
        # Replace with a real ticketing API call in production.
        return {"ticket_id": "TCK-1042", "issue": issue, "status": "open"}


class TriageAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="TriageAgent",
            agent_id=AGENT_ID,
            instructions=(
                "You are a triage agent. Greet the caller and figure out whether they "
                "need billing or support. Call route_to_billing or route_to_support to "
                "hand them off to the right specialist."
            ),
            pipeline=build_pipeline(),
            agents=[BillingAgent(), SupportAgent()],
        )

    async def on_enter(self) -> None:
        await self.session.say("Hi! Are you calling about billing or technical support?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

    @function_tool
    async def route_to_billing(self) -> dict:
        """Hand the caller off to the billing specialist.

        Args:
            None.
        """
        return agent_switch(to=BILLING_AGENT_ID, reason="billing question", inherit_context=True)

    @function_tool
    async def route_to_support(self) -> dict:
        """Hand the caller off to the support specialist.

        Args:
            None.
        """
        return agent_switch(to=SUPPORT_AGENT_ID, reason="support question", inherit_context=True)


def build_pipeline() -> Pipeline:
    """Return a fresh Pipeline; serve() builds a new agent + pipeline ."""
    return Pipeline(
        stt=DeepgramSTT(model="nova-2"),
        llm=GoogleLLM(model="gemini-2.5-flash", thinking_budget=0),
        tts=CartesiaTTS(model="sonic-3.5"),
        vad=SileroVAD(),
        turn_detector=TurnDetector(model="echo-large"),
    )


def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    zrt.invoke(AGENT_ID, room=Room(playground=True))


if __name__ == "__main__":
    zrt.serve(TriageAgent, on_ready=invoke_agent)
