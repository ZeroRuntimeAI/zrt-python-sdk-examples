"""
Customer support — e-commerce order support voice agent.

Feature:  STT -> LLM -> TTS cascade; resolve FAQs, look up orders, file tickets, offer callback.
Pipeline: Cartesia (STT) · Google Gemini (LLM) · SarvamAI (TTS) · Silero VAD · Namo turn detector
Env:      ZRT_AUTH_TOKEN, CARTESIA_API_KEY, GOOGLE_API_KEY, SARVAM_API_KEY
Run:      uv run use_case/customer_support.py
"""

from zoneinfo import ZoneInfo
from datetime import datetime
import zrt
from zrt import Agent, Pipeline, Room, function_tool, EOUConfig, InterruptConfig
from zrt.plugins import CartesiaSTT, GoogleLLM, SarvamAITTS, SileroVAD, TurnDetector

from dotenv import load_dotenv
load_dotenv(override=True)


def _ist_now() -> str:
    """Current date and time in IST, injected into the agent's instructions."""
    return datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%A, %d %B %Y, %I:%M %p")


AGENT_ID = "support-agent-py"



pipeline = Pipeline(
    stt=CartesiaSTT(model="ink-2"),
    llm=GoogleLLM(model="gemini-2.5-flash", thinking_budget=0),
    tts=SarvamAITTS(streaming=True),
    vad=SileroVAD(),
    turn_detector=TurnDetector(model="echo_large"),
    eou_config=EOUConfig(
        mode="ADAPTIVE", min_max_speech_wait_timeout=[0.2, 0.4]),
    interrupt_config=InterruptConfig(mode="HYBRID"),
)


class SupportAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="SupportAgent",
            agent_id=AGENT_ID,
            instructions=(
                f"Today's date and time is {_ist_now()} (IST). "
                "You are a calm, helpful customer support agent for Brightcart, an online store. "
                "Help customers with order status, returns, shipping, and common questions. Keep "
                "replies short and friendly. "
                "For order questions, ask for the order ID and call lookup_order. For general questions "
                "about shipping, returns, refunds, or warranty, call check_faq with the relevant topic. "
                "If the customer has a problem you cannot resolve, call create_ticket with a clear "
                "summary and an appropriate priority (low, medium, or high). If they are still stuck or "
                "prefer a person, offer a callback and call request_human_callback with their phone "
                "number. Always confirm details back to the customer. Never invent order or ticket data — "
                "use the tools."
            ),
            pipeline=pipeline,
        )

    async def on_enter(self) -> None:
        # Greet the customer as the store's support line.
        await self.session.say(
            "Hi, you've reached Brightcart support. I can help with your order, returns, shipping, or "
            "any questions. What can I do for you today?"
        )

    async def on_exit(self) -> None:
        await self.session.say("Thanks for contacting Brightcart support. Have a great day!")

    @function_tool
    async def lookup_order(self, order_id: str) -> dict:
        """Look up the status and details of a customer order.

        Args:
            order_id: The order identifier (e.g. "BC-77310").
        """
        # Replace with a real order-management system lookup in production.
        return {
            "order_id": order_id,
            "status": "shipped",
            "carrier": "UPS",
            "tracking_number": "1Z999AA10123456784",
            "estimated_delivery": "in 2 business days",
            "items": ["Wireless Headphones", "USB-C Cable"],
        }

    @function_tool
    async def check_faq(self, topic: str) -> dict:
        """Look up the answer to a common support question.

        Args:
            topic: The FAQ topic (e.g. "returns", "shipping", "refund", "warranty").
        """
        # Replace with a real knowledge-base / FAQ search in production.
        faqs = {
            "returns": "Returns are accepted within 30 days of delivery for unused items.",
            "shipping": "Standard shipping is 3-5 business days; express is 1-2 business days.",
            "refund": "Refunds are issued to the original payment method within 5-7 business days.",
            "warranty": "Electronics carry a 1 Year limited manufacturer warranty.",
        }
        answer = faqs.get(
            topic.lower(), "I don't have that one on file — let me create a ticket for you.")
        return {"topic": topic, "answer": answer}

    @function_tool
    async def create_ticket(self, issue: str, priority: str) -> dict:
        """Create a support ticket for an unresolved issue.

        Args:
            issue: A short description of the customer's problem.
            priority: Ticket priority — one of "low", "medium", or "high".
        """
        # Replace with a real ticketing system (Zendesk/Jira) write in production.
        return {
            "status": "ticket_created",
            "ticket_id": "TCK-40918",
            "issue": issue,
            "priority": priority,
        }

    @function_tool
    async def request_human_callback(self, phone: str) -> dict:
        """Schedule a callback from a human support agent.

        Args:
            phone: The customer's callback phone number.
        """
        # Replace with a real callback-queue / dialer integration in production.
        return {
            "status": "callback_scheduled",
            "phone": phone,
            "eta_minutes": 15,
        }


def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    zrt.invoke(AGENT_ID, room=Room(playground=True))


if __name__ == "__main__":

    zrt.serve(SupportAgent, on_ready=invoke_agent)
