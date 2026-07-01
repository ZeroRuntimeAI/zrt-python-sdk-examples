"""
Drive-through — fast-food order taker with snappy turn-taking.

Feature:  Order-taking agent that tracks a running order on the instance, upsells once,
          and confirms the total. Tuned for low-latency, barge-in-friendly conversation.
Pipeline: Cartesia (STT) · Groq (LLM) · Cartesia (TTS) · Silero VAD · Namo turn detector
          ADAPTIVE end-of-utterance + HYBRID interruption for snappy back-and-forth.
Env:      ZRT_AUTH_TOKEN, CARTESIA_API_KEY, GROQ_API_KEY
Run:      uv run use_case/drive_through.py
"""

from zoneinfo import ZoneInfo
from datetime import datetime
import zrt
from zrt import Agent, Pipeline, EOUConfig, InterruptConfig, Room, function_tool
from zrt.plugins import CartesiaSTT, CartesiaTTS, GroqLLM, SileroVAD, TurnDetector

from dotenv import load_dotenv
load_dotenv(override=True)


def _ist_now() -> str:
    """Current date and time in IST, injected into the agent's instructions."""
    return datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%A, %d %B %Y, %I:%M %p")


AGENT_ID = "drive-through-agent-py"


pipeline = Pipeline(
    stt=CartesiaSTT(model="ink-2"),
    llm=GroqLLM(model="llama-3.3-70b-versatile"),
    tts=CartesiaTTS(model="sonic-3.5"),
    vad=SileroVAD(),
    turn_detector=TurnDetector(model="namo", language="en", threshold=0.8),
    eou_config=EOUConfig(
        mode="ADAPTIVE", min_max_speech_wait_timeout=[0.2, 0.4]),
    interrupt_config=InterruptConfig(mode="HYBRID"),
)

class DriveThroughAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="DriveThroughAgent",
            agent_id=AGENT_ID,
            instructions=(
                f"Today's date and time is {_ist_now()} (IST). "
                "You are the order taker at a fast-food drive-through. Be quick, friendly, and "
                "concise. Take the customer's order item by item, calling add_item for each item "
                "with its quantity and remove_item if they change their mind. "
                "Upsell exactly once — after the first item or two, suggest a combo, drink, or "
                "dessert, but do not pester the customer if they decline. "
                "When they say they're done, call get_order_summary, read the items and total back "
                "to them, and once they confirm, call confirm_and_place_order. "
                "Keep every reply short — this is a drive-through, not a chat."
            ),
            pipeline=pipeline,
        )
        self._order: list[dict] = []

        @pipeline.on("llm_messages")
        def inject_live_order(data: dict) -> dict:

            lines = ", ".join(
                f"{l['quantity']}x {l['item']}" for l in self._order) or "nothing yet"
            context = {"role": "system",
                       "content": f"Live order so far: {lines}."}
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] Injecting live order context: {lines}")
            return {"messages": [*data["messages"], context]}

    async def on_enter(self) -> None:
        await self.session.say("Welcome to the drive-through! What can I get started for you?")

    async def on_exit(self) -> None:
        await self.session.say("Thanks, please pull around to the window. Have a great day!")

    @function_tool
    async def add_item(self, item: str, quantity: int) -> dict:
        """Add an item to the current order.

        Args:
            item: Name of the menu item to add (e.g. "cheeseburger").
            quantity: How many of this item to add.
        """
        # Replace with a real POS / menu-pricing lookup in production.
        self._order.append({"item": item, "quantity": quantity})
        return {"status": "added", "item": item, "quantity": quantity, "order_size": len(self._order)}

    @function_tool
    async def remove_item(self, item: str) -> dict:
        """Remove an item from the current order.

        Args:
            item: Name of the menu item to remove (e.g. "cheeseburger").
        """
        # Replace with a real POS update in production.
        before = len(self._order)
        self._order = [
            line for line in self._order if line["item"].lower() != item.lower()]
        removed = before - len(self._order)
        return {"status": "removed" if removed else "not_found", "item": item, "removed": removed}

    @function_tool
    async def get_order_summary(self) -> dict:
        """Summarize the current order with an estimated total.

        Args:
            None.
        """
        # Replace with real per-item pricing from the POS in production.
        unit_price = 5.00
        total_qty = sum(line["quantity"] for line in self._order)
        return {
            "items": self._order,
            "item_count": total_qty,
            "estimated_total": round(total_qty * unit_price, 2),
            "currency": "USD",
        }

    @function_tool
    async def confirm_and_place_order(self) -> dict:
        """Finalize and place the current order.

        Args:
            None.
        """
        # Replace with a real order-submission API call in production.
        total_qty = sum(line["quantity"] for line in self._order)
        return {
            "status": "placed",
            "order_id": "DT-7781",
            "items": self._order,
            "item_count": total_qty,
            "window": "pull forward to window 2",
        }


def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    zrt.invoke(AGENT_ID, room=Room(playground=True))


if __name__ == "__main__":
    zrt.serve(DriveThroughAgent, on_ready=invoke_agent)
