"""
Support chatbot: a text helpdesk for "Nimbus" driven over a "CHAT" pub/sub topic.

Feature:  A website helpdesk driven over room pub/sub, so the chat widget carries the
          whole conversation:
            - SUBSCRIBES to "CHAT": visitors who type in the widget publish their message on
              CHAT; the agent feeds it to the LLM with session.generate() (skipping its own
              echoes so it never answers itself).
            - PUBLISHES to "CHAT": every finished reply is mirrored to CHAT (via turn_complete)
              so the widget shows the agent's text.
          Domain tools (search_help, check_service_status, create_ticket, connect_to_human)
          resolve the request; their answers flow back to the widget through the same topic.
Pipeline: Google Gemini (LLM) — text-only over the CHAT pub/sub topic (no STT/TTS)
Env:      ZRT_AUTH_TOKEN, GOOGLE_API_KEY
Run:      uv run use_case/support_chatbot.py
"""

from dotenv import load_dotenv
from zrt.plugins import GoogleLLM
from zrt import Agent, Pipeline, Room, function_tool, PubSubPublishConfig
import zrt
import asyncio

load_dotenv(override=True)

AGENT_ID = "support-chatbot-agent"
AGENT_NAME = "NimbusHelpdesk"
CHAT_TOPIC = "CHAT"   # shared with the embed chat widget


def build_pipeline() -> Pipeline:
    """Return a fresh Pipeline; serve() builds a new agent + pipeline ."""
    return Pipeline(
        llm=GoogleLLM(model="gemini-2.5-flash", thinking_budget=0)
    )


class SupportChatbot(Agent):
    def __init__(self) -> None:
        super().__init__(
            name=AGENT_NAME,
            agent_id=AGENT_ID,
            instructions=(
                "You are Nimbus's helpdesk assistant; Nimbus is a project-management SaaS. "
                "Keep replies short, warm, and plain-text (this is shown in a chat widget). Answer how-to questions by calling search_help with the topic. "
                "If a visitor reports an outage or things being down, call check_service_status. "
                "If you cannot resolve their problem, call create_ticket with a clear summary and "
                "a priority (low, medium, or high). If they're frustrated or ask for a person, "
                "call connect_to_human. Always confirm details back. Never invent help content, "
                "ticket data, or system status; use the tools."
            ),
            pipeline=build_pipeline(),
        )
        self._tasks: set[asyncio.Task] = set()

    async def on_enter(self) -> None:
        # The room is joined by now; wire up the CHAT topic both ways.
        await self.session.subscribe_pubsub(CHAT_TOPIC, self._on_chat)
        self.session.on("turn_complete", self._on_turn_complete)
        # A persisted greeting so visitors who open the widget mid-session still see it.
        await self.session.publish_message(PubSubPublishConfig(
            topic=CHAT_TOPIC,
            message="Hi, you've reached Nimbus support. How can I help?",
            options={"persist": True},
        ))
        await self.session.say(
            "Hi, you've reached Nimbus support. You can type in the chat or just talk; "
            "what can I help you with?"
        )

    async def on_exit(self) -> None:
        pass

    # --- CHAT pub/sub wiring ----------------------------------------------

    def _on_chat(self, msg: dict) -> None:
        """A visitor typed in the widget. Synchronous callback; feed real messages to the
        LLM on a task, and skip the agent's own mirrored replies to avoid a loop."""
        text = (msg.get("message") or "").strip()
        if not text or msg.get("sender_name") == AGENT_NAME:
            return
        sender = msg.get("sender_name") or msg.get("sender_id") or "visitor"
        print(f"[chat] {sender}: {text}")
        self._spawn(self.session.generate(text))

    async def _on_turn_complete(self, payload: dict) -> None:
        """Mirror each finished reply to CHAT so the widget displays the agent's text."""
        text = payload.get("agent_transcript") if isinstance(
            payload, dict) else None
        if text:
            await self.session.publish_message(PubSubPublishConfig(topic=CHAT_TOPIC, message=text))

    def _spawn(self, coro) -> None:
        """Run a coroutine from the sync pub/sub callback, holding a reference so it
        isn't garbage-collected before it finishes."""
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # --- tools ---------------------------------------------------------------

    @function_tool
    async def search_help(self, topic: str) -> dict:
        """Look up a help article for a how-to question.

        Args:
            topic: The help topic (e.g. "import tasks", "billing", "integrations").
        """
        # Replace with a real knowledge-base search in production.
        return {
            "topic": topic,
            "title": f"How to: {topic}",
            "url": f"https://help.nimbus.example/{topic.lower().replace(' ', '-')}",
            "summary": "Open the board menu, choose the action, and follow the guided steps.",
        }

    @function_tool
    async def check_service_status(self) -> dict:
        """Check whether Nimbus services are operational.

        Args:
            None.
        """
        # Replace with a real status-page / health API lookup in production.
        return {"status": "operational", "incidents": [], "checked": "just now"}

    @function_tool
    async def create_ticket(self, summary: str, priority: str) -> dict:
        """Create a support ticket for an unresolved issue.

        Args:
            summary: A short description of the visitor's problem.
            priority: Ticket priority; one of "low", "medium", or "high".
        """
        # Replace with a real ticketing system (Zendesk/Jira) write in production.
        return {"status": "ticket_created", "ticket_id": "NMB-50918", "summary": summary, "priority": priority}

    @function_tool
    async def connect_to_human(self, reason: str) -> dict:
        """Hand the conversation off to a human support agent.

        Args:
            reason: A short note on why the visitor needs a person.
        """
        # Replace with a real live-chat / routing integration in production.
        return {"status": "handoff_started", "reason": reason, "eta_minutes": 5}


def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    zrt.invoke(AGENT_ID, room=Room(playground=True))


if __name__ == "__main__":
    zrt.serve(SupportChatbot, on_ready=invoke_agent)
