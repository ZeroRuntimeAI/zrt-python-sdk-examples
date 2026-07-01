"""
16 · Pub/Sub — broadcast the agent's replies to a room "CHAT" topic.

Feature:  Room pub/sub — out-of-band text messaging alongside the voice stream.
            - session.publish_message(PubSubPublishConfig(topic=..., message=..., options=...)):
              post text on a named topic to everyone in the room (options={"persist": True}
              keeps it in the room history).
            - session.subscribe_pubsub(topic, callback): receive messages on a topic. The
              callback gets {topic, message, payload, sender_id, sender_name, timestamp}.
          The agent mirrors each LLM reply to "CHAT" (via the turn_complete event), subscribes
          to "CHAT" to receive messages, and feeds a participant's typed message back to the LLM
          with session.generate(). A send_chat_message tool lets the model post on request. It
          still speaks (TTS) too, so the conversation shows up as both voice and text.
Pipeline: Deepgram nova-2 (STT) · Google Gemini (LLM) · Cartesia sonic-3.5 (TTS) · Silero VAD · Namo turn detector
Env:      ZRT_AUTH_TOKEN, GOOGLE_API_KEY
Run:      uv run features/pubsub.py
"""

import asyncio

import zrt
from zrt import Agent, Pipeline, Room, function_tool, PubSubPublishConfig
from zrt.plugins import GoogleLLM

from dotenv import load_dotenv
load_dotenv(override=True)

AGENT_ID = "pubsub-chat-agent-py16"
AGENT_NAME = "ChatAgent"
CHAT_TOPIC = "CHAT"   # the room pub/sub topic this agent publishes + subscribes on

pipeline = Pipeline(
    llm=GoogleLLM(model="gemini-2.5-flash", thinking_budget=0)
)

class ChatAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            name=AGENT_NAME,
            agent_id=AGENT_ID,
            instructions=(
                "You are a friendly chat assistant. Keep replies short. When the user asks you "
                "to post, broadcast, or send something to the chat, call send_chat_message."
            ),
            pipeline=pipeline,
        )
        self._tasks: set[asyncio.Task] = set()

    async def on_enter(self) -> None:
        # The room is joined by the time on_enter runs, so subscribe/publish here.
        await self.session.subscribe_pubsub(CHAT_TOPIC, self._on_chat)
        self.session.on("turn_complete", self._on_turn_complete)
        await self.session.publish_message(PubSubPublishConfig(
            topic=CHAT_TOPIC,
            message=f"{AGENT_NAME} has joined the room.",
            options={"persist": True},
        ))
        await self.session.say(
            "Hi! I'm in the room. Talk to me, or tell me to post something to the chat."
        )

    async def on_exit(self) -> None:
        pass

    # --- pub/sub -------------------------------------------------------------

    def _on_chat(self, msg: dict) -> None:
        """Receive every CHAT message (including our own echoes).

        This callback is SYNCHRONOUS, so async work (generate) is scheduled as a task.
        """
        sender = msg.get("sender_name") or msg.get("sender_id") or "unknown"
        text = (msg.get("message") or "").strip()
        print(f"[chat] {sender}: {text}")
        if not text or msg.get("sender_name") == AGENT_NAME:
            return
        self._spawn(self.session.generate(text))

    async def _on_turn_complete(self, payload: dict) -> None:
        """Publish each finished LLM reply to CHAT so subscribers receive the text."""
        text = payload.get("agent_transcript") if isinstance(
            payload, dict) else None
        if text:
            await self.session.publish_message(PubSubPublishConfig(topic=CHAT_TOPIC, message=text))

    def _spawn(self, coro) -> None:
        """Run a coroutine from the sync pub/sub callback."""
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # --- tools ---------------------------------------------------------------

    @function_tool
    async def send_chat_message(self, message: str) -> dict:
        """Publish a text message to the room's CHAT pub/sub topic so participants see it.

        Args:
            message: The message text to publish.
        """
        await self.session.publish_message(PubSubPublishConfig(topic=CHAT_TOPIC, message=message))
        return {"published": True, "topic": CHAT_TOPIC, "message": message}




def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    zrt.invoke(AGENT_ID, room=Room(playground=True))


if __name__ == "__main__":
    zrt.serve(ChatAgent, on_ready=invoke_agent)
