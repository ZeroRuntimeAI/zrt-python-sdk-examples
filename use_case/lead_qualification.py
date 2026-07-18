"""
Lead qualification: SaaS inbound SDR voice agent.

Feature:  STT -> LLM -> TTS cascade; BANT-style qualify, score, then demo or route.
Pipeline: Google (STT) · OpenAI (LLM) · Deepgram (TTS) · Silero VAD · Namo turn detector
Env:      ZRT_AUTH_TOKEN, GOOGLE_APPLICATION_CREDENTIALS, OPENAI_API_KEY, DEEPGRAM_API_KEY
Run:      uv run use_case/lead_qualification.py
"""

from zoneinfo import ZoneInfo
from datetime import datetime
import zrt
from zrt import Agent, Pipeline, Room, function_tool, EOUConfig, InterruptConfig
from zrt.plugins import DeepgramTTS, GoogleSTT, OpenAILLM, SileroVAD, TurnDetector

from dotenv import load_dotenv
load_dotenv(override=True)


def _ist_now() -> str:
    """Current date and time in IST, injected into the agent's instructions."""
    return datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%A, %d %B %Y, %I:%M %p")


AGENT_ID = "lead-qualification-agent"


def build_pipeline() -> Pipeline:
    """Return a fresh Pipeline; serve() builds a new agent + pipeline ."""
    return Pipeline(
        stt=GoogleSTT(model="chirp_3", location="us", stream=True),
        llm=OpenAILLM(model="gpt-5.4-nano-2026-03-17", streaming=True,
                      reasoning_effort="none", verbosity="low"),
        tts=DeepgramTTS(model="aura-2-andromeda-en", stream=True),
        vad=SileroVAD(),
        turn_detector=TurnDetector(model="echo-large"),
        eou_config=EOUConfig(
            mode="ADAPTIVE", min_max_speech_wait_timeout=[0.2, 0.4]),
        interrupt_config=InterruptConfig(mode="HYBRID"),
    )


class LeadQualAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="LeadQualAgent",
            agent_id=AGENT_ID,
            instructions=(
                f"Today's date and time is {_ist_now()} (IST). "
                "You are an inbound Sales Development Representative for Northwind Analytics, a SaaS "
                "data platform. Your job is to qualify inbound leads using a BANT framework: Budget, "
                "Authority, Need, and Timeline. Be warm, consultative, and concise; this is a phone "
                "conversation, not an interrogation. "
                "Ask about their budget range, their timeline to purchase, and whether they are the "
                "decision maker, then call qualify_lead with those answers. Next call score_lead to get "
                "a numeric score and tier. "
                "If the lead is qualified, offer to book a product demo: collect their email and a "
                "preferred slot and call book_demo. If they are not qualified or want to be routed, call "
                "route_to_sales with the tier so the right rep follows up. Always confirm the email back "
                "to the caller before booking. Never fabricate scores; use the tools."
            ),
            pipeline=build_pipeline(),
        )

    async def on_enter(self) -> None:
        # Greet the inbound lead and open qualification.
        await self.session.say(
            "Hi, thanks for reaching out to Northwind Analytics! I'd love to learn a bit about what "
            "you're looking to solve so I can point you in the right direction. What's prompting your "
            "interest today?"
        )

    async def on_exit(self) -> None:
        await self.session.say("Thanks so much for your time; we'll be in touch shortly. Take care!")

    @function_tool
    async def qualify_lead(self, budget: str, timeline: str, decision_authority: str) -> dict:
        """Capture BANT qualification answers from the lead.

        Args:
            budget: The lead's stated budget range (e.g. "$10k-$25k/yr").
            timeline: When the lead intends to buy (e.g. "this quarter").
            decision_authority: The lead's role in the decision (e.g. "final decision maker").
        """
        # Replace with a real CRM write / qualification engine in production.
        has_budget = any(c.isdigit() for c in budget)
        is_decision_maker = "decision" in decision_authority.lower(
        ) or "maker" in decision_authority.lower()
        qualified = has_budget and bool(timeline.strip()) and is_decision_maker
        return {
            "qualified": qualified,
            "budget": budget,
            "timeline": timeline,
            "decision_authority": decision_authority,
        }

    @function_tool
    async def score_lead(self, qualified: bool, budget: str, timeline: str) -> dict:
        """Compute a numeric lead score and assign a tier.

        Args:
            qualified: Whether the lead passed BANT qualification.
            budget: The lead's stated budget range.
            timeline: The lead's purchase timeline.
        """
        # Replace with a real lead-scoring model in production.
        score = 40 if qualified else 10
        if any(c.isdigit() for c in budget):
            score += 25
        if any(word in timeline.lower() for word in ("now", "week", "month", "quarter")):
            score += 25
        score = min(score, 100)
        tier = "hot" if score >= 80 else "warm" if score >= 50 else "cold"
        return {"score": score, "tier": tier, "qualified": qualified}

    @function_tool
    async def book_demo(self, email: str, slot: str) -> dict:
        """Book a product demo for a qualified lead.

        Args:
            email: The lead's email address for the calendar invite.
            slot: The requested demo slot (e.g. "Thursday 2pm PT").
        """
        # Replace with a real calendar / scheduling API in production.
        return {
            "status": "demo_booked",
            "email": email,
            "slot": slot,
            "confirmation_id": "DEMO-58219",
        }

    @function_tool
    async def route_to_sales(self, tier: str) -> dict:
        """Route the lead to the appropriate sales queue by tier.

        Args:
            tier: The lead tier from scoring ("hot", "warm", or "cold").
        """
        # Replace with a real CRM routing / round-robin assignment in production.
        owner = {
            "hot": "Senior AE - immediate follow-up",
            "warm": "AE - 24h follow-up",
            "cold": "Nurture sequence",
        }.get(tier.lower(), "General queue")
        return {"status": "routed", "tier": tier, "assigned_to": owner}


def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    zrt.invoke(AGENT_ID, room=Room(playground=True))


if __name__ == "__main__":

    zrt.serve(LeadQualAgent, on_ready=invoke_agent)
