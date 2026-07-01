"""
Medical triage — nurse-line agent that triages symptoms and routes to a specialist.

Feature:  A triage agent assesses symptoms and urgency, then hands off to a
          cardiology or general-medicine specialist via agent_switch(...).
Pipeline: Google (STT) · Google Gemini (LLM) · Cartesia (TTS) · Silero VAD · Namo turn detector
Env:      ZRT_AUTH_TOKEN, GOOGLE_APPLICATION_CREDENTIALS, GOOGLE_API_KEY, CARTESIA_API_KEY
Run:      uv run use_case/medical_triage.py
"""

from zoneinfo import ZoneInfo
from datetime import datetime
import zrt
from zrt import Agent, Pipeline, Room, function_tool, agent_switch
from zrt.plugins import CartesiaTTS, GoogleLLM, GoogleSTT, SileroVAD, TurnDetector

from dotenv import load_dotenv
load_dotenv(override=True)


def _ist_now() -> str:
    """Current date and time in IST, injected into the agent's instructions."""
    return datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%A, %d %B %Y, %I:%M %p")


TRIAGE_AGENT_ID = "triage-agent-py"
CARDIOLOGY_AGENT_ID = "cardiology-agent"
GENERAL_MEDICINE_AGENT_ID = "general-medicine-agent"
WORD_SUBSTITUTIONS = {
    "Dr.": "Doctor"
}


class CardiologyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="CardiologyAgent",
            agent_id=CARDIOLOGY_AGENT_ID,
            instructions=(
                f"Today's date and time is {_ist_now()} (IST). "
                "You are a cardiology intake specialist. The caller has been routed to you "
                "for heart-related concerns. Collect the patient's name and book a cardiology "
                "visit by calling schedule_cardiology. Speak calmly and reassuringly. "
                "If the caller describes life-threatening symptoms (severe chest pain, trouble "
                "breathing, fainting), tell them to call emergency services immediately. "
                "You are not a substitute for a licensed physician."
            ),
        )

    async def on_enter(self) -> None:
        await self.session.say(
            "You're now with cardiology intake. Can I get your name to set up a visit?"
        )

    @function_tool
    async def schedule_cardiology(self, patient: str) -> dict:
        """Book a cardiology appointment for a patient.

        Args:
            patient: Full name of the patient to schedule.
        """
        # Replace with a real scheduling system / EHR write in production.
        return {
            "status": "scheduled",
            "department": "cardiology",
            "patient": patient,
            "appointment_id": "CARD-2201",
            "provider": "Dr. Rao",
        }


class GeneralMedicineAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="GeneralMedicineAgent",
            agent_id=GENERAL_MEDICINE_AGENT_ID,
            instructions=(
                f"Today's date and time is {_ist_now()} (IST). "
                "You are a general-medicine intake specialist. The caller has been routed to you "
                "for non-cardiac concerns. Collect the patient's name and book a visit by calling "
                "schedule_general. Speak warmly and keep replies short. "
                "If the caller describes life-threatening symptoms, tell them to call emergency "
                "services immediately. You are not a substitute for a licensed physician."
            ),
        )

    async def on_enter(self) -> None:
        await self.session.say(
            "You're now with general medicine. May I have your name to book your visit?"
        )

    @function_tool
    async def schedule_general(self, patient: str) -> dict:
        """Book a general-medicine appointment for a patient.

        Args:
            patient: Full name of the patient to schedule.
        """
        # Replace with a real scheduling system / EHR write in production.
        return {
            "status": "scheduled",
            "department": "general-medicine",
            "patient": patient,
            "appointment_id": "GEN-5530",
            "provider": "Dr. Mehta",
        }


pipeline = Pipeline(
    stt=GoogleSTT(model="chirp_3", location="us", stream=True),
    llm=GoogleLLM(model="gemini-2.5-flash", thinking_budget=0),
    tts=CartesiaTTS(model="sonic-3.5"),
    vad=SileroVAD(),
    turn_detector=TurnDetector(model="namo", language="en", threshold=0.8),
    stt_word_substitutions=WORD_SUBSTITUTIONS
)

class TriageAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="TriageAgent",
            agent_id=TRIAGE_AGENT_ID,
            instructions=(
                f"Today's date and time is {_ist_now()} (IST). "
                "You are a nurse-line triage agent. Greet the caller, listen to their symptoms, "
                "and figure out where to send them. "
                "First call assess_symptoms to record what they describe, then call check_urgency "
                "to gauge how serious it is. "
                "SAFETY: If the caller describes any life-threatening emergency — severe chest pain, "
                "difficulty breathing, signs of stroke (face drooping, slurred speech, weakness), "
                "uncontrolled bleeding, or loss of consciousness — immediately tell them to hang up "
                "and call emergency services (911 or local equivalent). Do not delay with tools. "
                "You are NOT a substitute for a doctor and must not give a diagnosis. "
                "For chest pain, palpitations, or other cardiac symptoms, call route_to_cardiology. "
                "For all other concerns, call route_to_general. Keep replies short and calm."
            ),
            pipeline=pipeline,
            agents=[CardiologyAgent(), GeneralMedicineAgent()],
        )

    async def on_enter(self) -> None:
        await self.session.say(
            "Hi, you've reached the nurse triage line. I'm not a doctor, but I can help direct you. "
            "If this is a medical emergency, please hang up and call emergency services now. "
            "Otherwise, what symptoms are you experiencing?"
        )

    @function_tool
    async def assess_symptoms(self, symptoms: str) -> dict:
        """Record and categorize the caller's described symptoms.

        Args:
            symptoms: Free-text description of what the caller is experiencing.
        """
        # Replace with a real clinical triage / NLP service in production.
        return {
            "symptoms": symptoms,
            "category": "cardiac" if "chest" in symptoms.lower() else "general",
            "recorded": True,
        }

    @function_tool
    async def check_urgency(self, symptoms: str) -> dict:
        """Estimate the urgency level for the described symptoms.

        Args:
            symptoms: Free-text description of what the caller is experiencing.
        """
        # Replace with a real clinical decision-support service in production.
        text = symptoms.lower()
        if any(w in text for w in ("chest pain", "can't breathe", "fainting", "stroke")):
            level = "emergency"
        elif any(w in text for w in ("fever", "pain", "vomiting")):
            level = "urgent"
        else:
            level = "routine"
        return {"symptoms": symptoms, "urgency_level": level}

    @function_tool
    async def route_to_cardiology(self) -> dict:
        """Hand the caller off to the cardiology specialist.

        Args:
            None.
        """
        return agent_switch(to=CARDIOLOGY_AGENT_ID, reason="cardiac symptoms", inherit_context=True)

    @function_tool
    async def route_to_general(self) -> dict:
        """Hand the caller off to the general-medicine specialist.

        Args:
            None.
        """
        return agent_switch(to=GENERAL_MEDICINE_AGENT_ID, reason="non-cardiac symptoms", inherit_context=True)



def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    zrt.invoke(TRIAGE_AGENT_ID, room=Room(playground=True))


if __name__ == "__main__":
    zrt.serve(TriageAgent, on_ready=invoke_agent)
