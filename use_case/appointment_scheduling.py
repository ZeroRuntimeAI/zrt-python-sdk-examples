"""
Appointment scheduling — clinic receptionist voice agent.

Feature:  STT -> LLM -> TTS cascade; full book/reschedule/cancel/remind lifecycle.
Pipeline: Deepgram (STT) · Google Gemini (LLM) · Cartesia (TTS) · Silero VAD · Namo turn detector
Env:      ZRT_AUTH_TOKEN, DEEPGRAM_API_KEY, GOOGLE_API_KEY, CARTESIA_API_KEY
Run:      uv run use_case/appointment_scheduling.py
"""

from zoneinfo import ZoneInfo
from datetime import datetime
import zrt
from zrt import Agent, Pipeline, Room, function_tool, EOUConfig, InterruptConfig
from zrt.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD, TurnDetector

from dotenv import load_dotenv
load_dotenv(override=True)


def _ist_now() -> str:
    """Current date and time in IST, injected into the agent's instructions."""
    return datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%A, %d %B %Y, %I:%M %p")


AGENT_ID = "appointment-agent-py"
WORD_SUBSTITUTIONS = {
    "Dr.": "Doctor",
    "APT-10432": "A P T dash one zero four three two"
}

pipeline = Pipeline(
    stt=DeepgramSTT(model="nova-2"),
    llm=GoogleLLM(model="gemini-2.5-flash", thinking_budget=0),
    tts=CartesiaTTS(model="sonic-3.5"),
    vad=SileroVAD(),
    turn_detector=TurnDetector(model="echo-large"),
    eou_config=EOUConfig(
        mode="ADAPTIVE", min_max_speech_wait_timeout=[0.2, 0.4]),
    interrupt_config=InterruptConfig(mode="HYBRID"),
    stt_word_substitutions=WORD_SUBSTITUTIONS
)


class AppointmentAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="AppointmentAgent",
            agent_id=AGENT_ID,
            instructions=(
                f"Today's date and time is {_ist_now()} (IST). "
                "You are the friendly front-desk receptionist for Lakeside Family Clinic. "
                "You help callers book, reschedule, cancel, and get reminders for appointments. "
                "Speak warmly, keep replies short and natural, and confirm details out loud. "
                "Booking flow: ask for the desired date, call check_availability, then collect the "
                "patient's full name and a preferred time and call book_appointment. Read the returned "
                "appointment ID back to the caller so they can reference it later and make sure available time is in future, do not book or read the past time. "
                "To reschedule, ask for the existing appointment ID plus the new date and time (make sure it's in the future), then "
                "call reschedule_appointment. To cancel, ask for the appointment ID and confirm before "
                "calling cancel_appointment. If a caller wants a confirmation or reminder, call "
                "send_reminder with the appointment ID. Never invent availability or IDs — always use "
                "the tools. If anything is unclear, ask a brief clarifying question."
            ),
            pipeline=pipeline,
        )

    async def on_enter(self) -> None:
        # Greet the caller as the clinic the moment the session is live.
        await self.session.say(
            "Thank you for calling Lakeside Family Clinic. This is the front desk. "
            "Would you like to book, reschedule, or cancel an appointment?"
        )

    async def on_exit(self) -> None:
        await self.session.say("Take care, and we'll see you at the clinic. Goodbye!")

    @function_tool
    async def check_availability(self, date: str) -> dict:
        """Check open appointment slots for a given date.

        Args:
            date: The desired date in natural language or ISO form (e.g. "2026-07-02").
        """
        # Replace with a real scheduling system / DB lookup in production.
        return {
            "date": date,
            "open_slots": ["09:00", "10:30", "13:00", "15:30", "20:00"],
            "provider": "Dr. Mehta",
        }

    @function_tool
    async def book_appointment(self, patient_name: str, date: str, time: str) -> dict:
        """Book a new appointment for a patient.

        Args:
            patient_name: Full name of the patient.
            date: Appointment date (e.g. "2026-07-02").
            time: Appointment time (e.g. "10:30").
        """
        # Replace with a real booking API / DB write in production.
        return {
            "status": "booked",
            "appointment_id": "APT-10432",
            "patient_name": patient_name,
            "date": date,
            "time": time,
            "provider": "Dr. Mehta",
        }

    @function_tool
    async def reschedule_appointment(self, appointment_id: str, new_date: str, new_time: str) -> dict:
        """Move an existing appointment to a new date and time.

        Args:
            appointment_id: The existing appointment ID (e.g. "APT-10432").
            new_date: The new date (e.g. "2026-07-09").
            new_time: The new time (e.g. "13:00").
        """
        # Replace with a real scheduling API / DB update in production.
        return {
            "status": "rescheduled",
            "appointment_id": appointment_id,
            "new_date": new_date,
            "new_time": new_time,
        }

    @function_tool
    async def cancel_appointment(self, appointment_id: str) -> dict:
        """Cancel an existing appointment.

        Args:
            appointment_id: The appointment ID to cancel (e.g. "APT-10432").
        """
        # Replace with a real cancellation API / DB update in production.
        return {"status": "cancelled", "appointment_id": appointment_id}

    @function_tool
    async def send_reminder(self, appointment_id: str) -> dict:
        """Send a confirmation/reminder message for an appointment.

        Args:
            appointment_id: The appointment ID to remind about (e.g. "APT-10432").
        """
        # Replace with a real SMS/email reminder service in production.
        return {
            "status": "reminder_sent",
            "appointment_id": appointment_id,
            "channel": "sms",
        }



def invoke_agent() -> None:
    """Start a session once the agent is registered (fired by serve's on_ready)."""
    zrt.invoke(AGENT_ID, room=Room(playground=True))


if __name__ == "__main__":
    zrt.serve(AppointmentAgent, on_ready=invoke_agent)
