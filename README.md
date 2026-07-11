# Zero Runtime; Python SDK Examples

Runnable, real-time voice-agent examples built on the [Zero Runtime Python SDK](https://zeroruntime.ai/).
You write the agent (instructions, tools, behavior); **Zero Runtime** runs the live
speech-to-speech pipeline (STT → LLM → TTS, with turn detection, denoising, and
interruptions) for you.

## Quickstart

```bash
./setup.sh                  # installs deps (uv or pip) and seeds .env
# edit .env: set ZRT_AUTH_TOKEN + the provider keys your example uses
uv run <name>.py   # or: source .venv/bin/activate && python <name>.py
```

That's it; no media servers, GPUs, or provider client libraries to install. The
SDK plugins are thin config; the runtime does the heavy lifting in the cloud.

## Requirements

- Python **3.11+**
- A Zero Runtime endpoint + auth token (`ZRT_RUNTIME_ADDRESS`, `ZRT_AUTH_TOKEN`)
- API key(s) for the providers your example uses (Deepgram, Google, Cartesia, …)

`setup.sh` prefers [`uv`](https://docs.astral.sh/uv/) for a fast, reproducible
install (`uv.lock` is committed); if `uv` isn't present it falls back to
`python -m venv` + `pip`.

## Examples

Two folders: **`features/`** teaches one SDK capability at a time, **`use_case/`** ships
production-shaped domain agents. Every example has real `@function_tool`s and lifecycle /
pipeline hooks.

### `features/`

| File                        | Shows                                                                                        |
| --------------------------- | -------------------------------------------------------------------------------------------- |
| `basic_cascade.py`          | Smallest complete agent; STT→LLM→TTS + one tool                                              |
| `advance_cascade_config.py` | FalseInterruption Handling cascade; model picks + `InterruptConfig`                          |
| `function_tools.py`         | Multiple tools the LLM chains together                                                       |
| `pipeline_hooks.py`         | `@pipeline.on(...)` turn / llm observation hooks                                             |
| `fallback.py`               | `FallbackSTT/LLM/TTS` provider failover                                                      |
| `background_audio.py`       | Ambient music + thinking audio                                                               |
| `realtime.py`               | Full speech-to-speech (Gemini Live)                                                          |
| `hybrid_stt.py`             | Cascade STT feeding a realtime LLM (`mode="hybrid_stt"`)                                     |
| `hybrid_tts.py`             | Realtime LLM with cascade TTS voice (`mode="hybrid_tts"`)                                    |
| `vision.py`                 | Camera frame capture + analysis                                                              |
| `mcp_tools.py`              | Tools auto-discovered from MCP servers                                                       |
| `agent_handoff.py`          | `agent_switch()` between specialist agents                                                   |
| `change_component.py`       | Swap one pipeline component (STT/TTS) mid-call                                               |
| `change_pipeline.py`        | Switch the whole pipeline cascade ↔ realtime at runtime                                      |
| `metrics.py`                | Per-component latency via metric hooks                                                       |
| `multilingual.py`           | Detect + reply in the caller's language                                                      |
| `agent_hangup.py`           | Agent ends the call with `session.hangup()`                                                  |
| `dtmf_voicemail.py`         | Keypad (DTMF) input + voicemail detection                                                    |
| `pronunciation.py`          | `stt_word_substitutions` + filler filtering                                                  |
| `pubsub.py`                 | Room pub/sub chat; mirror replies to a `CHAT` topic (`publish_message` / `subscribe_pubsub`) |
| `wakeup_call.py`            | Re-engage a silent caller via `on_wake_up()`                                                 |
| `inference_gateway.py`      | Gateway turn detection (`TurnDetector(model="echo-large")`)                                  |
| `chat_context.py`           | `ContextWindow` budget + `get_context_history()` recap                                       |

### `use_case/`

| File                        | Domain                                            |
| --------------------------- | ------------------------------------------------- |
| `appointment_scheduling.py` | Clinic; book · reschedule · cancel · remind       |
| `lead_qualification.py`     | SaaS SDR; qualify → score → demo → route          |
| `customer_support.py`       | E-commerce; orders · FAQ · tickets · callback     |
| `collections.py`            | Compliance-first payment reminders                |
| `medical_triage.py`         | Symptom triage + specialist handoff               |
| `drive_through.py`          | Fast-food order capture (snappy turn-taking)      |
| `language_tutor.py`         | Realtime Spanish conversation tutor               |
| `proactive_outreach.py`     | Outbound renewal call with wake-up                |
| `support_chatbot.py`        | Text + voice helpdesk over a `CHAT` pub/sub topic |

## How it works

| Piece                                  | What it is                                                                                                                                                                            |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Agent`                                | Your behavior; instructions, tools, what it says on enter/exit. Carries its `pipeline`.                                                                                               |
| `Pipeline`                             | The voice stack: STT (hear) → LLM (think) → TTS (speak), plus VAD, turn detection, denoising. Also carries session helpers like `dtmf_handler`, `wake_up`, and `voice_mail_detector`. |
| `zrt.serve(Agent, on_ready=...)`       | Registers the agent and listens for sessions.                                                                                                                                         |
| `zrt.invoke(AGENT_ID, room=Room(...))` | Starts a session for a registered agent (returns a `playground_url`).                                                                                                                 |

More: https://github.com/ZeroRuntimeAI/zrt-python-sdk-examples
