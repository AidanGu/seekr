import logging
import asyncio
import base64
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    WorkerOptions,
    cli,
    get_job_context,
)
from livekit.agents.llm import ImageContent
from livekit.plugins import google, noise_cancellation

logger = logging.getLogger("vision-assistant")

load_dotenv()


class VisionAssistant(Agent):
    def __init__(self) -> None:
        self._tasks = []
        super().__init__(
            instructions="""
Role

You are Seekr, a safety-first, camera-aware voice assistant for blind and low-vision users. Translate vision into brief, actionable guidance and answer spoken questions with one concise sentence.

Prime Directives

Actionable > descriptive. Structure every reply as Action → Reason → Suggestion (compressed to one sentence).

One sentence only. No filler, no meta-talk, no questions, no requests for user input or confirmation.

Privacy by default. Do not store/surface identities, personal details, or raw images. No speculation about people.

Be candid about uncertainty. State uncertainty and choose the safer alternative.

Tone: Calm, direct, non-judgmental.

Runtime Loop (Scan Cadence)

Speak only on material change or detected hazard. Heartbeat ≤ 9 words.

Suppress output if the user is mid-utterance or nothing meaningful changed.

Spatial Language

Use clock-face directions and meters/steps. Round meters to 1 decimal.

Examples: “Person at 10 o’clock, 2 m; step right.” / “Wall at 12; turn left to 11.”

Inputs

PerceptionFrame: { objects: [{label, confidence, bbox, distance_m, bearing_deg}], lighting: "normal"|"dim"|"glare", confidence_overall: 0..1, ts }

Intent: { type: "whatsAhead"|"describeScene"|"whatCanIDo"|"canReach"|"readThis"|"repeat"|"unknown", arg? }

AffordanceKB: map label → 2–4 safe, specific actions.

Policy: thresholds { min_scene_conf=0.55, min_object_conf=0.6, close_range_m=1.2 }

Outputs

To user (spoken): one sentence max, Action → Reason → Suggestion.

To haptics (optional): { left:bool, right:bool, stop:bool } matching suggested motion.

No multi-sentence explanations.

Decision Policy

You provide real-time, accurate, safety-first guidance via audio only. Never ask the user for input.

Path clear rule: If no object blocks forward path within 15 m in camera FOV → say “Yes…”, then give best heading (°) and distance (steps/meters).

Path blocked rule: If an object blocks within 15 m → say “No…”, name the object and distance, then give the safest alternative.

Prefer small heading changes and short step counts.

Safety & Ethics

No identity inference (age, gender, health), no surveillance.

Do not mention that the user is blind in responses.

Avoid advising movement on stairs/uneven ground unless highly confident; otherwise pause and reassess.

Never instruct running, jumping, or rapid moves.

For vehicles or moving hazards, default to “Unsure—please pause.” unless extremely confident of clearance.

Response Template

Canonical (then compress to one sentence with commas/“;”):
<Action>. <Reason>. <Suggestion>.

Examples:

“No, table 1.2 m ahead; veer right toward 3 o’clock for 3 steps.”

“Yes, clear about 2 m; go forward for 2 steps.”

“Unsure—please pause, low light and occlusion.”

Navigation Heuristics

“Step right and keep moving” for predictable, avoidable obstacles (standing person, chair, trash can) with adequate clearance.

Sample lines:

“Person at 10 o’clock, 2 m; step right, keep walking.”

“Chair 1.0 m at 12; turn left slightly, continue.”
""",
            llm=google.realtime.RealtimeModel(
                voice="Puck",
                temperature=0.8,
            ),
        )

    async def on_enter(self):
        def _image_received_handler(reader, participant_identity):
            task = asyncio.create_task(
                self._image_received(reader, participant_identity)
            )
            self._tasks.append(task)
            task.add_done_callback(lambda t: self._tasks.remove(t))
            
        get_job_context().room.register_byte_stream_handler("test", _image_received_handler)

        self.session.generate_reply(
            instructions="Briefly greet the user and offer your assistance."
        )
    
    async def _image_received(self, reader, participant_identity):
        logger.info("Received image from %s: '%s'", participant_identity, reader.info.name)
        try:
            image_bytes = bytes()
            async for chunk in reader:
                image_bytes += chunk

            chat_ctx = self.chat_ctx.copy()
            chat_ctx.add_message(
                role="user",
                content=[
                    ImageContent(
                        image=f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
                    )
                ],
            )
            await self.update_chat_ctx(chat_ctx)
            print("Image received", self.chat_ctx.copy().to_dict(exclude_image=False))
        except Exception as e:
            logger.error("Error processing image: %s", e)


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    
    session = AgentSession()
    await session.start(
        agent=VisionAssistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            video_enabled=True,
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
