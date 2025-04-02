import logging
import os

from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    metrics,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import (
    cartesia,
    openai,
    deepgram,
    noise_cancellation,
    silero,
    turn_detector,
)


load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")


def prewarm(proc: JobProcess):
    try:
        proc.userdata["vad"] = silero.VAD.load()
        logger.info("VAD model prewarmed successfully.")
    except Exception as e:
        logger.error(f"Failed to prewarm VAD model: {e}")
        raise


async def entrypoint(ctx: JobContext):
    try:
        initial_ctx = llm.ChatContext().append(
            role="system",
            text=(
                "You are a voice assistant created by LiveKit. Your interface with users will be voice. "
                "You should use short and concise responses, and avoiding usage of unpronouncable punctuation. "
                "You were created as a demo to showcase the capabilities of LiveKit's agents framework."
            ),
        )

        logger.info(f"Connecting to room {ctx.room.name}")
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

        # Wait for the first participant to connect
        participant = await ctx.wait_for_participant()
        logger.info(f"Starting voice assistant for participant {participant.identity}")

        # Load configuration from environment variables or defaults
        min_delay = float(os.getenv("MIN_ENDPOINTING_DELAY", 0.5))
        max_delay = float(os.getenv("MAX_ENDPOINTING_DELAY", 5.0))
        llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")

        agent = VoicePipelineAgent(
            vad=ctx.proc.userdata["vad"],
            stt=deepgram.STT(),
            llm=openai.LLM(model=llm_model),
            tts=cartesia.TTS(),
            turn_detector=turn_detector.EOUModel(),
            min_endpointing_delay=min_delay,
            max_endpointing_delay=max_delay,
            noise_cancellation=noise_cancellation.BVC(),
            chat_ctx=initial_ctx,
        )

        usage_collector = metrics.UsageCollector()

        @agent.on("metrics_collected")
        def on_metrics_collected(agent_metrics: metrics.AgentMetrics):
            metrics.log_metrics(agent_metrics)
            usage_collector.collect(agent_metrics)

        agent.start(ctx.room, participant)

        # The agent should be polite and greet the user when it joins :)
        await agent.say("Hey, how can I help you today?", allow_interruptions=True)

    except Exception as e:
        logger.error(f"Error in entrypoint: {e}")
        raise


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
