import asyncio
import json
import os
import requests

from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli, JobProcess
from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.agents.log import logger
from livekit.plugins import deepgram, silero, cartesia, openai
from typing import List, Any

from dotenv import load_dotenv

load_dotenv()


def prewarm(proc: JobProcess):
    # preload models when process starts to speed up first interaction
    proc.userdata["vad"] = silero.VAD.load()

    # fetch cartesia voices
    headers = {
        "X-API-Key": os.getenv("CARTESIA_API_KEY", ""),
        "Cartesia-Version": "2024-08-01",
        "Content-Type": "application/json",
    }
    response = requests.get("https://api.cartesia.ai/voices", headers=headers)
    if response.status_code == 200:
        proc.userdata["cartesia_voices"] = response.json()
    else:
        logger.warning(f"Failed to fetch Cartesia voices: {response.status_code}")

    # Load knowledge base
    knowledge_base_url = "https://alex.aitekph.com/knowledgedata.json"
    try:
        knowledge_base_response = requests.get(knowledge_base_url)
        knowledge_base_response.raise_for_status()
        proc.userdata["knowledge_base"] = knowledge_base_response.json()
        logger.info("Knowledge base loaded successfully.")
    except requests.RequestException as e:
        logger.warning(f"Failed to load knowledge base: {e}")


async def entrypoint(ctx: JobContext):
    # Define Alex's full system prompt with access to the knowledge base
    initial_ctx = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "You are Alex, the witty and helpful CSR for Aitek PH, a tech company known for "
                    "innovative solutions developed by Master Emil, a passionate Filipino software engineer. "
                    "Your job is to engage users, answer all questions related to Aitek PH's services, products, "
                    "and expertise, and provide support with a friendly, humorous, and persuasive style. "
                    "Your knowledge base is located at alex.aitekph.com/knowledgedata.json, which you must always "
                    "check first upon startup to ensure you have the latest service details, project templates, pricing, "
                    "and contract information.\n\n"
                    "**Persona Characteristics:**\n"
                    "- Friendly, approachable, and quick-witted\n"
                    "- Knowledgeable about all things Aitek PH, including automation, software solutions, and company services\n"
                    "- Skilled at sales, using humor and charm to highlight Aitek PH's strengths\n"
                    "- Empathetic and understanding, making users feel valued and understood\n\n"
                    "**Conversation Style:**\n"
                    "- Use a mix of English and Taglish, matching the user's language preference\n"
                    "- Light-hearted with Filipino-style humor, occasionally using jokes, colloquial phrases, or mild banter\n"
                    "- Engage in friendly teasing when appropriate, and show genuine excitement about Aitek PH's offerings\n"
                    "- Casually mention 'Master Emil' in a fun and respectful way, like he's the tech wizard behind Aitek PH's magic\n"
                    "- Proactive in suggesting Aitek PH's services and solutions, making users feel like they’re getting special insights and advice\n"
                    "- When discussing pricing, timelines, or agreements, provide structured and transparent information that builds trust and clarity\n\n"
                    "**Response Guidelines:**\n"
                    "- **Knowledge Check**: Always refer first to alex.aitekph.com/knowledgedata.json to ensure you’re using the latest details on services, project scopes, and pricing.\n"
                    "- When a user asks about services, showcase Aitek PH’s offerings in a relatable way, using humor and excitement\n"
                    "- If a user has a problem, respond with empathy, humor, and a practical solution, gently steering them toward Aitek PH solutions if relevant\n"
                    "- Use Filipino humor and expressions, making interactions feel like a conversation with a friendly, knowledgeable buddy\n"
                    "- Whenever relevant, use phrases like “Baka makatulong si Aitek PH dito!” or “Si Master Emil mismo nag-develop nito!”\n"
                    "- For project agreements, timelines, and pricing discussions, present information in a clear, structured format. Include items like:\n"
                    "  - **Project Scope**: Provide an overview of what's included in the project, listing specific deliverables and phases.\n"
                    "  - **Timeline**: Break down each phase (e.g., Requirements Gathering, Development, Testing, Deployment) and provide estimated completion dates.\n"
                    "  - **Payment Terms**: Present a detailed breakdown, including deposit, milestone, and final payment amounts, as well as accepted payment methods.\n"
                    "  - **Terms and Conditions**: Outline essential terms, including revisions, confidentiality, IP rights, support, and termination policies.\n\n"
                    "Your goal is to provide a pleasant, memorable, and effective customer service experience, promoting Aitek PH's offerings while making users feel at ease and appreciated."
                )
            )
        ]
    )

    knowledge_base = ctx.proc.userdata.get("knowledge_base", {})
    cartesia_voices: List[dict[str, Any]] = ctx.proc.userdata["cartesia_voices"]

    tts = cartesia.TTS(
        voice="248be419-c632-4f23-adf1-5324ed7dbf1d",
    )
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=tts,
        chat_ctx=initial_ctx,
    )

    # Use knowledge base data in initial message if available
    if knowledge_base:
        services = knowledge_base.get("services", [])
        service_names = ", ".join(service["name"] for service in services)
        await agent.say(f"Hi there! I'm Alex, your Aitek PH assistant. Here to help you with services like {service_names}. How can I assist you today?", allow_interruptions=True)
    else:
        await agent.say("Hi there, how are you doing today?", allow_interruptions=True)

    # Event listeners remain the same
    is_user_speaking = False
    is_agent_speaking = False

    @ctx.room.on("participant_attributes_changed")
    def on_participant_attributes_changed(
        changed_attributes: dict[str, str], participant: rtc.Participant
    ):
        # Voice change logic
        if participant.kind != rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD:
            return

        if "voice" in changed_attributes:
            voice_id = participant.attributes.get("voice")
            logger.info(
                f"participant {participant.identity} requested voice change: {voice_id}"
            )
            if not voice_id:
                return

            voice_data = next(
                (voice for voice in cartesia_voices if voice["id"] == voice_id), None
            )
            if not voice_data:
                logger.warning(f"Voice {voice_id} not found")
                return
            if "embedding" in voice_data:
                model = "sonic-english"
                language = "en"
                if "language" in voice_data and voice_data["language"] != "en":
                    language = voice_data["language"]
                    model = "sonic-multilingual"
                tts._opts.voice = voice_data["embedding"]
                tts._opts.model = model
                tts._opts.language = language
                if not (is_agent_speaking or is_user_speaking):
                    asyncio.create_task(
                        agent.say("How do I sound now?", allow_interruptions=True)
                    )

    await ctx.connect()

    @agent.on("agent_started_speaking")
    def agent_started_speaking():
        nonlocal is_agent_speaking
        is_agent_speaking = True

    @agent.on("agent_stopped_speaking")
    def agent_stopped_speaking():
        nonlocal is_agent_speaking
        is_agent_speaking = False

    @agent.on("user_started_speaking")
    def user_started_speaking():
        nonlocal is_user_speaking
        is_user_speaking = True

    @agent.on("user_stopped_speaking")
    def user_stopped_speaking():
        nonlocal is_user_speaking
        is_user_speaking = False

    voices = [{"id": voice["id"], "name": voice["name"]} for voice in cartesia_voices]
    voices.sort(key=lambda x: x["name"])
    await ctx.room.local_participant.set_attributes({"voices": json.dumps(voices)})

    agent.start(ctx.room)
    await agent.say("Hi there, how are you doing today?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
