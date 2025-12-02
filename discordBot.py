# Packages
import discord
import datetime
import os
from dotenv import load_dotenv
load_dotenv()

from agent import get_agent_message
from memory_manager import HybridMemoryManager

# Initialize the bot
TOKEN =  os.getenv("DISCORD_TOKEN")
USE_RAG = os.getenv("USE_RAG", "false").lower() == "true"

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True

bot = discord.Client(intents=intents)

# Initialize memory manager (shared across all requests)
memory_manager = HybridMemoryManager(db_path="data/users.db")

# Initialize RAG system if enabled
rag_system = None
if USE_RAG:
    try:
        from rag_system import RAGSystem
        from agent_with_rag import get_agent_message_with_rag
        rag_system = RAGSystem(model_name="BAAI/bge-small-en-v1.5")
        print(f"---RAG system initialized with BGE embeddings---")
    except ImportError as e:
        print(f"Warning: RAG system disabled. Install sentence-transformers to enable.")
        print(f"Error: {e}")
        USE_RAG = False

# Bot Behaviors
@bot.event
async def on_ready():
    mode = "RAG-enhanced" if USE_RAG else "standard"
    print(f"---Connected Successfully to the server ({mode} mode)---")

@bot.event
async def on_message(message: discord.Message):

    if message.author == bot.user:
        return

    if message.channel.name.lower() == "general":
        if bot.user.mentioned_in(message):
            sender = message.author
            content = message.content.replace(bot.user.mention, "").strip()
            time_sent = message.created_at

            print(f"[{time_sent}] {sender.name}: {content}")

            if USE_RAG and rag_system:
                reply_text = get_agent_message_with_rag(
                    sender.name, content, time_sent,
                    memory_manager=memory_manager,
                    rag_system=rag_system
                )
            else:
                reply_text = get_agent_message(
                    sender.name, content, time_sent,
                    memory_manager=memory_manager
                )

            await message.channel.send(
                f"{sender.mention} {reply_text}"
            )




# Start the bot
bot.run(TOKEN)