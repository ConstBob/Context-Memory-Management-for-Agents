# Packages
import discord
import datetime
import os
from dotenv import load_dotenv
load_dotenv()

from agent import get_agent_message

# Initialize the bot
TOKEN =  os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True

bot = discord.Client(intents=intents)

# Bot Behaviors
@bot.event
async def on_ready():
    print(f"---Connected Successfully to the server---")

@bot.event
async def on_message(message: discord.Message):

    if message.author == bot.user: # Ignore the message from bot itself 
        return

    if message.channel.name.lower() == "general":
        if bot.user.mentioned_in(message): # Check if the bot was mentioned
            sender = message.author
            content = message.content.replace(bot.user.mention, "").strip()
            time_sent = message.created_at

            print(f"[{time_sent}] {sender.name}: {content}")
            reply_text = get_agent_message(sender.name, content, time_sent)

            await message.channel.send(
                f"{sender.mention} {reply_text}"
            )




# Start the bot
bot.run(TOKEN)