import discord
from tokens import DISCORD_BOT_TOKEN

"""
    Initialising the discord client
"""
client = discord.Client()
token = DISCORD_BOT_TOKEN


"""
    @Event
    Called with the script is run and the bot is active
"""
@client.event 
async def on_ready():
    print(f"Bot {client.user} is active!")

"""
    @Event
    Called when any message appears on the server

    @Todo: NLP Processing
"""
@client.event
async def on_message(message):
    print(message)
    username = str(message.author).split("#")[0]
    user_message = str(message.content)
    channel = str(message.channel.name)

    # Avoid infinite loop
    if message.author == client.user:
        return 

    print(f"{username} said: {user_message} in {channel}")
    await message.channel.send(f"Hello, {username}!")



"""
    Running the app
"""
client.run(token)