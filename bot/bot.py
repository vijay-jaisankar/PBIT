import discord
from tokens import DISCORD_BOT_TOKEN
import pandas as pd
import pickle 

"""
    Load model and train/test data
"""
model = None 
with open("lr.pkl", "rb") as f:
    model = pickle.load(f)

test_train = None
with open("test_train_data.pkl", "rb") as f2:
    test_train = pickle.load(f2)

# print(model)
# print(test_train)

"""
    Model - Submission Pipeline
"""
class SubmissionPipeline:
    def __init__(self, testDf, model,testTrainData):
        self.testDf = testDf
        self.model = model
        self.getTTData = testTrainData

    def run(self):
        self.predictions = self.model.predict(self.getTTData.get_X_test_custom(self.testDf))
        self.submission_df = pd.DataFrame({"target": self.predictions})
        return (self.submission_df)

"""
    Get Label of message
    Legend:
        - 0: Safe
        - 1: Flag
"""
def get_label(s):
    check_dict = {"comment_text" : [s]}
    check_df = pd.DataFrame(check_dict)
    submissionPipeline = SubmissionPipeline(check_df, model, test_train)
    label =  submissionPipeline.run()["target"][0]
    return label 


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

    # Get the label
    label = get_label(user_message)

    print(f"{username} said: {user_message} in {channel}")
    await message.channel.send(f"Hello, {username}! Label: {label}")



"""
    Running the app
"""
client.run(token)