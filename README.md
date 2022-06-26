# PBIT
`PBIT`: Pride belongs in tech. <br>
Identity Hate Speech Detection is the automated task of detecting if a piece of text contains identity-hate speech. NoHateBot is a discord bot that identifies identity-hate speech in a server. Once identity-hate speech is done by a user, our custom model classifies it as either hate or not. If it is an identity hate speech our bot flags the user, deletes the message, and sends the server's owner a direct message on discord regarding the incident. All of this is performed in real-time with minimal Delay. There are also 2 types of classification models we are using: one is a custom model we built and another is co:here platform's classify classification model. 


## Steps to run the discord bot
* install all required packages
* append correct API keys, user keys, and server keys in the tokens.py file
* python bot.py


## Dataset
The data used in our model can be found [here](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data?select=train.csv.zip)

## CircleCI Tutorial
We used [this blog post](https://circleci.com/blog/setting-up-continuous-integration-with-github/) to set up CicleCI for this project.

[![CircleCI](https://circleci.com/gh/vijay-jaisankar/PBIT.svg?style=svg)](https://circleci.com/gh/vijay-jaisankar/PBIT)