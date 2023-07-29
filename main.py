import requests
import json
import easygui

title = "Your StackOverFlow Question"
message = "Please enter your question:"
question = easygui.enterbox(message, title)

URL = "http://localhost:5000/predict_tags/" + question

response = requests.get(URL, headers={'Content-Type': 'application/json'})
print(response.text)

