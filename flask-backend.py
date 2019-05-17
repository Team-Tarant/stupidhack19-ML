from flask import Flask, jsonify, request

import torch
from pytorch_pretrained_bert import GPT2Tokenizer
from pytorch_pretrained_bert import GPT2LMHeadModel

import numpy as np

import os
from flask_cors import CORS, cross_origin
from io import BytesIO
import time

app = Flask(__name__)
CORS(app)

def storify_preds(predictions):
  story = []
  for i in range(predictions.shape[1]):
    story.append(tokenizer.decode([torch.argmax(predictions[0, i, :]).item()]))
  return "".join(story)

def predict(data_aids):
  global tokenizer, model
  # TODO use input string
  print(data_aids)
  text_1 = data_aids
  text_2 = "Do not tell about my mum"
  indexed_tokens_1 = tokenizer.encode(text_1)
  indexed_tokens_2 = tokenizer.encode(text_2)

  # Convert inputs to PyTorch tensors
  tokens_tensor_1 = torch.tensor([indexed_tokens_1])
  tokens_tensor_2 = torch.tensor([indexed_tokens_2])

  # If you have a GPU, put everything on cuda
  tokens_tensor_1 = tokens_tensor_1 #.to('cuda')
  tokens_tensor_2 = tokens_tensor_2 #.to('cuda')
  #model.to('cuda')

  # TODO return story, not single word
  # Predict all tokens
  with torch.no_grad():
      predictions_1, past = model(tokens_tensor_1)
      # past can be used to reuse precomputed hidden state in a subsequent predictions
      # (see beam-search examples in the run_gpt2.py example).
      predictions_2, past = model(tokens_tensor_2, past=past)

  # get the predicted last token
  predicted_index = torch.argmax(predictions_2[0, -1, :]).item()
  predicted_token = tokenizer.decode([predicted_index])

  return storify_preds(predictions_1)

@app.route('/ping')
def ping():
  print('someone is pinging')
  return 'pong'

@app.route('/get-story', methods=["POST"])
def get_story():
  data = request.data.decode("utf-8")
  print(data)
  return predict(data)


if __name__ == '__main__':
  global tokenizer
  global model
  #TODO make sure to use 345M model
  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  model = GPT2LMHeadModel.from_pretrained('gpt2')

  app.run("0.0.0.0", port=5000, debug=True)