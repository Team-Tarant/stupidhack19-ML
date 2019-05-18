from flask import Flask, jsonify, request

import torch
from pytorch_pretrained_bert import GPT2Tokenizer
from pytorch_pretrained_bert import GPT2LMHeadModel
import numpy as np

import os
from flask_cors import CORS, cross_origin
from io import BytesIO
import time
import argparse
import logging
from tqdm import trange

import torch
import torch.nn.functional as F
import numpy as np

app = Flask(__name__)
CORS(app)



def predict(data_aids):
  global tokenizer, model
  # TODO use input string
  if tokenizer is None:
    return 'VENAA NYT VITTU'
  text = run_model(data_aids)
  return text

@app.route('/ping')
def ping():
  print('someone is pinging')
  return 'pong'

@app.route('/get-story', methods=["POST"])
def get_story():
  data = request.data.decode("utf-8")
  print(data)
  return predict(data)


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def top_k_logits(logits, k):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)

def sample_sequence(model, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, device='cuda', sample=True):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
    prev = context
    output = context
    past = None
    with torch.no_grad():
        for i in trange(length):
            logits, past = model(prev, past=past)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            output = torch.cat((output, prev), dim=1)
    return output

def run_model(text):
    
  context_tokens = []
  context_tokens = tokenizer.encode(text)
  generated = 0
  for _ in range(args.nsamples // args.batch_size):
      out = sample_sequence(
          model=model, length=args.length,
          context=context_tokens,
          start_token=None,
          batch_size=args.batch_size,
          temperature=args.temperature, top_k=args.top_k, device=device
      )
      out = out[:, len(context_tokens):].tolist()
      for i in range(args.batch_size):
          generated += 1
          text = tokenizer.decode(out[i])
          print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
          print(text)
      print("=" * 80)
  return text

    


if __name__ == '__main__':
  global tokenizer
  global model
  #TODO make sure to use 345M model
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_name_or_path', type=str, default='gpt2', help='pretrained model name or path to local checkpoint')
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--nsamples", type=int, default=1)
  parser.add_argument("--batch_size", type=int, default=-1)
  parser.add_argument("--length", type=int, default=-1)
  parser.add_argument("--temperature", type=float, default=1.0)
  parser.add_argument("--top_k", type=int, default=5)
  parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
  args = parser.parse_args()
  print(args)
  if args.batch_size == -1:
      args.batch_size = 1
  assert args.nsamples % args.batch_size == 0

  np.random.seed(args.seed)
  torch.random.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
  model = GPT2LMHeadModel.from_pretrained("gpt2")
  model.to(device)

  if args.length == -1:
      args.length = model.config.n_ctx // 2
  elif args.length > model.config.n_ctx:
      raise ValueError("Can't get samples longer than window size: %s" % model.config.n_ctx)

  app.run("0.0.0.0", port=5000, debug=True)