import time
import zmq
import argparse
from perplexity_model import PerplexityModel
import json

parser = argparse.ArgumentParser("Start single perplexity model")
parser.add_argument("--gpus", help="gpus to host on", type=str)
parser.add_argument("--model_name", help="huggingface model name", type=str)
parser.add_argument("--port", help="port to communicate on", type=int)
args = parser.parse_args()

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind(f"tcp://*:{args.port}")

perplexity_model = PerplexityModel(args.model_name, gpus=json.loads(args.gpus))

while True:
    #  Wait for next request from client
    prompts = socket.recv_json()
    print(f"Received request: {prompts}")

    perplexity = perplexity_model.get_perplexity(prompts)

    socket.send_json(perplexity)