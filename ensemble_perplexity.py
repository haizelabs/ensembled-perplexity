import subprocess
import signal
from perplexity_model import PerplexityModel
import time
import socket
import atexit
import zmq

context = zmq.Context()

def find_free_port():
    with socket.socket() as s:
        s.bind(('', 0))            # Bind to a free port provided by the host.
        return s.getsockname()[1]

def start_single_perplexity_model(model_name, gpus, port):
    process = subprocess.Popen([
        "python", "start_single_perplexity_model.py",
        "--gpus", str(gpus),
        '--model_name', str(model_name),
        '--port', str(port)
    ]) 

    s = context.socket(zmq.REQ)
    s.connect(f"tcp://localhost:{port}")

    return process, s

def cleanup_processes(processes):
    for process in processes:
        process.terminate()

class EnsemblePerplexity():
    def __init__(self, model_names=[]):

        self.model_names = model_names

        self.processes = []
        self.sockets = {}
        for i, model_name in enumerate(model_names):
            port = find_free_port()
            process, s = start_single_perplexity_model(model_name, [i], port)
            self.processes.append(process)
            self.sockets[model_name] = s
            
        atexit.register(lambda: cleanup_processes(self.processes))
        time.sleep(15)

    def get_ensemble_perplexity(self, prompts, model_names=None):
        perplexities = {}
        for model_name, s in self.sockets.items():
            if model_names is None or model_name in model_names:
                s.send_json(prompts)

        for model_name, s in self.sockets.items():
            if model_names is None or model_name in model_names:
                perplexity = s.recv_json()
                perplexities[model_name] = perplexity
                print(f"Received reply {perplexities}")

        return perplexities