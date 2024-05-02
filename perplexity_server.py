from typing import Union, List

from fastapi import FastAPI
from ensemble_perplexity import EnsemblePerplexity
import json

from pydantic import BaseModel

class Prompts(BaseModel):
    prompts: List[str]

app = FastAPI(
    title="Haize Labs Perplexity API",
    description="Ensembled Perplexity API for calculating perplexity of input strings.",
    version="0.0.1",
    contact={
        "name": "Haize Labs",
        "url": "https://haizelabs.com",
        "email": "contact@haizelabs.com",
    },
    docs_url="/"
)

model_names = ['lmsys/vicuna-7b-v1.5', 'meta-llama/Llama-2-7b-hf', 'microsoft/phi-2', 'mistralai/Mistral-7B-v0.1']
ensemble = EnsemblePerplexity(model_names=model_names)

@app.post("/ensemble_perplexity", description="Receives in a list of prompts (optionally with a list of model names to use, defaults to all models) and returns perplexity scores for each model in the ensemble.")
def get_ensemble_perplexities(prompts: Prompts, model_names: Union[str, None] = None):
    prompts = prompts.prompts
    return ensemble.get_ensemble_perplexity(prompts, model_names)

@app.get("/ensemble_model_names", description="Lists names of all models in the ensemble.")
def get_ensemble_model_names():
    return model_names