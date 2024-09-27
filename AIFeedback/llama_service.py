import os

import torch
from fastapi import FastAPI
from huggingface_hub import login
from pydantic import BaseModel
from transformers import LlamaTokenizer, LlamaForCausalLM

hf_token = os.getenv('HF_TOKEN')

login(token=hf_token)
model_name = 'openlm-research/open_llama_3b'

tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto",
)
app = FastAPI()


class Prompt(BaseModel):
    text: str
    max_length: int = 50
    num_return_sequences: int = 1


@app.post("/generate")
async def generate_text(prompt: Prompt):
    input_ids = tokenizer(prompt.text, return_tensors="pt").input_ids.to("cuda")

    generation_output = model.generate(
        input_ids=input_ids, max_new_tokens=32
    )
    return {"generated_texts": tokenizer.decode(generation_output[0])}
