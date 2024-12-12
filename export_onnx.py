from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path
import torch


# Load model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Convert to ONNX
input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors="pt", dtype=torch.int64)

output_dir = Path("models/gpt2/1")  # Triton requires "model_name/version" structure
output_dir.mkdir(parents=True, exist_ok=True)
torch.onnx.export(
    model,
    (input_ids,),
    output_dir / "model.onnx",
    input_names=["input_ids"],
    output_names=["output"],
    dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"}},
    opset_version=14,
)
