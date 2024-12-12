from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time
import torch

# Load pre-trained model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Input text
prompts = [f"This is test prompt number {i}" for i in range(1, 10)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Record response times
response_times = []
results = []

for prompt in prompts:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Warm-up pass for the first prompt
    if len(response_times) == 0:
        _ = model.generate(input_ids, max_length=50)

    # Measure inference time
    start_time = time.time()
    output = model.generate(
        input_ids,
        max_length=50,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        do_sample=True,
    )
    end_time = time.time()

    # Record time and generated text
    response_time = end_time - start_time
    response_times.append(response_time)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    results.append({"prompt": prompt, "response_time": response_time, "generated_text": generated_text})

# Output average and detailed results
average_time = sum(response_times) / len(response_times)
print(f"Average Inference Time: {average_time:.4f} seconds")
