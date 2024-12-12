import time
import numpy as np
import tritonclient.http as httpclient

# Triton server URL
triton_url = "localhost:8000"

# Initialize the Triton client
client = httpclient.InferenceServerClient(url=triton_url)

# Prepare input data
input_ids = np.random.randint(0, 50257, size=(1, 10), dtype=np.int32)  # Random test input
inputs = [
    httpclient.InferInput("input_ids", input_ids.shape, "INT32")  # Match input type
]
inputs[0].set_data_from_numpy(input_ids)

# Define outputs
outputs = [
    httpclient.InferRequestedOutput("output")
]

# Warm-up pass to initialize the server
_ = client.infer(model_name="gpt2", inputs=inputs, outputs=outputs)

# Measure response time for multiple runs
response_times = []
num_requests = 100  # Number of requests to send

for _ in range(num_requests):
    start_time = time.time()
    response = client.infer(model_name="gpt2", inputs=inputs, outputs=outputs)
    end_time = time.time()

    response_time = end_time - start_time
    response_times.append(response_time)

# Calculate average response time
average_time = sum(response_times) / len(response_times)
print(f"Average Response Time: {average_time:.4f} seconds")

# Print an example output
output_logits = response.as_numpy("logits")
print(f"Example Output Shape: {output_logits.shape}")
