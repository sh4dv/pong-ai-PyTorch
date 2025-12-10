"""
Benchmark GPU usage to diagnose why training isn't using GPU properly.
This simulates the actual training workload.
"""

import torch
import numpy as np
import time
from dqn_model import DQN
from config import INPUT_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2, OUTPUT_SIZE

print("=" * 60)
print("GPU Training Benchmark")
print("=" * 60)

# Setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"\nDevice: {device}")

# Create model (same as training)
model = DQN(INPUT_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2, OUTPUT_SIZE).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = torch.nn.MSELoss()

print(f"Model on device: {next(model.parameters()).device}")

# Simulate training workload
batch_size = 256  # Same as your training
num_iterations = 100

print(f"\nRunning {num_iterations} training iterations with batch_size={batch_size}")
print("Watch Activity Monitor GPU now!\n")

# Warmup
for _ in range(10):
    states = torch.randn(batch_size, INPUT_SIZE).to(device)
    actions = torch.randint(0, OUTPUT_SIZE, (batch_size,)).to(device)
    rewards = torch.randn(batch_size).to(device)
    next_states = torch.randn(batch_size, INPUT_SIZE).to(device)
    dones = torch.randint(0, 2, (batch_size,)).float().to(device)
    
    current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = model(next_states).max(1)[0]
        target_q = rewards + 0.99 * next_q * (1 - dones)
    
    loss = criterion(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

if device.type == "mps":
    torch.mps.synchronize()

print("Warmup done. Starting benchmark...")
print("Check Activity Monitor → Window → GPU History NOW!")
print("-" * 60)

start_time = time.time()

for i in range(num_iterations):
    # Create batch (simulating replay buffer sampling)
    states = torch.randn(batch_size, INPUT_SIZE).to(device)
    actions = torch.randint(0, OUTPUT_SIZE, (batch_size,)).to(device)
    rewards = torch.randn(batch_size).to(device)
    next_states = torch.randn(batch_size, INPUT_SIZE).to(device)
    dones = torch.randint(0, 2, (batch_size,)).float().to(device)
    
    # Forward pass (same as training)
    current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Target Q values
    with torch.no_grad():
        next_q = model(next_states).max(1)[0]
        target_q = rewards + 0.99 * next_q * (1 - dones)
    
    # Backward pass
    loss = criterion(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    if (i + 1) % 20 == 0:
        print(f"Iteration {i + 1}/{num_iterations} - Loss: {loss.item():.4f}")

if device.type == "mps":
    torch.mps.synchronize()

elapsed = time.time() - start_time
iterations_per_sec = num_iterations / elapsed

print("-" * 60)
print(f"\nCompleted {num_iterations} training iterations")
print(f"Time: {elapsed:.2f}s")
print(f"Throughput: {iterations_per_sec:.1f} iterations/sec")
print(f"Per iteration: {elapsed/num_iterations*1000:.2f}ms")

print("\n" + "=" * 60)
print("Expected GPU behavior during this test:")
print("  • GPU frequency should jump to 900-1200 MHz")
print("  • GPU power should be 500-2000 mW")
print("  • GPU active residency should be >50%")
print("\nIf you saw low GPU usage (396 MHz, <100 mW):")
print("  ❌ PyTorch is NOT using GPU for actual computation")
print("  → This could be a PyTorch MPS backend issue")
print("=" * 60)
