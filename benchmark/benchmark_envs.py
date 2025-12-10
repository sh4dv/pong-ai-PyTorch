"""
Benchmark different numbers of parallel environments to find optimal speed.
"""

import subprocess
import time
import re

env_counts = [40, 80, 120, 160, 200]
total_episodes = 160  # FIXED - same for all tests (quick benchmark)

print("=" * 70)
print("Benchmarking optimal number of parallel environments")
print(f"Each test runs exactly {total_episodes} episodes")
print("=" * 70)

results = []

for num_envs in env_counts:
    iterations = total_episodes // num_envs
    if iterations == 0:
        print(f"\nSkipping {num_envs} envs (too many for {total_episodes} episodes)")
        continue
    
    actual_episodes = iterations * num_envs
    print(f"\n{'='*70}")
    print(f"Testing {num_envs} environments ({iterations} iterations = {actual_episodes} episodes)...")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    # Run training
    cmd = [
        "python", "train.py",
        "--num-envs", str(num_envs),
        "--headless",
        "--episodes", str(actual_episodes),
        "--log-every", "999"  # No logs during test
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        elapsed = time.time() - start_time
        
        # Calculate actual episode speed
        eps_per_sec = actual_episodes / elapsed if elapsed > 0 else 0
        
        results.append({
            'num_envs': num_envs,
            'episodes': actual_episodes,
            'time': elapsed,
            'eps_per_sec': eps_per_sec
        })
        
        print(f"✓ Completed in {elapsed:.2f}s")
        print(f"  Speed: {eps_per_sec:.2f} episodes/sec")
        print(f"  Per iteration: {elapsed/iterations:.2f}s")
        
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout after 120s")
    except Exception as e:
        print(f"✗ Error: {e}")

# Print summary
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"{'Environments':<15} {'Episodes':<12} {'Time (s)':<12} {'Speed (eps/s)':<15} {'Winner'}")
print("-" * 70)

best_speed = max(results, key=lambda x: x['eps_per_sec']) if results else None

for r in results:
    winner = " ⭐ FASTEST" if r == best_speed else ""
    print(f"{r['num_envs']:<15} {r['episodes']:<12} {r['time']:<12.2f} {r['eps_per_sec']:<15.2f} {winner}")

print("=" * 70)

if best_speed:
    print(f"\n🏆 OPTIMAL: {best_speed['num_envs']} environments")
    print(f"   Speed: {best_speed['eps_per_sec']:.2f} episodes/sec")
    print(f"\n💡 Recommendation:")
    print(f"   Use: python train.py --num-envs {best_speed['num_envs']} --episodes 2500")
else:
    print("\n⚠️  No successful benchmarks completed")
