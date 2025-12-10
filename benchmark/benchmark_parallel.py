"""
Parallel benchmark - run all tests simultaneously.
Tests both sync and async modes.
"""

import subprocess
import time
import threading
import queue

# Test configurations
test_configs = [
    {'envs': 40, 'async': False},
    {'envs': 80, 'async': False},
    {'envs': 120, 'async': False},
    {'envs': 160, 'async': False},
    {'envs': 40, 'async': True},
    {'envs': 80, 'async': True},
    {'envs': 120, 'async': True},
    {'envs': 160, 'async': True},
]

total_episodes = 160  # Same for all tests
results_queue = queue.Queue()

def run_test(config):
    """Run a single benchmark test."""
    num_envs = config['envs']
    use_async = config['async']
    mode = "async" if use_async else "sync"
    
    iterations = total_episodes // num_envs
    if iterations == 0:
        return
    
    actual_episodes = iterations * num_envs
    
    print(f"[{mode:5s} {num_envs:3d} envs] Starting...")
    
    start_time = time.time()
    
    # Build command
    cmd = [
        "python", "train.py",
        "--num-envs", str(num_envs),
        "--headless",
        "--episodes", str(actual_episodes),
        "--log-every", "999"
    ]
    
    if use_async:
        cmd.append("--async-envs")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=90
        )
        
        elapsed = time.time() - start_time
        eps_per_sec = actual_episodes / elapsed if elapsed > 0 else 0
        
        result_data = {
            'num_envs': num_envs,
            'mode': mode,
            'episodes': actual_episodes,
            'time': elapsed,
            'eps_per_sec': eps_per_sec,
            'success': True
        }
        
        print(f"[{mode:5s} {num_envs:3d} envs] ✓ {elapsed:.1f}s → {eps_per_sec:.2f} eps/s")
        
    except subprocess.TimeoutExpired:
        result_data = {
            'num_envs': num_envs,
            'mode': mode,
            'success': False
        }
        print(f"[{mode:5s} {num_envs:3d} envs] ✗ Timeout")
    except Exception as e:
        result_data = {
            'num_envs': num_envs,
            'mode': mode,
            'success': False
        }
        print(f"[{mode:5s} {num_envs:3d} envs] ✗ Error: {e}")
    
    results_queue.put(result_data)

# Start all tests in parallel
print("=" * 70)
print(f"Running {len(test_configs)} benchmark tests in PARALLEL")
print(f"Each test: {total_episodes} episodes")
print("=" * 70)
print()

overall_start = time.time()
threads = []

for config in test_configs:
    thread = threading.Thread(target=run_test, args=(config,))
    thread.start()
    threads.append(thread)
    time.sleep(0.5)  # Stagger starts slightly

# Wait for all threads to complete
for thread in threads:
    thread.join()

overall_elapsed = time.time() - overall_start

# Collect results
results = []
while not results_queue.empty():
    results.append(results_queue.get())

# Sort and display results
results = [r for r in results if r.get('success', False)]
results.sort(key=lambda x: (x['mode'], x['num_envs']))

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"Total benchmark time: {overall_elapsed:.1f}s (all tests ran in parallel)")
print()
print(f"{'Mode':<8} {'Envs':<8} {'Episodes':<12} {'Time (s)':<12} {'Speed (eps/s)':<15} {'Winner'}")
print("-" * 70)

# Find best for each mode
sync_results = [r for r in results if r['mode'] == 'sync']
async_results = [r for r in results if r['mode'] == 'async']

best_sync = max(sync_results, key=lambda x: x['eps_per_sec']) if sync_results else None
best_async = max(async_results, key=lambda x: x['eps_per_sec']) if async_results else None
best_overall = max(results, key=lambda x: x['eps_per_sec']) if results else None

for r in results:
    winner = ""
    if r == best_overall:
        winner = " 🏆 BEST OVERALL"
    elif r == best_sync and r['mode'] == 'sync':
        winner = " ⭐ Best Sync"
    elif r == best_async and r['mode'] == 'async':
        winner = " ⭐ Best Async"
    
    print(f"{r['mode']:<8} {r['num_envs']:<8} {r['episodes']:<12} {r['time']:<12.2f} {r['eps_per_sec']:<15.2f} {winner}")

print("=" * 70)

if best_overall:
    print(f"\n🏆 OPTIMAL CONFIGURATION:")
    print(f"   Mode: {best_overall['mode'].upper()}")
    print(f"   Environments: {best_overall['num_envs']}")
    print(f"   Speed: {best_overall['eps_per_sec']:.2f} episodes/sec")
    
    async_flag = " --async-envs" if best_overall['mode'] == 'async' else ""
    print(f"\n💡 Recommended command:")
    print(f"   python train.py --num-envs {best_overall['num_envs']}{async_flag} --episodes 2500 --headless")
