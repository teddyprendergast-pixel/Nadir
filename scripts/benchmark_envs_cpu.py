"""
===============================================================================
Nadir Robot — CPU Simulation Speed Benchmark (Steps Per Second)
===============================================================================
What this script does:
  This script tests your computer's hardware to find the fastest training settings.
  It runs short trial runs with different numbers of simulated robot clones
  (e.g., 128 vs 256 robots) and measures how many "Steps Per Second" (SPS)
  your CPU processor can achieve.

Why this matters for non-coders / beginners:
  - "SPS" (Steps Per Second) is like your simulation speedometer!
    If your SPS is 2,000, that means your computer can simulate 2,000 robot
    walking decisions every second.
    At that speed, 1,000,000 steps of practice takes just over 8 minutes!
  - If you simulate too few robots, you waste available CPU horsepower.
  - If you simulate too many, your computer runs out of fast cache memory
    and slows down.
  - This benchmark finds the "Goldilocks" sweet spot for your specific machine!
===============================================================================
"""

import subprocess
import re
import sys
import os
import time
import argparse

# Try importing psutil to monitor CPU percentage during the test
try:
    import psutil
except ImportError:
    psutil = None


def run_benchmark(num_devices: int = 16, env_counts: list = None):
    """
    Runs quick trial tests across different robot clone counts (environments)
    and prints a neat speed comparison leaderboard.
    """
    if env_counts is None:
        env_counts = [128, 256, 512]
    results = {}
    cpu_results = {}
    peak_cpu_results = {}
    
    print("=" * 70)
    print(f"CPU-OPTIMIZED MULTI-DEVICE BENCHMARK ({num_devices} Virtual CPU Devices)")
    print("=" * 70)
    print("This tests how fast your computer simulates robots across different counts.")
    print("Note: The 1st update is skipped from averages because JAX spends that time")
    print("compiling machine code (Just-In-Time compilation).")
    print("-" * 70)
    
    for num_envs in env_counts:
        # Check that environment count can be evenly divided across virtual devices
        if num_envs % num_devices != 0:
            print(f"Skipping num_envs = {num_envs} (must be evenly divisible by {num_devices} devices)")
            continue
            
        print(f"\nTesting {num_envs} parallel robots ({num_envs // num_devices} per CPU worker)...")
        
        # We run 3 updates total:
        # Update 1 = JIT machine-code compilation (slow)
        # Updates 2 & 3 = Steady-state maximum speed (real performance)
        num_steps = 24
        total_timesteps = 3 * num_steps * num_envs
        
        # Configure environment variables for child Python process
        env_vars = os.environ.copy()
        env_vars["PYTHONUNBUFFERED"] = "1"
        env_vars["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_devices} " + env_vars.get("XLA_FLAGS", "")
        
        # Ensure we invoke the virtual environment's Python executable (.venv)
        python_exe = sys.executable
        if ".venv" not in python_exe:
            python_exe = os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe")
            if not os.path.exists(python_exe):
                python_exe = "python"
                
        # Command to run a mini training session
        cmd = [
            python_exe,
            "-u",
            "-m",
            "nadir.training.train_cpu",
            "--num-devices", str(num_devices),
            "--num-envs", str(num_envs),
            "--num-minibatches", "4",
            "--solver", "CG",
            "--iterations", "12",
            "--total-timesteps", str(total_timesteps),
            "--log-interval", "1"
        ]
        
        # Launch the test in the background
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env_vars
        )
        
        sps_list = []
        cpu_samples = []
        stop_sampling = False
        
        # Background thread to record how much CPU power is being used (e.g. 85% vs 99%)
        def sample_cpu():
            if not psutil:
                return
            while not stop_sampling and process.poll() is None:
                try:
                    c = psutil.cpu_percent(interval=0.3)
                    if c > 2.0:
                        cpu_samples.append(c)
                except Exception:
                    break
        
        import threading
        sampler_thread = threading.Thread(target=sample_cpu, daemon=True)
        sampler_thread.start()
        
        # Read the training log outputs line by line and extract the SPS speed numbers
        for line in process.stdout:
            match = re.search(r"SPS:\s*(\d+)", line)
            if match:
                sps_val = float(match.group(1))
                sps_list.append(sps_val)
                print(f"  -> Update {len(sps_list)}: Speed = {sps_val:,.0f} steps/second")
                
        stop_sampling = True
        sampler_thread.join(timeout=1.0)
        process.wait()
        
        # Check if process exited normally
        if process.returncode != 0:
            print(f"Error encountered while testing {num_envs} envs. Exit code: {process.returncode}")
            results[num_envs] = 0
            cpu_results[num_envs] = 0
            peak_cpu_results[num_envs] = 0
            continue
            
        # Compute average and peak CPU usage
        avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
        peak_cpu = max(cpu_samples) if cpu_samples else 0.0
        cpu_results[num_envs] = avg_cpu
        peak_cpu_results[num_envs] = peak_cpu
        
        # Compute average SPS speed (excluding the 1st compilation update)
        if len(sps_list) > 1:
            avg_sps = sum(sps_list[1:]) / len(sps_list[1:])
            results[num_envs] = avg_sps
            print(f"  -> Average Steady-State Speed: {avg_sps:,.0f} SPS | Avg CPU: {avg_cpu:.1f}% | Peak CPU: {peak_cpu:.1f}%")
        elif len(sps_list) == 1:
            results[num_envs] = sps_list[0]
            print(f"  -> Speed: {sps_list[0]:,.0f} SPS (only 1 update) | Avg CPU: {avg_cpu:.1f}% | Peak CPU: {peak_cpu:.1f}%")
        else:
            print("  -> Could not parse SPS from output.")
            results[num_envs] = 0

    # -----------------------------------------------------------------------
    # Final Benchmark Leaderboard
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"FINAL BENCHMARK LEADERBOARD ({num_devices} Virtual CPU Devices)")
    print("=" * 70)
    print(f"{'Environments':<14} | {'Speed (SPS)':<18} | {'Avg CPU':<10} | {'Peak CPU':<10}")
    print("-" * 70)
    
    best_envs = None
    best_sps = -1
    
    for num_envs, sps in results.items():
        avg_str = f"{cpu_results.get(num_envs, 0):.1f}%"
        peak_str = f"{peak_cpu_results.get(num_envs, 0):.1f}%"
        print(f"{num_envs:<14} | {sps:<18,.0f} | {avg_str:<10} | {peak_str:<10}")
        if sps > best_sps:
            best_sps = sps
            best_envs = num_envs
            
    print("=" * 70)
    if best_envs:
        print(f"\n[RECOMMENDATION] Best setting for your computer: --num-envs {best_envs}")
        print(f"Achieved maximum throughput of {best_sps:,.0f} SPS with {peak_cpu_results.get(best_envs, 0):.1f}% peak CPU usage.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark CPU multi-device training speed")
    parser.add_argument("--num-devices", type=int, default=16,
                        help="Number of virtual CPU devices (default: 16)")
    parser.add_argument("--env-counts", type=int, nargs="+", default=[128, 256],
                        help="List of robot counts to test (e.g. --env-counts 128 256)")
    args = parser.parse_args()
    
    run_benchmark(num_devices=args.num_devices, env_counts=args.env_counts)
