import subprocess
import re
import sys
import os
import time
import argparse

try:
    import psutil
except ImportError:
    psutil = None

def run_benchmark(num_devices: int = 16, env_counts: list = None):
    if env_counts is None:
        env_counts = [128, 256, 512]
    results = {}
    cpu_results = {}
    peak_cpu_results = {}
    
    print("=" * 70)
    print(f"CPU-OPTIMIZED MULTI-DEVICE BENCHMARK ({num_devices} CPU Devices via pmap)")
    print("=" * 70)
    print("This tests different environment counts across CPU cores.")
    print("The first update is ignored because it includes JAX compilation time.")
    print("-" * 70)
    
    for num_envs in env_counts:
        if num_envs % num_devices != 0:
            print(f"Skipping num_envs = {num_envs} (not divisible by {num_devices} devices)")
            continue
            
        print(f"\nTesting num_envs = {num_envs} ({num_envs // num_devices} per device)...")
        
        # 3 updates total (1 compilation + 2 steady state)
        num_steps = 24
        total_timesteps = 3 * num_steps * num_envs
        
        env_vars = os.environ.copy()
        env_vars["PYTHONUNBUFFERED"] = "1"
        env_vars["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_devices} " + env_vars.get("XLA_FLAGS", "")
        
        # Path to python executable in .venv
        python_exe = sys.executable
        if ".venv" not in python_exe:
            python_exe = os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe")
            if not os.path.exists(python_exe):
                python_exe = "python"
                
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
        
        # Background thread to sample system-wide CPU utilization continuously
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
        
        # Read stdout line by line
        for line in process.stdout:
            match = re.search(r"SPS:\s*(\d+)", line)
            if match:
                sps_val = float(match.group(1))
                sps_list.append(sps_val)
                print(f"  -> Update {len(sps_list)}: SPS = {sps_val:,.0f}")
                
        stop_sampling = True
        sampler_thread.join(timeout=1.0)
        process.wait()
        
        if process.returncode != 0:
            print(f"Error running with {num_envs} envs. Exit code: {process.returncode}")
            results[num_envs] = 0
            cpu_results[num_envs] = 0
            peak_cpu_results[num_envs] = 0
            continue
            
        avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
        peak_cpu = max(cpu_samples) if cpu_samples else 0.0
        cpu_results[num_envs] = avg_cpu
        peak_cpu_results[num_envs] = peak_cpu
        
        if len(sps_list) > 1:
            avg_sps = sum(sps_list[1:]) / len(sps_list[1:])
            results[num_envs] = avg_sps
            print(f"  -> Average SPS (excluding compilation): {avg_sps:,.0f} | Avg CPU: {avg_cpu:.1f}% | Peak CPU: {peak_cpu:.1f}%")
        elif len(sps_list) == 1:
            results[num_envs] = sps_list[0]
            print(f"  -> SPS: {sps_list[0]:,.0f} (only 1 update) | Avg CPU: {avg_cpu:.1f}% | Peak CPU: {peak_cpu:.1f}%")
        else:
            print("  -> Could not parse SPS from output.")
            results[num_envs] = 0

    print("\n" + "=" * 70)
    print(f"BENCHMARK RESULTS ({num_devices} CPU Devices via pmap)")
    print("=" * 70)
    print(f"{'Environments':<14} | {'SPS (Throughput)':<18} | {'Avg CPU':<10} | {'Peak CPU':<10}")
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
        print(f"Optimal configuration for CPU: --num-envs {best_envs} --num-devices {num_devices}")
        print(f"Achieved maximum throughput of {best_sps:,.0f} SPS with {peak_cpu_results.get(best_envs, 0):.1f}% peak CPU utilization.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark CPU multi-device training")
    parser.add_argument("--num-devices", type=int, default=16, help="Number of virtual CPU devices (default: 16)")
    parser.add_argument("--env-counts", type=int, nargs="+", default=[128, 256], help="Environment counts to test (default: 128 256)")
    args = parser.parse_args()
    run_benchmark(num_devices=args.num_devices, env_counts=args.env_counts)
