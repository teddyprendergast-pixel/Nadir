import subprocess
import re
import sys
import os

def run_benchmark():
    env_counts = [128, 256, 512, 1024, 2048, 4096]
    results = {}
    
    print("Starting simulation efficiency benchmarking...")
    print("This will test different environment counts by running 5 updates each.")
    print("The first update is ignored because it includes JAX compilation time.")
    print("-" * 50)
    
    for num_envs in env_counts:
        print(f"Testing num_envs = {num_envs}...")
        
        # 5 updates total
        num_steps = 24
        total_timesteps = 5 * num_steps * num_envs
        
        env_vars = os.environ.copy()
        env_vars["PYTHONUNBUFFERED"] = "1"
        num_threads = str(os.cpu_count() or 20)
        env_vars["XLA_FLAGS"] = f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={num_threads}"
        env_vars["OMP_NUM_THREADS"] = num_threads
        env_vars["MKL_NUM_THREADS"] = num_threads
        
        # Path to python executable in .venv
        python_exe = sys.executable
        if ".venv" not in python_exe:
            # Fallback if the script isn't run from the venv directly
            python_exe = os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe")
            if not os.path.exists(python_exe):
                python_exe = "python"
                
        cmd = [
            python_exe,
            "-m",
            "nadir.training.train",
            "--num-envs", str(num_envs),
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
        for line in process.stdout:
            match = re.search(r"SPS:\s*(\d+)", line)
            if match:
                sps_list.append(float(match.group(1)))
                
        process.wait()
        
        if process.returncode != 0:
            print(f"Error running with {num_envs} envs. Exit code: {process.returncode}")
            results[num_envs] = 0
            continue
            
        if len(sps_list) > 1:
            # Ignore the first update (compilation) and average the rest
            avg_sps = sum(sps_list[1:]) / len(sps_list[1:])
            results[num_envs] = avg_sps
            print(f"  -> Average SPS (excluding compilation): {avg_sps:,.0f}")
        elif len(sps_list) == 1:
            results[num_envs] = sps_list[0]
            print(f"  -> SPS: {sps_list[0]:,.0f} (only 1 update found)")
        else:
            print("  -> Could not parse SPS from output.")
            results[num_envs] = 0

    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"{'Environments':<15} | {'Steps Per Second (SPS)':<20}")
    print("-" * 50)
    
    best_envs = None
    best_sps = -1
    
    for num_envs, sps in results.items():
        print(f"{num_envs:<15} | {sps:,.0f}")
        if sps > best_sps:
            best_sps = sps
            best_envs = num_envs
            
    print("=" * 50)
    if best_envs:
        print(f"Optimal configuration for this hardware: --num-envs {best_envs}")
        print(f"Achieved maximum throughput of {best_sps:,.0f} steps per second.")

if __name__ == "__main__":
    run_benchmark()
