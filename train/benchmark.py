import time
import numpy as np
from tank_env import TankEnv

def run_benchmark(steps=10000):
    print(f"Starting benchmark for {steps} steps...")
    env = TankEnv(arena_mode="open", opponent="stationary", spawn_powerups=False, seed=42)
    env.reset()
    
    # 预热管道 (规避 Node JIT 编译和子进程启动的开销)
    for _ in range(100):
        action = int(np.random.randint(0, env.action_space.n))
        env.step(action)
        
    start_time = time.perf_counter()
    
    terminations = 0
    for _ in range(steps):
        action = int(np.random.randint(0, env.action_space.n))
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            terminations += 1
            env.reset()
            
    end_time = time.perf_counter()
    env.close()
    
    duration = end_time - start_time
    fps = steps / duration
    print("-" * 30)
    print(f"Execution Time: {duration:.2f} seconds")
    print(f"Total Episodes: {terminations}")
    print(f"Throughput:     {fps:.0f} FPS (Steps/sec)")
    print("-" * 30)

if __name__ == "__main__":
    run_benchmark(10000)
