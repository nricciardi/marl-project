import os
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from pettingzoo.butterfly import cooperative_pong_v5
import supersuit as ss
from pprint import pprint
import json



BASE_DIR = "./checkpoints"


def env_creator(config=None, render_mode="rgb_array"):
    env = cooperative_pong_v5.env(render_mode=render_mode)
    
    # 1. Ridimensiona a 84x84 (Standard Atari)
    env = ss.resize_v1(env, x_size=84, y_size=84)
    
    # 2. Scala di grigi
    # Questo trasforma l'immagine da (84, 84, 3) a (84, 84) -> QUI nasceva l'errore 2D
    env = ss.color_reduction_v0(env, mode='full')
    
    # 3. FIX CRUCIALE: Frame Stacking
    # Prende gli ultimi 4 frame 2D e li impila insieme.
    # Nuova forma: (84, 84, 4).
    # RLlib ora lo riconosce come un input valido per la CNN (VisionNetwork).
    env = ss.frame_stack_v1(env, 4)
    
    # 4. Normalizza (0-255 -> 0.0-1.0)
    env = ss.dtype_v0(env, "float32")
    
    return PettingZooEnv(env)


ray.init()


# Register the environment
env_name = "coop_pong_v5"
ray.tune.register_env(env_name, env_creator)


# Configure PPO
config = (
    PPOConfig()
    .environment(env_name)
    .framework("torch")
    .resources(
        num_gpus=1,             # GPU per il Learner
        num_cpus_for_main_process=1   # 1 CPU Thread per ogni Worker
    )
    .env_runners(
        # TARIAMO SULLA TUA CPU (16 Thread totali)
        # Usiamo 10 worker paralleli. 
        # Lasciamo spazio al Learner (che usa molta CPU per preparare i dati GPU)
        num_env_runners=5,      
        
        # Vettorializzazione: ogni worker porta avanti 8 partite.
        # 10 workers * 8 envs = 80 partite simultanee -> Raccolta dati velocissima
        num_envs_per_env_runner=8, 
        
        # Raccogliamo "frammenti" di partite senza aspettare che finiscano tutte
        # Questo fluidifica il flusso dei dati
        rollout_fragment_length="auto",
    )
    .multi_agent(
        # Use Parameter Sharing: one policy for both paddle_0 and paddle_1
        policies={"shared_policy"},
        policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
    )
    .training(
        # Un batch size medio-grande (buon compromesso velocità/stabilità)
        train_batch_size=1024, 
        
        # Parametri PPO standard
        lr=1e-4, 
        gamma=0.99,
        lambda_=0.95,
        
        # Meno iterazioni sui dati vecchi = più velocità
        num_epochs=5, 
    )
    .update_from_dict({
        # La GPU NVIDIA lavora meglio con matrici grandi.
        # 2048 o 4096 è ideale per schede moderne.
        "sgd_minibatch_size": 256,
        "model": {
            "vf_share_layers": True,
        }
    })
)

# Build the algorithm
algo = config.build_algo()


last_checkpoint_path = None


print("Starting training...")
# Training loop

results = []
try:
    for i in range(10):
        print(f"=== Iteration {i} ===")
        result = algo.train()
        results.append(result)

        checkpoint_dir = os.path.join(BASE_DIR, str(i))

        os.makedirs(checkpoint_dir, exist_ok=True)

        env_runners_info = result["env_runners"]
        data = {
            "agent_episode_returns_mean": env_runners_info.get("agent_episode_returns_mean"),
            "episode_len_max": env_runners_info.get("episode_len_max"),
            "episode_len_mean": env_runners_info.get("episode_len_mean"),
            "episode_len_min": env_runners_info.get("episode_len_min"),
            "episode_return_max": env_runners_info.get("episode_return_max"),
            "episode_return_mean": env_runners_info.get("episode_return_mean"),
            "episode_return_min": env_runners_info.get("episode_return_min"),
            "module_episode_returns_mean": env_runners_info.get("module_episode_returns_mean"),
        }

        print("Result summary:")
        pprint(data)

        with open(os.path.join(checkpoint_dir, "result.json"), "w") as f:
            json.dump(data, f, indent=4)

        with open(os.path.join(checkpoint_dir, "result.txt"), "w") as f:
            pprint(result, stream=f)
            
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")

        algo.save_to_path(checkpoint_path)
        print(f"Checkpoint saved at: {checkpoint_path}")

        last_checkpoint_path = checkpoint_path
            
except KeyboardInterrupt:
    print("\nTraining stopped by user.")

print("Training completed.")
algo.stop()
ray.shutdown()