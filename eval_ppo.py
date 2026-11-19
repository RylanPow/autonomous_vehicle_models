# eval_ppo.py  — run this to evaluate a trained checkpoint; never trains
from ppo_policy import PPOPolicy          # imports the class, does NOT run train_ppo
from simulation import run_benchmark

if __name__ == "__main__":
    policy = PPOPolicy("checkpoints/ppo_metadrive.pt")   # loads weights, no training
    run_benchmark(policy, suite_name="quick_dev", render=True)