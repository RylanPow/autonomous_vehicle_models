# run this to evaluate a trained checkpoint
from ppo_policy import PPOPolicy, load_config
from simulation import run_benchmark

if __name__ == "__main__":
    config_path = "configs/ppo_config.yaml"
    cfg = load_config(config_path)

    policy = PPOPolicy(config_path=config_path)
    # or to evaluate a specific checkpoint file with that config:
    # policy = PPOPolicy(config_path=config_path, checkpoint_path="checkpoints/ppo_metadrive.pt")

    run_benchmark(
        policy,
        suite_name=cfg["experiment"]["eval_suite"],
        render=bool(cfg["experiment"]["render_eval"]),
    )