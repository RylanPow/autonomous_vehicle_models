import os
import time
import argparse
import multiprocessing as mp
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from metadrive.envs import MetaDriveEnv
from simulation import BasePolicy, run_benchmark


def load_config(config_path="configs/ppo_config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ortho_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class SquashedActorCritic(nn.Module):
    """Orthogonally initialized Tanh-squashed Gaussian Actor-Critic."""

    def __init__(self, obs_dim, act_dim, net_cfg):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        hidden_dim = int(net_cfg["hidden_dim"])
        self.log_std_min = float(net_cfg["log_std_min"])
        self.log_std_max = float(net_cfg["log_std_max"])

        self.actor_backbone = nn.Sequential(
            _ortho_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            _ortho_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.mu_head = _ortho_init(nn.Linear(hidden_dim, act_dim), std=0.01)
        with torch.no_grad():
            self.mu_head.bias[0] = 0.0   # Neutral steering
            self.mu_head.bias[1] = 0.6   # Forward throttle bias so it immediately drives & explores curves

        self.log_std = nn.Parameter(torch.full((act_dim,), float(net_cfg["init_log_std"])))

        self.critic_net = nn.Sequential(
            _ortho_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            _ortho_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            _ortho_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def _dist(self, obs):
        h = self.actor_backbone(obs)
        mean = self.mu_head(h)

        std = torch.exp(self.log_std.clamp(self.log_std_min, self.log_std_max))
        return Normal(mean, std)

    def get_action_and_value(self, obs, deterministic=False):
        dist = self._dist(obs)
        value = self.critic_net(obs).squeeze(-1)
        if deterministic:
            action = torch.tanh(dist.mean)
            return action, None, value

        u = dist.rsample()
        action = torch.tanh(u)
        log_prob = (dist.log_prob(u) - torch.log(1.0 - action.pow(2) + 1e-6)).sum(-1)
        return action, log_prob, value

    def evaluate_actions(self, obs, actions):
        dist = self._dist(obs)
        value = self.critic_net(obs).squeeze(-1)

        clipped_act = actions.clamp(-0.999, 0.999)
        u = 0.5 * torch.log((1.0 + clipped_act) / (1.0 - clipped_act))
        log_prob = (dist.log_prob(u) - torch.log(1.0 - clipped_act.pow(2) + 1e-6)).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value


class PPOPolicy(BasePolicy):
    """Wraps a trained PyTorch PPO checkpoint for simulation.py benchmarking."""

    def __init__(self, config_path="configs/ppo_config.yaml", checkpoint_path=None, device=None):
        cfg = load_config(config_path)
        exp_cfg = cfg["experiment"]
        env_cfg = cfg["env"]
        net_cfg = cfg["network"]

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path or exp_cfg["checkpoint_path"]
        self.decision_repeat = int(env_cfg["decision_repeat"])
        self.vehicle_config = dict(env_cfg.get("vehicle_config", {}))

        ckpt = None
        obs_dim, act_dim = None, None
        if os.path.exists(self.checkpoint_path):
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            if isinstance(ckpt, dict) and "model_state" in ckpt:
                obs_dim = ckpt["obs_dim"]
                act_dim = ckpt["act_dim"]
                net_cfg = ckpt.get("net_cfg", net_cfg)
                self.decision_repeat = ckpt.get("decision_repeat", self.decision_repeat)
                self.vehicle_config = ckpt.get("vehicle_config", self.vehicle_config)

        if obs_dim is None or act_dim is None:
            probe = MetaDriveEnv({
                "use_render": False,
                "start_seed": int(env_cfg["start_seed"]),
                "num_scenarios": 1,
                "vehicle_config": self.vehicle_config,
            })
            obs_dim = int(np.prod(probe.observation_space.shape))
            act_dim = int(np.prod(probe.action_space.shape))
            probe.close()

        self.name = f"{exp_cfg['name']}({obs_dim}D)"
        self.model = SquashedActorCritic(obs_dim=obs_dim, act_dim=act_dim, net_cfg=net_cfg).to(self.device)

        if ckpt is not None:
            state_dict = ckpt["model_state"] if isinstance(ckpt, dict) else ckpt
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(
                f"[PPO] Loaded '{self.checkpoint_path}' "
                f"(obs_dim={obs_dim}, repeat={self.decision_repeat}) on {self.device}"
            )
        else:
            print(f"[PPO WARNING] '{self.checkpoint_path}' not found; using untrained weights.")

    def act(self, obs, sensor_packet, env):
        obs_flat = np.asarray(obs, dtype=np.float32).reshape(-1)
        if obs_flat.shape[0] != self.model.obs_dim:
            raise ValueError(
                f"Observation dim mismatch: got {obs_flat.shape[0]}D from env, "
                f"but policy expects {self.model.obs_dim}D."
            )
        with torch.no_grad():
            obs_t = torch.as_tensor(obs_flat, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, _, _ = self.model.get_action_and_value(obs_t, deterministic=True)
        return action.squeeze(0).cpu().numpy().tolist()


def _shape_av_reward(env, base_reward, obs, action, prev_action, info, rew_cfg):
    shaped = float(base_reward)

    try:
        lane = env.agent.navigation.current_ref_lanes[0]
        _, lat_offset = lane.local_coordinates(env.agent.position[:2])
        clipped_offset = min(abs(float(lat_offset)), float(rew_cfg["lane_offset_clip"]))
        shaped -= float(rew_cfg["lane_center_weight"]) * clipped_offset
    except Exception:
        pass

    steer_delta = float(action[0] - prev_action[0])
    shaped -= float(rew_cfg["steer_delta_weight"]) * (steer_delta ** 2)

    lidar_norm = obs[-240:]
    front_min_m = float(np.min(np.concatenate([lidar_norm[:8], lidar_norm[-8:]]) * 50.0))
    speed = float(env.agent.speed)
    speed_factor = min(speed / 10.0, 1.0)
    #shaped -= float(rew_cfg["lane_center_weight"]) * clipped_offset * speed_factor
    warn_dist = float(rew_cfg["front_warn_dist_m"])
    if front_min_m < warn_dist and speed > float(rew_cfg["min_warn_speed_ms"]):
        proximity_ratio = (warn_dist - front_min_m) / warn_dist
        shaped -= float(rew_cfg["front_proximity_weight"]) * proximity_ratio * (speed / 10.0)

    if info.get("crash_vehicle", False) or info.get("out_of_road", False):
        shaped -= float(rew_cfg["terminal_crash_penalty"])

    return shaped


def _worker_loop(worker_id, conn, cfg, obs_dim, act_dim):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    env_cfg = cfg["env"]
    rew_cfg = cfg["reward_shaping"]
    scenarios_per_worker = int(env_cfg["scenarios_per_worker"])
    w_start_seed = int(env_cfg["start_seed"]) + worker_id * scenarios_per_worker

    env = MetaDriveEnv({
        "use_render": False,
        "start_seed": w_start_seed,
        "num_scenarios": scenarios_per_worker,
        "traffic_density": float(env_cfg["traffic_density"]),
        "decision_repeat": int(env_cfg["decision_repeat"]),
        "horizon": int(env_cfg["horizon"]),
        "vehicle_config": dict(env_cfg.get("vehicle_config", {})),
    })
    model = SquashedActorCritic(obs_dim=obs_dim, act_dim=act_dim, net_cfg=cfg["network"]).to("cpu")
    model.eval()

    obs, _ = env.reset()
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    prev_action = np.zeros(act_dim, dtype=np.float32)
    ep_ret = 0.0
    completed_episodes = []

    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "close":
                break
            elif cmd == "collect":
                state_dict, n_steps = payload
                model.load_state_dict(state_dict)

                obs_b, act_b, logp_b, rew_b, done_b, val_b = [], [], [], [], [], []
                for _ in range(n_steps):
                    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        act_t, logp_t, val_t = model.get_action_and_value(obs_t, deterministic=False)

                    act_np = act_t.squeeze(0).numpy()
                    next_obs, raw_rew, terminated, truncated, info = env.step(act_np.tolist())
                    next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
                    done = bool(terminated or truncated)

                    shaped_rew = _shape_av_reward(env, raw_rew, next_obs, act_np, prev_action, info, rew_cfg)
                    prev_action = act_np.copy()

                    obs_b.append(obs)
                    act_b.append(act_np)
                    logp_b.append(float(logp_t.item()))
                    rew_b.append(shaped_rew)
                    done_b.append(float(done))
                    val_b.append(float(val_t.item()))

                    ep_ret += shaped_rew
                    obs = next_obs

                    if done:
                        completed_episodes.append({
                            "return": ep_ret,
                            "success": float(info.get("arrive_dest", False)),
                            "route_comp": float(info.get("route_completion", 0.0)),
                        })
                        obs, _ = env.reset()
                        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
                        prev_action = np.zeros(act_dim, dtype=np.float32)
                        ep_ret = 0.0

                with torch.no_grad():
                    next_obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    _, _, next_val_t = model.get_action_and_value(next_obs_t, deterministic=True)
                    next_val = float(next_val_t.item())

                conn.send({
                    "obs": np.asarray(obs_b, dtype=np.float32),
                    "act": np.asarray(act_b, dtype=np.float32),
                    "logp": np.asarray(logp_b, dtype=np.float32),
                    "rew": np.asarray(rew_b, dtype=np.float32),
                    "done": np.asarray(done_b, dtype=np.float32),
                    "val": np.asarray(val_b, dtype=np.float32),
                    "next_val": next_val,
                    "episodes": completed_episodes,
                })
                completed_episodes = []
    finally:
        env.close()
        conn.close()


def _compute_gae(rew, val, done, next_val, gamma, gae_lambda):
    n = len(rew)
    adv = np.zeros(n, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(n)):
        non_term = 1.0 - done[t]
        nv = next_val if t == n - 1 else val[t + 1]
        delta = rew[t] + gamma * nv * non_term - val[t]
        gae = delta + gamma * gae_lambda * non_term * gae
        adv[t] = gae
    return adv, adv + val


def train_ppo(config_path="configs/ppo_config.yaml"):
    cfg = load_config(config_path)
    exp_cfg = cfg["experiment"]
    env_cfg = cfg["env"]
    net_cfg = cfg["network"]
    trn_cfg = cfg["training"]

    save_path = exp_cfg["checkpoint_path"]
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = mp.get_context("spawn")

    vehicle_config = dict(env_cfg.get("vehicle_config", {}))
    probe_env = MetaDriveEnv({
        "use_render": False,
        "start_seed": int(env_cfg["start_seed"]),
        "num_scenarios": 1,
        "traffic_density": float(env_cfg["traffic_density"]),
        "decision_repeat": int(env_cfg["decision_repeat"]),
        "vehicle_config": vehicle_config,
    })
    obs_dim = int(np.prod(probe_env.observation_space.shape))
    act_dim = int(np.prod(probe_env.action_space.shape))
    probe_env.close()

    model = SquashedActorCritic(obs_dim=obs_dim, act_dim=act_dim, net_cfg=net_cfg).to(device)
    lr_start = float(trn_cfg["lr_start"])
    lr_end = float(trn_cfg["lr_end"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_start, eps=float(trn_cfg["adam_eps"]))

    num_envs = int(trn_cfg["num_envs"])
    steps_per_env = int(trn_cfg["steps_per_env"])
    total_steps = int(trn_cfg["total_steps"])
    batch_size = int(trn_cfg["batch_size"])
    update_epochs = int(trn_cfg["update_epochs"])
    gamma = float(trn_cfg["gamma"])
    gae_lambda = float(trn_cfg["gae_lambda"])
    clip_eps = float(trn_cfg["clip_eps"])
    entropy_coef = float(trn_cfg["entropy_coef"])
    value_coef = float(trn_cfg["value_coef"])
    max_grad_norm = float(trn_cfg["max_grad_norm"])

    workers, conns = [], []
    for w_id in range(num_envs):
        parent_conn, child_conn = ctx.Pipe()
        p = ctx.Process(target=_worker_loop, args=(w_id, child_conn, cfg, obs_dim, act_dim))
        p.start()
        workers.append(p)
        conns.append(parent_conn)

    rollout_total = num_envs * steps_per_env
    step_count = 0
    recent_ep = []
    best_score = -1e9
    t_start = time.time()

    print(
        f"Training '{exp_cfg['name']}' (obs_dim={obs_dim}) across {num_envs} workers "
        f"for {total_steps:,} steps on {device}..."
    )

    try:
        while step_count < total_steps:
            frac = min(step_count / total_steps, 1.0)
            cur_lr = lr_start + frac * (lr_end - lr_start)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            for conn in conns:
                conn.send(("collect", (cpu_state, steps_per_env)))

            worker_batches = [conn.recv() for conn in conns]
            step_count += rollout_total

            all_obs, all_act, all_logp, all_adv, all_ret = [], [], [], [], []
            for b in worker_batches:
                adv, ret = _compute_gae(b["rew"], b["val"], b["done"], b["next_val"], gamma, gae_lambda)
                all_obs.append(b["obs"])
                all_act.append(b["act"])
                all_logp.append(b["logp"])
                all_adv.append(adv)
                all_ret.append(ret)
                recent_ep.extend(b["episodes"])

            obs_np = np.concatenate(all_obs, axis=0)
            act_np = np.concatenate(all_act, axis=0)
            logp_np = np.concatenate(all_logp, axis=0)
            adv_np = np.concatenate(all_adv, axis=0)
            ret_np = np.concatenate(all_ret, axis=0)
            adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)

            obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
            act_t = torch.as_tensor(act_np, dtype=torch.float32, device=device)
            old_logp_t = torch.as_tensor(logp_np, dtype=torch.float32, device=device)
            adv_t = torch.as_tensor(adv_np, dtype=torch.float32, device=device)
            ret_t = torch.as_tensor(ret_np, dtype=torch.float32, device=device)

            idxs = np.arange(rollout_total)
            for _ in range(update_epochs):
                np.random.shuffle(idxs)
                for start in range(0, rollout_total, batch_size):
                    mb = idxs[start : start + batch_size]
                    new_logp, entropy, val = model.evaluate_actions(obs_t[mb], act_t[mb])

                    ratio = torch.exp(new_logp - old_logp_t[mb])
                    surr1 = ratio * adv_t[mb]
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t[mb]
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = 0.5 * (ret_t[mb] - val).pow(2).mean()
                    loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy.mean()

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

            if len(recent_ep) >= 20:
                tail = recent_ep[-40:]
                mean_ret = float(np.mean([e["return"] for e in tail]))
                mean_route = float(np.mean([e["route_comp"] for e in tail]))
                mean_succ = float(np.mean([e["success"] for e in tail]))
                fps = int(step_count / max(time.time() - t_start, 1e-6))

                ckpt_score = mean_route * 100.0 + mean_succ * 50.0
                saved_tag = ""
                if ckpt_score > best_score and step_count >= int(trn_cfg["min_save_steps"]):
                    best_score = ckpt_score
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "obs_dim": obs_dim,
                            "act_dim": act_dim,
                            "decision_repeat": int(env_cfg["decision_repeat"]),
                            "vehicle_config": vehicle_config,
                            "net_cfg": net_cfg,
                        },
                        save_path,
                    )
                    saved_tag = " [SAVED BEST]"

                if (step_count // rollout_total) % int(trn_cfg["log_every_rollouts"]) == 0 or saved_tag:
                    print(
                        f"Steps: {step_count:7d}/{total_steps} | FPS: {fps:4d} | LR: {cur_lr:.1e} | "
                        f"Return: {mean_ret:6.1f} | Route: {mean_route*100:5.1f}% | "
                        f"Success: {mean_succ*100:5.1f}%{saved_tag}"
                    )

    finally:
        for conn in conns:
            try:
                conn.send(("close", None))
            except Exception:
                pass
        for p in workers:
            p.join(timeout=5)

    print(f"Training finished. Best model saved to '{save_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ppo_config.yaml", help="Path to YAML config file")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and run benchmark only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not args.eval_only:
        train_ppo(args.config)

    policy = PPOPolicy(config_path=args.config)
    run_benchmark(
        policy,
        suite_name=cfg["experiment"]["eval_suite"],
        render=bool(cfg["experiment"]["render_eval"]),
    )