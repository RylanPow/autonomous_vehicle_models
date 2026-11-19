import os
import time
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from metadrive.envs import MetaDriveEnv
from simulation import BasePolicy, run_benchmark


class SquashedActorCritic(nn.Module):
    """Tanh-squashed Gaussian policy + Value critic for continuous [-1, 1] control."""

    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.actor_backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.8))

        self.critic_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def _dist(self, obs):
        h = self.actor_backbone(obs)
        mean = self.mu_head(h)
        std = torch.exp(self.log_std.clamp(-2.5, 0.3))
        return Normal(mean, std)

    def get_action_and_value(self, obs, deterministic=False):
        dist = self._dist(obs)
        value = self.critic_net(obs).squeeze(-1)
        if deterministic:
            u = dist.mean
            action = torch.tanh(u)
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

    def __init__(self, checkpoint_path="checkpoints/ppo_metadrive.pt", obs_dim=None, act_dim=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.decision_repeat = 5

        ckpt = None
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(ckpt, dict) and "model_state" in ckpt:
                obs_dim = ckpt.get("obs_dim", obs_dim)
                act_dim = ckpt.get("act_dim", act_dim)
                self.decision_repeat = ckpt.get("decision_repeat", 5)

        if obs_dim is None or act_dim is None:
            probe = MetaDriveEnv({"use_render": False, "start_seed": 1000, "num_scenarios": 1})
            obs_dim = int(np.prod(probe.observation_space.shape))
            act_dim = int(np.prod(probe.action_space.shape))
            probe.close()

        self.name = f"PPO_Squashed({obs_dim}D)"
        self.model = SquashedActorCritic(obs_dim=obs_dim, act_dim=act_dim).to(self.device)

        if ckpt is not None:
            state_dict = ckpt["model_state"] if (isinstance(ckpt, dict) and "model_state" in ckpt) else ckpt
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"[PPO] Loaded checkpoint '{checkpoint_path}' (obs_dim={obs_dim}, act_dim={act_dim}, repeat={self.decision_repeat}) on {self.device}")
        else:
            print(f"[PPO WARNING] Checkpoint '{checkpoint_path}' not found; using random weights (obs_dim={obs_dim}).")

    def act(self, obs, sensor_packet, env):
        obs_flat = np.asarray(obs, dtype=np.float32).reshape(-1)
        expected_dim = self.model.obs_dim
        if obs_flat.shape[0] != expected_dim:
            raise ValueError(
                f"Observation dim mismatch: got {obs_flat.shape[0]}D from env, "
                f"but policy '{self.checkpoint_path}' expects {expected_dim}D. "
                f"Ensure training and benchmark use the same vehicle_config / observation space."
            )
        with torch.no_grad():
            obs_t = torch.as_tensor(obs_flat, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, _, _ = self.model.get_action_and_value(obs_t, deterministic=True)
        return action.squeeze(0).cpu().numpy().tolist()


def _worker_loop(worker_id, conn, start_seed, num_scenarios, traffic_density, decision_repeat, obs_dim, act_dim):
    """Worker process owning its own MetaDriveEnv instance with single-thread Torch ops."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    env = MetaDriveEnv({
        "use_render": False,
        "start_seed": start_seed,
        "num_scenarios": num_scenarios,
        "traffic_density": traffic_density,
        "decision_repeat": decision_repeat,
        "horizon": 1000,
    })
    model = SquashedActorCritic(obs_dim=obs_dim, act_dim=act_dim).to("cpu")
    model.eval()

    obs, _ = env.reset()
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
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
                    next_obs, rew, terminated, truncated, info = env.step(act_np.tolist())
                    next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
                    done = bool(terminated or truncated)

                    obs_b.append(obs)
                    act_b.append(act_np)
                    logp_b.append(float(logp_t.item()))
                    rew_b.append(float(rew))
                    done_b.append(float(done))
                    val_b.append(float(val_t.item()))

                    ep_ret += float(rew)
                    obs = next_obs

                    if done:
                        completed_episodes.append({
                            "return": ep_ret,
                            "success": float(info.get("arrive_dest", False)),
                            "route_comp": float(info.get("route_completion", 0.0)),
                        })
                        obs, _ = env.reset()
                        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
                        ep_ret = 0.0

                with torch.no_grad():
                    next_obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    _, _, next_val_t = model.get_action_and_value(next_obs_t, deterministic=True)
                    next_val = float(next_val_t.item())

                batch = {
                    "obs": np.asarray(obs_b, dtype=np.float32),
                    "act": np.asarray(act_b, dtype=np.float32),
                    "logp": np.asarray(logp_b, dtype=np.float32),
                    "rew": np.asarray(rew_b, dtype=np.float32),
                    "done": np.asarray(done_b, dtype=np.float32),
                    "val": np.asarray(val_b, dtype=np.float32),
                    "next_val": next_val,
                    "episodes": completed_episodes,
                }
                completed_episodes = []
                conn.send(batch)
    finally:
        env.close()
        conn.close()


def _compute_gae(rew, val, done, next_val, gamma=0.99, gae_lambda=0.95):
    n = len(rew)
    adv = np.zeros(n, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(n)):
        non_term = 1.0 - done[t]
        nv = next_val if t == n - 1 else val[t + 1]
        delta = rew[t] + gamma * nv * non_term - val[t]
        gae = delta + gamma * gae_lambda * non_term * gae
        adv[t] = gae
    ret = adv + val
    return adv, ret


def train_ppo(
    total_steps=300_000,
    num_envs=12,
    steps_per_env=256,
    batch_size=1024,
    update_epochs=10,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    traffic_density=0.05,
    decision_repeat=5,
    save_path="checkpoints/ppo_metadrive.pt",
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = mp.get_context("spawn")

    probe_env = MetaDriveEnv({
        "use_render": False,
        "start_seed": 1000,
        "num_scenarios": 1,
        "traffic_density": traffic_density,
        "decision_repeat": decision_repeat,
    })
    obs_dim = int(np.prod(probe_env.observation_space.shape))
    act_dim = int(np.prod(probe_env.action_space.shape))
    probe_env.close()

    model = SquashedActorCritic(obs_dim=obs_dim, act_dim=act_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    workers = []
    conns = []
    scenarios_per_worker = 25
    for w_id in range(num_envs):
        parent_conn, child_conn = ctx.Pipe()
        w_start_seed = 1000 + w_id * scenarios_per_worker
        p = ctx.Process(
            target=_worker_loop,
            args=(
                w_id, child_conn, w_start_seed, scenarios_per_worker,
                traffic_density, decision_repeat, obs_dim, act_dim,
            ),
        )
        p.start()
        workers.append(p)
        conns.append(parent_conn)

    rollout_total = num_envs * steps_per_env
    step_count = 0
    recent_ep = []
    t_start = time.time()

    print(
        f"Training PPO (obs_dim={obs_dim}, act_dim={act_dim}) across {num_envs} workers "
        f"({rollout_total} steps/rollout) on {device}..."
    )

    try:
        while step_count < total_steps:
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
                    loss = actor_loss + 0.5 * critic_loss - 0.005 * entropy.mean()

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "obs_dim": obs_dim,
                    "act_dim": act_dim,
                    "decision_repeat": decision_repeat,
                },
                save_path,
            )

            if recent_ep:
                tail = recent_ep[-30:]
                fps = int(step_count / max(time.time() - t_start, 1e-6))
                print(
                    f"Steps: {step_count:7d}/{total_steps} | FPS: {fps:4d} | "
                    f"Return: {np.mean([e['return'] for e in tail]):6.1f} | "
                    f"Route: {np.mean([e['route_comp'] for e in tail])*100:5.1f}% | "
                    f"Success: {np.mean([e['success'] for e in tail])*100:5.1f}%"
                )

    finally:
        for conn in conns:
            try:
                conn.send(("close", None))
            except Exception:
                pass
        for p in workers:
            p.join(timeout=5)

    print(f"Training finished. Saved model to '{save_path}'.")


if __name__ == "__main__":
    train_ppo(total_steps=300_000, num_envs=12, steps_per_env=256, batch_size=1024)
    policy = PPOPolicy("checkpoints/ppo_metadrive.pt")
    run_benchmark(policy, suite_name="quick_dev", render=True)