import sys, numpy as np, torch
from collections import Counter
import environment
environment.RETURN_START_PROB = 0.0
from environment import SeafloorEDNAEnv
from ppo import PPOAgent

agent = PPOAgent(n_map_channels=6, n_scalars=5, n_actions=9, grid_size=49, device=torch.device("cpu"))
agent.model.load_state_dict(torch.load(sys.argv[1], map_location="cpu",weights_only=True))
env = SeafloorEDNAEnv(grid_size=50, max_steps=500)
env.max_hotspot_dist = float(sys.argv[2]) if len(sys.argv) > 2 else None
greedy = len(sys.argv) > 3 and sys.argv[3] == "greedy"
print("mode:", "greedy" if greedy else "sampled")
reasons, samples, rewards = Counter(), [], []
for ep in range(200):
    env.reset(seed=50_000 + ep)
    env.canisters = 3
    obs, total = env._get_obs(), 0.0
    for t in range(500):
        with torch.no_grad():
            dist, _ = agent.model.forward(torch.tensor(obs["maps"])[None], torch.tensor(obs["scalars"])[None],
                                          torch.tensor(env.get_action_mask())[None])
        a = int(dist.probs.argmax()) if greedy else int(dist.sample())
        obs, r, term, trunc, info = env.step(a)
        total += r
        if term or trunc: break
    reasons[info.get("end_reason")] += 1; samples.append(info.get("samples_taken", 0)); rewards.append(total)
print(dict(reasons), "avg samples", np.mean(samples), "avg reward", round(float(np.mean(rewards)), 2))

