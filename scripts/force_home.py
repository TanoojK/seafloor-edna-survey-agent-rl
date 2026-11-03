import sys, numpy as np, torch
from collections import Counter
from environment import SeafloorEDNAEnv
from ppo import PPOAgent

agent = PPOAgent(n_map_channels=5, n_scalars=5, n_actions=9, grid_size=49, device=torch.device("cpu"))
agent.model.load_state_dict(torch.load(sys.argv[1], map_location=agent.device))
env = SeafloorEDNAEnv(grid_size=50, max_steps=500)

for mode in ("stochastic", "greedy"):
    reasons, steps_home = Counter(), []
    for ep in range(40):
        env.reset(seed=20_000 + ep)
        h = tuple(env.hotspots[0])
        env.agent_pos = h; env.sampled_mask[h] = 1.0
        env.canisters = 0; env.n_samples_taken = 1; env.has_left_spawn = True
        obs = env._get_obs()
        for t in range(500):
            with torch.no_grad():
                dist, _ = agent.model.forward(torch.tensor(obs["maps"][:5])[None], torch.tensor(obs["scalars"])[None],
                                              torch.tensor(env.get_action_mask())[None])
            a = int(dist.sample()) if mode == "stochastic" else int(dist.probs.argmax())
            obs, r, term, trunc, info = env.step(a)
            if term or trunc: break
        reasons[info.get("end_reason")] += 1
        if info.get("end_reason") == "returned_to_spawn": steps_home.append(t + 1)
    print(mode, dict(reasons), "median steps home:", np.median(steps_home) if steps_home else None)