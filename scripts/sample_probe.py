import sys, numpy as np, torch
import environment
environment.RETURN_START_PROB = 0.0
from environment import SeafloorEDNAEnv
from ppo import PPOAgent

agent = PPOAgent(n_map_channels=6, n_scalars=5, n_actions=9, grid_size=49, device=torch.device("cpu"))
agent.model.load_state_dict(torch.load(sys.argv[1], map_location="cpu", weights_only=True))
env = SeafloorEDNAEnv(grid_size=50, max_steps=500)
probs = {0: [], 1: [], 2: []}
for ep in range(40):
    env.reset(seed=40_000 + ep); env.canisters = 3
    obs = env._get_obs()
    for t in range(500):
        with torch.no_grad():
            dist, _ = agent.model.forward(torch.tensor(obs["maps"])[None], torch.tensor(obs["scalars"])[None],
                                          torch.tensor(env.get_action_mask())[None])
        if env.canisters > 0:
            ds = [max(abs(env.agent_pos[0] - h[0]), abs(env.agent_pos[1] - h[1]))
                  for h in env.hotspots if env.sampled_mask[h] == 0]
            if ds and min(ds) in probs:
                probs[min(ds)].append(float(dist.probs[0, 8]))
        obs, r, term, trunc, info = env.step(int(dist.sample()))
        if term or trunc: break
for d, v in probs.items():
    print(f"{d} tiles from an unsampled hotspot: n={len(v)}  mean P(SAMPLE)={np.mean(v) if v else float('nan'):.2f}")