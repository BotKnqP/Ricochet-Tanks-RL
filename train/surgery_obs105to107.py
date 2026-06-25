"""obs105 -> obs107 weight-transfer SURGERY (OpenAI-Five "Changing the Observation Space"): the 2 aim-LEAD dims were
INSERTED at obs index 94 (after the 4 velocity dims, before the trailing 11-d poison block). Correct surgery = INSERT 2
ZERO columns at 94 and RIGHT-SHIFT the 11 poison columns to 96-106. The 2 new columns get ZERO weight, so the surgered
net IGNORES the lead feature and is behaviourally IDENTICAL to the 105-d original -- only the SHAPE changes so it loads
into the 107-d env, ready to LEARN the lead feature via continued training.
Run:  python train/surgery_obs105to107.py <old_105.zip> <new_107.zip>
"""
import sys
import numpy as np
import torch
from stable_baselines3 import PPO

sys.path.insert(0, "train")
try:
    from train_bc import _SpaceEnv, N_ACTIONS
except ImportError:
    from .train_bc import _SpaceEnv, N_ACTIONS

OBS_OLD, OBS_NEW, AT, N_NEW = 105, 107, 94, 2
FIRST_LAYERS = {"mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight"}


def insert_cols(W):                                   # (64,105) -> (64,107)
    out = W.new_zeros((W.shape[0], OBS_NEW))
    out[:, :AT] = W[:, :AT]                            # [0..93]   copy (features + velocity)
    # out[:, 94:96] stays ZERO                          # [94..95]  new aim-lead (zero weight -> ignored)
    out[:, AT + N_NEW:] = W[:, AT:]                    # [96..106] <- old poison [94..104]
    return out


def surgery(old_path, new_path):
    old = PPO.load(old_path, device="cpu")
    assert old.observation_space.shape == (OBS_OLD,), old.observation_space.shape
    new = PPO("MlpPolicy", _SpaceEnv(OBS_NEW, N_ACTIONS), device="cpu")
    old_sd, new_sd = old.policy.state_dict(), new.policy.state_dict()
    out_sd, n_first = {}, 0
    for k in new_sd:
        assert k in old_sd, f"key {k} missing from old model"
        if k in FIRST_LAYERS:
            out_sd[k] = insert_cols(old_sd[k]); n_first += 1
            assert torch.all(out_sd[k][:, AT:AT + N_NEW] == 0), "lead cols not zero"
            assert torch.allclose(out_sd[k][:, :AT], old_sd[k][:, :AT]), "prefix mis-copied"
            assert torch.allclose(out_sd[k][:, AT + N_NEW:], old_sd[k][:, AT:]), "poison shift wrong"
        else:
            assert new_sd[k].shape == old_sd[k].shape, (k, tuple(new_sd[k].shape), tuple(old_sd[k].shape))
            out_sd[k] = old_sd[k].clone()
    assert n_first == 2, f"expected 2 first-layer matrices, transplanted {n_first}"
    new.policy.load_state_dict(out_sd, strict=True)
    new.save(new_path)
    parity_check(old, new)
    print(f"surgery OK: {old_path} (105) -> {new_path} (107)  [aim-lead zeroed, poison shifted, {n_first} first-layers]")
    return new


def parity_check(old, new, n=400):
    rng = np.random.default_rng(0)
    obs105 = rng.uniform(-1, 1, (n, OBS_OLD)).astype(np.float32)
    lead = rng.uniform(-1, 1, (n, N_NEW)).astype(np.float32)            # arbitrary lead -> must be IGNORED
    obs107 = np.concatenate([obs105[:, :AT], lead, obs105[:, AT:]], axis=1).astype(np.float32)
    a_old = np.asarray(old.predict(obs105, deterministic=True)[0])
    a_new = np.asarray(new.predict(obs107, deterministic=True)[0])
    match = float((a_old == a_new).mean())
    print(f"  parity: argmax match {match * 100:.1f}% over {n} states (lead randomized -> proven ignored)")
    assert match == 1.0, f"PARITY FAIL {match:.3f} -> column mapping wrong; do NOT trust this surgery"


if __name__ == "__main__":
    surgery(sys.argv[1], sys.argv[2])
