"""obs101 -> obs105 weight-transfer SURGERY (OpenAI-Five "Changing the Observation Space", arXiv:1912.06680 App.B).

The velocity dims were INSERTED at obs index 90 (before the trailing 11-d poison block), so a naive [W|0]
*append* would mis-map every poison feature. Correct surgery = INSERT 4 ZERO columns at 90 and RIGHT-SHIFT the
11 poison columns to 94-104. Because the 4 new columns get ZERO weight, the surgered net IGNORES velocity and is
behaviourally IDENTICAL to the 101-d original -- only the obs SHAPE changes so it loads into the 105-d env.
Run explicitly:  python train/surgery_obs101to105.py <old_101.zip> <new_105.zip>
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

OBS_OLD, OBS_NEW, VEL_AT, N_VEL = 101, 105, 90, 4
FIRST_LAYERS = {"mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight"}


def insert_cols(W):                                   # (64,101) -> (64,105)
    out = W.new_zeros((W.shape[0], OBS_NEW))
    out[:, :VEL_AT] = W[:, :VEL_AT]                    # [0..89]   copy verbatim
    # out[:, 90:94] stays ZERO                          # [90..93]  new velocity (zero weight -> ignored)
    out[:, VEL_AT + N_VEL:] = W[:, VEL_AT:]            # [94..104] <- old poison [90..100]
    return out


def surgery(old_path, new_path):
    old = PPO.load(old_path, device="cpu")
    assert old.observation_space.shape == (OBS_OLD,), old.observation_space.shape
    new = PPO("MlpPolicy", _SpaceEnv(OBS_NEW, N_ACTIONS), device="cpu")   # default net_arch [64,64] matches the champions
    old_sd, new_sd = old.policy.state_dict(), new.policy.state_dict()
    out_sd, n_first = {}, 0
    for k in new_sd:
        assert k in old_sd, f"key {k} missing from old model"
        if k in FIRST_LAYERS:
            out_sd[k] = insert_cols(old_sd[k]); n_first += 1
            assert torch.all(out_sd[k][:, VEL_AT:VEL_AT + N_VEL] == 0), "velocity cols not zero"
            assert torch.allclose(out_sd[k][:, :VEL_AT], old_sd[k][:, :VEL_AT]), "prefix mis-copied"
            assert torch.allclose(out_sd[k][:, VEL_AT + N_VEL:], old_sd[k][:, VEL_AT:]), "poison shift wrong"
        else:
            assert new_sd[k].shape == old_sd[k].shape, (k, tuple(new_sd[k].shape), tuple(old_sd[k].shape))
            out_sd[k] = old_sd[k].clone()
    assert n_first == 2, f"expected 2 first-layer matrices, transplanted {n_first}"
    new.policy.load_state_dict(out_sd, strict=True)
    new.save(new_path)
    parity_check(old, new)
    print(f"surgery OK: {old_path} (101) -> {new_path} (105)  [velocity zeroed, poison shifted, {n_first} first-layers]")
    return new


def parity_check(old, new, n=400):
    """BEHAVIORAL parity (catches a wrong VEL_AT/shift that the structural asserts miss): feed the SAME logical
    state through both nets and require identical argmax. obs105 inserts ARBITRARY velocity at 90-93 -> must be
    ignored (zero weight) -> argmax must match the 101-d net for every state."""
    rng = np.random.default_rng(0)
    obs101 = rng.uniform(-1, 1, (n, OBS_OLD)).astype(np.float32)
    vel = rng.uniform(-1, 1, (n, N_VEL)).astype(np.float32)
    obs105 = np.concatenate([obs101[:, :VEL_AT], vel, obs101[:, VEL_AT:]], axis=1).astype(np.float32)
    a_old = np.asarray(old.predict(obs101, deterministic=True)[0])
    a_new = np.asarray(new.predict(obs105, deterministic=True)[0])
    match = float((a_old == a_new).mean())
    print(f"  parity: argmax match {match * 100:.1f}% over {n} states (velocity randomized -> proven ignored)")
    assert match == 1.0, f"PARITY FAIL {match:.3f} -> column mapping is wrong; do NOT trust this surgery"


if __name__ == "__main__":
    surgery(sys.argv[1], sys.argv[2])
