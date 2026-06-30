from __future__ import annotations

from pathlib import Path
import tomllib


class Config:
    ROOT_DIR = Path(__file__).resolve().parents[2]

    ALGO = "ppo"
    ENV_ID = "LunarLander-v3"
    ORG = "sb3"
    LOG_FOLDER = ROOT_DIR / "logs"

    # LunarLander PPO training setup, aligned with RL Zoo defaults.
    POLICY = "MlpPolicy"
    ENV_WRAPPERS = [
        "agent_ppo.feature.reward_process.LunarLanderRewardWrapper",
    ]
    N_TIMESTEPS = 1_000_000
    N_ENVS = 16
    N_STEPS = 1024
    BATCH_SIZE = 64
    N_EPOCHS = 4
    LEARNING_RATE = 0.0003
    GAMMA = 0.999
    GAE_LAMBDA = 0.98
    CLIP_RANGE = 0.2
    ENT_COEF = 0.01
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    NORMALIZE = False
    NORM_OBS = False
    NORM_REWARD = False
    TRAIN_SEED = 0
    EVAL_SEED = 10_000
    EVAL_FREQ_STEPS = 25_000
    N_EVAL_EPISODES = 20
    EVAL_SUCCESS_REWARD = 200.0

    # Real environment simulation: noisy sensors and delayed control.
    REAL_OBS_NOISE_STD = 0.10
    REAL_ACTION_DELAY_STEPS = 8
    REAL_DEFAULT_ACTION = 0
    REAL_NOISE_SEED_OFFSET = 2026
    REAL_GUST_PROBABILITY = 0.18
    REAL_GUST_FORCE_X_STD = 3.0
    REAL_GUST_FORCE_Y_STD = 1.0
    REAL_GUST_DURATION_MIN = 8
    REAL_GUST_DURATION_MAX = 32

    # Final real score: (20% fuel + 40% precision + 40% stability) * completion.
    REAL_SCORE_FUEL_WEIGHT = 0.20
    REAL_SCORE_PRECISION_WEIGHT = 0.40
    REAL_SCORE_STABILITY_WEIGHT = 0.40
    REAL_SIDE_ENGINE_FUEL_RATIO = 0.25
    REAL_PRECISION_MAX_ERROR = 1.5
    REAL_STABILITY_MAX_ERROR = 1.5
    REAL_STABILITY_ANGLE_WEIGHT = 1.0
    REAL_STABILITY_ANGULAR_VEL_WEIGHT = 0.5
    REAL_STABILITY_HORIZONTAL_VEL_WEIGHT = 0.5
    REAL_COMPLETION_X_THRESHOLD = 0.25
    REAL_COMPLETION_ANGLE_THRESHOLD = 0.35

    # Reward design, aligned with Gymnasium LunarLander original reward.
    REWARD_DISTANCE_WEIGHT = -100.0
    REWARD_VELOCITY_WEIGHT = -100.0
    REWARD_ANGLE_WEIGHT = -100.0
    REWARD_LEG_CONTACT_BONUS = 10.0
    REWARD_MAIN_ENGINE_COST = 0.30
    REWARD_SIDE_ENGINE_COST = 0.03
    REWARD_CRASH = -100.0
    REWARD_LANDING = 100.0
    REWARD_SCALE = 1.0
    REWARD_BIAS = 0.0

    # LunarLander observation/action layout.
    OBS_DIM = 8
    STATE_SHAPE = (OBS_DIM,)
    STATE_DIM = OBS_DIM
    ACTION_NUM = 4
    VALUE_NUM = 1

    # Stable-Baselines3 MlpPolicy defaults for PPO.
    ACTOR_HIDDEN_LAYERS = [64, 64]
    CRITIC_HIDDEN_LAYERS = [64, 64]
    ACTIVATION_FN = "nn.Tanh"
    ORTHO_INIT = True

    HYPERPARAMS_FILE = ROOT_DIR / "agent_ppo" / "conf" / "ppo_lunarlander.yml"
    TRAIN_CONF_FILE = ROOT_DIR / "agent_ppo" / "conf" / "train_env_conf.toml"


def _get_nested(data, section: str, key: str, default):
    return data.get(section, {}).get(key, default)


def _apply_train_conf():
    if not Config.TRAIN_CONF_FILE.exists():
        return

    with Config.TRAIN_CONF_FILE.open("rb") as f:
        data = tomllib.load(f)

    Config.ALGO = data.get("algo", Config.ALGO)
    Config.ENV_ID = data.get("env", Config.ENV_ID)
    Config.ORG = data.get("organization", Config.ORG)
    Config.LOG_FOLDER = Config.ROOT_DIR / data.get("log_folder", str(Config.LOG_FOLDER.name))

    hyperparams_file = data.get("hyperparams_file")
    if hyperparams_file:
        Config.HYPERPARAMS_FILE = Config.ROOT_DIR / hyperparams_file

    Config.REWARD_SCALE = float(_get_nested(data, "reward", "scale", Config.REWARD_SCALE))
    Config.REWARD_BIAS = float(_get_nested(data, "reward", "bias", Config.REWARD_BIAS))
    Config.EVAL_SUCCESS_REWARD = float(_get_nested(data, "reward", "success_threshold", Config.EVAL_SUCCESS_REWARD))
    Config.REWARD_DISTANCE_WEIGHT = float(_get_nested(data, "reward", "distance_weight", Config.REWARD_DISTANCE_WEIGHT))
    Config.REWARD_VELOCITY_WEIGHT = float(_get_nested(data, "reward", "velocity_weight", Config.REWARD_VELOCITY_WEIGHT))
    Config.REWARD_ANGLE_WEIGHT = float(_get_nested(data, "reward", "angle_weight", Config.REWARD_ANGLE_WEIGHT))
    Config.REWARD_LEG_CONTACT_BONUS = float(_get_nested(data, "reward", "leg_contact_bonus", Config.REWARD_LEG_CONTACT_BONUS))
    Config.REWARD_MAIN_ENGINE_COST = float(_get_nested(data, "reward", "main_engine_cost", Config.REWARD_MAIN_ENGINE_COST))
    Config.REWARD_SIDE_ENGINE_COST = float(_get_nested(data, "reward", "side_engine_cost", Config.REWARD_SIDE_ENGINE_COST))
    Config.REWARD_CRASH = float(_get_nested(data, "reward", "crash", Config.REWARD_CRASH))
    Config.REWARD_LANDING = float(_get_nested(data, "reward", "landing", Config.REWARD_LANDING))

    Config.POLICY = _get_nested(data, "model", "policy", Config.POLICY)
    Config.OBS_DIM = int(_get_nested(data, "model", "observation_dim", Config.OBS_DIM))
    Config.STATE_SHAPE = (Config.OBS_DIM,)
    Config.STATE_DIM = Config.OBS_DIM
    Config.ACTION_NUM = int(_get_nested(data, "model", "action_num", Config.ACTION_NUM))
    Config.ACTOR_HIDDEN_LAYERS = list(_get_nested(data, "model", "actor_hidden_layers", Config.ACTOR_HIDDEN_LAYERS))
    Config.CRITIC_HIDDEN_LAYERS = list(_get_nested(data, "model", "critic_hidden_layers", Config.CRITIC_HIDDEN_LAYERS))
    Config.ORTHO_INIT = bool(_get_nested(data, "model", "ortho_init", Config.ORTHO_INIT))

    Config.N_TIMESTEPS = int(_get_nested(data, "train", "n_timesteps", Config.N_TIMESTEPS))
    Config.N_ENVS = int(_get_nested(data, "train", "n_envs", Config.N_ENVS))
    Config.N_STEPS = int(_get_nested(data, "train", "n_steps", Config.N_STEPS))
    Config.BATCH_SIZE = int(_get_nested(data, "train", "batch_size", Config.BATCH_SIZE))
    Config.N_EPOCHS = int(_get_nested(data, "train", "n_epochs", Config.N_EPOCHS))
    Config.LEARNING_RATE = float(_get_nested(data, "train", "learning_rate", Config.LEARNING_RATE))
    Config.GAMMA = float(_get_nested(data, "train", "gamma", Config.GAMMA))
    Config.GAE_LAMBDA = float(_get_nested(data, "train", "gae_lambda", Config.GAE_LAMBDA))
    Config.CLIP_RANGE = float(_get_nested(data, "train", "clip_range", Config.CLIP_RANGE))
    Config.ENT_COEF = float(_get_nested(data, "train", "ent_coef", Config.ENT_COEF))
    Config.VF_COEF = float(_get_nested(data, "train", "vf_coef", Config.VF_COEF))
    Config.MAX_GRAD_NORM = float(_get_nested(data, "train", "max_grad_norm", Config.MAX_GRAD_NORM))
    Config.NORMALIZE = bool(_get_nested(data, "train", "normalize", Config.NORMALIZE))
    Config.NORM_OBS = bool(_get_nested(data, "train", "norm_obs", Config.NORM_OBS))
    Config.NORM_REWARD = bool(_get_nested(data, "train", "norm_reward", Config.NORM_REWARD))
    Config.TRAIN_SEED = int(_get_nested(data, "train", "train_seed", Config.TRAIN_SEED))
    Config.EVAL_SEED = int(_get_nested(data, "train", "eval_seed", Config.EVAL_SEED))
    Config.EVAL_FREQ_STEPS = int(_get_nested(data, "train", "eval_freq_steps", Config.EVAL_FREQ_STEPS))
    Config.N_EVAL_EPISODES = int(_get_nested(data, "train", "n_eval_episodes", Config.N_EVAL_EPISODES))

    Config.REAL_OBS_NOISE_STD = float(_get_nested(data, "real", "obs_noise_std", Config.REAL_OBS_NOISE_STD))
    Config.REAL_ACTION_DELAY_STEPS = int(_get_nested(data, "real", "action_delay_steps", Config.REAL_ACTION_DELAY_STEPS))
    Config.REAL_DEFAULT_ACTION = int(_get_nested(data, "real", "default_action", Config.REAL_DEFAULT_ACTION))
    Config.REAL_NOISE_SEED_OFFSET = int(_get_nested(data, "real", "noise_seed_offset", Config.REAL_NOISE_SEED_OFFSET))
    Config.REAL_GUST_PROBABILITY = float(_get_nested(data, "real", "gust_probability", Config.REAL_GUST_PROBABILITY))
    Config.REAL_GUST_FORCE_X_STD = float(_get_nested(data, "real", "gust_force_x_std", Config.REAL_GUST_FORCE_X_STD))
    Config.REAL_GUST_FORCE_Y_STD = float(_get_nested(data, "real", "gust_force_y_std", Config.REAL_GUST_FORCE_Y_STD))
    Config.REAL_GUST_DURATION_MIN = int(_get_nested(data, "real", "gust_duration_min", Config.REAL_GUST_DURATION_MIN))
    Config.REAL_GUST_DURATION_MAX = int(_get_nested(data, "real", "gust_duration_max", Config.REAL_GUST_DURATION_MAX))

    Config.REAL_SCORE_FUEL_WEIGHT = float(_get_nested(data, "real_score", "fuel_weight", Config.REAL_SCORE_FUEL_WEIGHT))
    Config.REAL_SCORE_PRECISION_WEIGHT = float(_get_nested(data, "real_score", "precision_weight", Config.REAL_SCORE_PRECISION_WEIGHT))
    Config.REAL_SCORE_STABILITY_WEIGHT = float(_get_nested(data, "real_score", "stability_weight", Config.REAL_SCORE_STABILITY_WEIGHT))
    Config.REAL_SIDE_ENGINE_FUEL_RATIO = float(_get_nested(data, "real_score", "side_engine_fuel_ratio", Config.REAL_SIDE_ENGINE_FUEL_RATIO))
    Config.REAL_PRECISION_MAX_ERROR = float(_get_nested(data, "real_score", "precision_max_error", Config.REAL_PRECISION_MAX_ERROR))
    Config.REAL_STABILITY_MAX_ERROR = float(_get_nested(data, "real_score", "stability_max_error", Config.REAL_STABILITY_MAX_ERROR))
    Config.REAL_STABILITY_ANGLE_WEIGHT = float(_get_nested(data, "real_score", "stability_angle_weight", Config.REAL_STABILITY_ANGLE_WEIGHT))
    Config.REAL_STABILITY_ANGULAR_VEL_WEIGHT = float(_get_nested(data, "real_score", "stability_angular_vel_weight", Config.REAL_STABILITY_ANGULAR_VEL_WEIGHT))
    Config.REAL_STABILITY_HORIZONTAL_VEL_WEIGHT = float(_get_nested(data, "real_score", "stability_horizontal_vel_weight", Config.REAL_STABILITY_HORIZONTAL_VEL_WEIGHT))
    Config.REAL_COMPLETION_X_THRESHOLD = float(_get_nested(data, "real_score", "completion_x_threshold", Config.REAL_COMPLETION_X_THRESHOLD))
    Config.REAL_COMPLETION_ANGLE_THRESHOLD = float(_get_nested(data, "real_score", "completion_angle_threshold", Config.REAL_COMPLETION_ANGLE_THRESHOLD))


_apply_train_conf()
