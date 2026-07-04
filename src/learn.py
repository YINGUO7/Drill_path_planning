import os
import time
from typing import Callable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from Env2 import MillingEnvNurbs
from NN2 import MillingNurbsFeatureExtractor


def select_training_device():
    """选择训练设备。

    默认使用CPU，避免Windows下CUDA/PyTorch底层库直接崩溃。
    如需启用GPU，在终端中设置环境变量：SAC_DEVICE=cuda。
    """
    requested = os.environ.get("SAC_DEVICE", "cuda").strip().lower()
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        print("SAC_DEVICE=cuda was requested, but torch.cuda.is_available() is False. Fallback to CPU.")
    return "cpu"


class EvalInfoWrapper(gym.Wrapper):
    """保存评估episode结束时的info，供自定义EvalCallback写入TensorBoard。"""

    def __init__(self, env):
        super().__init__(env)
        self.eval_infos = []

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            self.eval_infos.append(info)
        return obs, reward, terminated, truncated, info


class CustomEvalCallback(EvalCallback):
    """评估时记录覆盖率、无效动作和按动作类型归一化的reward分量。"""

    def _on_step(self) -> bool:
        should_eval = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if should_eval:
            eval_t0 = time.perf_counter()
            print(f"[Eval] start | n_calls={self.n_calls} | timesteps={self.num_timesteps}")
        result = super()._on_step()
        if should_eval:
            eval_elapsed = time.perf_counter() - eval_t0
            env = self.eval_env.envs[0]
            while not hasattr(env, "eval_infos") and hasattr(env, "env"):
                env = env.env
            if hasattr(env, "eval_infos") and env.eval_infos:
                infos = env.eval_infos[-self.n_eval_episodes:]
                self.logger.record("eval_custom/coverage_ratio",
                                   np.mean([i.get("coverage_ratio", 0.0) for i in infos]))
                self.logger.record("eval_custom/invalid_actions",
                                   np.mean([i.get("episode_invalid_actions", 0.0) for i in infos]))
                self.logger.record("eval_reward_PER_ACTION/efficiency_per_valid_action",
                                   np.mean([i.get("ep_efficiency_reward_per_valid_action", 0.0) for i in infos]))
                self.logger.record("eval_reward_PER_ACTION/coverage_per_valid_action",
                                   np.mean([i.get("ep_coverage_reward_per_valid_action", 0.0) for i in infos]))
                self.logger.record("eval_reward_PER_ACTION/invalid_per_invalid_action",
                                   np.mean([i.get("ep_invalid_reward_per_invalid_action", 0.0) for i in infos]))
                self.logger.record("eval_perf/episode_wall_time_s",
                                   np.mean([i.get("episode_wall_time_s", 0.0) for i in infos]))
                env.eval_infos.clear()
            print(f"[Eval] end | elapsed={eval_elapsed:.2f}s | episodes={self.n_eval_episodes}")
        return result


def step_schedule(initial_value: float, decay_factor: float = 0.5) -> Callable[[float], float]:
    """分段学习率调度。"""

    def func(progress_remaining: float) -> float:
        if progress_remaining > 0.6:
            return initial_value
        if progress_remaining > 0.3:
            return initial_value * decay_factor
        return initial_value * (decay_factor ** 2)

    return func


class EpisodeLoggerForSAC(BaseCallback):
    """训练时在终端和TensorBoard记录episode级指标。"""

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is None or infos is None:
            return True

        for i, done in enumerate(dones):
            if not done:
                continue
            info = infos[i]
            episode = info.get("episode", {})
            reward = float(episode.get("r", info.get("episode_reward", 0.0)))
            length = int(episode.get("l", info.get("episode_length", 0)))
            coverage = float(info.get("coverage_ratio", 0.0))
            invalid = int(info.get("episode_invalid_actions", 0))
            wall_time = float(info.get("episode_wall_time_s", 0.0))

            eff_per_valid = float(info.get("ep_efficiency_reward_per_valid_action", 0.0))
            cov_per_valid = float(info.get("ep_coverage_reward_per_valid_action", 0.0))
            inv_per_invalid = float(info.get("ep_invalid_reward_per_invalid_action", 0.0))

            self.logger.record("episode/coverage_ratio", coverage)
            self.logger.record("episode/invalid_actions", invalid)
            self.logger.record("episode/episode_wall_time_s", wall_time)
            self.logger.record("episode/step_wall_time_ms", 1000.0 * wall_time / max(length, 1))
            self.logger.record("reward_PER_ACTION/efficiency_per_valid_action", eff_per_valid)
            self.logger.record("reward_PER_ACTION/coverage_per_valid_action", cov_per_valid)
            self.logger.record("reward_PER_ACTION/invalid_per_invalid_action", inv_per_invalid)

            print(
                f"Env {i} done | step={self.num_timesteps} | R={reward:.2f} | "
                f"Len={length} | coverage={coverage * 100:.3f}% | "
                f"remaining={int(info.get('remaining_cells', 0))} | invalid={invalid} | "
                f"wall={wall_time:.2f}s | {1000.0 * wall_time / max(length, 1):.1f}ms/step"
            )
        return True


def make_env(render_mode=None):
    """创建单个训练环境。"""
    return MillingEnvNurbs(render_mode=render_mode)


if __name__ == "__main__":
    version = 5
    gamma_list = [0.99, 0.98, 0.95, 0.90]
    total_timesteps = 1_000_000
    n_envs = 50
    eval_every_timesteps = 10_000
    device = select_training_device()
    print(f"Training device: {device}")

    for try_num, current_gamma in enumerate(gamma_list):
        print("=" * 60)
        print(f"Start SAC training | try={try_num} | gamma={current_gamma}")
        print(f"Parallel envs: {n_envs} | Eval every {eval_every_timesteps} total timesteps")
        print("=" * 60)

        tensorboard_log = f"./sac_milling_nurbs_tensorboard_{version}/"
        best_model_save_path = f"./trained_SAC_milling_nurbs_{version}/try_{try_num}_{current_gamma}/best_model/"
        checkpoint_save_path = f"./trained_SAC_milling_nurbs_{version}/try_{try_num}_{current_gamma}/checkpoints/"
        tb_log_name = f"try{try_num}_gamma_{current_gamma}"
        os.makedirs(best_model_save_path, exist_ok=True)
        os.makedirs(checkpoint_save_path, exist_ok=True)

        train_env = make_vec_env(
            lambda: Monitor(make_env(render_mode=None)),
            n_envs=n_envs,
        )
        eval_env = make_vec_env(
            lambda: Monitor(EvalInfoWrapper(make_env(render_mode=None))),
            n_envs=1,
        )

        policy_kwargs = dict(
            features_extractor_class=MillingNurbsFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[128, 128], qf=[128, 128]),
        )

        model = SAC(
            "MultiInputPolicy",
            train_env,
            learning_rate=step_schedule(3e-4, decay_factor=0.5),
            buffer_size=100_000,
            learning_starts=n_envs * 300,
            batch_size=128,
            gamma=current_gamma,
            train_freq=(1, "step"),
            gradient_steps=1,
            ent_coef="auto",
            target_entropy=-1.0,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=tensorboard_log,
            device=device,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=max(1, int(total_timesteps / 10 / n_envs)),
            save_path=checkpoint_save_path,
            name_prefix=f"SAC_env7_try{try_num}",
        )
        eval_callback = CustomEvalCallback(
            eval_env,
            best_model_save_path=best_model_save_path,
            eval_freq=max(1, int(eval_every_timesteps / n_envs)),
            n_eval_episodes=3,
            deterministic=True,
            render=False,
        )
        callback_list = CallbackList([
            EpisodeLoggerForSAC(),
            checkpoint_callback,
            eval_callback,
        ])

        model.learn(total_timesteps=total_timesteps, callback=callback_list, tb_log_name=tb_log_name)
        train_env.close()
        eval_env.close()
