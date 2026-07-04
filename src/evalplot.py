from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import SAC

from Env3 import MillingEnvNurbs
from NN3 import MillingNurbsFeatureExtractor  # noqa: F401


# ============================================================
# 手动配置区：只需要改这里
# ============================================================

MODEL_PATH = (
    "/root/autodl-tmp/tempertature_prediction/drill/ENV3/trained_SAC_milling_nurbs_1/try_2_0.95/best_model/best_model.zip"
)

SAVE_DIR = "./evaluation_results"

SEED = 42
DETERMINISTIC = True
SHOW_FIGURE = True          # 服务器无GUI时建议改为 False
RENDER_DURING_ROLLOUT = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# None 表示使用 env.max_episode_actions 作为安全上限
MAX_STEPS_GUARD = None


def summarize_episode(env, final_info, step_count, total_reward):
    """整理单个评估 episode 的核心指标。"""
    if len(getattr(env, "current_path", [])):
        env._refresh_full_sweep_state()

    covered = int(np.count_nonzero(env.inner_core_mask & (env.visited == 2)))
    remaining = int(np.count_nonzero(env.inner_core_mask & (env.visited == 0)))
    coverage = covered / max(env.total_planning_cells, 1)

    full_time = float(final_info.get("full_time", 0.0))
    if full_time <= 0.0 and len(getattr(env, "current_path", [])) >= 5:
        full_time = float(env.estimate_machining_time(env.current_path, mode="full", planner="vpop"))

    wall_time = float(final_info.get("episode_wall_time_s", 0.0))
    if wall_time <= 0.0 and hasattr(env, "episode_wall_start"):
        wall_time = float(time.perf_counter() - env.episode_wall_start)

    return {
        "steps": int(step_count),
        "total_reward": float(total_reward),
        "episode_wall_time_s": wall_time,
        "step_wall_time_ms": 1000.0 * wall_time / max(int(step_count), 1),
        "coverage_ratio": float(coverage),
        "covered_cells": covered,
        "remaining_cells": remaining,
        "invalid_actions": int(final_info.get("episode_invalid_actions", 0)),
        "valid_actions": int(final_info.get("episode_valid_actions", 0)),
        "local_time_sum": float(final_info.get("ep_local_time_sum", 0.0)),
        "full_time": full_time,
        "new_area_mm2_sum": float(final_info.get("ep_new_area_mm2_sum", 0.0)),
        "efficiency_reward_sum": float(final_info.get("ep_efficiency_reward_sum", 0.0)),
        "coverage_reward_sum": float(final_info.get("ep_coverage_reward_sum", 0.0)),
        "invalid_reward_sum": float(final_info.get("ep_invalid_reward_sum", 0.0)),
        "full_time_reward_sum": float(final_info.get("ep_full_time_reward_sum", 0.0)),
        "last_full_time_reward": float(final_info.get("full_time_reward", 0.0)),
        "last_raw_full_time_reward": float(final_info.get("raw_full_time_reward", 0.0)),
        "full_time_reward_floor": float(final_info.get("full_time_reward_floor", getattr(env, "full_time_reward_floor", 0.0))),
        "local_time_planner": str(final_info.get("local_time_planner", getattr(env, "local_time_planner", ""))),
        "time_eval_mode": str(final_info.get("time_eval_mode", getattr(env, "time_eval_mode", ""))),
        "final_reason": str(final_info.get("reason", "")),
        "terminated": bool(getattr(env, "terminated", False)),
        "truncated": bool(getattr(env, "truncated", False)),
    }


def print_summary(summary):
    """在终端打印评估结果。"""
    print("\n" + "=" * 76)
    print("SAC Evaluation Summary")
    print("=" * 76)
    print(f"steps:                  {summary['steps']}")
    print(f"episode wall time:      {summary['episode_wall_time_s']:.4f} s")
    print(f"step wall time:         {summary['step_wall_time_ms']:.3f} ms/step")
    print(f"total reward:           {summary['total_reward']:.3f}")
    print(f"coverage:               {summary['coverage_ratio'] * 100.0:.3f}%")
    print(f"covered / remaining:    {summary['covered_cells']} / {summary['remaining_cells']}")
    print(f"invalid / valid:        {summary['invalid_actions']} / {summary['valid_actions']}")
    print(f"local planner:          {summary['local_time_planner']}")
    print(f"time eval mode:         {summary['time_eval_mode']}")
    print(f"local time sum:         {summary['local_time_sum']:.4f} s")
    print(f"VPOp full-path time:    {summary['full_time']:.4f} s")
    print(f"new area sum:           {summary['new_area_mm2_sum']:.3f} mm^2")
    print(f"reward efficiency:      {summary['efficiency_reward_sum']:.3f}")
    print(f"reward coverage:        {summary['coverage_reward_sum']:.3f}")
    print(f"reward invalid:         {summary['invalid_reward_sum']:.3f}")
    print(f"reward full time sum:   {summary['full_time_reward_sum']:.3f}")
    print(f"reward full time last:  {summary['last_full_time_reward']:.3f}")
    print(f"reward full time raw:   {summary['last_raw_full_time_reward']:.3f}")
    print(f"reward full time floor: {summary['full_time_reward_floor']:.3f}")
    print(f"terminated/truncated:   {summary['terminated']} / {summary['truncated']}")
    print(f"final reason:           {summary['final_reason']}")
    print("=" * 76)


def save_figures(env, save_dir, model_path, summary):
    """保存最终 render 图像。"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(model_path)
    agent_info = f"{model_path.parent.parent.parent.name}_{model_path.parent.parent.name}"

    prefix = (
        f"SAC_Eval_{agent_info}_"
        f"Cov_{summary['coverage_ratio'] * 100.0:.2f}_"
        f"Time_{summary['full_time']:.2f}s_"
        f"Wall_{summary['episode_wall_time_s']:.2f}s_"
        f"R_{summary['total_reward']:.1f}_"
        f"Invalid_{summary['invalid_actions']}"
    )

    saved_paths = []

    if getattr(env, "fig", None) is not None:
        path = save_dir / f"{prefix}_path_coverage.png"
        env.fig.savefig(path, dpi=300, bbox_inches="tight")
        saved_paths.append(path)

    if getattr(env, "speed_fig", None) is not None:
        path = save_dir / f"{prefix}_full_speed_diagnostics.png"
        env.speed_fig.savefig(path, dpi=300, bbox_inches="tight")
        saved_paths.append(path)

    return saved_paths


def evaluate_and_plot():
    """加载模型，执行一个 episode，保存并显示最终图像。"""
    print(f"Loading SAC model: {MODEL_PATH}")
    print(f"Evaluation device: {DEVICE}")

    model = SAC.load(MODEL_PATH, device=DEVICE)
    print("Model loaded.")

    env = MillingEnvNurbs(render_mode=("plot" if RENDER_DURING_ROLLOUT else None))
    obs, _ = env.reset(seed=SEED)

    max_steps_guard = (
        int(MAX_STEPS_GUARD)
        if MAX_STEPS_GUARD is not None
        else int(getattr(env, "max_episode_actions", env.max_generation_steps + 50))
    )

    done = False
    step_count = 0
    total_reward = 0.0
    final_info = {}

    print(f"Start rollout | deterministic={DETERMINISTIC} | seed={SEED} | guard={max_steps_guard}")

    while not done and step_count < max_steps_guard:
        action, _ = model.predict(obs, deterministic=DETERMINISTIC)
        obs, reward, terminated, truncated, info = env.step(action)

        done = bool(terminated or truncated)
        step_count += 1
        total_reward += float(reward)
        final_info = info

    if not done:
        final_info = dict(final_info)
        final_info["reason"] = "max_steps_guard_reached"
        print(f"Warning: rollout stopped by max_steps_guard={max_steps_guard}.")

    # 只在结束后渲染最终图。不要为了画图强行修改 env.truncated。
    env.render_mode = "plot"
    try:
        env.render(force_speed_diagnostics=True)
    except TypeError:
        env.render()

    summary = summarize_episode(env, final_info, step_count, total_reward)
    print_summary(summary)

    saved_paths = save_figures(env, SAVE_DIR, MODEL_PATH, summary)
    for path in saved_paths:
        print(f"Saved figure: {path}")

    if SHOW_FIGURE:
        plt.ioff()
        plt.show(block=True)
    else:
        env.close()

    return summary


if __name__ == "__main__":
    evaluate_and_plot()
