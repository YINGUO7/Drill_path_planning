"""
test_baseline_temperature_field.py

功能：
    使用“基于温度场/PDE 等值线”的传统路径规划方法生成 baseline 路径，
    然后调用当前 Env 中的速度规划方法 estimate_machining_time() 计算完整路径加工时间。

使用方法：
    1) 把本文件复制到当前 ENV 代码目录下，例如 ENV3/。
    2) 修改下面的 ENV_MODULE_NAME，确保它对应当前环境文件，例如 Env3、Env1、Env2。
    3) 运行：python test_baseline_temperature_field.py

注意：
    本脚本不会训练 RL，也不会调用 step()。
    它只复用 Env 的几何、扫掠、固定加工区 mask 和速度规划函数。
"""

from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

try:
    import matplotlib.pyplot as plt
except Exception:  # 服务器极简环境下允许无 matplotlib
    plt = None


# =========================
# 手动配置区
# =========================
ENV_MODULE_NAME = "env"   # 例如："Env3"、"Env1"、"Env2"
ENV_CLASS_NAME = "MillingEnvNurbs"

SAVE_DIR = Path("./baseline_temperature_results")
SAVE_PREFIX = "temperature_field_baseline"

# baseline 路径生成参数。通常不需要和 RL 的 generation_points_per_turn 一致。
SAMPLES_PER_TURN = 360
DENSE_LEVEL_COUNT = 81
MAX_INTERNAL_TURNS = None       # None 表示使用 env.max_internal_turns
TRIM_LAST_TURN = True           # 覆盖首次达标后，二分截短最后一圈
RESAMPLE_SPACING_M = None       # None 表示使用 env.curve_sample_spacing；也可写 0.5e-3

# 速度规划参数。这里会调用 env.estimate_machining_time(...)
SPEED_PLANNER = "vpop"          # 当前 Env 支持："fast"、"strict"、"vpop"
SPEED_MODE = "full"

SHOW_FIGURE = False             # 服务器一般设为 False
SAVE_FIGURE = True
SAVE_NPZ = True


# =========================
# 基础工具函数
# =========================
def load_env_class():
    module = importlib.import_module(ENV_MODULE_NAME)
    return getattr(module, ENV_CLASS_NAME)


def polyline_length(path: np.ndarray) -> float:
    path = np.asarray(path, dtype=np.float64)
    if len(path) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def resample_polyline_by_arclength(path: np.ndarray, spacing: float) -> np.ndarray:
    """按弧长重采样折线路径，避免局部点太疏或太密影响速度规划。"""
    path = np.asarray(path, dtype=np.float64)
    if len(path) < 2:
        return path.copy()
    ds = np.linalg.norm(np.diff(path, axis=0), axis=1)
    keep = np.concatenate(([True], ds > 1e-12))
    path = path[keep]
    if len(path) < 2:
        return path.copy()
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))))
    total = float(s[-1])
    if total <= 1e-12:
        return path.copy()
    count = max(2, int(np.ceil(total / max(float(spacing), 1e-12))) + 1)
    s_new = np.linspace(0.0, total, count)
    x_new = np.interp(s_new, s, path[:, 0])
    y_new = np.interp(s_new, s, path[:, 1])
    return np.column_stack((x_new, y_new))


def solve_pde_temperature_field(env) -> np.ndarray:
    """在当前 Env 的 tool_center_feasible_mask 上求解 Poisson/PDE 标量场。"""
    mask = np.asarray(env.tool_center_feasible_mask, dtype=bool)
    ny, nx = mask.shape
    valid_y, valid_x = np.where(mask)
    num_valid = len(valid_y)
    if num_valid == 0:
        raise RuntimeError("tool_center_feasible_mask 为空，无法生成温度场 baseline。")

    index_map = np.full((ny, nx), -1, dtype=np.int32)
    index_map[valid_y, valid_x] = np.arange(num_valid)

    rows, cols, values = [], [], []
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nyb, nxb = valid_y + dy, valid_x + dx
        inside = (nyb >= 0) & (nyb < ny) & (nxb >= 0) & (nxb < nx)
        src = np.where(inside)[0]
        src = src[mask[nyb[src], nxb[src]]]
        rows.extend(src.tolist())
        cols.extend(index_map[nyb[src], nxb[src]].tolist())
        values.extend(np.ones(len(src), dtype=np.float64).tolist())

    rows.extend(np.arange(num_valid).tolist())
    cols.extend(np.arange(num_valid).tolist())
    values.extend((-4.0 * np.ones(num_valid, dtype=np.float64)).tolist())

    matrix = coo_matrix((values, (rows, cols)), shape=(num_valid, num_valid)).tocsr()
    rhs = np.full(num_valid, -(float(env.res) ** 2), dtype=np.float64)
    field_values = spsolve(matrix, rhs)

    temperature = np.zeros((ny, nx), dtype=np.float64)
    temperature[valid_y, valid_x] = field_values
    maximum = float(np.max(temperature))
    if maximum <= 0.0 or not np.isfinite(maximum):
        raise RuntimeError("PDE 温度场求解失败：最大值无效。")
    return temperature / maximum


def generate_temperature_field_baseline(env) -> Dict[str, np.ndarray | float | int]:
    """生成基于温度场等值线的中心向外 baseline 螺旋路径。"""
    t0 = time.perf_counter()
    temperature = solve_pde_temperature_field(env)

    x_coords = np.asarray(env.X_real_wp[0, :], dtype=np.float64)
    y_coords = np.asarray(env.Y_real_wp[:, 0], dtype=np.float64)

    center_row, center_col = np.unravel_index(np.argmax(temperature), temperature.shape)
    center_x, center_y = float(x_coords[center_col]), float(y_coords[center_row])
    path_center = np.array([center_x, center_y], dtype=np.float64)

    feasible_mask = np.asarray(env.tool_center_feasible_mask, dtype=bool)
    sorted_temperature = np.sort(temperature[feasible_mask])
    if len(sorted_temperature) == 0:
        raise RuntimeError("feasible temperature samples 为空。")

    def temperature_for_radius(normalized_radius: float) -> float:
        # 与之前 Env 中的 PDE baseline 逻辑保持一致：半径越大，对应越低的等值线温度。
        idx = int((1.0 - np.clip(normalized_radius, 0.0, 1.0) ** 2) * (len(sorted_temperature) - 1))
        return float(sorted_temperature[idx])

    interpolator = RegularGridInterpolator(
        (y_coords, x_coords), temperature, bounds_error=False, fill_value=0.0
    )

    samples_per_turn = int(SAMPLES_PER_TURN)
    dense_level_count = int(DENSE_LEVEL_COUNT)
    angles = np.linspace(0.0, 2.0 * np.pi, samples_per_turn, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)

    def contour_for_radius(normalized_radius: float) -> np.ndarray:
        target_temperature = temperature_for_radius(normalized_radius)
        low_r = np.zeros(samples_per_turn, dtype=np.float64)
        high_r = np.full(samples_per_turn, float(env.wp_size), dtype=np.float64)
        for _ in range(35):
            mid_r = 0.5 * (low_r + high_r)
            query = np.column_stack((center_y + mid_r * sin_a, center_x + mid_r * cos_a))
            mid_temperature = interpolator(query)
            inside = mid_temperature > target_temperature
            low_r = np.where(inside, mid_r, low_r)
            high_r = np.where(inside, high_r, mid_r)
        radius = 0.5 * (low_r + high_r)
        return np.column_stack((center_x + radius * cos_a, center_y + radius * sin_a))

    # 与之前温度场算法保持一致：中心固定圆作为第一层。
    requested_start_radius = float(getattr(env, "actual_start_radius", env.start_radius_ratio * env.tool_r))
    core_area = np.count_nonzero(feasible_mask) * float(env.res) ** 2
    equivalent_radius = float(np.sqrt(core_area / np.pi))
    inner_ratio = float(np.clip(requested_start_radius / max(equivalent_radius, 1e-12), 0.02, 0.20))

    dense_radii = np.linspace(inner_ratio, 1.0, dense_level_count)
    dense_contours = np.asarray([contour_for_radius(r) for r in dense_radii])
    seed_circle = path_center + requested_start_radius * np.column_stack((cos_a, sin_a))
    dense_contours[0] = seed_circle

    contour_steps = np.max(np.linalg.norm(np.diff(dense_contours, axis=0), axis=2), axis=1)
    contour_distance = np.concatenate(([0.0], np.cumsum(contour_steps)))

    def contour_at(distance: float) -> Tuple[np.ndarray, float]:
        idx = min(np.searchsorted(contour_distance, distance, side="right") - 1, len(dense_radii) - 2)
        idx = max(idx, 0)
        denom = max(contour_distance[idx + 1] - contour_distance[idx], 1e-12)
        alpha = float(np.clip((distance - contour_distance[idx]) / denom, 0.0, 1.0))
        contour = (1.0 - alpha) * dense_contours[idx] + alpha * dense_contours[idx + 1]
        radius_ratio = (1.0 - alpha) * dense_radii[idx] + alpha * dense_radii[idx + 1]
        return contour, float(radius_ratio)

    target_stepover = float(env.stepover_ratio * 2.0 * env.tool_r)
    max_turns = int(getattr(env, "max_internal_turns", 12) if MAX_INTERNAL_TURNS is None else MAX_INTERNAL_TURNS)
    internal_limit = max(0.0, float(contour_distance[-1]) - target_stepover)

    layer_distances = [0.0]
    while layer_distances[-1] < internal_limit and len(layer_distances) <= max_turns:
        layer_distances.append(min(layer_distances[-1] + target_stepover, internal_limit))

    layers_and_radii = [contour_at(d) for d in layer_distances]
    layers = [item[0] for item in layers_and_radii]
    layer_radii = np.asarray([item[1] for item in layers_and_radii], dtype=np.float64)

    inner_core_mask = np.asarray(env.inner_core_mask, dtype=bool)
    total_planning_cells = max(int(getattr(env, "total_planning_cells", np.count_nonzero(inner_core_mask))), 1)
    allowed_uncovered = int(getattr(env, "allowed_uncovered", max(1, np.ceil(total_planning_cells * (1.0 - env.coverage_target)))))

    raw_points = [layers[0][0]]
    selected = None

    print("正在生成温度场 baseline 螺旋...")
    print(f"  PDE中心: ({center_x * 1000:.3f}, {center_y * 1000:.3f}) mm")
    print(f"  目标刀间距: {target_stepover * 1000:.3f} mm")
    print(f"  内部待覆盖网格: {total_planning_cells}, 允许残留: {allowed_uncovered}")

    for turn in range(1, len(layers)):
        turn_points = []
        for j in range(samples_per_turn):
            phase = j / samples_per_turn
            alpha = phase ** 2 * (3.0 - 2.0 * phase)  # smoothstep
            turn_points.append((1.0 - alpha) * layers[turn - 1][j] + alpha * layers[turn][j])

        candidate_raw = np.vstack((np.asarray(raw_points), np.asarray(turn_points), layers[turn][0]))
        swept = env._rasterize_tool_sweep(candidate_raw)
        uncovered = int(np.count_nonzero(inner_core_mask & (~swept)))
        coverage = 1.0 - uncovered / total_planning_cells
        repeat_cells = int(np.count_nonzero(swept & np.asarray(getattr(env, "fixed_machined_mask", np.zeros_like(inner_core_mask)), dtype=bool)))
        print(f"  第{turn:02d}圈: coverage={coverage * 100:.4f}% remaining={uncovered} repeat={repeat_cells}")

        if uncovered > allowed_uncovered and turn < len(layers) - 1:
            raw_points.extend(turn_points)
            continue

        final_raw = candidate_raw
        if TRIM_LAST_TURN and uncovered <= allowed_uncovered:
            prefix = list(raw_points)
            low, high = 2, len(turn_points)
            while low < high:
                mid = (low + high) // 2
                trial = np.asarray(prefix + turn_points[:mid])
                trial_uncovered = int(np.count_nonzero(inner_core_mask & (~env._rasterize_tool_sweep(trial))))
                if trial_uncovered <= allowed_uncovered:
                    high = mid
                else:
                    low = mid + 1
            final_raw = np.asarray(prefix + turn_points[:low])
            swept = env._rasterize_tool_sweep(final_raw)
            uncovered = int(np.count_nonzero(inner_core_mask & (~swept)))
            coverage = 1.0 - uncovered / total_planning_cells

        selected = {
            "raw_path": np.asarray(final_raw, dtype=np.float64),
            "swept_mask": swept,
            "coverage_ratio": float(coverage),
            "remaining_cells": int(uncovered),
            "planned_turns": float(turn),
            "layer_radii": layer_radii[:turn + 1],
            "temperature": temperature,
            "path_center": path_center,
            "dense_radii": dense_radii,
        }
        if uncovered <= allowed_uncovered:
            break

        raw_points.extend(turn_points)

    if selected is None:
        raise RuntimeError("未能在最大圈数内生成 baseline 路径。")

    spacing = float(getattr(env, "curve_sample_spacing", 0.5e-3) if RESAMPLE_SPACING_M is None else RESAMPLE_SPACING_M)
    baseline_path = resample_polyline_by_arclength(selected["raw_path"], spacing)
    swept = env._rasterize_tool_sweep(baseline_path)
    remaining = int(np.count_nonzero(inner_core_mask & (~swept)))
    coverage = 1.0 - remaining / total_planning_cells

    selected["baseline_path"] = baseline_path
    selected["swept_mask"] = swept
    selected["coverage_ratio"] = float(coverage)
    selected["remaining_cells"] = int(remaining)
    selected["path_length"] = polyline_length(baseline_path)
    selected["generation_wall_time_s"] = float(time.perf_counter() - t0)
    return selected


def evaluate_baseline_with_env_speed_planner(env, baseline: Dict) -> Dict:
    """调用当前 Env 的速度规划方法计算完整路径加工时间。"""
    path = np.asarray(baseline["baseline_path"], dtype=np.float64)
    print("正在调用 Env.estimate_machining_time() 计算 baseline 完整加工时间...")
    t0 = time.perf_counter()
    full_time, profile = env.estimate_machining_time(
        path,
        mode=SPEED_MODE,
        return_profile=True,
        planner=SPEED_PLANNER,
    )
    speed_wall = time.perf_counter() - t0

    swept = np.asarray(baseline["swept_mask"], dtype=bool)
    inner_core_mask = np.asarray(env.inner_core_mask, dtype=bool)
    fixed_mask = np.asarray(getattr(env, "fixed_machined_mask", np.zeros_like(inner_core_mask)), dtype=bool)
    cavity_mask = np.asarray(env.cavity_mask, dtype=bool)
    total_planning_cells = max(int(getattr(env, "total_planning_cells", np.count_nonzero(inner_core_mask))), 1)

    covered = int(np.count_nonzero(inner_core_mask & swept))
    remaining = int(np.count_nonzero(inner_core_mask & (~swept)))
    repeat = int(np.count_nonzero(swept & fixed_mask))
    overcut = int(np.count_nonzero(swept & (~cavity_mask)))
    coverage = covered / total_planning_cells

    dynamics = profile.get("dynamics", {}) if isinstance(profile, dict) else {}
    result = {
        "full_time": float(full_time),
        "speed_planner_wall_time_s": float(speed_wall),
        "path_length": float(baseline["path_length"]),
        "coverage_ratio": float(coverage),
        "covered_cells": covered,
        "remaining_cells": remaining,
        "repeat_cells": repeat,
        "overcut_cells": overcut,
        "profile": profile,
        "max_speed": float(np.max(profile.get("speed", [0.0]))) if isinstance(profile, dict) and len(profile.get("speed", [])) else 0.0,
        "max_abs_ax": float(np.max(np.abs(dynamics.get("ax", [0.0])))) if len(dynamics.get("ax", [])) else 0.0,
        "max_abs_ay": float(np.max(np.abs(dynamics.get("ay", [0.0])))) if len(dynamics.get("ay", [])) else 0.0,
        "max_abs_jx": float(np.max(np.abs(dynamics.get("jx", [0.0])))) if len(dynamics.get("jx", [])) else 0.0,
        "max_abs_jy": float(np.max(np.abs(dynamics.get("jy", [0.0])))) if len(dynamics.get("jy", [])) else 0.0,
    }
    return result


def save_outputs(env, baseline: Dict, eval_result: Dict) -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    path = np.asarray(baseline["baseline_path"], dtype=np.float64)
    profile = eval_result["profile"]

    if SAVE_NPZ:
        npz_path = SAVE_DIR / f"{SAVE_PREFIX}.npz"
        save_dict = {
            "baseline_path": path,
            "raw_path": np.asarray(baseline["raw_path"], dtype=np.float64),
            "temperature": np.asarray(baseline["temperature"], dtype=np.float64),
            "path_center": np.asarray(baseline["path_center"], dtype=np.float64),
            "coverage_ratio": np.asarray(eval_result["coverage_ratio"]),
            "remaining_cells": np.asarray(eval_result["remaining_cells"]),
            "repeat_cells": np.asarray(eval_result["repeat_cells"]),
            "overcut_cells": np.asarray(eval_result["overcut_cells"]),
            "path_length": np.asarray(eval_result["path_length"]),
            "full_time": np.asarray(eval_result["full_time"]),
        }
        if isinstance(profile, dict):
            for key in ["s", "speed", "cap_s", "cap", "node_s", "node_v", "velocity_geom"]:
                if key in profile:
                    save_dict[f"profile_{key}"] = np.asarray(profile[key])
            dyn = profile.get("dynamics", {})
            for key in ["time", "ax", "ay", "jx", "jy"]:
                if key in dyn:
                    save_dict[f"dynamics_{key}"] = np.asarray(dyn[key])
        np.savez_compressed(npz_path, **save_dict)
        print(f"已保存数据: {npz_path}")

    if plt is None or not SAVE_FIGURE:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    ax.imshow(env.cavity_mask, origin="lower", extent=[0, env.wp_size * 1000, 0, env.wp_size * 1000], cmap="gray", alpha=0.25)
    ax.imshow(env.inner_core_mask, origin="lower", extent=[0, env.wp_size * 1000, 0, env.wp_size * 1000], cmap="Reds", alpha=0.15)
    ax.plot(path[:, 0] * 1000, path[:, 1] * 1000, lw=1.2, label="Temperature-field baseline")
    ax.scatter(path[0, 0] * 1000, path[0, 1] * 1000, s=35, marker="o", label="start")
    ax.scatter(path[-1, 0] * 1000, path[-1, 1] * 1000, s=35, marker="s", label="end")
    ax.set_aspect("equal")
    ax.set_title("Baseline path")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.legend()

    ax = axes[0, 1]
    im = ax.imshow(baseline["temperature"], origin="lower", extent=[0, env.wp_size * 1000, 0, env.wp_size * 1000])
    ax.set_aspect("equal")
    ax.set_title("PDE temperature field")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    if isinstance(profile, dict) and len(profile.get("s", [])):
        s_mm = np.asarray(profile.get("s")) * 1000.0
        ax.plot(s_mm, np.asarray(profile.get("speed")) * 1000.0, label="speed")
        if len(profile.get("cap", [])):
            ax.plot(np.asarray(profile.get("cap_s")) * 1000.0, np.asarray(profile.get("cap")) * 1000.0, "--", label="cap")
    ax.set_title("Feedrate profile")
    ax.set_xlabel("s (mm)")
    ax.set_ylabel("speed (mm/s)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    dyn = profile.get("dynamics", {}) if isinstance(profile, dict) else {}
    if len(dyn.get("time", [])):
        time_arr = np.asarray(dyn.get("time"))
        if len(dyn.get("ax", [])):
            ax.plot(time_arr, np.asarray(dyn.get("ax")), label="ax")
        if len(dyn.get("ay", [])):
            ax.plot(time_arr, np.asarray(dyn.get("ay")), label="ay")
        ax.axhline(env.acc_max, ls="--", lw=0.8)
        ax.axhline(-env.acc_max, ls="--", lw=0.8)
    ax.set_title("Axis acceleration")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("m/s²")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"coverage={eval_result['coverage_ratio']*100:.3f}%, "
        f"time={eval_result['full_time']:.3f}s, "
        f"length={eval_result['path_length']*1000:.1f}mm"
    )
    fig.tight_layout()
    fig_path = SAVE_DIR / f"{SAVE_PREFIX}_summary.png"
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    print(f"已保存图片: {fig_path}")

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)


def print_summary(eval_result: Dict, baseline: Dict) -> None:
    print("\n" + "=" * 80)
    print("Temperature-field baseline evaluation")
    print("=" * 80)
    print(f"coverage:              {eval_result['coverage_ratio'] * 100:.4f}%")
    print(f"covered / remaining:   {eval_result['covered_cells']} / {eval_result['remaining_cells']}")
    print(f"repeat cells:          {eval_result['repeat_cells']}")
    print(f"overcut cells:         {eval_result['overcut_cells']}")
    print(f"path length:           {eval_result['path_length'] * 1000:.3f} mm")
    print(f"full machining time:   {eval_result['full_time']:.6f} s")
    print(f"max speed:             {eval_result['max_speed'] * 1000:.3f} mm/s")
    print(f"max |ax| / |ay|:       {eval_result['max_abs_ax']:.6f} / {eval_result['max_abs_ay']:.6f} m/s^2")
    print(f"max |jx| / |jy|:       {eval_result['max_abs_jx']:.6f} / {eval_result['max_abs_jy']:.6f} m/s^3")
    print(f"path generation time:  {baseline['generation_wall_time_s']:.3f} s")
    print(f"speed planner time:    {eval_result['speed_planner_wall_time_s']:.3f} s")
    print("=" * 80)


def main():
    EnvClass = load_env_class()
    env = EnvClass(render_mode=None)
    baseline = generate_temperature_field_baseline(env)
    eval_result = evaluate_baseline_with_env_speed_planner(env, baseline)
    print_summary(eval_result, baseline)
    save_outputs(env, baseline, eval_result)


if __name__ == "__main__":
    main()
