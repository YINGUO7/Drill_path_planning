import numpy as np
import gymnasium as gym
import time
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib import rcParams
from scipy.interpolate import BSpline
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from scipy.signal import find_peaks


# rcParams["font.sans-serif"] = ["SimSun", "Microsoft YaHei", "Arial"]
# rcParams["font.serif"] = ["Times New Roman"]
# rcParams["axes.unicode_minus"] = False


class MillingEnvNurbs(gym.Env):
    """增量式NURBS/B-spline控制点生成环境。

    功能：
        从固定中心圆外侧开始，每个step只新增一个控制点。
        新控制点用增量式极坐标生成，然后用已有全部控制点形成一条完整open-uniform B-spline曲线。
        环境根据完整曲线的刀具扫掠结果计算新增覆盖、重复加工、过切和局部窗口时间奖励。

    输入：
        型腔几何参数、刀具参数、机床动态约束参数、控制点生成参数。

    输出：
        Gymnasium接口：reset()返回初始观测，step(action)返回下一观测、奖励、终止标志和诊断信息。
    """

    metadata = {"render_modes": ["human", "plot", None]}

    def __init__(self, wp_size_m=210e-3, pocket_side_m=190e-3, corner_radius_m=10e-3,
                 tool_dia_m=16e-3, v_max=0.10, acc_max=0.72, jerk_max=1.44,
                 resolution_m=0.5e-3, stepover_ratio=0.99, start_radius_ratio=0.90,
                 coverage_target=0.9998, max_internal_turns=12, render_mode="plot",
                 curve_sample_count=900, generation_points_per_turn=48,
                 curve_sample_spacing_m=0.5e-3, generation_time_window_points=240,
                 observation_grid_size=64,
                 time_eval_mode="local", coverage_efficiency_weight=0.01,
                 envelope_uncovered_weight=0.005, coverage_completion_bonus=200.0, remaining_cell_penalty=0.02,
                 invalid_action_penalty=-8.0, use_local_step_sweep=True,
                 local_sweep_tail_points=None, max_consecutive_invalid_actions=1000,
                 max_episode_actions=None,
                 radial_gap_safety_ratio=None, radial_gap_bin_count=75,
                 radial_gap_violation_penalty=None,
                 local_time_planner="fast", local_fast_jerk_factor=0.8,
                 full_time_reward_weight=50.0, full_time_reference=120.0,
                 full_time_reward_floor=-60.0,
                 profile_step=False, profile_interval=20):
        """初始化环境对象。

        输入：
            wp_size_m: 工件计算区域边长，单位m。
            pocket_side_m: 三角形型腔边长，单位m。
            corner_radius_m: 型腔圆角半径，单位m。
            tool_dia_m: 刀具直径，单位m。
            v_max: 进给速度上限，单位m/s。
            acc_max: 加速度上限，单位m/s^2。
            jerk_max: jerk上限，单位m/s^3。
            resolution_m: 栅格分辨率，单位m。
            stepover_ratio: 基准刀间距相对刀具直径的比例。
            start_radius_ratio: 固定中心圆半径相对刀具半径的比例。
            coverage_target: 内部规划区域目标覆盖率。
            max_internal_turns: 最大内部螺旋圈数估计，用于step截断。
            render_mode: "plot"、"human"或None。
            curve_sample_count: 完整曲线的最小采样点数。
            curve_sample_spacing_m: 完整曲线自适应采样的目标点间距。
            generation_points_per_turn: 每圈期望新增的控制点数量。
            generation_time_window_points: 局部时间奖励使用的曲线尾部采样点数量。
            observation_grid_size: 覆盖状态观测的下采样尺寸。
            time_eval_mode: step奖励中的时间估计模式，"local"使用尾部窗口，"full"使用完整路径。
            coverage_efficiency_weight: 单位时间新增覆盖效率奖励权重。
            envelope_uncovered_weight: 当前路径包络区域内未覆盖面积惩罚权重。
            coverage_completion_bonus: 达到覆盖目标时的终局奖励。
            remaining_cell_penalty: 截断但未覆盖完成时，对剩余网格数量的惩罚权重。
            invalid_action_penalty: 控制点越界或碰壁时的单步惩罚。
            use_local_step_sweep: True时，训练step只对曲线尾部局部路径做刀具扫掠。
            local_sweep_tail_points: 局部扫掠使用的尾部路径点数量；None时使用generation_time_window_points。
            max_consecutive_invalid_actions: 连续非法动作截断阈值，避免确定性eval在同一状态死循环。
            max_episode_actions: episode总动作数上限，包含非法动作；None时自动设为有效step上限的1.5倍。
            radial_gap_safety_ratio: 射线/极坐标bin相邻路径层最大间距系数；None时使用stepover_ratio。
            radial_gap_bin_count: 径向gap检查的角度分桶数量。
            radial_gap_violation_penalty: 径向gap超限时的惩罚；None时为1.5倍invalid_action_penalty。
            local_time_planner: "fast"、"strict"或"vpop"；fast用于训练step局部时间估计。
            local_fast_jerk_factor: fast局部时间估计中的jerk安全系数。
            full_time_reward_weight: episode结束时完整路径加工时间惩罚权重。
            full_time_reference: 完整路径加工时间归一化基准，单位s。
            full_time_reward_floor: 完整路径时间惩罚下限，防止极端坏路径产生爆炸reward。
            profile_step: True时打印step内部耗时分解。
            profile_interval: 每隔多少次step打印一次耗时。

        输出：
            无返回；初始化self属性。
        """
        super().__init__()
        self.render_mode = render_mode
        self.wp_size, self.pocket_side = float(wp_size_m), float(pocket_side_m)
        self.corner_radius, self.tool_r = float(corner_radius_m), 0.5 * float(tool_dia_m)
        self.v_max, self.acc_max, self.jerk_max = float(v_max), float(acc_max), float(jerk_max)
        self.res, self.stepover_ratio = float(resolution_m), float(stepover_ratio)
        self.start_radius_ratio, self.coverage_target = float(start_radius_ratio), float(coverage_target)
        self.max_internal_turns = int(max_internal_turns)
        self.curve_sample_count = int(curve_sample_count)
        self.curve_sample_spacing = float(curve_sample_spacing_m)
        self.generation_points_per_turn = int(generation_points_per_turn)
        self.generation_time_window_points = int(generation_time_window_points)
        self.observation_grid_size = int(observation_grid_size)
        self.time_eval_mode = str(time_eval_mode).lower()
        self.coverage_efficiency_weight = float(coverage_efficiency_weight)
        self.envelope_uncovered_weight = float(envelope_uncovered_weight)
        self.coverage_completion_bonus = float(coverage_completion_bonus)
        self.remaining_cell_penalty = float(remaining_cell_penalty)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.use_local_step_sweep = bool(use_local_step_sweep)
        self.local_sweep_tail_points = (None if local_sweep_tail_points is None else int(local_sweep_tail_points))
        self.max_consecutive_invalid_actions = max(1, int(max_consecutive_invalid_actions))
        self.requested_max_episode_actions = None if max_episode_actions is None else int(max_episode_actions)
        self.radial_gap_safety_ratio = float(self.stepover_ratio if radial_gap_safety_ratio is None else radial_gap_safety_ratio)
        self.radial_gap_bin_count = int(radial_gap_bin_count)
        self.radial_gap_violation_penalty = (
            1.5 * self.invalid_action_penalty if radial_gap_violation_penalty is None
            else float(radial_gap_violation_penalty)
        )
        self.local_time_planner = str(local_time_planner).lower()
        self.local_fast_jerk_factor = float(local_fast_jerk_factor)
        self.full_time_reward_weight = float(full_time_reward_weight)
        self.full_time_reference = float(full_time_reference)
        self.full_time_reward_floor = float(full_time_reward_floor)
        self.profile_step = bool(profile_step)
        self.profile_interval = max(1, int(profile_interval))
        self.profile_records = []

        self._validate_parameters()
        self.wp_grid_points = int(round(self.wp_size / self.res))
        self.center_wp = 0.5 * self.wp_size
        y_idx, x_idx = np.ogrid[:self.wp_grid_points, :self.wp_grid_points]
        self.Y_real_wp, self.X_real_wp = y_idx * self.res, x_idx * self.res
        grid_x = np.broadcast_to(self.X_real_wp, (self.wp_grid_points, self.wp_grid_points))
        grid_y = np.broadcast_to(self.Y_real_wp, (self.wp_grid_points, self.wp_grid_points))
        self.grid_points_flat = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        self.path_center = np.array([self.center_wp, self.center_wp], dtype=np.float64)

        self._build_rounded_triangle()
        self._precompute_tool_disk_offsets()
        self._build_fixed_machining_regions()
        self._build_observation_roi()
        self._init_generation_parameters()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "state": spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32),
            "coverage": spaces.Box(low=0, high=3,
                                   shape=(1, self.observation_grid_size, self.observation_grid_size),
                                   dtype=np.uint8),
        })
        self.fig = self.axes = self.coverage_image = None
        self.envelope_image = None
        self.envelope_contours = []
        self.coverage_path_line = self.coverage_local_time_path_line = None
        self.nurbs_path_line = self.control_polygon_line = self.local_time_path_line = None
        self.control_scatter = self.current_point_scatter = self.current_info_text = None
        self.curvature_line = self.action_dtheta_line = self.action_dr_line = None
        self.action_current_dtheta = self.action_current_dr = None
        self.speed_fig = self.speed_axes = None
        self.full_speed_line = self.full_cap_line = self.full_node_scatter = None
        self.full_ax_line = self.full_ay_line = None
        self.full_jx_line = self.full_jy_line = None
        self.full_time_line = self.full_curvature_line = None
        self.full_summary_text = None

    def _validate_parameters(self):
        """检查初始化参数合法性。

        输入：
            使用self中已经保存的初始化参数。

        输出：
            无返回；非法时抛出ValueError。
        """
        if self.corner_radius <= self.tool_r:
            raise ValueError("圆角半径必须大于刀具半径。")
        if self.pocket_side >= self.wp_size:
            raise ValueError("型腔边长必须小于计算区域尺寸。")
        if not 0.0 < self.stepover_ratio < 1.0:
            raise ValueError("stepover_ratio必须位于(0, 1)内。")
        if not 0.0 < self.start_radius_ratio <= 0.90:
            raise ValueError("start_radius_ratio必须位于(0, 0.90]内。")
        if not 0.0 < self.coverage_target <= 1.0:
            raise ValueError("coverage_target必须位于(0, 1]内。")
        if self.curve_sample_count < 50:
            raise ValueError("curve_sample_count至少为50。")
        if self.curve_sample_spacing <= 0.0:
            raise ValueError("curve_sample_spacing_m必须大于0。")
        if self.generation_points_per_turn < 6:
            raise ValueError("generation_points_per_turn至少为6。")
        if self.generation_time_window_points < 20:
            raise ValueError("generation_time_window_points至少为20。")
        if self.observation_grid_size < 16:
            raise ValueError("observation_grid_size至少为16。")
        if self.radial_gap_safety_ratio <= 0.0:
            raise ValueError("radial_gap_safety_ratio必须大于0。")
        if self.radial_gap_bin_count < 24:
            raise ValueError("radial_gap_bin_count至少为24。")
        if self.time_eval_mode not in ("local", "full"):
            raise ValueError("time_eval_mode必须为'local'或'full'。")
        if self.local_time_planner not in ("fast", "strict", "vpop"):
            raise ValueError("local_time_planner必须为'fast'、'strict'或'vpop'。")
        if not 0.0 < self.local_fast_jerk_factor <= 1.0:
            raise ValueError("local_fast_jerk_factor必须位于(0, 1]内。")
        if self.full_time_reference <= 0.0:
            raise ValueError("full_time_reference必须大于0。")
        if self.full_time_reward_floor > 0.0:
            raise ValueError("full_time_reward_floor必须小于或等于0。")

    def _precompute_tool_disk_offsets(self):
        """预计算刀具圆盘对应的栅格偏移。

        输入：
            无；使用tool_r和res。

        输出：
            self.tool_disk_row_offsets / self.tool_disk_col_offsets: 圆盘内像素偏移。

        说明：
            训练step的局部扫掠使用圆盘盖章法，避免每步建立KDTree并查询局部网格。
        """
        radius_px = int(np.ceil(self.tool_r / self.res))
        rr, cc = np.mgrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
        inside = (rr * self.res) ** 2 + (cc * self.res) ** 2 <= (self.tool_r + 0.5 * self.res) ** 2
        self.tool_disk_row_offsets = rr[inside].astype(np.int64)
        self.tool_disk_col_offsets = cc[inside].astype(np.int64)

    def _build_rounded_triangle(self):
        """构建圆角三角形型腔和刀具中心可行区域。

        输入：
            使用工件尺寸、型腔边长、圆角半径、刀具半径和网格坐标。

        输出：
            self.cavity_mask: 型腔区域mask。
            self.distance_to_cavity_wall: 到型腔外边界的距离场。
            self.tool_center_feasible_mask: 刀具中心可行区域mask。
        """
        cx = cy = self.center_wp
        r_in = self.pocket_side / (2.0 * np.sqrt(3.0))
        r_out, r_corner = 2.0 * r_in, self.corner_radius
        x, y = self.X_real_wp, self.Y_real_wp
        side_1 = y >= cy - r_in + r_corner
        side_2 = np.sqrt(3.0) * (x - cx) + y - cy <= r_out - 2.0 * r_corner
        side_3 = -np.sqrt(3.0) * (x - cx) + y - cy <= r_out - 2.0 * r_corner
        shrunk_triangle = side_1 & side_2 & side_3
        distance = distance_transform_edt(~shrunk_triangle) * self.res
        self.cavity_mask = distance <= r_corner
        self.distance_to_cavity_wall = distance_transform_edt(self.cavity_mask) * self.res
        self.tool_center_feasible_mask = self.cavity_mask & (self.distance_to_cavity_wall >= self.tool_r)
        self.total_valid_cells = int(np.count_nonzero(self.cavity_mask))

    def _rasterize_tool_sweep(self, path):
        """将刀具中心完整路径转为刀具扫掠区域。

        输入：
            path: 形状为(N, 2)的路径点数组，单位m。

        输出：
            swept_mask: bool矩阵，True表示该网格被刀具半径覆盖。

        说明：
            该函数会对整个工件网格查询到路径的最近距离，结果准确但开销较大。
            主要用于固定中心圆、episode结束、评估和render中的完整覆盖状态刷新。
        """
        path = np.asarray(path, dtype=np.float64)
        if len(path) == 0:
            return np.zeros_like(self.cavity_mask, dtype=bool)
        distance, _ = cKDTree(path).query(self.grid_points_flat)
        return distance.reshape(self.wp_grid_points, self.wp_grid_points) <= self.tool_r

    def _rasterize_tool_sweep_local(self, path, tail_points=None):
        """只对曲线尾部局部路径做刀具扫掠。

        输入：
            path: 形状为(N, 2)的完整路径点数组，单位m。
            tail_points: 使用最后多少个路径点；None时使用local_sweep_tail_points或generation_time_window_points。

        输出：
            swept_mask: bool矩阵，只有局部包围盒附近可能为True。

        说明：
            训练step中只需要评价新增控制点带来的局部覆盖变化。该函数先截取尾部路径，
            再沿路径段按栅格间距补点，对每个中心点使用预计算刀具圆盘offset做盖章。
            相比每步建立KDTree并查询局部网格，该方法更适合训练中的高频局部扫掠。
        """
        path = np.asarray(path, dtype=np.float64)
        if len(path) == 0:
            return np.zeros_like(self.cavity_mask, dtype=bool)

        if tail_points is None:
            tail_points = self.local_sweep_tail_points or self.generation_time_window_points
        tail_points = max(2, int(tail_points))
        local_path = path[-min(len(path), tail_points):]

        swept = np.zeros_like(self.cavity_mask, dtype=bool)
        if len(local_path) == 1:
            centers = local_path
        else:
            centers_parts = []
            max_step = max(0.75 * self.res, 1e-12)
            for p0, p1 in zip(local_path[:-1], local_path[1:]):
                seg_len = float(np.linalg.norm(p1 - p0))
                n = max(1, int(np.ceil(seg_len / max_step)))
                alpha = np.linspace(0.0, 1.0, n + 1, endpoint=True)
                seg = p0[None, :] + alpha[:, None] * (p1 - p0)[None, :]
                centers_parts.append(seg[:-1])
            centers_parts.append(local_path[-1:])
            centers = np.vstack(centers_parts)

        cols = np.rint(centers[:, 0] / self.res).astype(np.int64)
        rows = np.rint(centers[:, 1] / self.res).astype(np.int64)
        valid = (rows >= 0) & (rows < self.wp_grid_points) & (cols >= 0) & (cols < self.wp_grid_points)
        if not np.any(valid):
            return swept
        rows, cols = rows[valid], cols[valid]
        center_index = np.unique(rows * self.wp_grid_points + cols)
        rows = center_index // self.wp_grid_points
        cols = center_index % self.wp_grid_points

        rr = rows[:, None] + self.tool_disk_row_offsets[None, :]
        cc = cols[:, None] + self.tool_disk_col_offsets[None, :]
        valid_disk = (rr >= 0) & (rr < self.wp_grid_points) & (cc >= 0) & (cc < self.wp_grid_points)
        swept[rr[valid_disk], cc[valid_disk]] = True
        return swept

    def _step_sweep_mask(self, path):
        """获取训练step使用的扫掠mask。

        输入：
            path: 当前完整曲线采样点，单位m。

        输出：
            swept_mask: bool矩阵。

        说明：
            默认使用局部尾部扫掠以降低step耗时；如果关闭use_local_step_sweep，
            则退回完整扫掠，便于对照调试。
        """
        if self.use_local_step_sweep:
            return self._rasterize_tool_sweep_local(path)
        return self._rasterize_tool_sweep(path)

    def _refresh_full_sweep_state(self):
        """用当前完整路径刷新visited覆盖状态。

        输入：
            无；使用current_path、contour_pass_mask和seed_pass_mask。

        输出：
            full_swept: 当前完整路径的扫掠mask。

        说明：
            训练step为了速度只做局部扫掠；episode结束、评估和render需要完整路径覆盖结果时，
            调用该函数重新计算完整扫掠，并用固定加工区域+完整路径覆盖重建visited。
        """
        full_swept = self._rasterize_tool_sweep(self.current_path)
        visited = np.zeros_like(self.cavity_mask, dtype=np.uint8)
        visited[self.contour_pass_mask] = 1
        visited[self.seed_pass_mask] = 1
        visited[full_swept & self.inner_core_mask] = 2
        self.visited = visited
        return full_swept

    def _build_fixed_machining_regions(self):
        """构建固定已加工区域。

        输入：
            使用型腔mask、刀具半径、中心圆半径比例。

        输出：
            self.contour_pass_mask: 外圈精加工已覆盖区域。
            self.seed_pass_mask: 中心固定圆已覆盖区域。
            self.inner_core_mask: RL需要规划覆盖的内部区域。
        """
        self.contour_pass_mask = self.cavity_mask & (self.distance_to_cavity_wall <= 2.0 * self.tool_r)
        self.actual_start_radius = self.start_radius_ratio * self.tool_r
        theta = np.linspace(0.0, 2.0 * np.pi, max(240, self.generation_points_per_turn * 12), endpoint=True)
        self.seed_circle_path = self.path_center + self.actual_start_radius * np.column_stack((np.cos(theta), np.sin(theta)))
        self.seed_pass_mask = self.cavity_mask & self._rasterize_tool_sweep(self.seed_circle_path)
        self.fixed_machined_mask = self.contour_pass_mask | self.seed_pass_mask
        self.inner_core_mask = self.cavity_mask & (~self.fixed_machined_mask)
        self.total_planning_cells = int(np.count_nonzero(self.inner_core_mask))
        self.allowed_uncovered = max(1, int(np.ceil(self.total_planning_cells * (1.0 - self.coverage_target))))

    def _build_observation_roi(self):
        """构建覆盖观测使用的型腔ROI裁剪范围。

        输入：
            无；使用cavity_mask和刀具半径。

        输出：
            self.obs_roi: (row0, row1, col0, col1)，用于只观察型腔附近区域。
        """
        rows, cols = np.where(self.cavity_mask)
        if len(rows) == 0:
            self.obs_roi = (0, self.wp_grid_points, 0, self.wp_grid_points)
            return
        padding = max(2, int(np.ceil(1.5 * self.tool_r / self.res)))
        row0 = max(0, int(rows.min()) - padding)
        row1 = min(self.wp_grid_points, int(rows.max()) + padding + 1)
        col0 = max(0, int(cols.min()) - padding)
        col1 = min(self.wp_grid_points, int(cols.max()) + padding + 1)
        self.obs_roi = (row0, row1, col0, col1)

    def _init_generation_parameters(self):
        """初始化控制点生成参数。

        输入：
            使用刀具半径、stepover_ratio、generation_points_per_turn、max_internal_turns。

        输出：
            self.base_stepover: 基准刀间距。
            self.base_radial_step: 每个控制点的基准半径增量。
            self.min_radial_step / self.max_radial_step: 动作允许的有符号半径增量范围。
            self.base_dtheta: 参考角度增量。
            self.target_control_spacing: 控制点之间期望的近似空间步长。
            self.generation_radius_limit: 控制点最大半径。
            self.generation_radius_floor: 控制点最小半径。
            self.max_generation_steps: 最大step数量。
        """
        self.base_stepover = self.stepover_ratio * 2.0 * self.tool_r
        self.base_dtheta = 2.0 * np.pi / self.generation_points_per_turn
        self.min_dtheta = 0.65 * self.base_dtheta
        self.max_dtheta = 1.45 * self.base_dtheta
        self.target_control_spacing = self.base_stepover
        self.base_radial_step = self.base_stepover / self.generation_points_per_turn
        # 半径动作范围应围绕“单个控制点的基准半径增量”设置，而不是围绕整圈刀间距设置。
        # 这样动作尺度与generation_points_per_turn一致：每个step只调整当前控制点的局部径向推进。
        self.min_radial_step = -2.0 * self.base_radial_step
        self.max_radial_step = 6.0 * self.base_radial_step
        self.generation_radius_floor = 0.70 * self.actual_start_radius
        feasible_points = self.grid_points_flat[self.tool_center_feasible_mask.ravel()]
        self.generation_radius_limit = float(np.max(np.linalg.norm(feasible_points - self.path_center, axis=1)))
        self._precompute_boundary_radius()
        self.max_generation_steps = int(np.ceil(2.50 * self.max_internal_turns * self.generation_points_per_turn))
        if self.requested_max_episode_actions is None:
            self.max_episode_actions = int(np.ceil(2 * self.max_generation_steps))
        else:
            self.max_episode_actions = max(1, int(self.requested_max_episode_actions))

    def _point_inside_workspace(self, point):
        """判断点是否位于计算区域内部。

        输入：
            point: 二维点坐标，单位m。

        输出：
            inside: bool，True表示点没有越出工件计算区域。
        """
        return bool(0.0 <= point[0] <= self.wp_size and 0.0 <= point[1] <= self.wp_size)

    def _ray_boundary_radius(self, theta):
        """计算某个角度方向上的刀具中心可行边界半径。

        输入：
            theta: 极角，单位rad。

        输出：
            radius: 从path_center沿theta方向到可行区域边界的距离，单位m。
        """
        direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
        low, high = 0.0, self.generation_radius_limit
        high_point = self.path_center + high * direction
        if self._tool_center_is_feasible(high_point):
            return float(high)
        for _ in range(32):
            mid = 0.5 * (low + high)
            point = self.path_center + mid * direction
            if self._tool_center_is_feasible(point):
                low = mid
            else:
                high = mid
        return float(low)

    def _precompute_boundary_radius(self, count=2048):
        """预计算型腔归一化极坐标中的边界半径表。

        输入：
            count: 角度采样数量。

        输出：
            self.boundary_theta: 周期角度表。
            self.boundary_radius: 每个角度方向上的刀具中心可行边界半径。
        """
        theta = np.linspace(-np.pi, np.pi, int(count), endpoint=False)
        radius = np.array([self._ray_boundary_radius(t) for t in theta], dtype=np.float64)
        self.boundary_theta = np.concatenate((theta, [np.pi]))
        self.boundary_radius = np.concatenate((radius, [radius[0]]))
        self._precompute_corner_theta_from_boundary(theta, radius)
        vectors = self.grid_points_flat - self.path_center
        grid_radius = np.linalg.norm(vectors, axis=1)
        grid_theta = np.arctan2(vectors[:, 1], vectors[:, 0])
        grid_boundary_radius = self._boundary_radius(grid_theta)
        self.grid_theta_flat = grid_theta
        self.grid_radius_flat = grid_radius
        self.grid_rho_flat = grid_radius / np.maximum(grid_boundary_radius, 1e-12)

    def _precompute_corner_theta_from_boundary(self, theta, radius):
        """从边界半径表中提取三角型腔的拐角方向。

        输入：
            theta: 周期角度采样，不包含末尾重复点。
            radius: 对应角度方向上的可行边界半径。

        输出：
            无返回；写入 self.corner_theta。

        说明：
            在中心极坐标中，圆角三角型腔的三个顶点方向通常对应
            r_boundary(theta) 的三个局部峰值。方案3只需要知道这些
            方向，用于在拐角附近减小dtheta、增加控制点密度。
        """
        theta = np.asarray(theta, dtype=np.float64)
        radius = np.asarray(radius, dtype=np.float64)
        if len(theta) < 12:
            self.corner_theta = np.array([], dtype=np.float64)
            return

        left = np.roll(radius, 1)
        right = np.roll(radius, -1)
        peak_idx = np.flatnonzero((radius >= left) & (radius >= right))
        if len(peak_idx) == 0:
            peak_idx = np.argsort(radius)[-3:]

        # 先按半径从大到小取候选，再用最小角距避免同一个圆角峰附近重复取点。
        order = peak_idx[np.argsort(radius[peak_idx])[::-1]]
        min_sep = 2.0 * np.pi / 6.0
        selected = []
        for idx in order:
            angle = float(theta[idx])
            if all(self._angle_abs_diff(angle, old) >= min_sep for old in selected):
                selected.append(angle)
            if len(selected) >= 3:
                break

        if len(selected) < 3:
            for idx in np.argsort(radius)[::-1]:
                angle = float(theta[idx])
                if all(self._angle_abs_diff(angle, old) >= min_sep for old in selected):
                    selected.append(angle)
                if len(selected) >= 3:
                    break

        self.corner_theta = np.asarray(selected, dtype=np.float64)

    @staticmethod
    def _angle_abs_diff(a, b):
        """计算两个角度之间的最小绝对差，范围[0, pi]。"""
        return float(abs((float(a) - float(b) + np.pi) % (2.0 * np.pi) - np.pi))

    def _boundary_radius(self, theta):
        """查询任意角度方向上的刀具中心可行边界半径。

        输入：
            theta: 标量或数组角度，单位rad。

        输出：
            radius: 对应边界半径，单位m。
        """
        theta_array = np.asarray(theta, dtype=np.float64)
        wrapped = (theta_array + np.pi) % (2.0 * np.pi) - np.pi
        radius = np.interp(wrapped, self.boundary_theta, self.boundary_radius)
        return float(radius) if np.ndim(theta) == 0 else radius

    def _normalized_polar_point(self, theta, rho):
        """由型腔归一化极坐标生成控制点。

        输入：
            theta: 极角，单位rad。
            rho: 归一化半径，0表示中心，1表示当前角度上的可行边界。

        输出：
            point: 二维控制点坐标，单位m。
            radius: 实际物理半径，单位m。
            boundary_radius: 当前角度方向上的可行边界半径，单位m。
        """
        boundary_radius = max(self._boundary_radius(theta), 1e-12)
        rho = float(np.clip(rho, 0.0, 1.0))
        radius = rho * boundary_radius
        radial = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
        return self.path_center + radius * radial, float(radius), float(boundary_radius)

    def _path_envelope_mask(self, path):
        """计算当前螺旋路径形成的归一化极坐标包络区域。

        输入：
            path: 当前路径采样点数组，单位m。

        输出：
            envelope_mask: bool矩阵，True表示位于当前路径包络内部。
        """
        path = np.asarray(path, dtype=np.float64)
        if len(path) < 5:
            return np.zeros_like(self.cavity_mask, dtype=bool)
        vectors = path - self.path_center
        theta_unwrapped = np.unwrap(np.arctan2(vectors[:, 1], vectors[:, 0]))
        theta_wrapped = (theta_unwrapped + np.pi) % (2.0 * np.pi) - np.pi
        radius = np.linalg.norm(vectors, axis=1)
        boundary = self._boundary_radius(theta_wrapped)
        rho = np.clip(radius / np.maximum(boundary, 1e-12), 0.0, 1.0)

        bin_count = len(self.boundary_theta) - 1
        bin_width = 2.0 * np.pi / bin_count
        bin_index = np.floor((theta_wrapped + np.pi) / bin_width).astype(int) % bin_count
        rho_by_bin = np.full(bin_count, -np.inf, dtype=np.float64)
        np.maximum.at(rho_by_bin, bin_index, rho)
        valid = np.flatnonzero(np.isfinite(rho_by_bin))
        if len(valid) < 3:
            return np.zeros_like(self.cavity_mask, dtype=bool)

        theta_bins = self.boundary_theta[:-1]
        valid_theta = theta_bins[valid]
        valid_rho = rho_by_bin[valid]
        order = np.argsort(valid_theta)
        valid_theta, valid_rho = valid_theta[order], valid_rho[order]
        interp_theta = np.concatenate((valid_theta - 2.0 * np.pi, valid_theta, valid_theta + 2.0 * np.pi))
        interp_rho = np.concatenate((valid_rho, valid_rho, valid_rho))
        envelope_rho = np.interp(self.grid_theta_flat, interp_theta, interp_rho)
        envelope_flat = self.grid_rho_flat <= envelope_rho + 1e-9
        envelope_mask = envelope_flat.reshape(self.wp_grid_points, self.wp_grid_points)
        return envelope_mask & self.inner_core_mask

    def _radial_gap_constraint(self, path):
        """检查射线/极坐标bin上的相邻路径层径向gap。

        输入：
            path: 候选完整NURBS路径采样点，单位m。

        输出：
            ok: True表示所有已形成路径层之间的径向gap不超过约束。
            max_gap: 检测到的最大相邻径向gap，单位m。
            limit: 允许的最大径向gap，单位m。
            worst_theta: 最大gap所在角度，单位rad；无有效bin时为0。

        说明：
            该约束对应 d_gap(theta) <= eta * 2R。
            它把中心固定圆作为第一层路径，再把当前候选NURBS路径采样点加入。
            对每个角度bin内的半径排序，只检查相邻路径层之间的gap，不检查最外层到边界的gap，
            因为外侧区域可能还没有规划到。
        """
        path = np.asarray(path, dtype=np.float64)
        if len(path) < 5:
            return True, 0.0, float(self.radial_gap_safety_ratio * 2.0 * self.tool_r), 0.0

        points = np.vstack((self.seed_circle_path, path))
        vectors = points - self.path_center
        radius = np.linalg.norm(vectors, axis=1)
        theta = (np.arctan2(vectors[:, 1], vectors[:, 0]) + 2.0 * np.pi) % (2.0 * np.pi)

        bin_count = int(self.radial_gap_bin_count)
        bin_width = 2.0 * np.pi / bin_count
        bin_index = np.floor(theta / bin_width).astype(np.int64) % bin_count
        limit = float(self.radial_gap_safety_ratio * 2.0 * self.tool_r)

        max_gap = 0.0
        worst_bin = 0
        # NURBS采样点数量不大，按bin循环比构造稀疏结构更清楚，也足够快。
        for idx in range(bin_count):
            radii = radius[bin_index == idx]
            if len(radii) < 2:
                continue
            radii = np.sort(radii)
            # 合并同一层附近的密集采样点，避免同一条曲线的多个近邻点干扰层间gap。
            keep = np.concatenate(([True], np.diff(radii) > 0.25 * self.res))
            radii = radii[keep]
            if len(radii) < 2:
                continue
            gap = float(np.max(np.diff(radii)))
            if gap > max_gap:
                max_gap = gap
                worst_bin = idx

        worst_theta = (worst_bin + 0.5) * bin_width
        return max_gap <= limit, max_gap, limit, worst_theta

    def _time_eval_path(self, path, mode, local_window_length=None):
        """取得用于时间估计的路径片段。

        输入：
            path: 当前完整路径。
            mode: "local"或"full"。
            local_window_length: mode="local"时的尾部窗口弧长，单位m；None时使用固定点数回退逻辑。

        输出：
            eval_path: 实际参与速度规划和加工时间估计的路径片段。
        """
        path = np.asarray(path, dtype=np.float64)
        mode = str(mode).lower()
        if mode == "local":
            if local_window_length is not None and len(path) >= 2:
                target_length = max(float(local_window_length), 5.0 * self.curve_sample_spacing)
                s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))))
                start_s = max(0.0, s[-1] - target_length)
                start_idx = int(np.searchsorted(s, start_s, side="left"))
                start_idx = max(0, min(start_idx, len(path) - 2))
                # 至少保留若干点，避免速度规划中的曲率/jerk估计退化。
                start_idx = min(start_idx, max(0, len(path) - 8))
                return path[start_idx:]
            return path[-min(len(path), self.generation_time_window_points):]
        if mode == "full":
            return path
        raise ValueError("mode必须为'local'或'full'。")

    @staticmethod
    def _signed_interval_action(value, center, lower, upper):
        """将[-1, 1]动作映射到以center为零动作值的不对称区间。

        输入：
            value: 单个动作值，范围[-1, 1]。
            center: value=0时的输出值。
            lower: value=-1时的输出下限。
            upper: value=1时的输出上限。

        输出：
            mapped_value: 映射后的标量。
        """
        value = float(np.clip(value, -1.0, 1.0))
        return center + value * (upper - center) if value >= 0.0 else center + value * (center - lower)

    def _base_dtheta_for_radius(self, radius):
        """根据当前半径计算保持近似恒定空间步长所需的角度增量。

        输入：
            radius: 当前控制点相对中心的半径，单位m。

        输出：
            dtheta_center: 当前半径下的基准角度增量，单位rad。
        """
        radius = max(float(radius), self.actual_start_radius)
        dtheta = self.target_control_spacing / radius
        return float(np.clip(dtheta, self.min_dtheta, self.max_dtheta))

    def _corner_dtheta_scale(self, theta):
        """计算拐角区域的角度步长缩放系数。

        输入：
            theta: 当前控制点极角，单位rad。

        输出：
            scale: dtheta缩放系数。远离拐角约为1，靠近拐角小于1。

        说明：
            方案3的目标是在拐角附近增加控制点密度。这里不改变
            self.base_dtheta、self.min_dtheta、self.max_dtheta 等初始化参数，
            只在运行时根据当前角度对 dtheta_center 做局部缩小。
        """
        corners = getattr(self, "corner_theta", np.array([], dtype=np.float64))
        if len(corners) == 0:
            return 1.0

        nearest = min(self._angle_abs_diff(theta, corner) for corner in corners)
        # 影响宽度约为25度；中心处最多把dtheta压到55%。
        width = np.deg2rad(25.0)
        strength = 0.45
        score = np.exp(-0.5 * (nearest / max(width, 1e-12)) ** 2)
        return float(np.clip(1.0 - strength * score, 0.55, 1.0))

    def _action_intervals(self):
        """获取当前动作映射区间。

        输入：
            无；使用连续非法动作次数动态放大尝试范围。

        输出：
            dtheta_center, dtheta_lower, dtheta_upper, dr_lower, dr_upper: 当前动作映射中心和边界。
        """
        recovery_gain = min(5.0, 1.0 + 0.35 * self.invalid_action_steps)
        base_dtheta_center = self._base_dtheta_for_radius(self.current_radius)
        base_dtheta_lower = max(0.25 * self.base_dtheta,
                                base_dtheta_center - recovery_gain * (base_dtheta_center - self.min_dtheta))
        base_dtheta_upper = base_dtheta_center + recovery_gain * (self.max_dtheta - base_dtheta_center)
        corner_scale = self._corner_dtheta_scale(self.current_theta)
        dtheta_center = float(np.clip(base_dtheta_center * corner_scale,
                                      0.25 * self.base_dtheta, self.max_dtheta))
        dtheta_lower = float(np.clip(base_dtheta_lower * corner_scale,
                                     0.25 * self.base_dtheta, dtheta_center))
        dtheta_upper = float(np.clip(base_dtheta_upper * corner_scale,
                                     dtheta_center, self.max_dtheta))
        dr_lower = self.base_radial_step - recovery_gain * (self.base_radial_step - self.min_radial_step)
        dr_upper = self.base_radial_step + recovery_gain * (self.max_radial_step - self.base_radial_step)
        return dtheta_center, dtheta_lower, dtheta_upper, dr_lower, dr_upper

    def _map_action_to_increments(self, action):
        """将动作映射为极坐标增量。

        输入：
            action: 至少2维的动作数组。

        输出：
            dtheta: 本步角度增量。
            dr_requested: 本步请求半径增量。
            action: 裁剪后的二维动作。
        """
        action = np.asarray(action, dtype=np.float64).ravel()
        if action.size < 2:
            raise ValueError("action至少需要2个元素：[角度增量动作, 半径增量动作]。")
        action = np.clip(action[:2], -1.0, 1.0)
        dtheta_center, dtheta_lower, dtheta_upper, dr_lower, dr_upper = self._action_intervals()
        dtheta = self._signed_interval_action(action[0], dtheta_center, dtheta_lower, dtheta_upper)
        dr_requested = self._signed_interval_action(action[1], self.base_radial_step, dr_lower, dr_upper)
        return dtheta, dr_requested, action

    def _nearest_grid_index(self, point):
        """获取点对应的最近网格索引。

        输入：
            point: 二维点坐标，单位m。

        输出：
            row, col: 最近网格行列索引。
        """
        col = int(np.clip(round(point[0] / self.res), 0, self.wp_grid_points - 1))
        row = int(np.clip(round(point[1] / self.res), 0, self.wp_grid_points - 1))
        return row, col

    def _tool_center_is_feasible(self, point):
        """判断刀具中心点是否可行。

        输入：
            point: 二维点坐标，单位m。

        输出：
            feasible: bool，True表示刀具中心不越界。
        """
        point = np.asarray(point, dtype=np.float64)
        if not self._point_inside_workspace(point):
            return False
        row, col = self._nearest_grid_index(point)
        return bool(self.tool_center_feasible_mask[row, col])

    def _tool_centers_are_feasible(self, points):
        """批量判断刀具中心点是否可行。

        输入：
            points: 形状为(N, 2)的点数组，单位m。

        输出：
            feasible: 形状为(N,)的bool数组。
        """
        points = np.asarray(points, dtype=np.float64)
        if points.size == 0:
            return np.zeros(0, dtype=bool)
        inside = (
            (points[:, 0] >= 0.0) & (points[:, 0] <= self.wp_size) &
            (points[:, 1] >= 0.0) & (points[:, 1] <= self.wp_size)
        )
        feasible = np.zeros(len(points), dtype=bool)
        if not np.any(inside):
            return feasible
        cols = np.clip(np.rint(points[inside, 0] / self.res).astype(np.int64), 0, self.wp_grid_points - 1)
        rows = np.clip(np.rint(points[inside, 1] / self.res).astype(np.int64), 0, self.wp_grid_points - 1)
        feasible[inside] = self.tool_center_feasible_mask[rows, cols]
        return feasible

    def _make_open_uniform_tck(self, control_points):
        """基于控制点构建open-uniform B-spline曲线。

        输入：
            control_points: 形状为(N, 2)的控制点数组。

        输出：
            tck: (knots, [control_x, control_y], degree)格式；控制点不足2个时返回None。
        """
        control_points = np.asarray(control_points, dtype=np.float64)
        n = len(control_points)
        if n < 2:
            return None
        degree = min(3, n - 1)
        internal_count = n - degree - 1
        internal = np.linspace(0.0, 1.0, internal_count + 2)[1:-1] if internal_count > 0 else np.array([])
        knots = np.concatenate((np.zeros(degree + 1), internal, np.ones(degree + 1)))
        return knots, [control_points[:, 0].copy(), control_points[:, 1].copy()], degree

    def _resolve_curve_sample_count(self, tck, sample_count):
        """确定当前曲线采样点数。

        输入：
            tck: B-spline曲线表示。
            sample_count: 外部指定采样点数量；None时启用自适应采样。

        输出：
            count: 实际采样点数量。
        """
        if sample_count is not None:
            return int(sample_count)
        _, controls, _ = tck
        control_points = np.column_stack((controls[0], controls[1]))
        if len(control_points) < 2:
            return self.curve_sample_count
        polygon_length = float(np.sum(np.linalg.norm(np.diff(control_points, axis=0), axis=1)))
        adaptive_count = int(np.ceil(polygon_length / self.curve_sample_spacing)) + 1
        return max(self.curve_sample_count, adaptive_count)

    def _sample_tck(self, tck, sample_count=None):
        """采样B-spline曲线并估计曲率。

        输入：
            tck: _make_open_uniform_tck输出的曲线表示。
            sample_count: 采样点数量；None时使用self.curve_sample_count。

        输出：
            path: 形状为(M, 2)的曲线采样点。
            curvature: 每个采样点曲率，单位1/m。
            s: 累计弧长数组，单位m。
        """
        if tck is None:
            return np.empty((0, 2)), np.array([]), np.array([])
        knots, controls, degree = tck
        u = np.linspace(0.0, 1.0, self._resolve_curve_sample_count(tck, sample_count))
        sx = BSpline(knots, controls[0], degree)
        sy = BSpline(knots, controls[1], degree)
        x, y = sx(u), sy(u)
        path = np.column_stack((x, y))
        d1 = np.column_stack((sx.derivative(1)(u), sy.derivative(1)(u))) if degree >= 1 else np.zeros_like(path)
        d2 = np.column_stack((sx.derivative(2)(u), sy.derivative(2)(u))) if degree >= 2 else np.zeros_like(path)
        cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
        curvature = np.abs(cross) / np.maximum(np.linalg.norm(d1, axis=1) ** 3, 1e-12)
        s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))))
        return path, curvature, s

    def _current_curve(self):
        """由当前全部控制点生成完整曲线。

        输入：
            无；使用self.generated_control_points。

        输出：
            path: 当前完整曲线采样点。
            curvature: 当前完整曲线曲率。
            s: 当前完整曲线累计弧长。
            tck: 当前B-spline曲线表示。
        """
        tck = self._make_open_uniform_tck(self.generated_control_points)
        path, curvature, s = self._sample_tck(tck)
        return path, curvature, s, tck

    def _polyline_geometry(self, path):
        """从离散路径估计弧长、曲率和曲率变化率。

        输入：
            path: 形状为(N, 2)的路径点。

        输出：
            geom: 字典，包含s、curvature、curvature_rate；点数过少时返回None。
        """
        path = np.asarray(path, dtype=np.float64)
        if len(path) < 5:
            return None
        ds = np.linalg.norm(np.diff(path, axis=0), axis=1)
        keep = np.concatenate(([True], ds > 1e-10))
        path = path[keep]
        if len(path) < 5:
            return None
        s_raw = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))))
        if s_raw[-1] <= 1e-12:
            return None

        # 后续要用二阶、三阶几何导数，先按弧长等距重采样，避免参数采样不均匀放大差分噪声。
        s = np.linspace(0.0, s_raw[-1], len(path))
        path = np.column_stack((
            np.interp(s, s_raw, path[:, 0]),
            np.interp(s, s_raw, path[:, 1]),
        ))
        x, y = path[:, 0], path[:, 1]
        xs, ys = np.gradient(x, s, edge_order=2), np.gradient(y, s, edge_order=2)
        norm = np.maximum(np.sqrt(xs ** 2 + ys ** 2), 1e-12)
        xs, ys = xs / norm, ys / norm
        xss, yss = np.gradient(xs, s, edge_order=2), np.gradient(ys, s, edge_order=2)
        xsss, ysss = np.gradient(xss, s, edge_order=2), np.gradient(yss, s, edge_order=2)
        signed_curvature = xs * yss - ys * xss
        return {"s": s, "path": path, "xs": xs, "ys": ys, "xss": xss, "yss": yss,
                "xsss": xsss, "ysss": ysss, "signed_curvature": signed_curvature,
                "curvature": np.abs(signed_curvature),
                "curvature_rate": np.gradient(signed_curvature, s, edge_order=2)}

    def _feedrate_chord_cap(self, kappa, ts=0.002, delta=0.001e-3):
        """计算弦误差约束下的速度上限。

        输入：
            kappa: 曲率数组，单位1/m。
            ts: 插补周期，单位s。
            delta: 允许弦误差，单位m。

        输出：
            cap: 速度上限数组，单位m/s。
        """
        return np.sqrt(np.divide(8.0 * delta, kappa * ts ** 2, out=np.full_like(kappa, np.inf), where=kappa > 1e-12))

    @staticmethod
    def _trapz(y, x):
        """兼容不同NumPy版本的梯形积分。

        输入：
            y: 被积函数采样值。
            x: 自变量采样值。

        输出：
            integral: 梯形积分结果。
        """
        return np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)

    def _critical_curvature(self, ts=0.002, delta=0.001e-3):
        """计算论文式critical curvature阈值。

        输入：
            ts: 插补周期，单位s。
            delta: 允许弦误差，单位m。

        输出：
            kappa_cr: 弦误差、法向加速度、法向jerk三者共同决定的临界曲率。
        """
        k_chord = 8.0 * delta / ((self.v_max * ts) ** 2 + 4.0 * delta ** 2)
        k_acc = self.acc_max / max(self.v_max ** 2, 1e-12)
        k_jerk = np.sqrt(self.jerk_max / max(self.v_max ** 3, 1e-12))
        return float(min(k_chord, k_acc, k_jerk))

    def _local_speed_cap(self, geom, jerk_factor=0.90):
        """计算路径各采样点的局部速度上限。

        输入：
            geom: _polyline_geometry输出的路径几何字典。
            jerk_factor: jerk上限安全系数。

        输出：
            cap: 每个采样点的局部速度上限，单位m/s。
        """
        kappa = geom["curvature"]
        rate = np.abs(geom["curvature_rate"])
        cap_feed = np.full_like(kappa, self.v_max)
        cap_chord = self._feedrate_chord_cap(kappa)
        cap_acc = np.sqrt(np.divide(0.90 * self.acc_max, kappa, out=np.full_like(kappa, np.inf), where=kappa > 1e-12))
        jerk_limit = float(jerk_factor) * self.jerk_max
        cap_normal_jerk = np.cbrt(np.divide(jerk_limit, kappa ** 2, out=np.full_like(kappa, np.inf), where=kappa > 1e-12))
        cap_rate_jerk = np.cbrt(np.divide(jerk_limit, rate, out=np.full_like(kappa, np.inf), where=rate > 1e-12))
        return np.minimum.reduce((cap_feed, cap_chord, cap_acc, cap_normal_jerk, cap_rate_jerk))

    def _axis_constraint_speed_cap(self, geom, jerk_factor=0.92):
        """计算VPOp风格的单轴局部速度上限。

        输入：
            geom: _polyline_geometry输出的路径几何字典。
            jerk_factor: 轴向jerk安全系数。

        输出：
            cap: 同时考虑进给、弦误差、X/Y轴速度、几何加速度和几何jerk的速度上限。

        说明：
            Beudaert等人的VPOp方法核心是直接约束每个驱动轴：
            q_dot = q_s * s_dot
            q_ddot = q_ss * s_dot^2 + q_s * s_ddot
            q_dddot = q_sss * s_dot^3 + 3*q_ss*s_dot*s_ddot + q_s*s_dddot

            这里先把与s_ddot、s_dddot无关的单轴几何项写进局部速度上限，
            后续再通过_axis_dynamics和迭代修正检查完整轴向加速度/jerk。
        """
        base_cap = self._local_speed_cap(geom, jerk_factor=jerk_factor)
        xs, ys = np.abs(geom["xs"]), np.abs(geom["ys"])
        xss, yss = np.abs(geom["xss"]), np.abs(geom["yss"])
        xsss, ysss = np.abs(geom["xsss"]), np.abs(geom["ysss"])

        cap_vx = np.divide(self.v_max, xs, out=np.full_like(base_cap, np.inf), where=xs > 1e-12)
        cap_vy = np.divide(self.v_max, ys, out=np.full_like(base_cap, np.inf), where=ys > 1e-12)

        # 当切向加速度为0时，轴向几何加速度项为 q_ss * v^2。
        cap_ax_geom = np.sqrt(np.divide(0.90 * self.acc_max, xss, out=np.full_like(base_cap, np.inf), where=xss > 1e-12))
        cap_ay_geom = np.sqrt(np.divide(0.90 * self.acc_max, yss, out=np.full_like(base_cap, np.inf), where=yss > 1e-12))

        # 当切向加速度和切向jerk为0时，轴向几何jerk项为 q_sss * v^3。
        axis_jerk = float(jerk_factor) * self.jerk_max
        cap_jx_geom = np.cbrt(np.divide(axis_jerk, xsss, out=np.full_like(base_cap, np.inf), where=xsss > 1e-12))
        cap_jy_geom = np.cbrt(np.divide(axis_jerk, ysss, out=np.full_like(base_cap, np.inf), where=ysss > 1e-12))

        return np.minimum.reduce((
            base_cap, cap_vx, cap_vy, cap_ax_geom, cap_ay_geom, cap_jx_geom, cap_jy_geom
        ))

    def _initial_planning_points(self, geom, cap):
        """根据曲率峰值和速度上限低谷生成速度规划分段点。

        输入：
            geom: _polyline_geometry输出的路径几何字典。
            cap: 局部速度上限数组。

        输出：
            points: 排序后的分段点索引，包含起点和终点。
        """
        n = len(geom["s"])
        if n <= 2:
            return np.arange(n, dtype=int)
        distance = max(8, n // 180)
        kappa_cr = self._critical_curvature()
        paper_points, _ = find_peaks(geom["curvature"], height=kappa_cr, distance=distance)
        cap_valleys, _ = find_peaks(-cap, distance=distance)
        points = np.unique(np.concatenate(([0], paper_points, cap_valleys, [n - 1]))).astype(int)
        return points

    def _sine_transition(self, vs, ve, ts=0.002):
        """生成两个速度节点之间的正弦型S加减速速度曲线。

        输入：
            vs: 起始速度，单位m/s。
            ve: 结束速度，单位m/s。
            ts: 插补周期，单位s。

        输出：
            t: 时间采样。
            v: 速度采样。
        """
        dv = abs(float(ve) - float(vs))
        if dv < 1e-12:
            return np.array([0.0]), np.array([float(vs)])
        n_acc = np.pi * dv / max(2.0 * self.acc_max * ts, 1e-12)
        n_jerk = np.pi * np.sqrt(dv / max(2.0 * self.jerk_max, 1e-12)) / ts
        n = max(1, int(np.ceil(max(n_acc, n_jerk))))
        t = np.linspace(0.0, n * ts, n + 1)
        q = t / max(n * ts, 1e-12)
        shape = 0.5 * (np.sin(np.pi * (q - 0.5)) + 1.0)
        v = float(vs) + np.sign(float(ve) - float(vs)) * dv * shape
        v[-1] = float(ve)
        return t, v

    def _transition_distance(self, vs, ve):
        """计算正弦型S加减速从vs过渡到ve需要的弧长。

        输入：
            vs: 起始速度。
            ve: 结束速度。

        输出：
            distance: 过渡距离，单位m。
        """
        t, v = self._sine_transition(vs, ve)
        return float(self._trapz(v, t))

    def _reachable_speed(self, v0, length, upper):
        """给定段长，计算从v0出发可加速达到的最高节点速度。

        输入：
            v0: 起点速度。
            length: 段长。
            upper: 目标节点速度上限。

        输出：
            speed: 可达速度。
        """
        if upper <= v0 or self._transition_distance(v0, upper) <= length:
            return float(upper)
        low, high = float(v0), float(upper)
        for _ in range(45):
            mid = 0.5 * (low + high)
            if self._transition_distance(v0, mid) <= length:
                low = mid
            else:
                high = mid
        return float(low)

    def _controllable_speed(self, v1, length, upper):
        """给定段长，计算能减速到v1的最高前一节点速度。

        输入：
            v1: 下一节点速度。
            length: 段长。
            upper: 当前节点速度上限。

        输出：
            speed: 可控速度。
        """
        if upper <= v1 or self._transition_distance(upper, v1) <= length:
            return float(upper)
        low, high = float(v1), float(upper)
        for _ in range(45):
            mid = 0.5 * (low + high)
            if self._transition_distance(mid, v1) <= length:
                low = mid
            else:
                high = mid
        return float(low)

    def _scan_node_speeds(self, geom, points, cap, start_speed, end_speed):
        """对分段点速度做前向可达和后向可控扫描。

        输入：
            geom: 路径几何字典。
            points: 分段点索引。
            cap: 局部速度上限。
            start_speed: 起点速度。
            end_speed: 终点速度。

        输出：
            node_s: 分段点弧长。
            node_v: 扫描后的分段点速度。
        """
        s_node = geom["s"][points]
        node_cap = np.array([np.min(cap[max(0, p - 4):min(len(cap), p + 5)]) for p in points], dtype=np.float64)
        node_cap[0] = min(node_cap[0], start_speed)
        node_cap[-1] = min(node_cap[-1], end_speed)
        node_v = node_cap.copy()
        node_v[0] = node_cap[0]
        for i in range(len(node_v) - 1):
            length = s_node[i + 1] - s_node[i]
            node_v[i + 1] = min(node_v[i + 1], self._reachable_speed(node_v[i], length, node_cap[i + 1]))
        node_v[-1] = min(node_v[-1], end_speed)
        for i in range(len(node_v) - 2, -1, -1):
            length = s_node[i + 1] - s_node[i]
            node_v[i] = min(node_v[i], self._controllable_speed(node_v[i + 1], length, node_v[i]))
        return s_node, node_v

    def _build_sine_block(self, s0, s1, vs, ve, vf_limit):
        """在两个分段点之间构建正弦S型速度块并返回该段耗时。

        输入：
            s0, s1: 段起止弧长。
            vs, ve: 段起止速度。
            vf_limit: 段内允许峰值速度。

        输出：
            block_time: 该段估计加工时间，单位s。
        """
        length = float(s1 - s0)
        if length <= 1e-12:
            return 0.0
        low = max(float(vs), float(ve))
        high = max(low, float(vf_limit))
        if self._transition_distance(vs, high) + self._transition_distance(high, ve) > length:
            for _ in range(45):
                mid = 0.5 * (low + high)
                if self._transition_distance(vs, mid) + self._transition_distance(mid, ve) <= length:
                    low = mid
                else:
                    high = mid
            vf = low
        else:
            vf = high
        t_acc, v_acc = self._sine_transition(vs, vf)
        t_dec, v_dec = self._sine_transition(vf, ve)
        d_acc = self._trapz(v_acc, t_acc)
        d_dec = self._trapz(v_dec, t_dec)
        cruise = max(0.0, length - d_acc - d_dec)
        cruise_time = cruise / max(vf, 1e-12)
        return float(t_acc[-1] + cruise_time + t_dec[-1])

    def _build_sine_block_profile(self, s0, s1, vs, ve, vf_limit):
        """构建单个正弦S型速度块的弧长-速度采样。

        输入：
            s0, s1: 段起止弧长，单位m。
            vs, ve: 段起止速度，单位m/s。
            vf_limit: 段内峰值速度上限，单位m/s。

        输出：
            block_time: 该段耗时，单位s。
            s_profile: 段内弧长采样，单位m。
            v_profile: 段内速度采样，单位m/s。
        """
        length = float(s1 - s0)
        if length <= 1e-12:
            return 0.0, np.array([s0]), np.array([vs])
        low = max(float(vs), float(ve))
        high = max(low, float(vf_limit))
        if self._transition_distance(vs, high) + self._transition_distance(high, ve) > length:
            for _ in range(45):
                mid = 0.5 * (low + high)
                if self._transition_distance(vs, mid) + self._transition_distance(mid, ve) <= length:
                    low = mid
                else:
                    high = mid
            vf = low
        else:
            vf = high

        t_acc, v_acc = self._sine_transition(vs, vf)
        t_dec, v_dec = self._sine_transition(vf, ve)
        s_acc = np.concatenate(([0.0], np.cumsum(0.5 * (v_acc[:-1] + v_acc[1:]) * np.diff(t_acc))))
        s_dec_local = np.concatenate(([0.0], np.cumsum(0.5 * (v_dec[:-1] + v_dec[1:]) * np.diff(t_dec))))
        d_acc = float(s_acc[-1])
        d_dec = float(s_dec_local[-1])
        cruise = max(0.0, length - d_acc - d_dec)
        cruise_time = cruise / max(vf, 1e-12)

        s_parts = [s0 + s_acc]
        v_parts = [v_acc]
        if cruise > 1e-10 and vf > 1e-12:
            cruise_steps = max(2, int(np.ceil(cruise_time / 0.002)))
            s_cruise = np.linspace(s0 + d_acc, s1 - d_dec, cruise_steps)
            v_cruise = np.full_like(s_cruise, vf)
            s_parts.append(s_cruise[1:])
            v_parts.append(v_cruise[1:])
        s_parts.append(s1 - d_dec + s_dec_local[1:])
        v_parts.append(v_dec[1:])

        block_time = float(t_acc[-1] + cruise_time + t_dec[-1])
        return block_time, np.concatenate(s_parts), np.concatenate(v_parts)

    def _build_velocity_profile_on_geom(self, geom, cap, points, node_v):
        """由分段点速度生成整条路径采样点上的速度。

        输入：
            geom: 路径几何字典。
            cap: 当前局部速度上限。
            points: 分段点索引。
            node_v: 分段点速度。

        输出：
            v_geom: 插值到geom["s"]上的速度。
            s_profile: 分段S曲线弧长采样。
            v_profile: 分段S曲线速度采样。
        """
        s_profile, v_profile = [], []
        for i in range(len(points) - 1):
            i0, i1 = int(points[i]), int(points[i + 1])
            segment_cap = cap[i0:i1 + 1]
            vf_limit = min(self.v_max, float(np.min(segment_cap)))
            _, s_seg, v_seg = self._build_sine_block_profile(
                geom["s"][i0], geom["s"][i1], node_v[i], node_v[i + 1], vf_limit
            )
            if i > 0:
                s_seg, v_seg = s_seg[1:], v_seg[1:]
            s_profile.extend(s_seg.tolist())
            v_profile.extend(v_seg.tolist())

        if len(s_profile) < 2:
            v_geom = np.full_like(geom["s"], node_v[0] if len(node_v) else 0.0)
            return v_geom, np.asarray(s_profile), np.asarray(v_profile)

        s_profile = np.asarray(s_profile, dtype=np.float64)
        v_profile = np.asarray(v_profile, dtype=np.float64)
        order = np.argsort(s_profile)
        s_profile, v_profile = s_profile[order], v_profile[order]
        s_unique, keep = np.unique(s_profile, return_index=True)
        v_unique = v_profile[keep]
        v_geom = np.interp(geom["s"], s_unique, v_unique)
        return v_geom, s_unique, v_unique

    def _apply_global_axis_safety_scale(self, geom, velocity, s_profile, v_profile, node_v, cap=None):
        """对未收敛的速度曲线做全局安全缩放。

        输入：
            geom: 路径几何字典。
            velocity: geom采样点速度。
            s_profile/v_profile: render使用的弧长-速度曲线。
            node_v: 分段节点速度。

        输出：
            cap: 可选局部速度上限数组。

        输出：
            velocity, s_profile, v_profile, node_v, dyn, converged_by_scale:
            缩放后的速度相关数组、动力学结果和是否满足约束。

        说明：
            局部axis-check是启发式修正，遇到尖锐几何三阶导数时可能在限定迭代内不收敛。
            全局缩放利用近似比例关系：加速度随速度缩放因子平方变化，jerk随三次方变化。
            它不是时间最优，但作为最终兜底可以优先保证机床动态约束。
        """
        v = np.asarray(velocity, dtype=np.float64).copy()
        vp = None if v_profile is None else np.asarray(v_profile, dtype=np.float64).copy()
        nv = None if node_v is None else np.asarray(node_v, dtype=np.float64).copy()
        dyn = self._axis_dynamics(geom, v)
        converged_by_scale = False

        for _ in range(8):
            max_acc = max(float(np.max(np.abs(dyn["ax"]))), float(np.max(np.abs(dyn["ay"]))), 1e-12)
            max_jerk = max(float(np.max(np.abs(dyn["jx"]))), float(np.max(np.abs(dyn["jy"]))), 1e-12)
            acc_scale = np.sqrt(0.98 * self.acc_max / max_acc) if max_acc > self.acc_max else 1.0
            jerk_scale = np.cbrt(0.98 * self.jerk_max / max_jerk) if max_jerk > self.jerk_max else 1.0
            cap_scale = 1.0
            if cap is not None:
                cap_arr = np.asarray(cap, dtype=np.float64)
                over = v > cap_arr
                if np.any(over):
                    cap_scale = float(0.98 * np.min(cap_arr[over] / np.maximum(v[over], 1e-12)))
            scale = float(min(1.0, acc_scale, jerk_scale, cap_scale))
            if scale >= 0.999:
                converged_by_scale = True
                break
            v *= scale
            if vp is not None:
                vp *= scale
            if nv is not None:
                nv *= scale
            dyn = self._axis_dynamics(geom, v)

        return v, s_profile, vp, nv, dyn, converged_by_scale

    def _axis_dynamics(self, geom, velocity):
        """计算路径速度对应的切向、法向和X/Y轴动力学。

        输入：
            geom: 包含xs、ys、xss、yss、xsss、ysss的路径几何字典。
            velocity: 与geom["s"]同长度的路径速度，单位m/s。

        输出：
            dyn: 字典，包含时间、切向加速度、切向jerk、法向jerk、X/Y加速度和X/Y jerk。
        """
        s = geom["s"]
        v = np.asarray(velocity, dtype=np.float64)
        ds = np.diff(s)
        dt = 2.0 * ds / np.maximum(v[:-1] + v[1:], 1e-12)
        time_axis = np.concatenate(([0.0], np.cumsum(dt)))

        a_seg = (v[1:] ** 2 - v[:-1] ** 2) / np.maximum(2.0 * ds, 1e-12)
        at = np.zeros_like(v)
        if len(a_seg):
            at[0], at[-1] = a_seg[0], a_seg[-1]
            if len(v) > 2:
                at[1:-1] = 0.5 * (a_seg[:-1] + a_seg[1:])

        if len(time_axis) >= 3 and time_axis[-1] > 1e-12:
            jt = np.gradient(at, time_axis, edge_order=2)
        else:
            jt = np.zeros_like(v)

        k = geom["signed_curvature"]
        kp = geom["curvature_rate"]
        jt_frenet = jt - k ** 2 * v ** 3
        jn_frenet = 3.0 * k * v * at + kp * v ** 3

        xs, ys = geom["xs"], geom["ys"]
        xss, yss = geom["xss"], geom["yss"]
        xsss, ysss = geom["xsss"], geom["ysss"]
        ax = xss * v ** 2 + xs * at
        ay = yss * v ** 2 + ys * at
        jx = xsss * v ** 3 + 3.0 * xss * v * at + xs * jt
        jy = ysss * v ** 3 + 3.0 * yss * v * at + ys * jt
        return {
            "time": time_axis,
            "at": at,
            "jt_feed": jt,
            "jt_frenet": jt_frenet,
            "jn_frenet": jn_frenet,
            "ax": ax,
            "ay": ay,
            "jx": jx,
            "jy": jy,
            "total_time": float(time_axis[-1]) if len(time_axis) else 0.0,
        }

    @staticmethod
    def _integrate_speed_profile_time(s_profile, v_profile, speed_floor=1e-9):
        """从连续弧长-速度曲线积分加工时间，避免几何采样点末端零速导致时间爆炸。

        输入：
            s_profile: 速度曲线弧长采样，单位m。
            v_profile: 速度曲线速度采样，单位m/s。
            speed_floor: 时间积分速度下限，只用于数值保护。

        输出：
            total_time: 积分得到的加工时间，单位s。
        """
        s_profile = np.asarray(s_profile, dtype=np.float64)
        v_profile = np.asarray(v_profile, dtype=np.float64)
        if len(s_profile) < 2 or len(v_profile) < 2:
            return 0.0
        order = np.argsort(s_profile)
        s_profile = s_profile[order]
        v_profile = v_profile[order]
        s_unique, keep = np.unique(s_profile, return_index=True)
        v_unique = v_profile[keep]
        if len(s_unique) < 2:
            return 0.0
        ds = np.diff(s_unique)
        v_mid = 0.5 * (v_unique[:-1] + v_unique[1:])
        valid = ds > 1e-12
        if not np.any(valid):
            return 0.0
        return float(np.sum(ds[valid] / np.maximum(v_mid[valid], float(speed_floor))))

    def _axis_checked_feedrate_profile(self, geom, mode, return_profile=False, max_iter=12):
        """执行VPOp风格的X/Y轴约束速度规划。

        输入：
            geom: 路径几何字典。
            mode: "local"或"full"，决定起止速度。
            return_profile: True时返回速度曲线和动力学诊断。
            max_iter: 局部降速修正最大迭代次数。

        输出：
            total_time: 加工时间估计。
            profile: return_profile=True时返回完整诊断字典。

        说明：
            该函数复现论文“约束交集”的二维工程版本：
            1. 直接把X/Y轴速度、轴向几何加速度、轴向几何jerk写入局部速度上限；
            2. 以曲率峰值和速度上限低谷作为约束敏感点；
            3. 对分段点速度做前向可达/后向可控扫描；
            4. 生成速度曲线后显式计算X/Y轴加速度和jerk；
            5. 对超限区域追加分段点并压低局部上限，最后用全局缩放兜底。

            注意：
            论文原VPOp是在固定时间步内对s_{j+1}求解轴约束区间并带回退二分。
            当前环境为了训练效率采用“轴约束速度上限 + 迭代轴检查”的等价近似，
            但约束对象已经从切向/法向近似切换为X/Y单轴约束。
        """
        base_cap = self._axis_constraint_speed_cap(geom)
        cap = base_cap.copy()
        paper_points = self._initial_planning_points(geom, cap)
        extra_points = np.array([], dtype=int)
        points = paper_points.copy()

        if mode == "full":
            start_speed = min(self._seed_circle_speed_cap(), cap[0])
            end_speed = 0.0
        else:
            start_speed = cap[0]
            end_speed = cap[-1]

        converged = False
        s_profile = v_profile = node_s = node_v = None
        dyn = None
        v_geom = np.minimum(cap.copy(), self.v_max)

        for it in range(max(1, int(max_iter))):
            points = np.unique(points).astype(int)
            node_s, node_v = self._scan_node_speeds(geom, points, cap, start_speed, end_speed)
            v_geom, s_profile, v_profile = self._build_velocity_profile_on_geom(geom, cap, points, node_v)
            dyn = self._axis_dynamics(geom, v_geom)

            acc_ratio = np.maximum(np.abs(dyn["ax"]), np.abs(dyn["ay"])) / max(self.acc_max, 1e-12)
            jerk_ratio = np.maximum(np.abs(dyn["jx"]), np.abs(dyn["jy"])) / max(self.jerk_max, 1e-12)
            cap_violation = np.maximum(v_geom - cap, 0.0)
            bad_metric = np.maximum.reduce((
                acc_ratio,
                jerk_ratio,
                1.0 + cap_violation / max(self.v_max, 1e-12),
            ))

            if (np.max(acc_ratio) <= 1.02 and np.max(jerk_ratio) <= 1.02
                    and np.max(cap_violation) <= 1e-5):
                converged = True
                break

            bad_peaks, _ = find_peaks(bad_metric, height=1.0, distance=max(4, len(v_geom) // 300))
            if len(bad_peaks) == 0:
                bad_peaks = np.array([int(np.argmax(bad_metric))])
            worst = bad_peaks[np.argsort(bad_metric[bad_peaks])[-min(32, len(bad_peaks)):]]
            extra_points = np.unique(np.concatenate((extra_points, worst))).astype(int)
            points = np.unique(np.concatenate(([0], paper_points, extra_points, [len(v_geom) - 1]))).astype(int)

            radius = max(10, len(v_geom) // 220)
            for idx in worst:
                factor_acc = 0.90 / max(acc_ratio[idx], 1.0) ** 0.5
                factor_jerk = 0.90 / max(jerk_ratio[idx], 1.0) ** (1.0 / 3.0)
                factor = np.clip(min(0.96, factor_acc, factor_jerk), 0.18, 0.96)
                lo, hi = max(0, idx - radius), min(len(v_geom), idx + radius + 1)
                cap[lo:hi] = np.minimum(cap[lo:hi], v_geom[lo:hi] * factor)

        if dyn is None:
            dyn = self._axis_dynamics(geom, v_geom)
        max_acc = max(float(np.max(np.abs(dyn["ax"]))), float(np.max(np.abs(dyn["ay"]))), 0.0)
        max_jerk = max(float(np.max(np.abs(dyn["jx"]))), float(np.max(np.abs(dyn["jy"]))), 0.0)
        if max_acc > 1.02 * self.acc_max or max_jerk > 1.02 * self.jerk_max:
            v_geom, s_profile, v_profile, node_v, dyn, scaled_ok = self._apply_global_axis_safety_scale(
                geom, v_geom, s_profile, v_profile, node_v, cap=cap
            )
            converged = converged or scaled_ok
        # 加工总时间应由连续弧长-速度曲线积分得到。
        # 不再使用dyn["total_time"]作为full_time来源，因为v_geom在终点附近可能被插值成多个0速度点，
        # 导致非零长度段除以接近0的速度，产生1e9量级的虚假加工时间。
        if s_profile is not None and v_profile is not None and len(s_profile) >= 2 and len(v_profile) >= 2:
            total_time = self._integrate_speed_profile_time(s_profile, v_profile, speed_floor=1e-9)
            dyn["total_time"] = float(total_time)
        else:
            total_time = float(dyn["total_time"])
        if not return_profile:
            return total_time

        return total_time, {
            "s": np.asarray(s_profile if s_profile is not None else geom["s"], dtype=np.float64),
            "speed": np.asarray(v_profile if v_profile is not None else v_geom, dtype=np.float64),
            "cap_s": geom["s"],
            "cap": cap,
            "base_cap": base_cap,
            "node_s": np.asarray(node_s if node_s is not None else [], dtype=np.float64),
            "node_v": np.asarray(node_v if node_v is not None else [], dtype=np.float64),
            "velocity_geom": v_geom,
            "dynamics": dyn,
            "paper_points": paper_points,
            "extra_points": extra_points,
            "converged": converged,
            "iterations": it + 1,
            "mode": mode,
            "planner": "vpop_axis_constraints",
        }

    def _seed_circle_speed_cap(self):
        """计算中心固定圆允许的出口速度上限。

        输入：
            无；使用actual_start_radius和机床约束。

        输出：
            speed_cap: 中心圆速度上限，单位m/s。
        """
        radius = max(self.actual_start_radius, 1e-12)
        kappa = 1.0 / radius
        cap_feed = self.v_max
        cap_chord = float(self._feedrate_chord_cap(np.array([kappa]))[0])
        cap_acc = np.sqrt(0.90 * self.acc_max * radius)
        cap_jerk = np.cbrt(0.90 * self.jerk_max * radius ** 2)
        return float(min(cap_feed, cap_chord, cap_acc, cap_jerk))

    def _full_time_reward(self, full_time):
        """把完整路径加工时间转换为有界终局奖励。

        输入：
            full_time: strict完整路径速度规划得到的加工时间，单位s。

        输出：
            clipped_reward: 实际加入环境reward的有界惩罚。
            raw_reward: 未限幅的原始惩罚，用于诊断。

        说明：
            极端坏路径可能让strict速度规划在局部速度接近0时得到巨大加工时间。
            这种信息说明路径很差，但不应该以1e8量级直接进入SAC的Q学习目标。
        """
        raw_reward = -self.full_time_reward_weight * float(full_time) / max(self.full_time_reference, 1e-12)
        clipped_reward = float(np.clip(raw_reward, self.full_time_reward_floor, 0.0))
        return clipped_reward, float(raw_reward)

    def _estimate_machining_time_fast(self, path, mode="local", return_profile=False, local_window_length=None):
        """快速估计局部加工时间。

        输入：
            path: 路径点数组，单位m。
            mode: "local"或"full"；训练step通常使用"local"。
            return_profile: True时返回简化速度曲线诊断。
            local_window_length: mode="local"时的尾部窗口弧长，单位m。

        输出：
            machining_time: 加工时间估计，单位s。
            profile: return_profile=True时返回简化诊断。

        说明：
            该函数用于训练step的稠密reward。它保留弦误差、法向加速度、近似法向jerk、
            曲率变化率jerk和前后向加速度扫描，但不做X/Y轴axis-check迭代。
        """
        path = np.asarray(path, dtype=np.float64)
        if len(path) < 5:
            empty = {"s": np.array([]), "speed": np.array([]), "cap_s": np.array([]), "cap": np.array([]),
                     "node_s": np.array([]), "node_v": np.array([]), "mode": str(mode).lower(),
                     "planner": "fast"}
            return (0.0, empty) if return_profile else 0.0

        mode = str(mode).lower()
        eval_path = self._time_eval_path(path, mode, local_window_length=local_window_length)
        geom = self._polyline_geometry(eval_path)
        if geom is None or geom["s"][-1] <= 1e-12:
            empty = {"s": np.array([]), "speed": np.array([]), "cap_s": np.array([]), "cap": np.array([]),
                     "node_s": np.array([]), "node_v": np.array([]), "mode": mode, "planner": "fast"}
            return (0.0, empty) if return_profile else 0.0

        cap = self._axis_constraint_speed_cap(geom, jerk_factor=self.local_fast_jerk_factor)
        ds = np.diff(geom["s"])
        v = cap.copy()
        if mode == "full":
            v[0] = min(self._seed_circle_speed_cap(), v[0])
            v[-1] = 0.0
        for i in range(len(ds)):
            v[i + 1] = min(v[i + 1], np.sqrt(max(v[i] ** 2 + 2.0 * self.acc_max * ds[i], 0.0)))
        if mode == "full":
            v[-1] = 0.0
        for i in range(len(ds) - 1, -1, -1):
            v[i] = min(v[i], np.sqrt(max(v[i + 1] ** 2 + 2.0 * self.acc_max * ds[i], 0.0)))

        total_time = float(np.sum(2.0 * ds / np.maximum(v[:-1] + v[1:], 1e-9)))
        if not return_profile:
            return total_time
        return total_time, {
            "s": geom["s"],
            "speed": v,
            "cap_s": geom["s"],
            "cap": cap,
            "node_s": np.array([]),
            "node_v": np.array([]),
            "velocity_geom": v,
            "mode": mode,
            "planner": "fast",
        }

    def estimate_machining_time(self, path, mode="local", return_profile=False, local_window_length=None,
                                planner=None):
        """用分段正弦S型速度规划估计给定路径的加工时间。

        输入：
            path: 形状为(N, 2)的路径点数组，单位m。
            mode: "local"或"full"。
                "local": 只取路径末尾generation_time_window_points个采样点，适合step中的稠密奖励。
                "full": 使用完整路径，适合episode结束、评估或需要全局加工时间时。
            return_profile: True时额外返回速度规划诊断曲线。
            local_window_length: mode="local"时按尾部弧长截取局部路径；None时使用固定点数回退。
            planner: "fast"、"strict"或"vpop"；None时local使用self.local_time_planner，full使用vpop。

        输出：
            machining_time: 估计加工时间，单位s。
            profile: return_profile=True时返回字典，包含弧长、速度、局部速度上限和节点速度。

        说明：
            该实现采用VPOp论文的核心约束思想：
            1. 速度规划约束对象是X/Y单轴速度、单轴加速度和单轴jerk；
            2. 局部速度上限由轴向约束、弦误差和曲率相关约束共同决定；
            3. 规划后显式检查X/Y轴动力学，超限则追加约束点并重规划；
            4. 训练step可使用fast近似，episode结束和render默认使用vpop/strict轴约束版本。
        """
        path = np.asarray(path, dtype=np.float64)
        if len(path) < 5:
            empty = {"s": np.array([]), "speed": np.array([]), "cap_s": np.array([]), "cap": np.array([]),
                     "node_s": np.array([]), "node_v": np.array([]), "mode": str(mode).lower()}
            return (0.0, empty) if return_profile else 0.0

        mode = str(mode).lower()
        planner = (self.local_time_planner if planner is None and mode == "local" else "vpop") if planner is None else str(planner).lower()
        if planner == "fast":
            return self._estimate_machining_time_fast(
                path, mode=mode, return_profile=return_profile, local_window_length=local_window_length
            )
        if planner not in ("strict", "vpop"):
            raise ValueError("planner必须为'fast'、'strict'或'vpop'。")
        eval_path = self._time_eval_path(path, mode, local_window_length=local_window_length)

        geom = self._polyline_geometry(eval_path)
        if geom is None or geom["s"][-1] <= 1e-12:
            empty = {"s": np.array([]), "speed": np.array([]), "cap_s": np.array([]), "cap": np.array([]),
                     "node_s": np.array([]), "node_v": np.array([]), "mode": mode}
            return (0.0, empty) if return_profile else 0.0
        return self._axis_checked_feedrate_profile(geom, mode, return_profile=return_profile)

    def _full_path_speed_diagnostics(self):
        """计算完整路径的速度、XY加速度、XY jerk、累计时间和曲率诊断。

        输入：
            无；使用self.current_path。

        输出：
            diag: 字典，包含完整路径的弧长、时间、速度、局部速度上限、节点速度、
                  X/Y加速度、X/Y jerk、曲率和总耗时。

        说明：
            render第二个图使用该函数。它不是step中的局部奖励时间，而是对当前完整路径
            调用mode='full'速度规划后得到的完整路径时间诊断。
        """
        path = np.asarray(getattr(self, "current_path", np.empty((0, 2))), dtype=np.float64)
        empty = {
            "s": np.array([]), "time": np.array([]), "speed": np.array([]), "cap_s": np.array([]),
            "cap": np.array([]), "node_s": np.array([]), "node_v": np.array([]), "ax": np.array([]),
            "ay": np.array([]), "jx": np.array([]), "jy": np.array([]), "curvature": np.array([]),
            "total_time": 0.0,
        }
        if len(path) < 5:
            return empty

        total_time, profile = self.estimate_machining_time(path, mode="full", return_profile=True)
        s_profile = np.asarray(profile.get("s", []), dtype=np.float64)
        v_profile = np.asarray(profile.get("speed", []), dtype=np.float64)
        if len(s_profile) < 2 or len(v_profile) < 2:
            empty["total_time"] = float(total_time)
            return empty

        geom = self._polyline_geometry(path)
        if geom is None:
            empty["total_time"] = float(total_time)
            return empty

        dyn = profile.get("dynamics", {})
        velocity_geom = np.asarray(profile.get("velocity_geom", []), dtype=np.float64)
        if len(velocity_geom) == len(geom["s"]):
            display_speed = np.interp(s_profile, geom["s"], velocity_geom)
            display_time = np.interp(s_profile, geom["s"], np.asarray(dyn.get("time", np.zeros_like(geom["s"]))))
            display_ax = np.interp(s_profile, geom["s"], np.asarray(dyn.get("ax", np.zeros_like(geom["s"]))))
            display_ay = np.interp(s_profile, geom["s"], np.asarray(dyn.get("ay", np.zeros_like(geom["s"]))))
            display_jx = np.interp(s_profile, geom["s"], np.asarray(dyn.get("jx", np.zeros_like(geom["s"]))))
            display_jy = np.interp(s_profile, geom["s"], np.asarray(dyn.get("jy", np.zeros_like(geom["s"]))))
            kappa = np.interp(s_profile, geom["s"], geom["curvature"])
        else:
            display_speed = v_profile
            display_time = np.zeros_like(s_profile)
            display_ax = display_ay = display_jx = display_jy = np.zeros_like(s_profile)
            kappa = np.interp(s_profile, geom["s"], geom["curvature"])

        return {
            "s": s_profile,
            "time": display_time,
            "speed": display_speed,
            "cap_s": np.asarray(profile.get("cap_s", []), dtype=np.float64),
            "cap": np.asarray(profile.get("cap", []), dtype=np.float64),
            "node_s": np.asarray(profile.get("node_s", []), dtype=np.float64),
            "node_v": np.asarray(profile.get("node_v", []), dtype=np.float64),
            "ax": display_ax,
            "ay": display_ay,
            "jx": display_jx,
            "jy": display_jy,
            "curvature": kappa,
            "total_time": float(dyn.get("total_time", total_time)),
            "converged": bool(profile.get("converged", False)),
            "iterations": int(profile.get("iterations", 0)),
        }

    def _estimate_local_window_time(self, current_path):
        """估计当前完整曲线尾部窗口的局部加工时间。

        输入：
            current_path: 当前完整B-spline曲线采样点。

        输出：
            local_time: 尾部窗口的估计加工时间，单位s。
        """
        return self.estimate_machining_time(current_path, mode="local")

    def _coverage_matrix(self):
        """生成render使用的覆盖状态矩阵。

        输入：
            无；使用当前visited和固定mask。

        输出：
            matrix: uint8矩阵，编码未加工、固定加工和新增加工区域。
        """
        matrix = np.zeros_like(self.cavity_mask, dtype=np.uint8)
        matrix[self.inner_core_mask] = 1
        matrix[self.contour_pass_mask] = 2
        matrix[self.seed_pass_mask] = 4
        if hasattr(self, "visited"):
            matrix[self.inner_core_mask & (self.visited == 2)] = 3
        return matrix

    def _downsample_category_map(self, category_map):
        """将高分辨率类别图下采样为观测用低分辨率类别图。

        输入：
            category_map: 形状为(H, W)的uint8矩阵。

        输出：
            small: 形状为(observation_grid_size, observation_grid_size)的uint8矩阵。

        说明：
            使用每个低分辨率像素对应区域的中心点采样，避免step中进行大量Python循环。
        """
        category_map = np.asarray(category_map, dtype=np.uint8)
        size = self.observation_grid_size
        y_idx = np.clip(((np.arange(size) + 0.5) * category_map.shape[0] / size).astype(np.int64),
                        0, category_map.shape[0] - 1)
        x_idx = np.clip(((np.arange(size) + 0.5) * category_map.shape[1] / size).astype(np.int64),
                        0, category_map.shape[1] - 1)
        return category_map[np.ix_(y_idx, x_idx)].astype(np.uint8)

    def _coverage_observation(self):
        """构造覆盖状态的单通道ROI类别图观测。

        输入：
            无；使用visited、inner_core_mask、fixed_machined_mask和cavity_mask。

        输出：
            coverage: 形状为(1, G, G)的uint8数组。
                0: 忽略区域，包括型腔外区域。
                1: 目标区域中仍未覆盖的部分。
                2: RL路径已经覆盖的目标区域。
                3: 固定已加工区域，包括外圈精加工和中心圆。
        """
        category = np.zeros_like(self.cavity_mask, dtype=np.uint8)
        category[self.inner_core_mask & (self.visited == 0)] = 1
        category[self.inner_core_mask & (self.visited == 2)] = 2
        category[self.fixed_machined_mask] = 3
        row0, row1, col0, col1 = self.obs_roi
        roi_category = category[row0:row1, col0:col1]
        return self._downsample_category_map(roi_category)[None, :, :]

    def _get_obs(self):
        """构造当前观测向量。

        输入：
            无；使用当前控制点状态、覆盖状态和上一step诊断量。

        输出：
            obs: Dict观测，包含标量state和覆盖栅格coverage。
        """
        covered = np.count_nonzero(self.inner_core_mask & (self.visited == 2))
        remaining = np.count_nonzero(self.inner_core_mask & (self.visited == 0))
        row, col = self._nearest_grid_index(self.current_point)
        wall_margin = self.distance_to_cavity_wall[row, col] - self.tool_r
        last = self.last_info
        state = np.array([
            (self.current_point[0] - self.center_wp) / self.wp_size,
            (self.current_point[1] - self.center_wp) / self.wp_size,
            np.clip(getattr(self, "current_rho", 0.0), 0.0, 1.0),
            np.sin(self.current_theta),
            np.cos(self.current_theta),
            covered / max(self.total_planning_cells, 1),
            np.clip(wall_margin / max(self.tool_r, 1e-12), -1.0, 1.0),
            np.clip(last.get("local_time", 0.0), 0.0, 2.0) / 2.0,
            np.clip(last.get("new_area_mm2", 0.0) / 200.0, 0.0, 1.0),
            np.clip(self.current_step / max(self.max_generation_steps, 1), 0.0, 1.0),
        ], dtype=np.float32)
        return {
            "state": np.clip(state, -1.0, 1.0).astype(np.float32),
            "coverage": self._coverage_observation(),
        }

    def reset(self, seed=None, options=None):
        """重置episode。

        输入：
            seed: 随机种子。
            options: Gymnasium兼容参数，当前未使用。

        输出：
            observation: 初始观测。
            info: 空字典。
        """
        super().reset(seed=seed)
        self.visited = np.zeros_like(self.cavity_mask, dtype=np.uint8)
        self.visited[self.contour_pass_mask] = 1
        self.visited[self.seed_pass_mask] = 1
        self.episode_wall_start = time.perf_counter()
        self.current_step = 0
        self.generated_control_points = []
        self.current_path = np.empty((0, 2), dtype=np.float64)
        self.current_curvature = np.array([], dtype=np.float64)
        self.current_envelope_mask = np.zeros_like(self.cavity_mask, dtype=bool)
        self.current_local_time_path = np.empty((0, 2), dtype=np.float64)
        self.current_tck = None
        self.terminated = self.truncated = False
        self.no_new_cut_steps = 0
        self.invalid_action_steps = 0
        self.episode_action_count = 0
        self.episode_invalid_actions = 0
        self.total_generation_time = 0.0
        self.episode_reward = 0.0
        self.ep_efficiency_reward_sum = 0.0
        self.ep_coverage_reward_sum = 0.0
        self.ep_invalid_reward_sum = 0.0
        self.ep_full_time_reward_sum = 0.0
        self.ep_new_area_mm2_sum = 0.0
        self.ep_local_time_sum = 0.0
        self.profile_records = []
        self.last_info = {}
        self.render_history = {
            "step": [],
            "local_time": [],
            "eval_time": [],
            "full_time": [],
            "total_time": [],
            "coverage_efficiency": [],
            "new_area_mm2": [],
            "reward": [],
        }
        self.current_feedrate_profile = {"s": np.array([]), "speed": np.array([]), "cap_s": np.array([]),
                                         "cap": np.array([]), "node_s": np.array([]), "node_v": np.array([]),
                                         "mode": self.time_eval_mode}
        self.current_point = self.seed_circle_path[0].copy()
        self.generated_control_points.append(self.current_point.copy())
        vector = self.current_point - self.path_center
        self.current_radius = float(np.linalg.norm(vector))
        self.current_theta = float(np.arctan2(vector[1], vector[0]))
        self.current_boundary_radius = self._boundary_radius(self.current_theta)
        self.current_rho = float(np.clip(self.current_radius / max(self.current_boundary_radius, 1e-12), 0.0, 1.0))
        return self._get_obs(), {}

    def step(self, action):
        """新增一个控制点，并用全部控制点重建完整B-spline曲线。

        输入：
            action: 二维数组，取值范围[-1, 1]。
                action[0]: 控制正向角度增量。
                action[1]: 控制有符号半径增量；允许局部向内回摆。

        输出：
            observation: 下一状态。
            reward: 本步奖励。
            terminated: 覆盖率达标时为True。
            truncated: 步数超限、连续无新增覆盖或连续非法动作过多时为True。
            info: 覆盖、局部时间、曲率和控制点诊断信息。
        """
        step_t0 = time.perf_counter()
        if self.terminated or self.truncated:
            return self._get_obs(), 0.0, self.terminated, self.truncated, self.last_info
        self.episode_action_count += 1

        # 增量式型腔归一化极坐标控制点生成：
        # 1) action[0]只决定本步前进角度，且始终为正，保证路径绕中心单向推进。
        # 2) action[1]先映射为物理半径增量，再换算成归一化半径rho的增量。
        # 3) 控制点由 center + rho * r_boundary(theta) * radial 生成，因此路径会随型腔形状变形。
        dtheta, dr_requested, action = self._map_action_to_increments(action)
        theta_new = self.current_theta + dtheta
        boundary_new = self._boundary_radius(theta_new)
        rho_floor = min(1.0, self.generation_radius_floor / max(boundary_new, 1e-12))
        rho_unclipped = self.current_rho + dr_requested / max(boundary_new, 1e-12)
        rho_new = float(np.clip(rho_unclipped, rho_floor, 1.0))
        new_point, radius_new, boundary_new = self._normalized_polar_point(theta_new, rho_new)
        dr = radius_new - self.current_radius
        t_action = time.perf_counter()

        if not self._tool_center_is_feasible(new_point):
            # 非法动作不执行：不追加控制点、不更新当前半径/角度、不更新覆盖。
            # 智能体会收到负奖励；若连续非法动作过多，则截断episode，避免确定性eval死循环。
            self.invalid_action_steps += 1
            self.episode_invalid_actions += 1
            invalid_truncated = self.invalid_action_steps >= self.max_consecutive_invalid_actions
            action_count_truncated = self.episode_action_count >= self.max_episode_actions
            self.truncated = bool(invalid_truncated or action_count_truncated)
            if invalid_truncated:
                reason = "too_many_invalid_actions"
            elif action_count_truncated:
                reason = "max_episode_actions"
            else:
                reason = "invalid_control_point"
            remaining = int(np.count_nonzero(self.inner_core_mask & (self.visited == 0)))
            covered = int(np.count_nonzero(self.inner_core_mask & (self.visited == 2)))
            current_length = 0.0
            if len(self.current_path) >= 2:
                current_length = float(np.sum(np.linalg.norm(np.diff(self.current_path, axis=0), axis=1)))
            reward = self.invalid_action_penalty
            full_time = 0.0
            full_time_reward = 0.0
            raw_full_time_reward = 0.0
            if self.truncated and len(self.current_path) >= 5:
                # 非法动作导致episode结束时，也用当前已生成路径做一次完整strict时间评价。
                # 本次非法控制点不执行，评价对象仍是上一条有效路径。
                self._refresh_full_sweep_state()
                full_time = self.estimate_machining_time(self.current_path, mode="full", planner="strict")
                full_time_reward, raw_full_time_reward = self._full_time_reward(full_time)
                reward += full_time_reward
            self.ep_full_time_reward_sum += full_time_reward
            self.episode_reward += reward
            self.ep_invalid_reward_sum += self.invalid_action_penalty
            self.last_info = {
                "reason": reason,
                "action_executed": False,
                "invalid_action_steps": self.invalid_action_steps,
                "episode_invalid_actions": self.episode_invalid_actions,
                "new_cells": 0,
                "new_area_mm2": 0.0,
                "repeat_cells": 0,
                "repeat_area_mm2": 0.0,
                "overcut_cells": 0,
                "infeasible_curve_points": 0,
                "curve_length": current_length,
                "local_time": 0.0,
                "eval_time": 0.0,
                "full_time": full_time,
                "local_time_path_length": 0.0,
                "local_time_planner": self.local_time_planner,
                "coverage_efficiency": 0.0,
                "efficiency_reward": 0.0,
                "coverage_reward": 0.0,
                "full_time_reward": full_time_reward,
                "raw_full_time_reward": raw_full_time_reward,
                "envelope_uncovered_cells": 0,
                "envelope_uncovered_area_mm2": 0.0,
                "envelope_coverage_ratio": 0.0,
                "max_kappa": 0.0,
                "max_dkappa": 0.0,
                "remaining_cells": remaining,
                "coverage_ratio": covered / max(self.total_planning_cells, 1),
                "coverage_reached": False,
                "control_points": np.asarray(self.generated_control_points).copy(),
                "attempted_control_point": new_point.copy(),
                "action": action.copy(),
                "dtheta": float(dtheta),
                "dr_requested": float(dr_requested),
                "equivalent_stepover": float(dr * 2.0 * np.pi / max(dtheta, 1e-12)),
                "dr": float(dr),
                "radius": float(radius_new),
                "rho": float(rho_new),
                "boundary_radius": float(boundary_new),
            }
            self._attach_episode_metrics(self.last_info, reward)
            self._record_render_history(self.last_info, reward)
            if self.render_mode in ("plot", "human"):
                self.render()
            obs = self._get_obs()
            t_end = time.perf_counter()
            self._record_step_profile({
                "action": t_action - step_t0,
                "invalid_bookkeeping": t_end - t_action,
                "total": t_end - step_t0,
            }, invalid=True)
            return obs, float(reward), False, self.truncated, self.last_info

        previous_point = self.current_point.copy()
        previous_envelope_mask = self.current_envelope_mask.copy()
        local_window_length = 1.5 * max(float(np.linalg.norm(new_point - previous_point)), self.curve_sample_spacing)

        self.generated_control_points.append(new_point.copy())
        path, curvature, s, tck = self._current_curve()
        t_curve = time.perf_counter()
        infeasible_curve_points = int(np.count_nonzero(~self._tool_centers_are_feasible(path)))
        t_feasible = time.perf_counter()

        radial_gap_ok, radial_gap_max, radial_gap_limit, radial_gap_theta = self._radial_gap_constraint(path)
        t_radial_gap = time.perf_counter()
        if not radial_gap_ok:
            # 候选路径会让相邻螺旋层径向间距超过刀具允许stepover，不执行该动作。
            self.generated_control_points.pop()
            self.invalid_action_steps += 1
            self.episode_invalid_actions += 1
            invalid_truncated = self.invalid_action_steps >= self.max_consecutive_invalid_actions
            action_count_truncated = self.episode_action_count >= self.max_episode_actions
            self.truncated = bool(invalid_truncated or action_count_truncated)
            if invalid_truncated:
                reason = "too_many_invalid_actions_radial_gap"
            elif action_count_truncated:
                reason = "max_episode_actions"
            else:
                reason = "radial_gap_violation"

            remaining = int(np.count_nonzero(self.inner_core_mask & (self.visited == 0)))
            covered = int(np.count_nonzero(self.inner_core_mask & (self.visited == 2)))
            current_length = 0.0
            if len(self.current_path) >= 2:
                current_length = float(np.sum(np.linalg.norm(np.diff(self.current_path, axis=0), axis=1)))

            reward = self.radial_gap_violation_penalty
            full_time = 0.0
            full_time_reward = 0.0
            raw_full_time_reward = 0.0
            if self.truncated and len(self.current_path) >= 5:
                self._refresh_full_sweep_state()
                full_time = self.estimate_machining_time(self.current_path, mode="full", planner="strict")
                full_time_reward, raw_full_time_reward = self._full_time_reward(full_time)
                reward += full_time_reward

            self.ep_full_time_reward_sum += full_time_reward
            self.episode_reward += reward
            self.ep_invalid_reward_sum += self.radial_gap_violation_penalty
            self.last_info = {
                "reason": reason,
                "action_executed": False,
                "radial_gap_violation": True,
                "radial_gap_max": float(radial_gap_max),
                "radial_gap_limit": float(radial_gap_limit),
                "radial_gap_theta": float(radial_gap_theta),
                "invalid_action_steps": self.invalid_action_steps,
                "episode_invalid_actions": self.episode_invalid_actions,
                "new_cells": 0,
                "new_area_mm2": 0.0,
                "repeat_cells": 0,
                "repeat_area_mm2": 0.0,
                "overcut_cells": 0,
                "infeasible_curve_points": int(infeasible_curve_points),
                "curve_length": current_length,
                "local_time": 0.0,
                "eval_time": 0.0,
                "full_time": full_time,
                "local_time_path_length": 0.0,
                "local_time_planner": self.local_time_planner,
                "coverage_efficiency": 0.0,
                "efficiency_reward": 0.0,
                "coverage_reward": 0.0,
                "full_time_reward": full_time_reward,
                "raw_full_time_reward": raw_full_time_reward,
                "envelope_uncovered_cells": 0,
                "envelope_uncovered_area_mm2": 0.0,
                "envelope_coverage_ratio": 0.0,
                "max_kappa": float(np.max(curvature)) if len(curvature) else 0.0,
                "max_dkappa": 0.0,
                "remaining_cells": remaining,
                "coverage_ratio": covered / max(self.total_planning_cells, 1),
                "coverage_reached": False,
                "control_points": np.asarray(self.generated_control_points).copy(),
                "attempted_control_point": new_point.copy(),
                "action": action.copy(),
                "dtheta": float(dtheta),
                "dr_requested": float(dr_requested),
                "equivalent_stepover": float(dr * 2.0 * np.pi / max(dtheta, 1e-12)),
                "dr": float(dr),
                "radius": float(radius_new),
                "rho": float(rho_new),
                "boundary_radius": float(boundary_new),
            }
            self._attach_episode_metrics(self.last_info, reward)
            self._record_render_history(self.last_info, reward)
            if self.render_mode in ("plot", "human"):
                self.render()
            obs = self._get_obs()
            t_end = time.perf_counter()
            self._record_step_profile({
                "action": t_action - step_t0,
                "curve": t_curve - t_action,
                "feasible": t_feasible - t_curve,
                "radial_gap": t_radial_gap - t_feasible,
                "invalid_bookkeeping": t_end - t_radial_gap,
                "total": t_end - step_t0,
            }, invalid=True)
            return obs, float(reward), False, self.truncated, self.last_info

        swept = self._step_sweep_mask(path)
        t_sweep = time.perf_counter()
        previously_cut = self.visited != 0
        new_mask = swept & self.inner_core_mask & (self.visited == 0)
        repeat_mask = swept & self.cavity_mask & previously_cut
        overcut_mask = swept & (~self.cavity_mask)
        self.visited[new_mask] = 2

        new_cells = int(np.count_nonzero(new_mask))
        repeat_cells = int(np.count_nonzero(repeat_mask))
        overcut_cells = int(np.count_nonzero(overcut_mask))
        t_masks = time.perf_counter()
        local_time_path = self._time_eval_path(path, "local", local_window_length=local_window_length)
        local_time, local_profile = self.estimate_machining_time(
            path, mode="local", return_profile=True, local_window_length=local_window_length
        )
        if self.time_eval_mode == "local":
            eval_time = local_time
        else:
            eval_time = self.estimate_machining_time(path, mode=self.time_eval_mode)
        self.total_generation_time += local_time
        t_time = time.perf_counter()
        envelope_mask = self._path_envelope_mask(path)
        envelope_cells_mask = envelope_mask & self.inner_core_mask
        new_envelope_mask = envelope_mask & (~previous_envelope_mask)
        reward_envelope_cells_mask = new_envelope_mask & self.inner_core_mask
        envelope_uncovered_mask = reward_envelope_cells_mask & (self.visited == 0)
        envelope_cells = int(np.count_nonzero(envelope_cells_mask))
        reward_envelope_cells = int(np.count_nonzero(reward_envelope_cells_mask))
        envelope_uncovered_cells = int(np.count_nonzero(envelope_uncovered_mask))
        envelope_uncovered_area_mm2 = envelope_uncovered_cells * self.res ** 2 * 1e6
        envelope_coverage_ratio = 1.0 - envelope_uncovered_cells / max(reward_envelope_cells, 1)
        t_envelope = time.perf_counter()

        self.current_point = new_point.copy()
        self.current_radius = float(radius_new)
        self.current_boundary_radius = float(boundary_new)
        self.current_rho = float(rho_new)
        self.current_theta = float(theta_new)
        self.current_path, self.current_curvature, self.current_tck = path, curvature, tck
        self.current_envelope_mask = envelope_mask
        self.current_local_time_path = local_time_path
        self.current_feedrate_profile = local_profile
        self.current_step += 1
        self.invalid_action_steps = 0
        self.no_new_cut_steps = self.no_new_cut_steps + 1 if new_cells == 0 else 0

        new_area_mm2 = new_cells * self.res ** 2 * 1e6
        repeat_area_mm2 = repeat_cells * self.res ** 2 * 1e6
        max_kappa = float(np.max(curvature)) if len(curvature) else 0.0
        max_dkappa = float(np.max(np.abs(np.gradient(curvature, s, edge_order=2)))) if len(s) >= 3 and s[-1] > 1e-12 else 0.0

        coverage_efficiency = new_area_mm2 / max(local_time, 1e-6)
        efficiency_reward = self.coverage_efficiency_weight * coverage_efficiency
        self.ep_efficiency_reward_sum += efficiency_reward
        self.ep_new_area_mm2_sum += new_area_mm2
        self.ep_local_time_sum += local_time

        remaining = int(np.count_nonzero(self.inner_core_mask & (self.visited == 0)))
        covered = int(np.count_nonzero(self.inner_core_mask & (self.visited == 2)))
        coverage_reached = remaining <= self.allowed_uncovered
        self.terminated = coverage_reached
        valid_step_truncated = self.current_step >= self.max_generation_steps
        action_count_truncated = self.episode_action_count >= self.max_episode_actions
        no_new_cut_truncated = self.no_new_cut_steps >= 50
        self.truncated = valid_step_truncated or action_count_truncated or no_new_cut_truncated
        truncation_reason = ""
        if valid_step_truncated:
            truncation_reason = "max_valid_generation_steps"
        elif action_count_truncated:
            truncation_reason = "max_episode_actions"
        elif no_new_cut_truncated:
            truncation_reason = "no_new_cut_steps"

        t_reward_start = time.perf_counter()
        # 训练step使用局部扫掠以降低耗时；episode结束或截断时，用完整路径重新刷新覆盖状态。
        # 这样终局统计和评估不会依赖局部扫掠近似。
        if self.terminated or self.truncated:
            self._refresh_full_sweep_state()
            envelope_uncovered_mask = reward_envelope_cells_mask & (self.visited == 0)
            envelope_uncovered_cells = int(np.count_nonzero(envelope_uncovered_mask))
            envelope_uncovered_area_mm2 = envelope_uncovered_cells * self.res ** 2 * 1e6
            envelope_coverage_ratio = 1.0 - envelope_uncovered_cells / max(reward_envelope_cells, 1)
            remaining = int(np.count_nonzero(self.inner_core_mask & (self.visited == 0)))
            covered = int(np.count_nonzero(self.inner_core_mask & (self.visited == 2)))
            coverage_reached = remaining <= self.allowed_uncovered
            self.terminated = coverage_reached
        t_terminal_full_sweep = time.perf_counter()

        coverage_reward = -self.envelope_uncovered_weight * envelope_uncovered_area_mm2
        if self.terminated:
            coverage_reward += self.coverage_completion_bonus
        elif self.truncated:
            coverage_reward -= self.remaining_cell_penalty * remaining
        self.ep_coverage_reward_sum += coverage_reward
        t_reward_calc = time.perf_counter()

        full_time = 0.0
        full_time_reward = 0.0
        raw_full_time_reward = 0.0
        if self.terminated or self.truncated:
            # 只有episode结束时才执行完整路径strict速度规划。训练step仍使用fast local planner，
            # 避免把完整axis-check规划放进每一步，导致环境采样速度明显下降。
            full_time = self.estimate_machining_time(path, mode="full", planner="strict")
            full_time_reward, raw_full_time_reward = self._full_time_reward(full_time)
        self.ep_full_time_reward_sum += full_time_reward
        t_terminal_full_time = time.perf_counter()

        reward = efficiency_reward + coverage_reward + full_time_reward

        self.episode_reward += reward
        self.last_info = {
            "new_cells": new_cells,
            "new_area_mm2": new_area_mm2,
            "repeat_cells": repeat_cells,
            "repeat_area_mm2": repeat_area_mm2,
            "overcut_cells": overcut_cells,
            "infeasible_curve_points": infeasible_curve_points,
            "curve_length": float(s[-1]) if len(s) else 0.0,
            "local_time": local_time,
            "eval_time": eval_time,
            "full_time": full_time,
            "time_eval_mode": self.time_eval_mode,
            "local_time_window_length": float(local_window_length),
            "local_time_path_length": float(np.sum(np.linalg.norm(np.diff(local_time_path, axis=0), axis=1))) if len(local_time_path) >= 2 else 0.0,
            "local_time_planner": self.local_time_planner,
            "coverage_efficiency": coverage_efficiency,
            "efficiency_reward": efficiency_reward,
            "coverage_reward": coverage_reward,
            "full_time_reward": full_time_reward,
            "raw_full_time_reward": raw_full_time_reward,
            "new_envelope_cells": reward_envelope_cells,
            "envelope_uncovered_cells": envelope_uncovered_cells,
            "envelope_uncovered_area_mm2": envelope_uncovered_area_mm2,
            "envelope_coverage_ratio": envelope_coverage_ratio,
            "max_kappa": max_kappa,
            "max_dkappa": max_dkappa,
            "remaining_cells": remaining,
            "coverage_ratio": covered / max(self.total_planning_cells, 1),
            "coverage_reached": coverage_reached,
            "radial_gap_violation": False,
            "radial_gap_max": float(radial_gap_max),
            "radial_gap_limit": float(radial_gap_limit),
            "radial_gap_theta": float(radial_gap_theta),
            "sweep_mode": "local" if self.use_local_step_sweep else "full",
            "reason": "coverage_reached" if self.terminated else (
                truncation_reason if self.truncated else "running"
            ),
            "action_executed": True,
            "invalid_action_steps": self.invalid_action_steps,
            "episode_invalid_actions": self.episode_invalid_actions,
            "control_points": np.asarray(self.generated_control_points).copy(),
            "new_control_point": new_point.copy(),
            "action": action.copy(),
            "dtheta": float(dtheta),
            "dr_requested": float(dr_requested),
            "equivalent_stepover": float(dr * 2.0 * np.pi / max(dtheta, 1e-12)),
            "dr": float(dr),
            "radius": float(radius_new),
            "rho": float(rho_new),
            "boundary_radius": float(boundary_new),
        }
        self._attach_episode_metrics(self.last_info, reward)
        self._record_render_history(self.last_info, reward)
        t_info_pack = time.perf_counter()
        if self.render_mode in ("plot", "human"):
            self.render()
        t_render = time.perf_counter()
        obs = self._get_obs()
        t_obs = time.perf_counter()
        self._record_step_profile({
            "action": t_action - step_t0,
            "curve": t_curve - t_action,
            "feasible": t_feasible - t_curve,
            "radial_gap": t_radial_gap - t_feasible,
            "sweep": t_sweep - t_radial_gap,
            "mask_update": t_masks - t_sweep,
            "time_plan": t_time - t_masks,
            "envelope": t_envelope - t_time,
            "terminal_full_sweep": t_terminal_full_sweep - t_reward_start,
            "reward_calc": t_reward_calc - t_terminal_full_sweep,
            "terminal_full_time": t_terminal_full_time - t_reward_calc,
            "info_pack": t_info_pack - t_terminal_full_time,
            "render": t_render - t_info_pack,
            "reward_info": t_render - t_envelope,
            "obs": t_obs - t_render,
            "total": t_obs - step_t0,
        }, invalid=False)
        return obs, float(reward), self.terminated, self.truncated, self.last_info

    def _record_step_profile(self, timing, invalid=False):
        """记录并按间隔打印step耗时。

        输入：
            timing: 字典，键为阶段名称，值为耗时秒数。
            invalid: 当前动作是否为非法动作。

        输出：
            无返回；更新self.profile_records并按profile_interval打印。
        """
        if not self.profile_step:
            return
        timing = dict(timing)
        timing["step"] = int(self.episode_action_count)
        timing["valid_step"] = int(self.current_step)
        timing["invalid"] = bool(invalid)
        self.profile_records.append(timing)
        if self.episode_action_count % self.profile_interval != 0 and not invalid:
            return
        parts = [
            f"{key}={value * 1000.0:.2f}ms"
            for key, value in timing.items()
            if isinstance(value, (int, float)) and key not in ("step", "valid_step", "invalid")
        ]
        print(
            f"[Env7 step profile] action_step={timing['step']} valid_step={timing['valid_step']} "
            f"invalid={timing['invalid']} | " + " | ".join(parts)
        )

    def _attach_episode_metrics(self, info, last_reward):
        """向info中写入episode级统计量。

        输入：
            info: 当前step的诊断字典。
            last_reward: 当前step总奖励。

        输出：
            无返回；原地更新info。
        """
        length = max(int(getattr(self, "episode_action_count", self.current_step)), 1)
        invalid_count = int(getattr(self, "episode_invalid_actions", 0))
        valid_count = max(length - invalid_count, 1)
        covered = int(np.count_nonzero(self.inner_core_mask & (self.visited == 2)))
        remaining = int(np.count_nonzero(self.inner_core_mask & (self.visited == 0)))
        coverage_ratio = covered / max(self.total_planning_cells, 1)
        info.update({
            "episode_reward": float(self.episode_reward),
            "episode_length": int(length),
            "episode_wall_time_s": float(time.perf_counter() - getattr(self, "episode_wall_start", time.perf_counter())),
            "last_reward": float(last_reward),
            "coverage_ratio": float(coverage_ratio),
            "remaining_cells": int(remaining),
            "ep_efficiency_reward_sum": float(self.ep_efficiency_reward_sum),
            "ep_coverage_reward_sum": float(self.ep_coverage_reward_sum),
            "ep_invalid_reward_sum": float(self.ep_invalid_reward_sum),
            "ep_full_time_reward_sum": float(self.ep_full_time_reward_sum),
            "full_time_reward_floor": float(self.full_time_reward_floor),
            "ep_efficiency_reward_mean": float(self.ep_efficiency_reward_sum / length),
            "ep_coverage_reward_mean": float(self.ep_coverage_reward_sum / length),
            "ep_invalid_reward_mean": float(self.ep_invalid_reward_sum / length),
            "ep_efficiency_reward_per_valid_action": float(self.ep_efficiency_reward_sum / valid_count),
            "ep_coverage_reward_per_valid_action": float(self.ep_coverage_reward_sum / valid_count),
            "ep_invalid_reward_per_invalid_action": float(
                self.ep_invalid_reward_sum / invalid_count if invalid_count > 0 else 0.0
            ),
            "ep_new_area_mm2_sum": float(self.ep_new_area_mm2_sum),
            "ep_local_time_sum": float(self.ep_local_time_sum),
            "episode_invalid_actions": invalid_count,
            "episode_valid_actions": int(valid_count),
            "max_generation_steps": int(getattr(self, "max_generation_steps", 0)),
            "max_episode_actions": int(getattr(self, "max_episode_actions", 0)),
        })

    def _record_render_history(self, info, reward):
        """记录render诊断历史。

        输入：
            info: step生成的诊断字典。
            reward: 当前step奖励。

        输出：
            无返回；更新self.render_history。
        """
        if not hasattr(self, "render_history"):
            return
        self.render_history["step"].append(float(self.current_step))
        self.render_history["local_time"].append(float(info.get("local_time", 0.0)))
        self.render_history["eval_time"].append(float(info.get("eval_time", 0.0)))
        self.render_history["full_time"].append(float(info.get("full_time", 0.0)))
        self.render_history["total_time"].append(float(self.total_generation_time))
        self.render_history["coverage_efficiency"].append(float(info.get("coverage_efficiency", 0.0)))
        self.render_history["new_area_mm2"].append(float(info.get("new_area_mm2", 0.0)))
        self.render_history["reward"].append(float(reward))

    def get_generated_path(self):
        """获取当前完整B-spline路径。

        输入：
            无。

        输出：
            path: 形状为(N, 2)的当前曲线采样点。
        """
        return np.asarray(self.current_path, dtype=np.float64)

    def get_generated_control_points(self):
        """获取当前所有控制点。

        输入：
            无。

        输出：
            control_points: 形状为(N, 2)的控制点数组。
        """
        return np.asarray(self.generated_control_points, dtype=np.float64)

    def _set_axis(self, axis, title):
        """设置绘图坐标轴。

        输入：
            axis: matplotlib坐标轴对象。
            title: 图标题。

        输出：
            无返回。
        """
        size = self.wp_size * 1000.0
        axis.set(title=title, xlabel="X (mm)", ylabel="Y (mm)", xlim=(0, size), ylim=(0, size))
        axis.set_aspect("equal")

    def _plot_cavity_boundary(self, axis, linewidth=1.2):
        """绘制型腔边界。

        输入：
            axis: matplotlib坐标轴对象。
            linewidth: 线宽。

        输出：
            无返回。
        """
        x_mm, y_mm = self.X_real_wp[0] * 1000.0, self.Y_real_wp[:, 0] * 1000.0
        axis.contour(x_mm, y_mm, self.cavity_mask.astype(float), levels=[0.5], colors="black", linewidths=linewidth)

    def _initialize_render(self):
        """初始化render窗口。

        输入：
            无。

        输出：
            无返回；创建覆盖图、控制点/曲线图、动作映射图和曲率图。
        """
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15.5, 11.0))
        self.fig.suptitle("Incremental Control-Point NURBS Generation Test", fontsize=15)
        extent = [0, self.wp_size * 1000, 0, self.wp_size * 1000]
        cmap = mcolors.ListedColormap(["#D9D9D9", "#FFB3B3", "#17BECF", "#2CA02C", "#F2C94C"])

        ax_cov, ax_path, ax_map, ax_curv = self.axes.ravel()

        self.coverage_image = ax_cov.imshow(self._coverage_matrix(), origin="lower", extent=extent, cmap=cmap, vmin=0, vmax=4)
        envelope_cmap = mcolors.ListedColormap([(0.0, 0.0, 0.0, 0.0), (0.50, 0.0, 0.95, 0.16)])
        self.envelope_image = ax_cov.imshow(np.zeros_like(self.cavity_mask, dtype=np.uint8), origin="lower",
                                            extent=extent, cmap=envelope_cmap, vmin=0, vmax=1)
        self._plot_cavity_boundary(ax_cov, linewidth=1.4)
        ax_cov.plot(self.seed_circle_path[:, 0] * 1000, self.seed_circle_path[:, 1] * 1000, color="#F2A900", linewidth=1.7, label="Fixed seed circle")
        self.coverage_path_line, = ax_cov.plot([], [], color="black", linewidth=1.25, label="Current NURBS path")
        self.coverage_local_time_path_line, = ax_cov.plot([], [], color="#FFD400", linewidth=2.4, label="Local timing path")
        patches = [mpatches.Patch(color="#FFB3B3", label="Inner uncovered"),
                   mpatches.Patch(color="#17BECF", label="Prefinished contour"),
                   mpatches.Patch(color="#2CA02C", label="New covered"),
                   mpatches.Patch(color="#F2C94C", label="Seed covered"),
                   mpatches.Patch(color=(0.50, 0.0, 0.95, 0.16), label="Path envelope")]
        handles, labels = ax_cov.get_legend_handles_labels()
        ax_cov.legend(handles + patches, labels + [p.get_label() for p in patches], loc="upper right", fontsize=8)
        self._set_axis(ax_cov, "Coverage Update")

        self._plot_cavity_boundary(ax_path, linewidth=1.2)
        ax_path.plot(self.seed_circle_path[:, 0] * 1000, self.seed_circle_path[:, 1] * 1000, color="#F2A900", linewidth=1.5, label="Fixed seed circle")
        self.nurbs_path_line, = ax_path.plot([], [], color="#1F77B4", linewidth=1.5, label="Current NURBS path")
        self.local_time_path_line, = ax_path.plot([], [], color="#FFD400", linewidth=2.4, label="Local timing path")
        self.control_polygon_line, = ax_path.plot([], [], color="0.55", linewidth=0.9, linestyle="--", label="Control polygon")
        self.control_scatter = ax_path.scatter([], [], facecolors="none", edgecolors="#7B2CBF", s=35, label="Control points")
        self.current_point_scatter = ax_path.scatter([], [], color="magenta", marker="s", s=58, label="New control point")
        self.current_info_text = ax_path.text(0.02, 0.02, "", transform=ax_path.transAxes, va="bottom", ha="left", fontsize=9,
                                                   bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7"))
        ax_path.legend(loc="upper right", fontsize=8)
        self._set_axis(ax_path, "NURBS Path and Incremental Control Points")

        action_grid = np.linspace(-1.0, 1.0, 201)
        self.action_dtheta_line, = ax_map.plot([], [], color="#1F77B4", label="dtheta (deg)")
        self.action_dr_line, = ax_map.plot([], [], color="#D62728", label="dr (mm)")
        self.action_current_dtheta = ax_map.axvline(0.0, color="#1F77B4", linestyle=":", linewidth=1.0)
        self.action_current_dr = ax_map.axvline(0.0, color="#D62728", linestyle=":", linewidth=1.0)
        ax_map.axhline(0.0, color="0.75", linewidth=0.8)
        ax_map.set(xlabel="action value", ylabel="mapped increment", xlim=(-1.0, 1.0),
                   title="Action Mapping Distribution")
        ax_map.grid(True, linestyle=":")
        ax_map.legend(fontsize=8)
        self._action_grid = action_grid

        self.curvature_line, = ax_curv.plot([], [], color="#1F77B4", label="NURBS curvature")
        ax_curv.set(title="Curvature Profile", xlabel="Arc length (mm)", ylabel="Curvature (1/m)")
        ax_curv.grid(True, linestyle=":")
        ax_curv.legend(fontsize=8)
        self.fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show(block=False)

    def _initialize_speed_render(self):
        """初始化完整路径速度诊断窗口。

        输入：
            无。

        输出：
            无返回；创建完整速度、XY加速度、XY jerk、累计时间和曲率图。
        """
        self.speed_fig, self.speed_axes = plt.subplots(3, 2, figsize=(15.5, 11.0))
        self.speed_fig.suptitle("Full-Path Feedrate, Axis Dynamics and Curvature Diagnostics", fontsize=14)
        ax_speed, ax_acc, ax_jerk, ax_time, ax_curv, ax_info = self.speed_axes.ravel()

        self.full_cap_line, = ax_speed.plot([], [], color="0.65", linestyle="--", label="Full-path speed cap")
        self.full_speed_line, = ax_speed.plot([], [], color="#1F77B4", linewidth=1.6, label="Planned full-path speed")
        self.full_node_scatter = ax_speed.scatter([], [], s=18, color="#D62728", label="Speed nodes")
        ax_speed.axhline(self.v_max * 1000.0, color="red", linestyle=":", linewidth=1.0, label="Vmax")
        ax_speed.set(title="Full-Path Feedrate Profile", xlabel="Arc length (mm)", ylabel="Speed (mm/s)")
        ax_speed.grid(True, linestyle=":")
        ax_speed.legend(fontsize=8)

        self.full_ax_line, = ax_acc.plot([], [], color="#1F77B4", label="X acceleration")
        self.full_ay_line, = ax_acc.plot([], [], color="#FF7F0E", label="Y acceleration")
        ax_acc.axhline(self.acc_max * 1000.0, color="red", linestyle=":", linewidth=1.0)
        ax_acc.axhline(-self.acc_max * 1000.0, color="red", linestyle=":", linewidth=1.0)
        ax_acc.set(title="X/Y Axis Acceleration", xlabel="Arc length (mm)", ylabel="Acceleration (mm/s2)")
        ax_acc.grid(True, linestyle=":")
        ax_acc.legend(fontsize=8)

        self.full_jx_line, = ax_jerk.plot([], [], color="#1F77B4", label="X jerk")
        self.full_jy_line, = ax_jerk.plot([], [], color="#FF7F0E", label="Y jerk")
        ax_jerk.axhline(self.jerk_max * 1000.0, color="red", linestyle=":", linewidth=1.0)
        ax_jerk.axhline(-self.jerk_max * 1000.0, color="red", linestyle=":", linewidth=1.0)
        ax_jerk.set(title="X/Y Axis Jerk", xlabel="Arc length (mm)", ylabel="Jerk (mm/s3)")
        ax_jerk.grid(True, linestyle=":")
        ax_jerk.legend(fontsize=8)

        self.full_time_line, = ax_time.plot([], [], color="#2CA02C", linewidth=1.6, label="Cumulative time")
        ax_time.set(title="Cumulative Time Along Full Path", xlabel="Arc length (mm)", ylabel="Time (s)")
        ax_time.grid(True, linestyle=":")
        ax_time.legend(fontsize=8)

        self.full_curvature_line, = ax_curv.plot([], [], color="#9467BD", linewidth=1.4, label="Curvature")
        ax_curv.set(title="Curvature Along Full Path", xlabel="Arc length (mm)", ylabel="Curvature (1/m)")
        ax_curv.grid(True, linestyle=":")
        ax_curv.legend(fontsize=8)

        ax_info.axis("off")
        self.full_summary_text = ax_info.text(0.02, 0.98, "", transform=ax_info.transAxes,
                                              va="top", ha="left", fontsize=10,
                                              bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.7"))

        self.speed_fig.tight_layout(rect=[0, 0, 1, 0.94])
        plt.show(block=False)

    def _render_speed_diagnostics(self):
        """刷新完整路径速度诊断窗口。

        输入：
            无。

        输出：
            无返回；根据当前完整路径重新计算速度规划和轴动力学诊断。
        """
        if self.speed_fig is None:
            self._initialize_speed_render()

        ax_speed, ax_acc, ax_jerk, ax_time, ax_curv, _ = self.speed_axes.ravel()
        diag = self._full_path_speed_diagnostics()
        s = np.asarray(diag.get("s", []), dtype=np.float64)
        s_mm = s * 1000.0
        v = np.asarray(diag.get("speed", []), dtype=np.float64)
        cap_s = np.asarray(diag.get("cap_s", []), dtype=np.float64)
        cap = np.asarray(diag.get("cap", []), dtype=np.float64)
        node_s = np.asarray(diag.get("node_s", []), dtype=np.float64)
        node_v = np.asarray(diag.get("node_v", []), dtype=np.float64)

        self.full_speed_line.set_data(s_mm, v * 1000.0)
        self.full_cap_line.set_data(cap_s * 1000.0, cap * 1000.0)
        if len(node_s):
            self.full_node_scatter.set_offsets(np.column_stack((node_s * 1000.0, node_v * 1000.0)))
        else:
            self.full_node_scatter.set_offsets(np.empty((0, 2)))
        ax_speed.relim()
        ax_speed.autoscale_view()

        self.full_ax_line.set_data(s_mm, np.asarray(diag.get("ax", [])) * 1000.0)
        self.full_ay_line.set_data(s_mm, np.asarray(diag.get("ay", [])) * 1000.0)
        self.full_jx_line.set_data(s_mm, np.asarray(diag.get("jx", [])) * 1000.0)
        self.full_jy_line.set_data(s_mm, np.asarray(diag.get("jy", [])) * 1000.0)
        self.full_time_line.set_data(s_mm, np.asarray(diag.get("time", [])))
        self.full_curvature_line.set_data(s_mm, np.asarray(diag.get("curvature", [])))

        for axis in (ax_acc, ax_jerk, ax_time, ax_curv):
            axis.relim()
            axis.autoscale_view()

        total_time = float(diag.get("total_time", 0.0))
        max_speed = float(np.max(v) * 1000.0) if len(v) else 0.0
        max_ax = float(np.max(np.abs(diag.get("ax", []))) * 1000.0) if len(diag.get("ax", [])) else 0.0
        max_ay = float(np.max(np.abs(diag.get("ay", []))) * 1000.0) if len(diag.get("ay", [])) else 0.0
        max_jx = float(np.max(np.abs(diag.get("jx", []))) * 1000.0) if len(diag.get("jx", [])) else 0.0
        max_jy = float(np.max(np.abs(diag.get("jy", []))) * 1000.0) if len(diag.get("jy", [])) else 0.0
        path_len = float(s[-1] * 1000.0) if len(s) else 0.0
        self.full_summary_text.set_text(
            f"Full-path diagnostics\n\n"
            f"Path length: {path_len:.3f} mm\n"
            f"Full-path time: {total_time:.4f} s\n"
            f"Axis-check converged: {diag.get('converged', False)}\n"
            f"Axis-check iterations: {diag.get('iterations', 0)}\n"
            f"Max speed: {max_speed:.3f} mm/s\n"
            f"Max |Ax|: {max_ax:.3f} mm/s2\n"
            f"Max |Ay|: {max_ay:.3f} mm/s2\n"
            f"Max |Jx|: {max_jx:.3f} mm/s3\n"
            f"Max |Jy|: {max_jy:.3f} mm/s3\n\n"
            f"Limits\n"
            f"Vmax: {self.v_max * 1000.0:.3f} mm/s\n"
            f"Amax: {self.acc_max * 1000.0:.3f} mm/s2\n"
            f"Jmax: {self.jerk_max * 1000.0:.3f} mm/s3"
        )
        self.speed_fig.canvas.draw_idle()
        self.speed_fig.canvas.flush_events()

    def render(self, force_speed_diagnostics=False):
        """刷新render窗口。

        输入：
            force_speed_diagnostics: 是否强制刷新完整路径速度/加速度/jerk诊断图。
                默认False，仅在episode结束时绘制第二个figure；
                测试脚本可以设为True，以便在未终止时查看当前完整路径诊断。

        输出：
            无返回；更新覆盖、当前完整曲线和控制点。
        """
        if len(getattr(self, "current_path", [])):
            self._refresh_full_sweep_state()
        if self.fig is None:
            self._initialize_render()
        ax_cov, _, ax_map, ax_curv = self.axes.ravel()
        self.coverage_image.set_data(self._coverage_matrix())
        self.envelope_image.set_data(self.current_envelope_mask.astype(np.uint8))
        for contour in self.envelope_contours:
            contour.remove()
        self.envelope_contours = []
        if np.any(self.current_envelope_mask):
            x_mm, y_mm = self.X_real_wp[0] * 1000.0, self.Y_real_wp[:, 0] * 1000.0
            contour_set = ax_cov.contour(x_mm, y_mm, self.current_envelope_mask.astype(float),
                                         levels=[0.5], colors="#7B2CBF", linewidths=1.25, alpha=0.55)
            if hasattr(contour_set, "collections"):
                self.envelope_contours = list(contour_set.collections)
            else:
                self.envelope_contours = [contour_set]
        path = self.get_generated_path() * 1000.0
        local_time_path = getattr(self, "current_local_time_path", np.empty((0, 2))) * 1000.0
        controls = self.get_generated_control_points() * 1000.0
        if len(path):
            self.coverage_path_line.set_data(path[:, 0], path[:, 1])
            self.nurbs_path_line.set_data(path[:, 0], path[:, 1])
        if len(local_time_path):
            self.coverage_local_time_path_line.set_data(local_time_path[:, 0], local_time_path[:, 1])
            self.local_time_path_line.set_data(local_time_path[:, 0], local_time_path[:, 1])
        else:
            self.coverage_local_time_path_line.set_data([], [])
            self.local_time_path_line.set_data([], [])
        if len(controls):
            self.control_polygon_line.set_data(controls[:, 0], controls[:, 1])
            self.control_scatter.set_offsets(controls)
            self.current_point_scatter.set_offsets(controls[-1].reshape(1, 2))
        covered = np.count_nonzero(self.inner_core_mask & (self.visited == 2))
        remaining = np.count_nonzero(self.inner_core_mask & (self.visited == 0))
        coverage = covered / max(self.total_planning_cells, 1)
        ax_cov.set_title(f"Coverage Update | step {self.current_step} | coverage {coverage * 100:.3f}% | remaining {remaining}")
        self.current_info_text.set_text(
            f"step: {self.current_step}\n"
            f"control points: {len(controls)}\n"
            f"action executed: {self.last_info.get('action_executed', True)}\n"
            f"invalid steps: {self.last_info.get('invalid_action_steps', 0)}\n"
            f"episode invalid: {self.last_info.get('episode_invalid_actions', 0)}\n"
            f"new cells: {self.last_info.get('new_cells', 0)}\n"
            f"repeat cells: {self.last_info.get('repeat_cells', 0)}\n"
            f"dr: {self.last_info.get('dr', 0.0) * 1000.0:.3f} mm\n"
            f"requested dr: {self.last_info.get('dr_requested', 0.0) * 1000.0:.3f} mm\n"
            f"radius: {self.last_info.get('radius', 0.0) * 1000.0:.3f} mm\n"
            f"rho: {self.last_info.get('rho', getattr(self, 'current_rho', 0.0)):.4f}\n"
            f"boundary r: {self.last_info.get('boundary_radius', getattr(self, 'current_boundary_radius', 0.0)) * 1000.0:.3f} mm\n"
            f"equiv. stepover: {self.last_info.get('equivalent_stepover', 0.0) * 1000.0:.3f} mm\n"
            f"local path: {self.last_info.get('local_time_path_length', 0.0) * 1000.0:.2f} mm\n"
            f"local time: {self.last_info.get('local_time', 0.0):.4f} s\n"
            f"radial gap: {self.last_info.get('radial_gap_max', 0.0) * 1000.0:.2f}/"
            f"{self.last_info.get('radial_gap_limit', self.radial_gap_safety_ratio * 2.0 * self.tool_r) * 1000.0:.2f} mm\n"
            f"eval time ({self.last_info.get('time_eval_mode', self.time_eval_mode)}): "
            f"{self.last_info.get('eval_time', 0.0):.4f} s\n"
            f"env. uncut: {self.last_info.get('envelope_uncovered_area_mm2', 0.0):.2f} mm2\n"
            f"env. cov.: {self.last_info.get('envelope_coverage_ratio', 0.0) * 100.0:.2f}%\n"
            f"eff.: {self.last_info.get('coverage_efficiency', 0.0):.2f} mm2/s\n"
            f"full time: {self.last_info.get('full_time', 0.0):.4f} s\n"
            f"R_eff/R_cov: {self.last_info.get('efficiency_reward', 0.0):.2f}/"
            f"{self.last_info.get('coverage_reward', 0.0):.2f}\n"
            f"R_full: {self.last_info.get('full_time_reward', 0.0):.2f}\n"
            f"R_full_raw: {self.last_info.get('raw_full_time_reward', 0.0):.2f}"
        )
        if len(self.current_curvature) and len(self.current_path) >= 2:
            s_mm = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(self.current_path, axis=0), axis=1)))) * 1000.0
            self.curvature_line.set_data(s_mm, self.current_curvature)
            ax_curv.relim()
            ax_curv.autoscale_view()

        dtheta_center, dtheta_lower, dtheta_upper, dr_lower, dr_upper = self._action_intervals()
        dtheta_values = [self._signed_interval_action(x, dtheta_center, dtheta_lower, dtheta_upper) for x in self._action_grid]
        dr_values = [self._signed_interval_action(x, self.base_radial_step, dr_lower, dr_upper) for x in self._action_grid]
        self.action_dtheta_line.set_data(self._action_grid, np.rad2deg(dtheta_values))
        self.action_dr_line.set_data(self._action_grid, np.asarray(dr_values) * 1000.0)
        action_for_plot = self.last_info.get("action", np.zeros(2)) if self.last_info else np.zeros(2)
        self.action_current_dtheta.set_xdata([action_for_plot[0], action_for_plot[0]])
        self.action_current_dr.set_xdata([action_for_plot[1], action_for_plot[1]])
        ax_map.relim()
        ax_map.autoscale_view(scalex=False, scaley=True)

        # 完整路径速度/加速度/jerk诊断默认只在路径规划结束后绘制。
        # 测试入口可通过force_speed_diagnostics=True查看未结束episode的当前完整路径诊断，
        # 但不应为了渲染而篡改self.truncated/self.terminated状态。
        if self.terminated or self.truncated or force_speed_diagnostics:
            self._render_speed_diagnostics()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        """关闭render窗口。

        输入：
            无。

        输出：
            无返回。
        """
        if self.fig is not None:
            plt.ioff()
            plt.close(self.fig)
            self.fig = None
        if self.speed_fig is not None:
            plt.close(self.speed_fig)
            self.speed_fig = None


if __name__ == "__main__":
    TEST_RANDOM_SEED = None
    TEST_STEPS = 20000
    SHOW_FINAL_RENDER = True
    BLOCK_FINAL_FIGURE = True

    env = MillingEnvNurbs(render_mode="plot", generation_points_per_turn=24,
                          generation_time_window_points=240,
                          profile_step=False, profile_interval=10)
    rng = np.random.default_rng(TEST_RANDOM_SEED)
    obs, _ = env.reset(seed=TEST_RANDOM_SEED)
    print("开始测试Env7：每个step新增一个控制点，并重建完整NURBS/B-spline曲线。")
    print(f"observation keys: {list(obs.keys())}, state shape: {obs['state'].shape}, "
          f"coverage shape: {obs['coverage'].shape}, action space: {env.action_space}")

    for i in range(min(TEST_STEPS, env.max_episode_actions)):
        action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        print("terminated:", terminated)
        print("truncated:", truncated)
        print("reason:", info["reason"])
        print("invalid_action_steps:", info["invalid_action_steps"])
        print("episode_action_count:", env.episode_action_count)
        print("max_episode_actions:", env.max_episode_actions)
        if i % 5 == 0 or terminated or truncated:
            print(f"step {i + 1:03d} | reward {reward:8.3f} | new {info.get('new_cells', 0):4d} | "
                  f"coverage {info.get('coverage_ratio', 0.0) * 100:7.3f}% | "
                  f"remaining {info.get('remaining_cells', 0):5d} | controls {len(env.get_generated_control_points()):3d}")
        if terminated or truncated:
            print(f"测试结束：terminated={terminated}, truncated={truncated}, reason={info.get('reason', '')}")
            break

    print(f"当前NURBS路径采样点数：{len(env.get_generated_path())}")
    print(f"当前控制点数量：{len(env.get_generated_control_points())}")
    if env.profile_records:
        keys = [key for key in env.profile_records[-1].keys()
                if key not in ("step", "valid_step", "invalid")]
        print("\n平均step耗时统计：")
        for key in keys:
            values = [record[key] for record in env.profile_records if key in record]
            if values:
                print(f"{key:>18s}: {np.mean(values) * 1000.0:9.3f} ms")

    if SHOW_FINAL_RENDER:
        # 测试过程保持render_mode=None以避免拖慢step耗时统计；
        # 测试结束后再渲染最终路径和完整速度诊断图。
        env.render_mode = "plot"
        if not (env.terminated or env.truncated):
            print("测试循环达到TEST_STEPS上限，但环境仍未terminated/truncated；这不是环境截断。")
        env.render(force_speed_diagnostics=True)

        if BLOCK_FINAL_FIGURE:
            print("最终render窗口已打开。关闭图窗后脚本才会退出。")
            plt.ioff()
            plt.show(block=True)
