import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches


class MillingEnvRenderer:
    """Matplotlib renderer for MillingEnvNurbs.

    This class holds only visualization and render-history logic. It delegates
    environment data/method access back to the env object, so existing code can
    still use env.fig, env.speed_fig, env.render(), and env.close().
    """

    def __init__(self, env):
        object.__setattr__(self, "env", env)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def __setattr__(self, name, value):
        if name == "env":
            object.__setattr__(self, name, value)
        else:
            setattr(self.env, name, value)

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

