import numpy as np
from scipy.signal import find_peaks


class SpeedPlanner:
    """速度规划与加工时间计算模块。

    说明：
        该类从环境中拆出速度规划相关逻辑。
        self.env 保存环境对象；未在本类中定义的属性和方法会转发到 env。
        因此原速度规划代码中使用的 self.v_max、self.acc_max、self._time_eval_path 等接口保持不变。
    """

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

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

