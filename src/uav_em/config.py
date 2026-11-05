"""统一研究配置：位置用米，时间用秒，干扰功率用dBm，能量用瓦时。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class AircraftConfig:
    """定高平面运动的研究假设，不对应经过标定的实际机型。

    altitude_m为相对统一地面高度m，正数，用于传播距离；不开放高度优化。
    vertical_clearance_m为上下净空m，非负；仅逐栋建筑高度碰撞使用，等于净空边界也视为不可通行。
    """

    # 沿折线航路的恒定速度，m/s；暂不模拟转弯加速度和风。
    speed_mps: float = 15.0
    altitude_m: float = 30.0
    vertical_clearance_m: float = 5.0
    # 飞行和咨询时悬停的功率，W。
    flight_power_w: float = 350.0
    hover_power_w: float = 300.0
    # 初始能量和不得用于普通任务飞行的余量，Wh。
    initial_energy_wh: float = 100.0
    reserve_energy_wh: float = 15.0
    # 几何障碍膨胀距离，m；不是电磁安全距离。
    clearance_m: float = 5.0


@dataclass(frozen=True)
class SimulationConfig:
    """执行、观测与模型预算；各字段彼此独立，不混同采样频率。

    source_refinement_rounds为同一决策点最多咨询轮数，整数且至少1；每轮计入max_advisor_calls。
    warning_window_seconds为趋势观察窗s，warning_rise_db为提前复核累计增量dB，均为正数。
    预警仍受minimum_review_seconds约束，不是新的模型行动或学习门控。
    """

    # 每次最大飞行推进时间；每轮在当前位置测量一次。
    step_seconds: float = 1.0
    # 常规决策间隔；直接咨询模型，不使用学习门控。
    review_seconds: float = 10.0
    # 新测量明显偏离已有估计时可提前咨询，但至少间隔此时长。
    minimum_review_seconds: float = 3.0
    # 触发提前决策的测量创新量绝对值，dB；不是中断时间阈值。
    innovation_trigger_db: float = 7.0
    warning_window_seconds: float = 5.0
    warning_rise_db: float = 6.0
    # 任务仿真时长上限及每次决策固定计入的虚拟悬停时间。
    max_seconds: float = 180.0
    decision_delay_seconds: float = 0.5
    # 模型调用上限；耗尽后使用明确记录的本地数值回退。
    max_advisor_calls: int = 24
    source_refinement_rounds: int = 3
    # 搜索预算分别是A*扩展数和RRT*采样提议数，二者不能等同。
    astar_expansion_budget: int = 5000
    rrt_attempt_budget: int = 700
    # 一次任务允许的主动补测次数，以及每次可执行的最长补测航行时间。
    max_probes: int = 2
    probe_budget_seconds: float = 12.0


@dataclass(frozen=True)
class MapConfig:
    """二维定高航路，节点坐标为列／行号乘resolution_m（米）。

    geometry_mode为planar或fixed_altitude；前者禁止所有楼体轮廓，后者按逐栋高度和上下净空筛选。
    buildings为局部米制简单多边形棱柱，含id/footprint_m/height_m，可选庭院holes_m、基底base_m、损耗loss_db。
    obstacles_m始终为额外的平面矩形禁入区，不用其统一传播高度推断可否飞越。
    provenance记录来源、许可、坐标基准和缺失高度处理，不是电磁观测或在线真值信息。
    """

    # 已知几何地图的列数、行数，以及节点间距m。
    columns: int = 41
    rows: int = 31
    resolution_m: float = 20.0
    # 起终点位置；必须位于边界内且不与膨胀障碍相交。
    start_m: tuple[float, float] = (40.0, 300.0)
    goal_m: tuple[float, float] = (760.0, 300.0)
    # 已知矩形障碍，每项为[xmin,ymin,xmax,ymax]，m。
    obstacles_m: tuple[tuple[float, float, float, float], ...] = (
        (280.0, 240.0, 360.0, 360.0),
        (520.0, 100.0, 580.0, 260.0),
    )
    geometry_mode: str = "planar"
    buildings: tuple[dict, ...] = ()
    provenance: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ElectromagneticConfig:
    """同一窄观测频带的非相干接收功率；不建模服务站或通信链路。

    carrier_frequency_hz为载频Hz，reference_distance_m为远场参考距离m，均须为有限正数。
    propagation_mode为log_distance或single_reflection；前者保留经验基线，后者为主配置的有限路径近似。
    path_loss_exponent为对数距离损耗指数，[2,6]；single_reflection必须为2，避免重复经验衰减。
    relative_permittivity为墙／地面统一相对介电常数，有限且>=1；conductivity_s_per_m为电导率S/m，非负。
    反射采用非极化平均Fresnel功率系数；只含单次墙／地面反射，不计算相干相位或绕射。
    obstacle_height_m和obstacle_loss_db为所有矩形的统一高度m和每次穿透损耗dB，非负。
    power_grid可为总接收干扰dBm静态栅格；路径在load_config时解析，必须power_only、空fields且频率／机高一致。
    fields可为传播源（eirp_dbm、height_m）或显式高斯基线（sigma_m、peak_dbm），不混用参数。
    EIRP是观测频带内等效全向辐射功率dBm；源高m非负，其活动时间和速度仅供真值生成。
    """

    carrier_frequency_hz: float = 2.4e9
    reference_distance_m: float = 1.0
    path_loss_exponent: float = 2.2
    obstacle_height_m: float = 20.0
    obstacle_loss_db: float = 6.0
    propagation_mode: str = "log_distance"
    relative_permittivity: float = 4.0
    conductivity_s_per_m: float = 0.005
    power_grid: dict | None = None

    # 均匀背景功率，dBm；用于无局部观测时的先验，不代表已测安全。
    background_dbm: float = -100.0
    # 当前点干扰功率测量的独立高斯噪声标准差，dB。
    measurement_noise_db: float = 0.5
    # 干扰软参考值，dBm；超过后增加代价，不是失效或禁飞阈值。
    interference_reference_dbm: float = -80.0
    # 初期局部干扰场：保留弱干扰绕行通道；真值脚本不进入模型上下文。
    fields: tuple[dict, ...] = field(default_factory=lambda: (
        {"center_m": [410.0, 390.0], "eirp_dbm": -2.0, "height_m": 1.5,
         "start_seconds": 0.0, "end_seconds": 200.0, "velocity_mps": [0.0, 0.0]},
        {"center_m": [620.0, 350.0], "eirp_dbm": -5.0, "height_m": 1.5,
         "start_seconds": 24.0, "end_seconds": 200.0, "velocity_mps": [0.0, -0.4]},
    ))


@dataclass(frozen=True)
class SensorConfig:
    """理想化已标定360°阵列的同步波束功率快照，不是空间圆盘真值扫描。

    azimuth_bins为均分360°的波束数，整数且在[4,360]；方向以东为0°，逆时针增加。
    beamwidth_deg为半功率全波束宽度°，(0,180]；front_back_db为方向抑制上限dB，非负。
    detection_threshold_dbm为有限的波束检测灵敏度dBm，不是干扰危险阈值；还须通过背景及角对比检测。
    minimum_excess_db为相对背景最小检测增量dB，正数；noise_db为波束功率噪声标准差dB，非负。
    minimum_contrast_db为角峰相对扫描中位功率的最小突出量dB，正数，抑制平坦旁瓣的伪方向。
    mode为power_bearing（功率与测向）或power_only（只有点功率），不是自动从功率推出来向。
    bearing_error_deg为整次扫描的方位标定误差稳态标准差°，[0,45]。
    calibration_correlation_seconds为方位误差相关时间s，非负；0退回独立快照误差。
    同一主瓣内角峰按beamwidth_deg去重，是分辨能力近似，不是源数估计或置信区间。
    参数均未由实测标定；无测距能力或固定探测半径。
    """

    mode: str = "power_bearing"
    azimuth_bins: int = 24
    beamwidth_deg: float = 45.0
    front_back_db: float = 25.0
    detection_threshold_dbm: float = -94.0
    minimum_excess_db: float = 3.0
    minimum_contrast_db: float = 3.0
    noise_db: float = 1.0
    bearing_error_deg: float = 3.0
    calibration_correlation_seconds: float = 5.0


@dataclass(frozen=True)
class BeliefConfig:
    """干扰功率测量的局部外推；误差尺度是启发式，不是校准概率。

    source_max_fit_rmse_db是源假设窗口拟合RMSE上限dB，必须为有限正数。
    source_count_penalty_db2是每源复杂度惩罚dB²，必须有限且非负；均非物理安全阈值。
    source_bearing_weight_db2为归一化方向Huber误差的权重dB²，有限正数；默认9，不是概率。
    source_max_bearing_rmse_deg为方向残差均方根容限°，(0,180]，默认45；不是定位精度。
    """

    # 点观测可支持局部估计的最大距离；不是传感器能直接观测一个圆盘。
    support_radius_m: float = 40.0
    # 超过此年龄的测量不再支持当前估计，位置重新标记为未知。
    max_age_seconds: float = 35.0
    # 测量位置的误差下限及未知位置的误差尺度，dB。
    measured_error_db: float = 1.0
    unknown_error_db: float = 15.0
    # 外推距离和数据年龄引入的额外启发式误差。
    error_per_meter_db: float = 0.08
    error_per_second_db: float = 0.12
    source_max_fit_rmse_db: float = 8.0
    source_count_penalty_db2: float = 1.0
    source_bearing_weight_db2: float = 9.0
    source_max_bearing_rmse_deg: float = 45.0


@dataclass(frozen=True)
class ProjectConfig:
    """固定配置快照；环境脚本仅用于真值驱动和事后回放。"""

    seed: int = 7
    aircraft: AircraftConfig = field(default_factory=AircraftConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    map: MapConfig = field(default_factory=MapConfig)
    electromagnetic: ElectromagneticConfig = field(default_factory=ElectromagneticConfig)
    belief: BeliefConfig = field(default_factory=BeliefConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        for section in (self.aircraft, self.simulation, self.map, self.electromagnetic, self.belief, self.sensor):
            for name, value in asdict(section).items():
                if name in ("buildings", "provenance", "power_grid"):
                    continue
                if section is self.electromagnetic and name == "propagation_mode":
                    if value not in ("log_distance", "single_reflection"):
                        raise ValueError("propagation_mode must be log_distance or single_reflection")
                    continue
                if section is self.map and name == "geometry_mode":
                    if value not in ("planar", "fixed_altitude"):
                        raise ValueError("geometry_mode must be planar or fixed_altitude")
                    continue
                if section is self.sensor and name == "mode":
                    if value not in ("power_only", "power_bearing"):
                        raise ValueError("sensor mode must be power_only or power_bearing")
                    continue
                if name not in ("start_m", "goal_m", "obstacles_m", "fields") and (
                        isinstance(value, bool) or not isinstance(value, (int, float))):
                    raise ValueError(f"configuration field must be numeric: {name}")
                if isinstance(value, (int, float)) and (isinstance(value, bool) or not math.isfinite(value)):
                    raise ValueError(f"invalid numeric configuration: {name}")
        positive = (self.aircraft.speed_mps, self.aircraft.flight_power_w,
                    self.aircraft.hover_power_w, self.simulation.step_seconds,
                    self.simulation.review_seconds, self.simulation.minimum_review_seconds,
                    self.simulation.innovation_trigger_db, self.simulation.max_seconds,
                    self.simulation.probe_budget_seconds, self.map.resolution_m,
                    self.belief.support_radius_m, self.belief.max_age_seconds,
                    self.belief.measured_error_db, self.belief.unknown_error_db)
        positive += (self.belief.source_max_fit_rmse_db, self.belief.source_bearing_weight_db2,
                     self.belief.source_max_bearing_rmse_deg)
        positive += (self.aircraft.altitude_m, self.electromagnetic.carrier_frequency_hz,
                     self.electromagnetic.reference_distance_m, self.sensor.beamwidth_deg,
                     self.sensor.minimum_excess_db, self.simulation.warning_window_seconds,
                     self.simulation.warning_rise_db, self.sensor.minimum_contrast_db)
        if any(value <= 0 for value in positive):
            raise ValueError("durations, resolutions and positive physical parameters must be positive")
        nonnegative = (self.aircraft.clearance_m, self.aircraft.vertical_clearance_m, self.simulation.decision_delay_seconds,
                       self.electromagnetic.measurement_noise_db, self.belief.error_per_meter_db,
                       self.belief.error_per_second_db, self.belief.source_count_penalty_db2)
        nonnegative += (self.electromagnetic.obstacle_height_m, self.electromagnetic.obstacle_loss_db,
                        self.sensor.front_back_db, self.sensor.noise_db, self.sensor.bearing_error_deg,
                        self.sensor.calibration_correlation_seconds)
        if any(value < 0 for value in nonnegative):
            raise ValueError("margins, delays, noise and uncertainty growth must be nonnegative")
        if self.electromagnetic.interference_reference_dbm <= self.electromagnetic.background_dbm:
            raise ValueError("interference reference must exceed background power")
        if not 2 <= self.electromagnetic.path_loss_exponent <= 6:
            raise ValueError("path loss exponent must be in [2, 6]")
        if self.electromagnetic.relative_permittivity < 1 or self.electromagnetic.conductivity_s_per_m < 0:
            raise ValueError("relative permittivity must be >= 1 and conductivity must be nonnegative")
        if (self.electromagnetic.propagation_mode == "single_reflection"
                and self.electromagnetic.path_loss_exponent != 2):
            raise ValueError("single_reflection requires path_loss_exponent=2")
        if self.electromagnetic.reference_distance_m < 299792458 / self.electromagnetic.carrier_frequency_hz:
            raise ValueError("reference distance must be at least one wavelength")
        if self.sensor.beamwidth_deg > 180 or self.sensor.bearing_error_deg > 45:
            raise ValueError("invalid beamwidth or bearing error")
        if self.belief.source_max_bearing_rmse_deg > 180:
            raise ValueError("source bearing RMSE limit must not exceed 180 degrees")
        if not 0 <= self.aircraft.reserve_energy_wh < self.aircraft.initial_energy_wh:
            raise ValueError("energy must satisfy 0 <= reserve < initial")
        if self.simulation.minimum_review_seconds > self.simulation.review_seconds:
            raise ValueError("minimum review interval exceeds regular interval")
        for name, value, minimum in (
            ("seed", self.seed, 0), ("columns", self.map.columns, 2), ("rows", self.map.rows, 2),
            ("max_advisor_calls", self.simulation.max_advisor_calls, 0),
            ("source_refinement_rounds", self.simulation.source_refinement_rounds, 1),
            ("max_probes", self.simulation.max_probes, 0),
            ("astar_expansion_budget", self.simulation.astar_expansion_budget, 1),
            ("rrt_attempt_budget", self.simulation.rrt_attempt_budget, 1),
            ("azimuth_bins", self.sensor.azimuth_bins, 4),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.sensor.azimuth_bins > 360:
            raise ValueError("at most 360 azimuth beams")
        for point in (self.map.start_m, self.map.goal_m):
            if len(point) != 2 or any(isinstance(value, bool) or not math.isfinite(value) for value in point):
                raise ValueError("positions must contain two finite coordinates")
        for box in self.map.obstacles_m:
            if len(box) != 4 or not all(math.isfinite(value) for value in box) or box[0] >= box[2] or box[1] >= box[3]:
                raise ValueError("invalid obstacle rectangle")
        from .buildings import Building
        if not isinstance(self.map.buildings, (list, tuple)) or not isinstance(self.map.provenance, dict):
            raise ValueError("buildings must be a sequence and provenance must be an object")
        buildings = [Building.from_dict(item) for item in self.map.buildings]
        if len({item.id for item in buildings}) != len(buildings):
            raise ValueError("building ids must be unique")
        if self.electromagnetic.power_grid is not None:
            from .power_grid import validate_power_grid
            validate_power_grid(self.electromagnetic.power_grid, self)
        common = {"center_m", "start_seconds", "end_seconds", "velocity_mps"}
        for item in self.electromagnetic.fields:
            if not isinstance(item, dict) or set(item) not in (
                    common | {"sigma_m", "peak_dbm"}, common | {"eirp_dbm", "height_m"}):
                raise ValueError("invalid synthetic field schema")
            required = set(item)
            for name in ("center_m", "velocity_mps"):
                if len(item[name]) != 2 or not all(not isinstance(value, bool)
                        and isinstance(value, (int, float)) and math.isfinite(value) for value in item[name]):
                    raise ValueError("invalid field position or velocity")
            if not all(not isinstance(item[name], bool) and isinstance(item[name], (int, float))
                       and math.isfinite(item[name]) for name in required - {"center_m", "velocity_mps"}):
                raise ValueError("invalid field number")
            if (item.get("sigma_m", 1) <= 0 or item.get("height_m", 0) < 0
                    or not -150 <= item.get("eirp_dbm", item.get("peak_dbm")) <= 80
                    or item["start_seconds"] < 0 or item["end_seconds"] <= item["start_seconds"]):
                raise ValueError("invalid field scale or activity interval")


def load_config(path: str | Path) -> ProjectConfig:
    """严格读取独立配置，不悄悄忽略拼错的字段或重复JSON键。"""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate configuration key: {key}")
            result[key] = value
        return result

    value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)
    classes = {"aircraft": AircraftConfig, "simulation": SimulationConfig,
               "map": MapConfig, "electromagnetic": ElectromagneticConfig, "belief": BeliefConfig,
               "sensor": SensorConfig}
    if not isinstance(value, dict) or set(value) - {"seed", *classes}:
        raise ValueError("unknown project configuration field")
    if isinstance(value.get("electromagnetic"), dict):
        grid = value["electromagnetic"].get("power_grid")
        if isinstance(grid, dict) and "path" in grid:
            if set(grid) - {"path", "provenance"} or not isinstance(grid["path"], str):
                raise ValueError("external power grid requires path and optional provenance only")
            from .power_grid import load_power_grid
            import hashlib
            grid_path = Path(path).resolve().parent / grid["path"]
            resolved = load_power_grid(grid_path)
            provenance = {**resolved.get("provenance", {}), **grid.get("provenance", {}),
                          "input_file": grid_path.name, "input_sha256": hashlib.sha256(grid_path.read_bytes()).hexdigest(),
                          "quantity": "total_received_interference_power_dbm", "time_model": "static"}
            resolved["provenance"] = provenance
            value["electromagnetic"]["power_grid"] = resolved
    sections = {name: cls(**value.get(name, {})) for name, cls in classes.items()}
    config = ProjectConfig(seed=value.get("seed", 7), **sections)
    config.validate()
    return config
