"""模型只产生受限策略文本；模拟响应、真实模型与回退必须分别记账。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Protocol

import numpy as np

from .planning import PLANNERS, PROFILES, SamplingRegion
from .source_model import SourceHypothesis, parse_source_hypothesis


class Advisor(Protocol):
    def decide(self, context: dict) -> str:
        """返回原始JSON文本，不允许通过修改上下文改变仿真状态。"""
        ...


@dataclass(frozen=True)
class StagePlan:
    """模型的阶段目标与待验证假设；文字条件不作为自动执行程序。"""

    objective: str
    focus: str
    assumption: str
    revision_condition: str


@dataclass(frozen=True)
class Decision:
    """解析后的策略提议，不是已经通过数值复核的飞行指令。"""

    # 当前仅有continue／replan／probe；后期行动须先实现算子与验证，不能只改提示词。
    action: str
    # planner和profile限定为planning.py已有的规划器和软代价偏好。
    planner: str
    profile: str
    reason: str
    # 仅probe可用，必须匹配本次上下文提供的补测候选编号。
    target_id: str | None = None
    # 可选随机树采样盒，只影响搜索分布，不是安全区域或干扰源识别结果。
    regions: tuple[SamplingRegion, ...] = ()
    stage_plan: StagePlan | None = None
    source_hypothesis: SourceHypothesis | None = None


def parse_decision(response: str, bounds, probe_ids: set[str]) -> Decision:
    """拒绝未知字段、重复键、非有限数、越界区域和未经批准的算子。"""
    if not isinstance(response, str) or len(response) > 4096:
        raise ValueError("response must be JSON text of at most 4096 characters")

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate key: {key}")
            value[key] = item
        return value

    def invalid_constant(value):
        raise ValueError(f"nonfinite number: {value}")

    value = json.loads(response, object_pairs_hook=unique, parse_constant=invalid_constant)
    required = {"action", "planner", "profile", "reason"}
    if not isinstance(value, dict) or not required <= set(value) <= required | {
            "target_id", "regions", "stage_plan", "source_hypothesis"}:
        raise ValueError("decision requires action/planner/profile/reason and optional target_id/regions/stage_plan/source_hypothesis")
    if value["action"] not in ("continue", "replan", "probe"):
        raise ValueError("unsupported action")
    if value["planner"] not in PLANNERS or value["profile"] not in PROFILES:
        raise ValueError("unsupported planner or profile")
    if not isinstance(value["reason"], str) or len(value["reason"]) > 600:
        raise ValueError("reason must be short text")
    target = value.get("target_id")
    if value["action"] == "probe":
        if not isinstance(target, str) or target not in probe_ids:
            raise ValueError("probe must select a currently available target_id")
    elif "target_id" in value:
        raise ValueError("target_id is only valid for probe")
    proposals = value.get("regions", [])
    if not isinstance(proposals, list) or len(proposals) > 3:
        raise ValueError("at most three sampling regions are supported")
    if proposals and (value["planner"] != "rrt_star" or value["action"] == "continue"):
        raise ValueError("regions require an active rrt_star search")
    regions = []
    for item in proposals:
        if not isinstance(item, dict) or set(item) != {"low_m", "high_m", "weight"}:
            raise ValueError("invalid region schema")
        for key in ("low_m", "high_m"):
            if (not isinstance(item[key], list) or len(item[key]) != 2
                    or any(isinstance(number, bool) or not isinstance(number, (int, float))
                           or not math.isfinite(number) for number in item[key])):
                raise ValueError("region corners must contain two finite numbers")
        weight = item["weight"]
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError("region weight must be finite and positive")
        low, high = np.asarray(item["low_m"]), np.asarray(item["high_m"])
        if np.any(low < 0) or np.any(high > bounds) or np.any(high <= low):
            raise ValueError("sampling region outside map or empty")
        regions.append(SamplingRegion(tuple(float(number) for number in low),
                                      tuple(float(number) for number in high), float(weight)))
    if not math.isfinite(sum(region.weight for region in regions)):
        raise ValueError("total sampling weight must be finite")
    stage_plan = None
    if "stage_plan" in value:
        stage = value["stage_plan"]
        fields = {"objective", "focus", "assumption", "revision_condition"}
        if not isinstance(stage, dict) or set(stage) != fields:
            raise ValueError("stage_plan requires objective/focus/assumption/revision_condition")
        if any(not isinstance(stage[key], str) or not stage[key].strip() or len(stage[key]) > 160
               for key in fields):
            raise ValueError("stage_plan fields must be nonempty strings of at most 160 characters")
        if stage["focus"] not in ("goal", "south", "middle", "north"):
            raise ValueError("stage_plan focus must be goal or an available corridor id")
        stage_plan = StagePlan(**{key: stage[key].strip() for key in fields})
    source = parse_source_hypothesis(value["source_hypothesis"], bounds) if "source_hypothesis" in value else None
    return Decision(value["action"], value["planner"], value["profile"], value["reason"],
                    target, tuple(regions), stage_plan, source)


def build_prompt(context: dict) -> str:
    """明确当前能力与未知信息，既不要求生成代码，也不声称模型能保证安全。"""
    instruction = (
        "你辅助定高无人机在部分可观测的时变电磁干扰环境中优化航路，不研究与服务站的连接。"
        "干扰功率单位dBm，数值越大干扰越强，例如-60强于-90。超参考值仅增加软代价，不等于平台失效。"
        "只依据给定观测和估计；未知区域不是已确认安全区域。源位置只能作为待验证假设，"
        "不得当作已知事实，不得编造未来事件。"
        "只输出一个JSON对象，不要代码块或额外文字。必须有action、planner、profile、reason。"
        "action只能是continue、replan、probe；planner只能是astar、rrt_star；"
        "profile只能是balanced、avoid_interference、explore。reason简短。"
        "优先使用preferred_planner；astar仅为可选网格对照，rrt_star支持候选采样区域。"
        "probe必须额外给target_id，只能取probe_candidates中当前列出的id；非probe不得有target_id。"
        "仅rrt_star重新搜索时可额外给regions，最多3项，每项只有low_m、high_m、weight。"
        "其他情况不需要regions。首次没有航路时必须replan或probe。"
        "避免重复无效探测；探测是获取局部信息，不是得到整个地图。"
        "你负责阶段计划，传统规划器负责数值求解；不要输出代码或未实现的行动。"
        "action必须从本轮available_actions选择；补测完成或到期后不得continue，"
        "因为到补测点的路线已结束，必须replan或选择当前可用的新probe。"
        "请附stage_plan对象，只有objective、focus、assumption、revision_condition四个字段。"
        "objective说明当前阶段目标，focus只能为goal、south、middle、north；"
        "assumption说明待验证假设而非未观测事实，revision_condition说明什么观测或求解反馈会使你改计划。"
        "三个说明字段各不超过160字，尽量简短。stage_plan只是记忆，不自动执行未来动作或改变复核时机。"
        "先检查execution_feedback和上一stage_plan：区分原提议、实际接受行动、补测结果与数值回退；"
        "再结合走廊摘要、候选点approach_route和goal_distance_reduction_m决定继续、补测或重规划。"
        "上一计划若suspended，先检查导致暂停的错误或实际执行差异，不得声称该计划已成功完成。"
        "求解失败不证明走廊不存在，低干扰点也不证明整条未知走廊安全。"
        "source_inference含带时间戳点测量、源假设、功率与方向拟合误差、新观测预测残差，均不是隐藏真值。"
        "directional_sensing仅在available为true时提供阵列接收的远源方向线索；没有阵列不能从点功率得到方向。"
        "方位误差可能跨轮相关，连续相似角峰不等于独立验证；波束宽度不是置信区间。东方0度、逆时针，"
        "每行peaks_deg_dbm为方位与波束接收功率；不是远处位置功率或已知源坐标，峰数不是源数。"
        "结合多位置时序、sensing_warning的增强趋势及当前航路判断是否提前补测或绕行；"
        "没有检测不表示前方安全。有角峰或持续增强时，先判断现有源假设能否解释证据，"
        "能提出可检验解释则提交候选源；证据不足可不提交，须在reason或assumption中说明歧义及补测意图。"
        "可额外提交source_hypothesis，只有reference_seconds和sources两个字段；"
        "reference_seconds必须等于本轮time_seconds。sources可为空，至多3项，每项只有"
        "center_m、eirp_dbm、height_m、velocity_mps，分别为推测位置m、频带内辐射功率dBm、源高m、速度m/s，"
        "具体限制见source_inference.limits。背景已由先验固定，不输出背景或源出现消失脚本。"
        "数值层用public_propagation_model中的公开距离损耗和遮挡近似计算候选源的接收功率，毫瓦叠加；"
        "这些是建模假设，不是任何真实源参数。eirp_dbm不是当前位置接收功率。"
        "单源格式示意（不是位置建议，不可照抄数值）："
        '{"center_m":[200,200],"eirp_dbm":0,"height_m":1.5,"velocity_mps":[0,0]}。'
        "sigma_m/peak_dbm仅用于显式高斯合成基线，不与传播源字段混写，同一假设不混用两类源。"
        "中心(t)=center_m+velocity_mps*(t-reference_seconds)，每条测量须用其自身时间。"
        "reason和阶段假设应对应当前测量及未知情况，不要反复套用泛化理由。"
        "不能用一个很大的固定源强行解释跨时间变化；比较移动、位置和源数量的解释，"
        "数据不足时说明歧义并考虑现有probe候选，不能声称已定位真实发射机。"
        "source_refinement反馈同时检验已有功率与检测方向，rejected_bad_bearing表示方向不一致；"
        "新观测source_bearing_check在更新假设前计算。方向检验用候选源合成角峰，不知道真实源ID，"
        "相关扫描按时间块汇总。拒绝后可修正位置、功率、运动或源数，但拟合好不等于真实定位或未来预测正确。"
        "每次仍须给出一个完整行动；同一决策点迭代期间没有新观测或飞行，消耗真实调用预算。"
        "若不需要修改源假设可省略该字段，保留既有假设；不要把未观测区域宣称为安全。"
        "例：{\"action\":\"replan\",\"planner\":\"rrt_star\",\"profile\":\"avoid_interference\","
        "\"reason\":\"比较替代航路\",\"stage_plan\":{\"objective\":\"向目标搜索替代航路\","
        "\"focus\":\"goal\",\"assumption\":\"未知部分尚待观测\",\"revision_condition\":\"新测量偏离估计或求解失败时复核\"}}\n"
    )
    if context.get("probe_candidates"):
        example = {"action": "probe", "planner": context.get("preferred_planner", "rrt_star"), "profile": "balanced",
                   "target_id": context["probe_candidates"][0]["id"], "reason": "在预算内补充局部观测"}
        instruction += "补测格式例子（是否采用由你判断）：" + json.dumps(example, ensure_ascii=False) + "\n"
    else:
        instruction += "当前无补测候选，禁止action=probe。\n"
    instruction += "生成前检查：如果action=probe，target_id绝对不可省略，必须逐字使用当前候选id。\n上下文：\n"
    return instruction + json.dumps(context, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class MockAdvisor:
    """确定性工程夹具，不是语言模型，不用于宣称模型优化收益。"""

    def __init__(self):
        self.calls = 0

    def decide(self, context: dict) -> str:
        self.calls += 1
        value = {"action": "replan", "planner": context.get("preferred_planner", "rrt_star"),
                 "profile": "avoid_interference",
                 "reason": "固定工程响应：按当前估计降低干扰代价，不代表模型能力"}
        if self.calls == 2 and context["probe_candidates"]:
            value.update(action="probe", target_id=context["probe_candidates"][0]["id"])
        elif context["active_probe"] is not None:
            value.update(action="continue")
        previous_stage = context.get("stage_plan")
        value["stage_plan"] = (previous_stage["proposal"] if value["action"] == "continue" and previous_stage else {
            "objective": "固定响应：补充局部观测" if value["action"] == "probe" else "固定响应：向目标重新搜索",
            "focus": context["probe_candidates"][0].get("corridor_id", "goal") if value["action"] == "probe" else "goal",
            "assumption": "工程夹具，不代表模型推理；未知区域仍未确认",
            "revision_condition": "新测量、补测结束或求解失败后复核",
        })
        return json.dumps(value, ensure_ascii=False)


@dataclass(frozen=True)
class LocalModelConfig:
    """仅加载已有本地模型，不下载权重、不执行远端代码。

    generation_seconds为可选生成软时限，单位s；None不限时，否则须有限且大于0。
    不限时仍受max_new_tokens限制，不保证输出完整或符合协议。
    """

    model_path: str
    # auto优先CUDA，显存不足时应显式调整，不自动改用其他模型。
    device: str = "auto"
    # 输入超限直接报错，不静默截断任务约束。
    max_input_tokens: int = 8000
    max_new_tokens: int = 1024
    generation_seconds: float | None = None

    def __post_init__(self):
        if not isinstance(self.model_path, str) or not self.model_path.strip():
            raise ValueError("model_path must be a local directory")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("invalid model device")
        if any(type(value) is not int or value <= 0 for value in (self.max_input_tokens, self.max_new_tokens)):
            raise ValueError("token budgets must be positive integers")
        if self.generation_seconds is not None and (
                isinstance(self.generation_seconds, bool) or not math.isfinite(self.generation_seconds)
                or self.generation_seconds <= 0):
            raise ValueError("generation time budget must be None or finite and positive")


class LocalModelAdvisor:
    """复用本地Qwen类因果语言模型；保留原始输出、词元统计与失败记录。"""

    def __init__(self, config: LocalModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.torch = None
        self.device = None
        self.load_seconds = 0.0
        self.last_call: dict = {}

    def load(self) -> None:
        if self.model is not None:
            return
        directory = Path(self.config.model_path).resolve()
        if not (directory / "config.json").is_file():
            raise ValueError(f"incomplete local model directory: {directory}")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exception:
            raise RuntimeError("local inference requires the toolkit environment or requirements-llm.txt") from exception
        device = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            str(directory), local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        model.eval()
        self.model, self.tokenizer, self.torch, self.device = model, tokenizer, torch, device
        self.load_seconds = time.perf_counter() - started

    def decide(self, context: dict) -> str:
        self.last_call = {"backend": "local", "status": "started"}
        started = time.perf_counter()
        try:
            self.load()
            messages = [{"role": "user", "content": build_prompt(context)}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            self.last_call["model_input_text"] = text
            inputs = self.tokenizer(text, return_tensors="pt")
            input_tokens = int(inputs["input_ids"].shape[-1])
            self.last_call["input_tokens"] = input_tokens
            limit = getattr(self.model.config, "max_position_embeddings", None)
            if (input_tokens > self.config.max_input_tokens
                    or limit is not None and input_tokens + self.config.max_new_tokens > limit):
                raise ValueError("model input exceeds token budget; context was not truncated")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.inference_mode():
                output = self.model.generate(
                    **inputs, max_new_tokens=self.config.max_new_tokens,
                    max_time=self.config.generation_seconds, do_sample=False,
                    pad_token_id=(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                                  else self.tokenizer.eos_token_id),
                )
            generated = output[0, input_tokens:]
            self.last_call.update(status="generated", output_tokens=int(generated.shape[-1]),
                                  hit_token_limit=int(generated.shape[-1]) >= self.config.max_new_tokens)
            return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        except Exception as exception:
            self.last_call.update(status="error", error=type(exception).__name__)
            raise
        finally:
            self.last_call["elapsed_seconds"] = time.perf_counter() - started
