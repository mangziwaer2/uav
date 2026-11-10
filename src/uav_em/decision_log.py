"""把已记录的可见输入、原始响应与执行结果导出为中文文本，不参与决策。"""

from __future__ import annotations

import json


def render_decision_log(episode: dict) -> str:
    """仅展示决策记录，不读取评价配置、完整真值或回放地图。"""
    lines = [
        "无人机电磁干扰实验：决策日志",
        "本文件在仿真结束后生成，不是实时日志。",
        "numeric=无模型数值基线；mock=固定模拟响应；local=真实本地模型；budget_fallback=调用预算回退。",
        "codex_subagent=在线子agent决策LLM；offline_algorithm=离线设计的固定数值算法，无在线LLM。",
        "未知不等于低干扰；模型提议不等于实际执行；到达不代表模型有效或飞行安全。",
    ]

    def append_json(title, value):
        lines.extend([title, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)])

    for index, entry in enumerate(episode.get("decisions", []), start=1):
        context = entry.get("context", {})
        runtime = dict(entry.get("model_runtime", {}))
        model_input = runtime.pop("model_input_text", None)
        lines.extend([
            "", "=" * 72,
            f"决策 {index} | 仿真时间 {entry['time_seconds']:.3f} s",
            f"触发原因：{entry.get('trigger')} | 后端：{entry.get('backend')}",
            f"模型调用编号：{entry.get('advisor_call')} | 咨询实际耗时：{entry.get('advisor_seconds', 0):.3f} s",
        ])
        sensing_settings = context.get("directional_sensing", {})
        propagation = context.get("source_inference", {}).get("public_propagation_model")
        if propagation:
            mode = propagation.get("propagation_mode", "log_distance")
            lines.append("【电磁传播：" + ("自由空间距离衰减＋等效穿楼损耗＋单次墙／地面Fresnel反射；路径功率非相干叠加"
                         if mode == "single_reflection" else "对数距离＋等效穿楼损耗") + "】")
            lines.append("【测向提供来波角峰，不是源位置；反射来波可能不指向源，未知区域不因测向变成已测】")
        sensing = sensing_settings.get("recent_scans", [])
        if sensing_settings.get("available") is False:
            lines.append("【感知能力：仅当前位置功率测量，未配备测向阵列】")
        if sensing:
            latest_scan = sensing[-1]
            peaks = latest_scan["peaks_deg_dbm"]
            lines.append("【范围感知：接收远源方向，不是扫描周围地图】")
            lines.append(f"观测时刻：{latest_scan['time_seconds']:.3f} s；东为0°，逆时针；"
                         + ("，".join(f"方向{angle:.1f}° / 波束功率{power:.2f} dBm" for angle, power in peaks)
                            if peaks else "未检出超过灵敏度的角峰（不代表周围安全）"))
            lines.append("无直接距离或源坐标；峰数不等于源数。")
            if latest_scan.get("calibration_correlation_seconds", 0) > 0:
                lines.append("相邻扫描的方位误差相关；重复方向不是独立定位证据，扇区宽度不是置信区间。")
        if context.get("sensing_warning"):
            append_json("【提前复核依据：只使用已接收观测】", context["sensing_warning"])
        append_json("【输入：策略可见上下文；null表示未提供数值，不代表低干扰】", context)
        if model_input is not None:
            lines.extend(["【本地模型分词前实际输入文本，含聊天模板】", model_input])
        elif entry.get("backend") == "local":
            lines.append("【实际提示词未记录；不能将可见上下文冒充完整模型输入】")
        raw = entry.get("raw_response")
        if runtime.get("response_is_action_codebook_mapping"):
            if runtime.get("selection_mode") in ("reasoned_generate", "budgeted_reasoning_logits"):
                lines.extend(["【模型生成原文；思考可能未完成，不能冒充最终动作】", str(runtime.get("raw_model_response")),
                              "【最终动作字母】", str(runtime.get("final_action_letter"))])
                if runtime.get("action_readout_input_text") is not None:
                    lines.extend(["【动作读出的实际输入；包含模型思考及明确记录的控制token】",
                                  runtime["action_readout_input_text"],
                                  "【由控制器关闭思考】" + str(runtime.get("thinking_closed_by_controller")),
                                  "【插入的控制token，非模型生成】" + str(runtime.get("inserted_control_token_ids"))])
            else:
                lines.extend(["【模型原始动作字母；logits模式为合法选项中最高分字母，未生成解释】",
                              str(runtime.get("raw_model_response"))])
            lines.append("【动作码本映射后的规范指令；不是模型自由生成的JSON】")
        else:
            lines.append("【原始输出：未经修复】")
        if raw is None:
            lines.append("未取得原始响应；是否调用过模型须结合后端、调用编号和错误判断。")
        elif isinstance(raw, str):
            lines.append(raw)
        else:
            append_json("非文本响应：", raw)
        append_json("【解析后的原提议；旧记录缺失时为null】", entry.get("proposed_decision"))
        append_json("【解析后／覆盖后的决策；不一定是模型原始提议】", entry.get("decision"))
        append_json("【阶段计划状态；active不表示目标已实现或假设已证实】", {
            "update": entry.get("stage_plan_update"), "state": entry.get("stage_plan_after"),
        })
        lines.extend([
            "【实际执行与复核】",
            f"接受行动：{entry.get('accepted_action')} | 实际航路算法：{entry.get('accepted_planner')}",
            f"覆盖原因：{entry.get('override')} | 模型／解析错误：{entry.get('error')}",
            f"规划失败：{entry.get('planning_failure')}",
            f"接受航路的balanced代价：{entry.get('accepted_route_balanced_cost')}",
        ])
        for result in episode.get("plans", []):
            if result["time_seconds"] == entry["time_seconds"]:
                append_json("本次传统算子求解摘要：", {
                    key: result.get(key) for key in (
                        "purpose", "planner", "profile", "regions", "waypoint_m", "reason", "length_m", "cost",
                        "iterations", "iteration_unit", "elapsed_seconds",
                    )
                })
        if runtime:
            append_json("【模型运行信息】", runtime)
        trace = entry.get("source_refinement_trace", [])
        for round_index, consultation in enumerate(trace, start=1):
            lines.append(f"【源假设咨询子轮 {round_index}；调用编号 {consultation['advisor_call']}；无额外飞行或观测】")
            if round_index < len(trace):
                append_json("本子轮实际可见上下文：", consultation["context"])
                model_runtime = dict(consultation.get("model_runtime", {}))
                actual_input = model_runtime.pop("model_input_text", None)
                if actual_input is not None:
                    lines.extend(["本子轮实际模型输入：", actual_input])
                lines.extend(["本子轮原始输出：", consultation.get("raw_response") or "未取得响应"])
                append_json("本子轮运行信息：", model_runtime)
            append_json("源假设数值验证；拟合误差不是新点预测误差：", consultation.get("source_evaluation"))
            fit = (consultation.get("source_evaluation") or {}).get("candidate_fit")
            if fit and fit.get("bearing_fit", {}).get("scan_count", 0):
                bearing = fit["bearing_fit"]
                lines.append(f"候选源功率RMSE：{fit['rmse_db']:.2f} dB；方向RMSE：{bearing['rmse_deg']:.1f}°；"
                             f"含角峰扫描{bearing['scan_count']}次，相关时间块{bearing['block_count']}个（不是独立样本证明）。")
            lines.append(f"子轮错误：{consultation.get('error')}")
        if "source_model_after" in entry:
            append_json("【当前源模型及更新前的新点预测残差；不是已知真实源】", entry["source_model_after"])
            checks = entry["source_model_after"].get("new_point_checks", [])
            if any(check.get("source_bearing_check") is not None for check in checks):
                lines.append("新观测方向残差在接收新测量、更新假设之前计算；与本轮拟合分数分开，不是定位真值误差。")
    bearing_events = [event for event in episode.get("events", []) if event.get("source_bearing_check") is not None]
    if bearing_events:
        lines.append("\n【飞行期间新观测方向残差；按更新前源版本预测，不是定位真值误差】")
        for event in bearing_events:
            bearing = event["source_bearing_check"]
            lines.append(f"时刻{event['time_seconds']:.3f}s；源版本{event['source_revision']}；"
                         f"功率残差{event['source_residual_db']:.2f}dB；"
                         f"方向RMSE{bearing['mean_square_deg2'] ** 0.5:.1f}°。")
    append_json("\n【实验指标；暴露为事后评价，不是模型输入】", episode.get("metrics", {}))
    return "\n".join(lines) + "\n"
