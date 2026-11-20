# M5独立决策会话公开合同

你负责三个固定公开状态各pilot/refine两次决策，总计6个请求。无新飞行/测量；目标是用有限源假设作业改善对剩余路线风险的任务决策。只能读取本public下contract.md、control.json、已发布requests及自己responses；可写session_ready.json、responses。不得浏览scenario、inputs、runs、reports、evaluation、predictions、replay、其他研究/目录、父会话、联网或额外传感。共享FS只是流程边界。各状态独立，前一状态不是下一状态新增观测。

请求含当前点测、GP辅助软点、原拟合/源池、逐世界路线及下一15米动作分歧、最近出生后预测误差。dBm越大越强，目标T+Q兼顾时间和沿路功率。GP软点不是独立证据；源是假设世界，不是真定位；权重是相对的，不证明绝对可靠。原候选评分只对原预测有效，新提议工具结果在选择时未知。不能用训练误差降低替代未来准确或任务增益。

pilot必须propose一项job：family取static/translate/strength_change；region取task_disk/left_half/right_half；max_bases为整数1或2。基于剔除末3条已知点的前缀拟合，无warm start，无几何细化。随后refine请求显示其对已知末3点的回顾诊断，并不是新传感。refine可keep或propose两项job，并可指定其中0或1项追加有界几何细化。full拟合只用最近12真实点和原full GP软点，warm start取原事件出生前源池。几何工具优化幅度、中心(每轴±40m)、尺度(÷/×√2，20–240m)，不改速度/变化率；原区域/目标/预算不变，失败保留基础拟合。

区域以对应拟合末点为中心160m圆域；left_half为当前位置朝goal的左侧(叉积非负)，right_half反之，均裁剪已知地图。动态字典只有四轴±3m/s、强度±0.5dB/s；是等效假设而非已知物理机制。每个世界最多两个基元，世界间不叠加。

三项均成功后替代原事件三新生模型，出生前池占0.75、新世界各0.25/3，出生后证据置0；不能改该规则。keep或任何无效/超时/失败保留原出生后池。随后同一数值规划器重评原候选并取最优，你不能手设参数/权重/路线或新增作业。

每次仅写一个JSON对象，恰有request_id、action、jobs、geometry_job、reason。pilot: action=propose，jobs恰1项，geometry_job=null；refine: propose时jobs恰2项、geometry_job=null/0/1，keep时jobs=[]且geometry_job=null。每job恰有family、region、max_bases。reason简短可审核，1–1000字符，无需展示隐含推理过程。request_id须匹配本轮，不能复制旧请求绑定。

会话应在首请求发布前就绪；收到请求先写匹配agent_id/request_id的started_monotonic回执，再读取大请求、作决定，保存唯一原始回复和ready。必须在同一活动轮连续等待并服务6次，不等父会话逐次唤醒。60秒启动、启动后180秒回复；不重试、不预读未发布状态，不读取数值对照或真值。完成6次或control=completed即结束。
