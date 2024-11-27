### 1 背景
在能源系统中，面对多样化的能源需求（如电力、热力、冷需求）和不确定的可再生能源供应（如风电、光伏），如何实现能源的高效调度成为一个重要问题。多能系统的调度优化需要在以下目标之间进行权衡：

- 满足能源需求：保证电力、热力和冷需求的实时供应。
- 优化储能设备使用：合理利用储能设备（如电池）进行充放电。
- 碳捕集与成本控制：在满足低碳目标的同时，降低碳捕集设备的运行成本。
- 传统方法往往基于线性规划或动态规划，难以应对系统的非线性、不确定性和高维特性。强化学习（Reinforcement Learning, RL） 提供了一种新思路，通过智能体与环境的交互，不断优化策略，动态适应复杂的能源调度场景。
模型框架

### 2.1 状态空间
状态空间用于描述当前多能系统的运行状态，本文设计了 6 个状态变量：

- 电力需求（Power Demand）
- 热需求（Heat Demand）
- 冷需求（Cooling Demand）
- 储能状态（SOC, State of Charge）
- 风电输出（Wind Power）
- 光伏输出（PV Power）
这些状态变量的取值范围被归一化到 [0,1]

### 2.2 动作空间
动作空间表示智能体可以采取的控制动作，包括：
- 储能充放电功率（Storage Power）：取值范围 [−1,1]，正值表示充电，负值表示放电。
- 碳捕集设备加载率（Carbon Capture Rate）：取值范围 [0,1]，表示碳捕集设备的运行负荷率。

### 2.3 奖励函数
奖励函数用于评估每一步智能体动作的优劣，包含两个部分：

- 能量平衡奖励
- 碳捕集成本奖励
- 总奖励函数
### 2.4 学习目标
智能体通过 SAC 算法学习一个最优策略 (\pi(a|s))，以最大化累积奖励

DRL_test.html下载在浏览器打开，可以打印为pdf
