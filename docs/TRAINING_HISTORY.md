
# 固定出生点冠军(v3champ)的实验记录

> [English version](TRAINING_HISTORY_en.md)

由于后期误删了大量reward变化等重要数据和多个历史版本的智能体，信息有限。
严格成绩评估见[`LADDER.md`](LADDER.md)




> 模型文件:`models/auto/league_robust_v3_best_105.zip`(浏览器里叫 **v3champ / 固定出生点冠军**)
> 观测 105 维 · 动作 18 离散 · 规则 survival_v1 · 固定出生点(蓝左红右,对称)
> 本文记录从零到冠军的每一步:做了什么改动、提升了多少、用的什么 reward。

# 最终成绩(120 局/对手 × 4 个 seed-base)

| # | 对手 | 胜率 | 开火率 | 命中/局 | 击杀耗时 | 过 5 成? |
|---|---|---|---|---|---|---|
| 1 | laika(闪避) | **0.775** | 6.4% | 2.52 | 10.4s | ✅ |
| 2 | easy_laika | **0.775** | 3.0% | 1.93 | 28.9s | ✅ |
| 3 | stationary | **0.983** | 4.8% | 2.73 | 22.1s | ✅ |
| 4 |aggressivepro(莽撞) | **0.458** | 7.3% | 1.51 | 8.9s | 五五开 |

**均值 0.748 · 最差 0.458 · 3/4 稳过 5 成。** 三个闪避型 laika 稳赢;莽撞 pro 五五开(它是另一堵「鲁莽对穿」的墙,不是闪避墙)。

---

## 训练链路总览(每一步的改动与提升)

| # | 阶段 | 核心算法 | 关键改动 | 量化提升 |
|---|---|---|---|---|
| 0 | 环境 + 脚本对手 | — | moba1v1duel 场景、5 个脚本 bot、毒圈/回血/护盾 | 提供可学习的专家与对手 |
| 1 | **行为克隆 BC** | 监督学习(火力加权交叉熵) | 用脚本专家采集胜局 demo;**shell_decay 物理对齐** | **laika 14% → 53%**(原始最大瓶颈) |
| 2 | **all-scripts DAgger** | DAgger + maximin gate | 脚本对战矩阵 → 专家地图(每个对手配一个 ≥0.95 的蓝方专家) | maximin **→ 0.20**(4/5 脚本被打穿) |
| 3 | **自博弈 self-play** | 锚定 PPO + 冻结的自己进池 | obs 对称 → 蓝方策略直接当红方打 | **laika 0.30 → 0.64**(第一个能抬 laika 的方法) |
| 4 | **域随机化联赛 v2** | 锚定 PPO + 多准则 gate | paramBot 生态(9 预设 + 每局随机参数);426k 步 | held-out **0.284 → 0.414**;maximin 0.02 → 0.12;pro 0.38 → 0.53 |
| 5 | **域随机化联赛 v3** | + anchor-to-policy | 扩展 randomized 家族 + seed 雪崩混合;600k 步 | held-out **0.414 → 0.526**;maximin 0.07 → **0.20**(6 准则全过 = 冠军) |
| 6 | **观测手术** | 零填充迁移 | obs 101 → 105(加 4 维速度);parity 100% | 部署版 `_best_105`,固定出生点 laika ~0.77 |

下面逐阶段展开。

---

### 阶段 0 — 环境与脚本对手
搭引擎 `game_core.js`、`1v1duel` 对枪场景(毒圈逼近、呼吸式回血、5 发 2 倍武器、转向 -30%),以及 5 个脚本对手(由弱到强):
- **stationary**(不动靶) · **easy_laika**(慢闪避) · **laika**(防守闪避型狙击手) · **laika-aggressive**(激进) · **laika-aggressive-pro**(纪律性激进)。

这一步不训练,只是把「专家」和「对手」准备好。

### 阶段 1 — 行为克隆(BC):laika 14% → 53%
让脚本专家(laika-aggressive)当蓝方采集**胜局轨迹**(obs→action),用**火力加权交叉熵**做监督克隆(开火动作权重 ×4~16,避免学成只会乱走不开枪)。

> **本阶段最大突破不是网络,是物理:** 训练桥早期 `shell_decay`(炮弹随距离衰减)和浏览器不一致,导致克隆出来的策略在真实物理下打 laika 只有 **14%**。把 `shell_decay=True` 设为统一标准后,**同一个克隆直接 14% → 53%**。这是整个项目的第一个、也是收益最大的修复——印证了「先保证仿真一致,再谈算法」。

### 阶段 2 — all-scripts DAgger:打穿 4/5 脚本,撞上多任务墙
BC 是开环克隆,一旦偏离专家轨迹就崩(协变量偏移)。**DAgger** 让学生自己跑、专家在它走到的状态上打标签,反复迭代。

- 先跑**脚本对战矩阵**,发现**玩家 0(蓝方)结构性优势**:每个对手都存在一个胜率 ≥0.95 的蓝方专家(stationary←aggro、easy←pro、laika←aggro、aggro←pro、pro←aggro)。据此建**专家地图**,每个对手用它的克星当标注器。
- 用 **maximin gate**(只有当「最差对手」也变好时才提升 checkpoint)挑选,得到 `bc_dagger_allscripts_v3`。
- **结果:maximin 0.20** —— laika/easy/stationary/pro 都还行,但 **laika-aggressive 卡在 0.04**。

> **关键诊断:** 单独给 laika-aggressive 训一个**专才**,胜率能到 **0.88**(但会忘掉其它对手 = 灾难性遗忘)。所以 0.04 不是「打不过」,而是**单一身份盲策略的多任务冲突**:对 laika 要稳准狙、对 aggro 要躲让自爆,两种相反策略一个网络扛不住,会来回摇摆。

### 阶段 3 — 自博弈 self-play:laika 0.30 → 0.64
场景左右对称,所以**蓝方训出来的策略可以直接从红方视角(obs1)当对手打**。把**冻结的旧版自己**放进对手池,配 **maximin gate** 做锚定 PPO。

- DAgger 的标签是「矛盾的」(不同对手要求相反动作);RL 的**胜负奖励是无歧义的**,于是 RL 学到了 DAgger 学不到的**条件策略**。
- **laika 0.30 → 0.64**,稳健(多 seed)。这是第一个真正把 laika 抬起来的方法——之前纯 PPO 只会侵蚀 warm-start,纯 DAgger 会平台期。

### 阶段 4 — 域随机化联赛 v2:泛化 0.284 → 0.414
为了不过拟合那 5 个固定脚本,引入 **paramBot 生态**:9 个原型预设(rusher/kiter/sniper/charger/precision/spammer/counter/baiter/turtle)+ **每局随机抽参数**(域随机化)。对手池以 randomized 为主。

- gate 升级为**多准则**(不只看胜率):① 留出(held-out)对手的泛化、② 自伤率、③ 被对手捡战利品的脆弱性,外加 maximin 地板和固定脚本不退化。
- 426k 步:**held-out 泛化 0.284 → 0.414**;maximin 0.02 → 0.12(charger/spammer 被攻克);pro 0.38 → 0.53;laika 守住 0.90。

### 阶段 5 — 域随机化联赛 v3:冠军诞生(maximin 0.07 → 0.20)
在 v2 基础上三个改动:
1. **anchor-to-policy** —— CE 锚点不再拉向狭窄的 laika demo,而是拉向「热启动起点策略自己的 argmax」,抗漂移的同时不把策略拽回老套路。
2. **扩展 randomized 家族** —— 增加 spammer/baiter/wall-sniper 原型 + **seed 雪崩混合**(去相关,避免不同 env 撞同一批随机参数)。
3. 修复分配器(保证每个列出的对手 ≥1 个 env,不再把小权重对手悄悄归零)。

- 600k 步:**held-out 0.414 → 0.526**;maximin 0.07 → **0.20**;`v3_best` 在**全部 6 个准则**上提升。
- 这就是 **`league_robust_v3_best`** —— 冠军。

### 阶段 6 — 观测手术:101 → 105
后期给观测加了 **4 维速度**(改善预判瞄准),obs 从 101 升到 105。用 `surgery_obs101to105.py` 给冠军的输入层**插 4 列零权重**(parity 校验 100% 一致),得到部署版 **`league_robust_v3_best_105`**——也就是浏览器里的 v3champ。它在固定出生点对三个闪避 laika 稳定 ~0.77。

---

## reward 参数(`game_core.js` `REWARD_DEFAULTS`, 行 466–480)

冠军全程用的是**基础稀疏奖励**,所有「塑形项」都是 0.0(休眠),靠胜负 + 命中 + 自伤 + 时间惩罚把策略推出来:

```
// 终局
win:            +1.0     // 赢
loss:           -1.0     // 输
winBySelfHit:   +0.15    // 对手自爆而赢(奖励打折,避免坐等)
timeoutPenalty: -0.7     // 超时/平局判罚(逼迫主动解决)
draw:            0.0

// 战斗(每步)
hit:            +0.35    // 命中敌人
hitShield:      +0.18    // 命中带护盾的敌人(收益减半)
selfHit:        -0.16    // 跳弹打到自己
selfHitShield:  -0.12
selfDefeat:     -0.12    // 因自伤而死

// 资源 / 节奏
powerup:        +0.04    // 捡到战利品
timePenalty:    -0.001   // 每步轻微紧迫感
poisonHurt:      0.0     // (休眠)毒圈停留惩罚

// 塑形项 —— 全部 0.0(休眠,冠军未使用)
closeRangeHit:   0.0     // 近距对穿惩罚
cleanTrade:      0.0     // 不挨打的命中奖励
backwardPenalty: 0.0     // 倒车惩罚
approachCoef:    0.0     // 势函数式接近奖励
aimBonus:        0.0     // 对准奖励
proximityBonus:  0.0  (proximityRange 200)   // 贴脸压制
cornerCoef:      0.0  (cornerRange 120)       // 封堵逃跑角度
```

**PPO 侧额外两件套**(不在上表,是训练器逻辑):
- **CE-anchor(λ_ce)** —— 在 PPO 损失里加一项交叉熵,把策略拉向 BC/起点基座,抗灾难性遗忘。
- **critic warm-up** —— 先冻结 actor、只热身 critic 若干步,等价值函数稳了再放开策略,避免早期乱更新毁掉 warm-start。

> 设计取向:**奖励尽量稀疏**(胜负为主、命中为辅),把「怎么赢」留给 RL 自己探索;所有会改变最优解的塑形项(贴脸、封角、接近)都关掉——它们在后续「攻闪避墙」的实验里试过,会把策略带偏,故冠军不用。


## 参考文献
## References

**Imitation & behavior cloning**
- Pomerleau, D. A. (1988). *ALVINN: An Autonomous Land Vehicle in a Neural Network.* NIPS 1988. https://papers.nips.cc/paper/1988/hash/812b4ba287f5ee0bc9d43bbf5bbe87fb-Abstract.html
- Ross, S., Gordon, G. J., & Bagnell, J. A. (2011). *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (DAgger). AISTATS 2011. https://arxiv.org/abs/1011.0686

**Reinforcement learning algorithms**
- Mnih, V., et al. (2013). *Playing Atari with Deep Reinforcement Learning* (DQN). https://arxiv.org/abs/1312.5602
- Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning.* Nature 518:529–533. https://www.nature.com/articles/nature14236
- Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms* (PPO). https://arxiv.org/abs/1707.06347
- Schulman, J., et al. (2015). *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (GAE). https://arxiv.org/abs/1506.02438
- Raffin, A., et al. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations.* JMLR 22(268). https://jmlr.org/papers/v22/20-1364.html

**Multi-agent, self-play & league training**
- Lowe, R., et al. (2017). *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments* (MADDPG). https://arxiv.org/abs/1706.02275
- Heinrich, J., & Silver, D. (2016). *Deep Reinforcement Learning from Self-Play in Imperfect-Information Games* (NFSP). https://arxiv.org/abs/1603.01121
- Bansal, T., et al. (2017). *Emergent Complexity via Multi-Agent Competition.* https://arxiv.org/abs/1710.03748
- Vinyals, O., et al. (2019). *Grandmaster level in StarCraft II using multi-agent reinforcement learning* (AlphaStar). Nature 575:350–354. https://www.nature.com/articles/s41586-019-1724-z
- OpenAI, et al. (2019). *Dota 2 with Large Scale Deep Reinforcement Learning* (OpenAI Five; observation surgery). https://arxiv.org/abs/1912.06680
- Lanctot, M., et al. (2017). *A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning* (PSRO). https://arxiv.org/abs/1711.00832
- Timbers, F., et al. (2020). *Approximate Exploitability: Learning a Best Response.* https://arxiv.org/abs/2004.09677
- Zhang, R., et al. (2024). *A Survey on Self-Play Methods in Reinforcement Learning.* https://arxiv.org/abs/2408.01072

**Curriculum, generalization & emergent strategy**
- Florensa, C., et al. (2017). *Reverse Curriculum Generation for Reinforcement Learning.* CoRL 2017. https://arxiv.org/abs/1707.05300
- Florensa, C., et al. (2018). *Automatic Goal Generation for Reinforcement Learning Agents* (GoalGAN). https://arxiv.org/abs/1705.06366
- Kurach, K., et al. (2019). *Google Research Football: A Novel Reinforcement Learning Environment* (incl. the "Football Academy"). https://arxiv.org/abs/1907.11180
- Baker, B., et al. (2019). *Emergent Tool Use From Multi-Agent Autocurricula* (hide-and-seek). https://arxiv.org/abs/1909.07528
- Cobbe, K., et al. (2018). *Quantifying Generalization in Reinforcement Learning* (CoinRun). https://arxiv.org/abs/1812.02341

**Pursuit–evasion / predator–prey (the cornering hypothesis)**
- Janosov, M., Virágh, C., Vásárhelyi, G., & Vicsek, T. (2017). *Group chasing tactics: how to catch a faster prey.* New J. Phys. 19:053003. https://arxiv.org/abs/1701.00284
- de Souza, C., et al. (2020). *Decentralized Multi-Agent Pursuit using Deep Reinforcement Learning.* https://arxiv.org/abs/2010.08193  *(note: the repo's earlier notes mis-cited this id for the Janosov paper above)*
- Xu, S., & Dang, Z. (2025). *Emergent behaviors in multiagent pursuit evasion games within a bounded 2D grid world.* Sci. Rep. 15:29376. https://www.nature.com/articles/s41598-025-15057-x

**Tooling / multi-agent environments & continual learning**
- Terry, J. K., et al. (2021). *PettingZoo: Gym for Multi-Agent Reinforcement Learning.* https://arxiv.org/abs/2009.14471
- Raiman, J., et al. (2019). *Neural Network Surgery with Sets.* https://arxiv.org/abs/1912.06719
- Rolnick, D., et al. (2019). *Experience Replay for Continual Learning* (CLEAR). https://arxiv.org/abs/1811.11682

**Same-domain corroboration**
- Ackermann, T., Spang, M., & Gardi, H. A. (2025). *Reinforcement Learning Agent for a 2D Shooter Game.* https://arxiv.org/abs/2509.15042
