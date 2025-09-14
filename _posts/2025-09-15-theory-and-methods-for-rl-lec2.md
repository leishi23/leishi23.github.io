---
layout: post
title:  "Theory and Methods for Reinforcement Learning — Lec.2"
date:   2025-09-14
desc: "Theory and Methods for Reinforcement Learning — Lec.2"
keywords: "Machine learning"
categories: [Machine learning]
tags: [Machine learning]
icon: icon-html
---

# Theory and Methods for Reinforcement Learning — Lec.2


> **_Keypoints:_**  
- Markov Chains  
- Markov Decision Processes (MDPs)  
- Policies  
- Performance Criteria  
- Value Functions  
- Occupancy Measure  
- Bellman Equations  
- Solving MDPs: VI & PI  
- From Planning to RL  
- Monte Carlo vs TD Learning  

---


## Markov Chains Refresher

### Definition: Markov Chain
- A time-homogeneous Markov chain is a stochastic process \(\{X_0, X_1, \dots\}\) on a countable state space satisfying the Markov property:

$$
\mathbb{P}(X_{t+1}=j\mid X_t=i, X_{t-1},\dots,X_0)
= \mathbb{P}(X_{t+1}=j\mid X_t=i)
= P_{ij}.
$$

### Markov Process (Tuple)
- \(\langle \mathcal{S}, P, \mu \rangle\):
  - **State set**: \(\mathcal{S}\)
  - **Transition model**: \(P_{ss'} = \mathbb{P}(s'\mid s): \mathcal{S} \to \Delta(\mathcal{S})\)
  - **Initial distribution**: \(s_0 \sim \mu \in \Delta(\mathcal{S})\)

### Stationary Distribution (Ergodic Case)
- If the chain is irreducible and aperiodic with finite states (ergodic), there exists a unique stationary distribution \(d^*\) and \(\{X_t\}\) converges to it:

$$
\lim_{t\to\infty} \big(P^t\big)_{ij} = d^*_j,\quad \forall i,j.
$$

- Vector form: with \(P\) the transition matrix and \(d^*\) a row vector,

$$
d^* = d^* P,\quad \text{i.e., } d^* \text{ is the left principal eigenvector of } P.
$$

> **Difference with Markov Property**: 虽然 Markov Property 只依赖当前状态，但如果你从不同的初始状态出发，短期内分布是不同的。但是在 遍历 (ergodic) 的条件下，随着 \(t \to \infty\)，这些由不同初始状态引起的差异会逐渐消失，最终所有状态的分布都会收敛到同一个 **stationary distribution**。

## Markov Decision Processes (MDPs)

### Controlled Markov Property
$$
\mathbb{P}(s_{t+1}=s'\mid s_t=s, a_t=a, \dots, s_0, a_0)
= \mathbb{P}(s_{t+1}=s'\mid s_t=s, a_t=a)
= P(s'\mid s,a).
$$

### Definition: MDP
- Tuple \((\mathcal{S}, \mathcal{A}, P, r, \mu, \gamma)\):
  - **States**: \(\mathcal{S}\)
  - **Actions**: \(\mathcal{A}\)
  - **Transition**: \(P(s'\mid s,a): \mathcal{S}\times \mathcal{A} \to \Delta(\mathcal{S})\)
  - **Reward**: \(r(s,a): \mathcal{S}\times \mathcal{A} \to \mathbb{R}\)
  - **Initial distribution**: \(s_0\sim \mu\in\Delta(\mathcal{S})\)
  - **Discount factor**: \(\gamma\in(0,1)\)

## Policies

### Definition: Policy
- A policy maps histories \(h_t=(s_0,a_0,\dots,s_{t-1},a_{t-1},s_t)\) to actions.
    - Deterministic policy:
      - Stationary policy: \( \pi:\mathcal{S}\to\mathcal{A}, \ a_t=\pi(s_t)\)
      - Markov policy: \( \pi_t:\mathcal{S}\to\mathcal{A}, \ a_t=\pi_t(s_t)\)
      - History-dependent policy: \( \pi_t:\mathcal{H}_t\to\mathcal{A}, \ a_t=\pi_t(h_t)\)
    - Stochastic policy:
      - Stationary policy: \( \pi:\mathcal{S}\to\Delta(\mathcal{A}), \ a_t\sim\pi(\cdot\mid s_t)\)
      - Markov policy: \( \pi_t:\mathcal{S}\to\Delta(\mathcal{A}), \ a_t\sim\pi_t(\cdot\mid s_t)\)
      - History-dependent policy: \( \pi_t:\mathcal{H}_t\to\Delta(\mathcal{A}), \ a_t\sim\pi_t(\cdot\mid h_t)\)

### Remarks
- Infinite-horizon discounted objectives admit an optimal stationary deterministic policy.
- Finite-horizon objectives generally require a (nonstationary) deterministic Markov policy.

## Performance Criteria

### Reminder: Objective Choices
- Finite horizon: cumulative reward; average reward.
- Infinite horizon: discounted reward; average reward.

### Discounted Return 
$$
J(\pi) 
= \mathbb{E}\Big[ \sum_{t=0}^{\infty} \gamma^{t}\, r(s_t, a_t) \;\Big|\; s_0\sim\mu,\; a_t\sim\pi(\cdot\mid s_t),\; s_{t+1}\sim P(\cdot\mid s_t,a_t) \Big].
$$

### Observations
- If \(\gamma=1\), total reward can diverge (e.g., cyclic processes).
- With \(\gamma\in(0,1)\) and bounded rewards (\(\lvert r\rvert<\infty\)), the return is finite.

## Value Functions Definitions

### State-Value Function
$$
V^{\pi}(s) 
:= \mathbb{E}\!\left[ \sum_{t=0}^{\infty} \gamma^t\, r(s_t,a_t) \;\middle|\; s_0=s,\, \pi \right].
$$

### Action-Value Function (Q-function)
$$
Q^{\pi}(s,a)
:= \mathbb{E}\!\left[ \sum_{t=0}^{\infty} \gamma^t\, r(s_t,a_t) \;\middle|\; s_0=s,\, a_0=a,\, \pi \right].
$$

## Relations between \(V^{\pi}\) and \(Q^{\pi}\)
- For any policy \(\pi: \mathcal{S}\to\Delta(\mathcal{A})\):

$$
Q^{\pi}(s,a) 
= r(s,a) + \gamma \sum_{s'\in\mathcal{S}} P(s'\mid s,a)\, V^{\pi}(s'),
\tag{1}
$$

$$
V^{\pi}(s) 
= \sum_{a\in\mathcal{A}} \pi(a\mid s)\, Q^{\pi}(s,a).
\tag{2}
$$



## Occupancy Measure Definition
For initial distribution \(\mu\) and policy \(\pi\), the (discounted) occupancy measure is
$$
\lambda^{\pi}_{\mu}(s,a)
= (1-\gamma) \sum_{t=0}^{\infty} \gamma^t\, \mathbb{P}[\, s_t=s,\ a_t=a\mid s_0\sim\mu,\, \pi\,].
$$

> Intuition: \(\lambda^{\pi}_{\mu}(s,a)\) represents the (normalized) discounted frequency of visiting state-action pair \((s,a)\) when starting from initial distribution \(\mu\) and following policy \(\pi\).

### Relation to Value
$$
V^{\pi}(\mu) = \frac{1}{1-\gamma} \sum_{s,a}  \lambda^{\pi}_{\mu}(s,a)\, r(s,a) = \frac{<\lambda^{\pi}_{\mu}, r>}{1-\gamma}.
$$

本质上, 衡量从状态 \(s \sim \mu\) 开始并遵循策略 \(\pi\) 时的预期累计奖励的\(V^{\pi}(s)\)就是每个状态-动作pair \((s,a)\) 的奖励 \(r(s,a)\) 按照其在轨迹中出现的频率（由occupancy \(\lambda^{\pi}_{\mu}(s,a)\) 给出）加权求和

## Optimal Value Functions
- Optimal state-value function:
$$
V^*(s) := \max_{\pi \in \Pi} V^{\pi}(s)
$$
- Optimal action-value function:
$$
Q^*(s,a) := \max_{\pi \in \Pi} Q^{\pi}(s,a)
$$
- \(V^*\) and \(Q^*\) satisfy the Bellman optimality equations:
$$
V^*(s) = \max_{a\in\mathcal{A}} Q^*(s,a) \\
Q^*(s,a) = r(s,a) + \gamma \sum_{s'\in\mathcal{S}} P(s'\mid s,a)\, V^*(s').
$$


## Solving MDPs: find the optimal policy
- **目标**：寻找最优策略 \(\pi^\star\) 使得
  \[
  V^{\pi^\star}(s)=V^\star(s):=\max_{\pi} V^\pi(s),\quad \forall s\in\mathcal S.
  \]
  注：最优策略可能不唯一，但 **\(V^\star\)** 唯一

## Bellman Optimality Conditions
  \[
  V^\star(s)=\max_{a\in\mathcal A}\Big[r(s,a)+\gamma\sum_{s'}P(s'|s,a)V^\star(s')\Big].
  \]

## Existence of an Optimal Policy
- Infinite horizon MDP 中，存在**deterministic and stationary**的最优策略 \(\pi^\star\)，使得
  \[
  V^{\pi^\star}(s)=V^\star(s),\quad Q^{\pi^\star}(s,a)=Q^\star(s,a).
  \]
  moreover：
  \[
  \pi^\star(s)=\arg\max_{a\in\mathcal A} Q^\star(s,a).
  \]


## Bellman Consistency
- 
  \[
  V^\pi(s)=\mathbb E_{a\sim\pi(\cdot|s)}\!\Big[r(s,a)+\gamma\sum_{s'}P(s'|s,a)V^\pi(s')\Big].
  \]
- 
  \[
  V_\pi=R_\pi+\gamma P_\pi V_\pi,
  \]
  where:  
  \(V_\pi\in\mathbb R^{n}\), \( (V_\pi)_s=V^\pi(s)\)；  
  \(R_\pi\in\mathbb R^{n}\), \( (R_\pi)_s=\sum_{a}\pi(a|s)r(s,a)\)；  
  \(P_\pi\in\mathbb R^{n\times n}\), \( (P_\pi)_{s,s'}=\sum_{a}\pi(a|s)P(s'|s,a)\)。


## Policy Evaluation（Close-form solution）
- 
  \[
  V_\pi=(I-\gamma P_\pi)^{-1}R_\pi.
  \]
Compt. cost is \(\mathcal O(|\mathcal S|^3+|\mathcal S|^2|\mathcal A|)\)。


## Bellman Expectation Operator and Fixed-point Persepective
- Let \(\Gamma^\pi:\mathbb R^{|\mathcal S|}\to\mathbb R^{|\mathcal S|}\),
  \[
  \Gamma^\pi V:=R_\pi+\gamma P_\pi V.
  \]
  - \(V_\pi\) 是 \(\Gamma^\pi\) 的 fixed point, i.e., \(V_\pi=\Gamma^\pi V_\pi\).
  - \(\Gamma^\pi\) is the iteration operator for policy evaluation.

## Bellman Optimality Equations & Operator
-  \[
  \begin{aligned}
  V^\star(s)&=\max_{a}\Big[r(s,a)+\gamma\sum_{s'}P(s'|s,a)V^\star(s')\Big],\\
  Q^\star(s,a)&=r(s,a)+\gamma\sum_{s'}P(s'|s,a)\max_{a'}Q^\star(s',a').
  \end{aligned}
  \]

- Bellman optimality operator \(\Gamma:\mathbb R^{|\mathcal S|}\to\mathbb R^{|\mathcal S|}\),
  \[
  (TV)(s)=\max_{a}\Big[r(s,a)+\gamma\sum_{s'}P(s'|s,a)V(s')\Big],
  \]
  \(\Gamma\)代表一种运算，也即算子,\(V^\star\) 是 \(\Gamma\) 的不动点

## Contraction of \(\Gamma\)
- for any \(V,V'\in\mathbb R^{|\mathcal S|}\),
\[
\|\Gamma V' - \Gamma V\|_\infty \le \gamma \|V' - V\|_\infty.
\]

## Solving MDP：Value Iteration 与 Policy Iteration

### Value Iteration (VI)
- 算法（Fiexed-point iteration）：
  1) 选取初值 `V_0`（如 `V_0(s)=0`）。
  2) 迭代更新：
     $$
     V_{t+1} = \Gamma\,V_t.
     $$
  3) 直到收敛或满足终止条件（如 \(\|V_{t+1}-V_t\|_\infty \le \varepsilon\)）。
- Find \(\pi^*\) from \(V^*\)：
  $$
  \pi^*(s) = \arg\max_{a\in\mathcal A} \Big[ r(s,a) + \gamma \sum_{s'} P(s'\mid s,a) V(s') \Big].
  $$
- Alternatively: 直接在 \(Q\)-空间迭代：
  $$
  Q_{t+1}(s,a) = r(s,a) + \gamma \sum_{s'\in\mathcal S} P(s'\mid s,a) \max_{a'\in\mathcal A} Q_t(s',a'),\quad
  \pi(s) \in \arg\max_{a} Q_t(s,a).
  $$
> Why we use \(Q\) here? Because \(Q^{\pi}(s,a)\) **directly** gives the value of taking action \(a\) in state \(s\).
- 
  $$
  Q^{\pi}(s,a) = r(s,a) + \gamma \sum_{s'} P(s'\mid s,a) V^{\pi}(s'),\tag{1}
  $$
  $$
  V^{\pi}(s) = \sum_{a\in\mathcal A} \pi(a\mid s)\, Q^{\pi}(s,a).\tag{2}
  $$
- 线性收敛：
  $$
  \|V_t - V^*\|_\infty \le \gamma^t\, \|V_0 - V^*\|_\infty.
  $$

> Value iteration first finds the optimal value function \(V^*\) (or \(Q^*\)), then derives the optimal policy \(\pi^*\) by greedy selection.

### Policy Improvement 
- Intuition: Start from an initial policy \(\pi\)，iterately perform:
  - Evaluate \(V^{\pi}\) (or \(Q^{\pi}\)) for current policy \(\pi\).
  - Improve policy \(\pi\) by making it greedy w.r.t. \(V^{\pi}\) (or \(Q^{\pi}\)).
- 若存在策略 \(\pi'\) 使得对所有 \(s\in\mathcal S\)，
  $$
  Q^{\pi}(s,\pi'(s)) \ge V^{\pi}(s),
  $$
  则 \(V^{\pi'}(s) \ge V^{\pi}(s)\) 对任意 \(s\) 成立；若 \(\pi\) 非最优，则不等式变为\(>\),严格成立
- 推论：可用“对 \(V^{\pi}\) 贪心”的方式改进策略：
  $$
  \pi_{t+1}(s) \in \arg\max_{a\in\mathcal A} Q^{\pi_t}(s,a).
  $$

### Policy Iteration (PI)
- 算法（直接在策略空间迭代）：
  1) 初始化策略 \(\pi_0\)。
  2) 对 \(t=0,1,2,\dots\) 重复：
     - 策略评估（求 \(V^{\pi_t}\)）：
       - 迭代法：\(V \leftarrow \Gamma^{\pi_t} V\) 直至收敛，其中
       - 闭式解：\(
         V^{\pi_t} = (I - \gamma P^{\pi_t})^{-1} R^{\pi_t}.
         \)
     - 策略提升（贪心更新）：
       $$
       \pi_{t+1}(s) \in \arg\max_{a\in\mathcal A} [ r(s,a) + \gamma \sum_{s'} P(s'\mid s,a) V^{\pi_t}(s') ].
       $$
- 线性收敛：
  $$
  \|V^{\pi_t} - V^*\|_\infty \le \gamma^t\, \|V^{\pi_0} - V^*\|_\infty.
  $$

### VI vs. PI

| Algorithm            | Value Update                                                                 | Policy Update   |
|----------------------|-------------------------------------------------------------------------------|-----------------|
| Value Iteration (VI) | \(V_{t+1} = \mathcal{T} V_t\)                                                 | None            |
| Policy Iteration (PI)| \(V_{t+1} = \mathbb{E}_{\pi_t}\!\left[ r(s,a) + \gamma \sum_{s' \in S} P(s' \mid s,a)\, V(s') \right]\) | Greedy Policy   |


| Algorithm            | Per iteration cost                        | Number of iterations                                                                 | Output                                      |
|----------------------|--------------------------------------------|---------------------------------------------------------------------------------------|---------------------------------------------|
| Value Iteration (VI) | \(\mathcal{O}(\|S\|^2 \|A\|)\)            | \(T = \mathcal{O}\left(\frac{\log(\epsilon(1-\gamma))}{\log \gamma}\right)\)          | \(V_T\) such that \(\|V_T - V^*\| \leq \epsilon\) |
| Policy Iteration (PI)| \(\mathcal{O}(\|S\|^3 + \|S\|^2 \|A\|)\) | \(T = \mathcal{O}\left(\frac{\|S\|(\|A\|-1)}{1-\gamma}\right)\)                       | \(V^*\) and \(\pi^*\)                        |

- PI converges in a finite number of iterations, whereas VI does not.  
- These solution mythologies are broadly known as model-based RL.  


## From Planning to RL

### Fundamental Challenges:
- VI and PI require knowledge of the transition model \(P\) and reward function \(r\).
  - Needs sampling approaches
- The computation and memory costs grow rapidly with the size of the state and action spaces (the "curse of dimensionality").
  - Needs new representations

### Categories of RL Algorithms:
- **Model-based RL**: Learn the model \(P\) and \(r\), then plan using VI or PI.
  - Experience \(\{(s,a,s',r)\}\) → Learn transition \(P(s'|s,a)\) and reward \(r(s,a)\) → Policy \(\pi(a|s)\)
- **Model-free RL**: Directly learn the policy \(\pi\) or value functions \(V^\pi\), \(Q^\pi\) from experience without explicitly modeling \(P\) and \(r\).
  - Experience \(\{(s,a,s',r)\}\) → Learn policy \(\pi(a|s)\) or value functions \(V^\pi(s)\), \(Q^\pi(s,a)\) → Action selection \(\pi(a|s)\)
- **On-policy vs. Off-policy**:
  - On-policy: Collect data by interacting with the environment. Exploritation-exploration trade-off.
  - Off-policy: Learn from a fixed dataset of experiences.

### Representation Learning:
- Use function approximators (e.g., neural networks) to represent policies and value functions.


### Mean Estimation

- 目标：给定样本序列 \(X_1, X_2, …, X_n\)，估计均值 \(\mu=\mathbb E[X]\)。
- Sampl Average）：
  $$
  \hat\mu_n = \frac{1}{n}\sum_{i=1}^n X_i. \\
  \hat\mu_n = \hat\mu_{n-1} + \frac{1}{n}\big(X_n - \hat\mu_{n-1}\big).
  $$
- Stochastic Approximation：
  $$
  \mu_{n+1} = \mu_n + \alpha_n\big(X_{n+1} - \mu_n\big),\quad n=0,1,2,\dots
  $$
  where \(\alpha_n\) is the step size.

### Model-free Prediction
- Goal: Under a given policy \(\pi\), estimate \(V^{\pi}(s)\) or \(Q^{\pi}(s,a)\) only from trajectories (episodes):
  $$
  V^{\pi}(s) := \mathbb{E}\Big[\sum_{t=0}^{\infty} \gamma^t\, r(s_t,a_t)\,\Big|\, s_0=s\Big].
  $$

### Monte Carlo (MC) Method
- Idea: Use the mean of returns \(G_t = r_t + \gamma r_{t+1} + \dots\) observed after visiting \(s\) to estimate \(V^{\pi}(s)\).
- Algorithm (pseudo-code):  
  1) For each episode \(\tau=\{s_0,a_0,r_0,s_1,\dots\}\) generated by \(\pi\):  
  2) For each state visit \(s_t\) in the episode:  
     - Compute return \(G_t = r_t + \gamma r_{t+1} + \cdots\)  
     - Update visit count \(n_{s_t} \leftarrow n_{s_t}+1\)  
     - Update value:
       $$
       V(s_t) \leftarrow V(s_t) + \frac{1}{n_{s_t}}\big(G_t - V(s_t)\big).
       $$
- Key points:  
  - No model required; estimates are independent across states; slow for long episodes.  
  - Convergence: if each state is visited infinitely often, then \(V \to V^{\pi}\).  

### Temporal Difference (TD) Learning
- From Bellman consistency:
  $$
  V^{\pi}(s) = \mathbb{E}_{a\sim\pi(\cdot|s)}\Big[r(s,a) + \gamma\, \mathbb{E}_{s'\sim P(\cdot|s,a)} V^{\pi}(s')\Big].
  $$
- Online incremental update:
  $$
  V(s_t) \leftarrow V(s_t) + \alpha_t \underbrace{(r_t + \gamma V(s_{t+1}) - V(s_t))}_{\text{TD error:=} \delta_t}
  $$
- Algorithm (pseudo-code):  
  1) For each step \(t\) in episode \(\tau=\{s_0,a_0,r_0,s_1,\dots\}\:  
     1) Observe \((s_t,a_t,r_t,s_{t+1})\) sampled by \(\pi\).  
     2) Update \(V\) by the above rule. 
- Uses one-step bootstrap, low variance but biased target.  
- Model-free; works with incomplete episodes; suitable for continuing tasks.  
- Convergence: if each state is visited infinitely often and \(\alpha_t\) satisfies Robbins–Monro conditions, then \(V \to V^{\pi}\).  

### Comparison: DP vs MC vs TD
- DP: no sampling, exploit Markov property. 
- MC: sampling, model-free, no Markov property. Unbiased, higher variance as it relies many random steps.
- TD: sampling, model-free, exploit Markov property, online. Biased, lower variance as it only relies on next step.

> MC error can be written as the sum of TD errors over the episode:
$$
\begin{aligned}
  G_t - V(s_t) &= r_{t} + \gamma G_{t+1} - V(s_t) \\
  &= r_{t} + \gamma G_{t+1} - V(s_t) + \gamma V(s_{t+1}) - \gamma V(s_{t+1}) \\
  &= r_{t} + \gamma V(s_{t+1}) - V(s_t) + \gamma (G_{t+1} - V(s_{t+1})) \\
  &= \delta_t + \gamma \delta_{t+1} + \gamma^2 (G_{t+2} - V(s_{t+2})) \\
  &= ... \\
  &= \sum_{k=t}^{T-1} \gamma^{k-t} \delta_k.
\end{aligned}
$$


### Multi-step TD: n-step Returns and Updates
- Definition (given episode termination time \(T\)):
  $$
  \begin{aligned}
  G_t^{(1)} &= r_{t+1} + \gamma V(s_{t+1}) \quad &\text{(TD(0) one-step)}\\
  G_t^{(2)} &= r_{t+1} + \gamma r_{t+2} + \gamma^2 V(s_{t+2})\\
  G_t^{(n)} &= r_{t+1} + \cdots + \gamma^{n-1} r_{t+n} + \gamma^n V(s_{t+n}),\\
  G_t^{(\infty)} &= r_{t+1} + \cdots + \gamma^{T-t-1} r_T \quad &\text{(MC, when } t+n\ge T)
  \end{aligned}
  $$
- n-step TD update:
  $$
  V(s_t) \leftarrow V(s_t) + \alpha_t \underbrace{\big(G_t^{(n)} - V(s_t)\big)}_{\text{n-step TD error}}.
  $$

### TD(λ) and Eligibility Traces
- λ-return (geometric average of all n-step returns):
  $$
  G_t^{\lambda} = (1-\lambda)\sum_{n=1}^{\infty} \lambda^{n-1}\, G_t^{(n)}.
  $$
- Update (forward view):
  $$
  V(s_t) \leftarrow V(s_t) + \alpha\,\big(G_t^{\lambda} - V(s_t)\big).
  $$
- Backward view (efficient implementation):  
  - TD error: \(\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)\); eligibility trace:
    $$
    e_t(s) = \gamma\lambda\, e_{t-1}(s) + \mathbf 1\{s_t=s\}.
    $$
  - Parameter update:
    $$
    V(s) \leftarrow V(s) + \alpha\, \delta_t\, e_t(s),\quad \forall s.
    $$
  - Properties: \(\lambda=0\) reduces to TD(0); \(\lambda=1\) approximates MC; intermediate \(\lambda\) often improves convergence.  

### SARSA (On-policy TD Control for Q-estimation)
- Goal: Estimate \(Q^{\pi}\) in policy evaluation/improvement based on interaction samples.  
- Update:
  $$
  Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha\, \Big[r_t + \gamma\, Q(s_{t+1}, a_{t+1}) - Q(s_t,a_t)\Big].
  $$
- Pseudo-code:  
  1) Observe \((s_t,a_t,r_t,s_{t+1},a_{t+1})\) sampled by current on-policy strategy.  
  2) Update \(Q\) by the above rule.  

### Summary: DP vs MC vs TD (Key Points)
- Model requirement: DP requires a model; MC/TD do not.  
- Bootstrap: DP/TD yes; MC no.  
- Update timing: DP/TD can update immediately; MC requires complete episodes.  
- Use of Markov property: DP/TD yes; MC no.  
- Bias/variance: MC is unbiased but high variance; TD is biased but low variance.  
