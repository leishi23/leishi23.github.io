---
layout: post
title:  "HF Deep RL Course — Unit 2 (Q-Learning)"
date:   2026-08-14
desc: "HF Deep RL Course — Unit 2 (Q-Learning)"
keywords: "Machine learning"
categories: [Machine learning]
tags: [Machine learning]
icon: icon-html
---

# HF Deep RL Course — Unit 2 (Q-Learning)

> Source: [Hugging Face Deep RL Course, Unit 2](https://huggingface.co/learn/deep-rl-course/unit2/introduction) — notes from *Introduction* through *Additional Readings*, reorganized for review. Continues from [Unit 1]({{ '/machine learning/2026/08/13/HF_DeepRL_Unit1.html' | replace: ' ', '%20' | prepend: site.baseurl }}).

> **_Keypoints:_**

- Recap: policy-based vs value-based
- Two types of value functions: $V(s)$ vs $Q(s,a)$
- Why we still have a policy (Greedy / Epsilon-Greedy)
- The Bellman equation (recursive value estimation)
- Monte Carlo vs Temporal Difference (TD)
- Q-Learning: the algorithm, Q-table, update rule
- Worked numeric example (2 timesteps)
- Off-policy vs On-policy (Q-Learning vs Sarsa)
- Glossary + self-check + additional readings

---

## 0. What this unit is for

Unit 1 covered the RL framework broadly. This unit **dives into value-based methods** and studies the **first real RL algorithm: Q-Learning** — implemented from scratch and trained on:

1. **Frozen-Lake-v1** (non-slippery) — go from start **(S)** to goal **(G)**, walking only on frozen tiles **(F)**, avoiding holes **(H)**.
2. **An autonomous taxi** — learn to navigate a city and transport passengers from A to B.

> This unit is **fundamental for Deep Q-Learning** — the first Deep RL algorithm that played Atari and beat human level on some games (Breakout, Space Invaders, …).

---

## 1. Short recap

The agent learns by **interacting through trial and error**, receiving rewards as its **only feedback**, with the goal of **maximizing expected cumulative reward** (the reward hypothesis).

The agent's decision-making process is the **policy $\pi$**: given a state, it outputs an action (or a distribution over actions). The goal is the **optimal policy $\pi^{*}$**.

Two families of methods:

| | **Policy-based** | **Value-based** |
|---|---|---|
| What is trained | The **policy directly** | A **value function** |
| What it learns | Which **action** to take given a state | Which **state (or state-action) is more valuable** |
| Policy | Learned — behavior defined **by training** | **Not** learned — we must **specify it by hand** (e.g. Greedy) |
| Optimality via | Finding $\pi^{*}$ directly | Finding $Q^{*}$ or $V^{*}$ ⇒ which **gives** $\pi^{*}$ |

> **Key insight:** whichever method you use, **you always have a policy**. In value-based methods you just don't *train* it — it's a simple pre-specified function (e.g. Greedy) that reads the value function to pick actions.

---

## 2. Two Types of Value-Based Methods

The value of a state is the **expected discounted return** if the agent **starts at that state and then acts according to the policy**.

### The state-value function
For each state, outputs the expected return if the agent **starts at that state** and follows the policy forever after:

$$
V_{\pi}(s) = \mathbb{E}_{\pi}\!\left[\, G_t \mid S_t = s \,\right]
$$

### The action-value function (the Q function)
For each **state–action pair**, outputs the expected return if the agent **starts in that state, takes that action**, and then follows the policy forever after:

$$
Q_{\pi}(s,a) = \mathbb{E}_{\pi}\!\left[\, G_t \mid S_t = s,\; A_t = a \,\right]
$$

### The difference

| | **State-value $V(s)$** | **Action-value $Q(s,a)$** |
|---|---|---|
| Computes value of | a **state** $S_t$ | a **state–action pair** $(S_t, A_t)$ |
| Answers | "How good is it to *be* here?" | "How good is it to take *this action* here?" |
| Table size | one entry per state | one entry per (state, action) |

In **either case the returned value is the expected return**.

> ⚠️ **The problem:** to compute *each* value we'd have to **sum all rewards** an agent can get starting from that state — computationally expensive. **That's where the Bellman equation comes in.**

### Deriving the policy: Greedy & Epsilon-Greedy
Since the policy isn't trained, we specify it:
- **Greedy policy** — always take the action with the **highest** value (pure exploitation):

$$
\pi(s) = \arg\max_{a} Q(s,a)
$$

- **Epsilon-Greedy policy** — what's actually used in practice, because it handles the **exploration/exploitation trade-off** (see §5).

---

## 3. The Bellman Equation

**Problem it solves:** computing $V(S_t)$ naively means summing the whole future reward sequence — and we **re-do that repeated computation** for every state.

**The idea (recursive, like dynamic programming):** the value of any state is

$$
\boxed{\,V_{\pi}(s) = \mathbb{E}_{\pi}\!\left[\, R_{t+1} + \gamma\, V_{\pi}(S_{t+1}) \mid S_t = s \,\right]\,}
$$

i.e. **the immediate reward $R_{t+1}$ + the discounted value of the state that follows $\gamma V(S_{t+1})$.**

Unrolling it:
- $V(S_t) = R_{t+1} + \gamma V(S_{t+1})$
- $V(S_{t+1}) = R_{t+2} + \gamma V(S_{t+2})$
- … and so on.

> **To recap:** instead of computing each value as the **sum of the expected return** (a long process), we compute it as **immediate reward + discounted value of the next state**.

**Think about the role of $\gamma$** (the course leaves this as an exercise):

| $\gamma$ | Effect |
|---|---|
| $\gamma = 0$ | Value = immediate reward only — completely **myopic** |
| $\gamma$ low (e.g. 0.1) | Future barely matters; very short-sighted |
| $\gamma = 1$ | No discounting — all future rewards weigh equally (may not converge in continuing tasks) |
| $\gamma \gg 1$ (e.g. a million) | Values **explode / diverge** — this is why we require $\gamma \in [0,1]$ |

---

## 4. Monte Carlo vs Temporal Difference Learning

Both are **strategies for training** the value/policy function from experience. The difference is **when** they learn:

- **Monte Carlo** uses **an entire episode** of experience before learning.
- **Temporal Difference** uses **only one step** $(S_t, A_t, R_{t+1}, S_{t+1})$ to learn.

### Monte Carlo: learning at the end of the episode
Wait until the episode ends, compute the actual return $G_t$, and use it as the **target**:

$$
V(S_t) \leftarrow V(S_t) + \alpha \left[\, G_t - V(S_t) \,\right]
$$

**Worked example** (from the course): $V$ initialized to 0 everywhere, learning rate $\alpha = 0.1$, discount $\gamma = 1$ (no discount). The mouse explores randomly, and the episode ends after >10 steps. The reward sequence gives:

$$
G_0 = R_1 + R_2 + R_3 + \dots = 1 + 0 + 0 + 0 + 0 + 0 + 1 + 1 + 0 + 0 = 3
$$

$$
V(S_0) \leftarrow V(S_0) + \alpha\,[\,G_0 - V(S_0)\,] = 0 + 0.1 \times [\,3 - 0\,] = \mathbf{0.3}
$$

Then start a new episode with this new knowledge. Running more and more episodes, **the agent plays better and better**.

### TD Learning: learning at each step
TD waits for only **one interaction** to form a **TD target** and update immediately. Since we haven't seen the whole episode, we don't have $G_t$ — so we **estimate** it with $R_{t+1} + \gamma V(S_{t+1})$:

$$
V(S_t) \leftarrow V(S_t) + \alpha \Big[\, \underbrace{R_{t+1} + \gamma V(S_{t+1})}_{\text{TD target}} - V(S_t) \,\Big]
$$

The bracketed quantity is the **TD error**. This is called **bootstrapping**, because the update is based **in part on an existing estimate $V(S_{t+1})$** rather than a complete sample $G_t$. This variant is **TD(0)** or **one-step TD**.

**Worked example:** $V$ initialized to 0, $\alpha = 0.1$, $\gamma = 1$. The mouse goes left and eats cheese, so $R_{t+1} = 1$:

$$
V(S_0) \leftarrow V(S_0) + \alpha\,[\,R_1 + \gamma V(S_1) - V(S_0)\,] = 0 + 0.1 \times [\,1 + 1 \times 0 - 0\,] = \mathbf{0.1}
$$

### Side-by-side

| | **Monte Carlo** | **TD Learning** |
|---|---|---|
| Updates | At the **end of the episode** | At **each step** |
| Target | **Actual** discounted return $G_t$ | **Estimated** return (TD target) $R_{t+1} + \gamma V(S_{t+1})$ |
| Needs a complete episode? | **Yes** | **No** |
| Bootstrapping? | No | **Yes** |
| Bias / Variance | Unbiased, **higher variance** | Biased, **lower variance** |
| Works on continuing tasks? | No (needs termination) | Yes |

> **Summary:** MC uses the **actual accurate discounted return** of the episode; TD replaces the unknown $G_t$ with an **estimated return, the TD target**.

---

## 5. Q-Learning

### What is Q-Learning?
> Q-Learning is an **off-policy, value-based method that uses a TD approach to train its action-value function**.

- **Off-policy** — different policies for acting and updating (see §7).
- **Value-based** — finds the optimal policy *indirectly* by training an action-value function.
- **TD approach** — updates at **each step**, not at the end of the episode.

**Q-Learning is the algorithm we use to train the Q-function** — an action-value function giving the value of being at a state and taking a specific action there.

The **Q comes from "the Quality"** (the value) of that action at that state.

**Value vs reward — don't confuse them:**
- The **value** of a state / state-action pair = **expected cumulative reward** if the agent starts there and then follows its policy.
- The **reward** = the **immediate feedback** from the environment after an action.

### The Q-table
Internally the Q-function is encoded by a **Q-table** — a table where **each cell is a state-action pair value**. Think of it as the **memory / cheat sheet** of the Q-function.

In the maze example, the state is only the mouse's position, so a $2\times3$ grid gives **6 rows**, one per position, with one column per action. Initialized to **0** everywhere.

So: **given a state and an action, the Q-function looks inside the Q-table to output the value.**

The chain that makes it all work:

$$
\text{optimal } Q\text{-function } Q^{*} \;\Longrightarrow\; \text{optimal Q-table} \;\Longrightarrow\; \text{optimal policy } \pi^{*}
$$

because if we know $Q^{*}$, we **know the best action to take at each state**. At the start the Q-table is useless (arbitrary/zero values), but as the agent **explores and we update it, it approximates the optimal policy better and better**.

### The Q-Learning algorithm

**Step 1 — Initialize the Q-table.** One entry per state-action pair, **usually 0**.

**Step 2 — Choose an action using the epsilon-greedy strategy.** With initial $\varepsilon = 1.0$:

$$
a_t =
\begin{cases}
\text{random action} & \text{with probability } \varepsilon \quad (\textbf{exploration})\\[4pt]
\arg\max_{a} Q(S_t, a) & \text{with probability } 1-\varepsilon \quad (\textbf{exploitation})
\end{cases}
$$

Early in training $\varepsilon$ is high, so **we mostly explore**. As the Q-table's estimates improve, we **progressively decay $\varepsilon$** — less exploration, more exploitation.

**Step 3 — Perform $A_t$, observe $R_{t+1}$ and $S_{t+1}$.**

**Step 4 — Update $Q(S_t, A_t)$.** Using the TD target (bootstrap: immediate reward + discounted value of the next state, taking the action that **maximizes** the current Q at the next state):

$$
\boxed{\,Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \Big[\, R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t) \,\Big]\,}
$$

To do this update we need only $S_t, A_t, R_{t+1}, S_{t+1}$.

> **The crucial subtlety:** the $\max_a$ inside the target uses a **greedy** policy — *not* epsilon-greedy. It always takes the highest-valued action. But when we then **act** in the new state, we use **epsilon-greedy again**. **That is exactly why Q-Learning is off-policy.**

---

## 6. Worked Q-Learning Example (2 timesteps)

**Setup.** You're a mouse in a tiny maze, always starting at the same point. Goal: **eat the big pile of cheese** (bottom-right) and avoid the poison.

- Episode ends if: we eat the **poison**, eat the **big cheese**, or take **more than five steps**.
- Learning rate $\alpha = 0.1$; discount $\gamma = 0.99$.

**Reward function:**

| Event | Reward |
|---|---|
| Going to a state with **no** cheese | **+0** |
| Going to a state with a **small** cheese | **+1** |
| Going to the state with the **big pile** of cheese | **+10** |
| Going to the state with the **poison** (and dying) | **−10** |
| Taking more than five steps | **+0** |

The optimal policy we hope to learn: **right, right, down**.

**Step 1 — Initialize the Q-table** to 0 everywhere. For now it's useless.

### Training timestep 1
- **Step 2 (choose action):** $\varepsilon = 1.0$ is big ⇒ take a **random action**. We go **right**.
- **Step 3 (act):** we get a **small cheese**, so $R_{t+1} = 1$, and we're in a new state.
- **Step 4 (update):** with all Q-values still 0, $\max_a Q(S_1, a) = 0$:

$$
Q(S_0, \text{right}) \leftarrow 0 + 0.1\Big[\, 1 + 0.99 \times 0 - 0 \,\Big] = \mathbf{0.1}
$$

### Training timestep 2
- **Step 2 (choose action):** take a **random action again**, since $\varepsilon = 0.99$ is still big. *(Note we decayed $\varepsilon$ slightly — as training progresses we want less and less exploration.)* We take **down**. **This is a bad action — it leads to the poison.**
- **Step 3 (act):** we eat poison ⇒ $R_{t+1} = -10$, and **we die**.
- **Step 4 (update):**

$$
Q(S_1, \text{down}) \leftarrow 0 + 0.1\Big[\, -10 + 0.99 \times 0 - 0 \,\Big] = \mathbf{-1.0}
$$

Because we're dead, we **start a new episode**. But notice: **with just two exploration steps the agent already got smarter** — it has learned that "right" from the start is mildly good ($+0.1$) and that "down" there is bad ($-1.0$).

> As we keep exploring/exploiting and updating with the TD target, **the Q-table gives a better and better approximation**, and at the end of training we have an estimate of the **optimal Q-function**.

---

## 7. Off-policy vs On-policy

The difference is subtle but important:

| | **Off-policy** | **On-policy** |
|---|---|---|
| Definition | **Different** policy for **acting** (inference) and **updating** (training) | The **same** policy for acting and updating |
| Example | **Q-Learning** — acts with **epsilon-greedy**, updates using the **greedy** $\max_a Q(S_{t+1},a)$ | **Sarsa** — the **epsilon-greedy** policy also selects the next state-action pair used in the update |
| Update target | $R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a)$ | $R_{t+1} + \gamma\, Q(S_{t+1}, A_{t+1})$ |

So in Q-Learning: **acting policy** = epsilon-greedy; **updating policy** = greedy. In Sarsa both are epsilon-greedy.

---

## 8. Q-Learning Recap (one screen)

*Q-Learning* is the RL algorithm that:
- Trains a **Q-function**, an **action-value function** encoded in internal memory by a **Q-table** containing all state-action pair values.
- Given a state and action, **searches its Q-table** for the corresponding value.
- When training is done, we have an **optimal Q-function ⇔ optimal Q-table**.
- And an optimal Q-function means an **optimal policy**, since we know for each state the **best action to take**.
- In the beginning the Q-table is **useless** (arbitrary/zero values), but as we explore and update, it gives a **better and better approximation**.

### The whole loop in one line

$$
S_t \xrightarrow[\ \varepsilon\text{-greedy}\ ]{} A_t \longrightarrow (R_{t+1}, S_{t+1}) \longrightarrow Q(S_t,A_t) \mathrel{+}= \alpha\big[R_{t+1} + \gamma \max_a Q(S_{t+1},a) - Q(S_t,A_t)\big]
$$

---

## 9. Glossary

**Strategies to find the optimal policy**
- **Policy-based methods** — the policy is usually trained with a neural network to select what action to take given a state. The network outputs the action instead of using a value function; it's re-adjusted from experience to give better actions.
- **Value-based methods** — a value function is trained to output the value of a state or state-action pair. This value **doesn't define the action**: we must specify the agent's behavior given the value function's output (e.g. a **Greedy Policy** always taking the action leading to the biggest reward).

**Two main value-based strategies**
- **The state-value function** — for each state, the expected return if the agent starts in that state and follows the policy until the end.
- **The action-value function** — for each state **and action** pair, the expected return if the agent starts in that state, **takes that action**, then follows the policy forever after.

**Epsilon-greedy strategy**
- Balances exploration and exploitation.
- Chooses the action with the highest expected reward with probability $1-\varepsilon$.
- Chooses a random action with probability $\varepsilon$.
- $\varepsilon$ is typically **decreased over time** to shift focus toward exploitation.

**Greedy strategy**
- Always chooses the action expected to lead to the highest reward, based on current knowledge (**only exploitation**).
- Includes **no exploration**.
- Can be **disadvantageous** in environments with uncertainty or unknown optimal actions.

**Off-policy vs on-policy**
- **Off-policy** — a different policy is used at training time and inference time.
- **On-policy** — the same policy is used during training and inference.

**MC and TD**
- **Monte Carlo (MC)** — learning at the **end of the episode**; update the value/policy function from a **complete episode**.
- **Temporal Difference (TD)** — learning at **each step**; update without requiring a complete episode.

---

## 10. Self-check (quick review questions)

1. In value-based methods we never train a policy — so **where does the policy come from**?
2. Write both $V_{\pi}(s)$ and $Q_{\pi}(s,a)$ from memory. What exactly differs?
3. What computational problem does the **Bellman equation** solve, and what does it replace the long sum with?
4. What happens to values if $\gamma = 0$? If $\gamma = 1$? If $\gamma$ is huge?
5. MC vs TD: **when** does each update, and what does each use as the target?
6. What is **bootstrapping**, and which of MC/TD does it?
7. Why does TD have **lower variance** but **more bias** than MC?
8. Write the **Q-Learning update rule** from memory.
9. In the update target, is the action chosen **greedily** or **epsilon-greedily**? Why does this answer make Q-Learning **off-policy**?
10. In the worked example, verify: why is $Q(S_0,\text{right}) = 0.1$ and $Q(S_1,\text{down}) = -1.0$?
11. Why do we **decay $\varepsilon$** over training?
12. How does Sarsa's update differ from Q-Learning's, in one formula?
13. Why does an optimal Q-table immediately give you an optimal policy?

---

## 11. Additional Readings

These are **optional** if you want to go deeper.

**Monte Carlo and TD Learning**
- [Why do temporal difference (TD) methods have lower variance than Monte Carlo methods?](https://stats.stackexchange.com/questions/355820/why-do-temporal-difference-td-methods-have-lower-variance-than-monte-carlo-met)
- [When are Monte Carlo methods preferred over temporal difference ones?](https://stats.stackexchange.com/questions/336974/when-are-monte-carlo-methods-preferred-over-temporal-difference-ones)

**Q-Learning**
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2020.pdf) — Richard Sutton & Andrew G. Barto, **Chapters 5, 6 and 7**
- [Foundations of Deep RL Series, L2: Deep Q-Learning](https://youtu.be/Psrhxy88zww) — Pieter Abbeel

---

> **Next up:** **Deep Q-Learning** — replacing the Q-table with a neural network that *approximates* Q-values, so we can scale to state spaces far too large to tabulate (e.g. Atari from pixels).
