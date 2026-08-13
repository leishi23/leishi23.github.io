---
layout: post
title:  "HF Deep RL Course — Unit 1 (Introduction to Deep RL)"
date:   2026-08-13
desc: "HF Deep RL Course — Unit 1 (Introduction to Deep RL)"
keywords: "Machine learning"
categories: [Machine learning]
tags: [Machine learning]
icon: icon-html
---

# HF Deep RL Course — Unit 1 (Introduction to Deep RL)

> Source: [Hugging Face Deep RL Course, Unit 1](https://huggingface.co/learn/deep-rl-course/unit1/introduction) — notes from *Introduction* through *Additional Readings*, reorganized for review.

> **_Keypoints:_**

- What is RL? (learning from interaction)
- The RL framework: the state–action–reward loop
- The reward hypothesis & Markov property
- Observations vs. States; Action spaces
- Rewards, discounting & the return
- Types of tasks: episodic vs. continuing
- Exploration / Exploitation trade-off
- Two approaches: policy-based vs. value-based
- What the "Deep" adds
- Glossary + self-check + additional readings

---

## 1. What is Reinforcement Learning?

### The big picture
An **agent** (an AI) learns from an **environment** by **interacting with it** (trial and error) and **receiving rewards** (positive or negative) as feedback.

The course's intuition: hand your little brother a controller for a game he's never played.
- He presses a button → gets a coin → **+1 reward**. He learns *get the coins*.
- He presses again → touches an enemy → dies → **−1 reward**. He learns *avoid the enemies*.

**Without any supervision**, he gets better and better. That's how humans and animals learn — **through interaction**. RL is just a *computational approach* to learning from actions.

### Formal definition
> Reinforcement learning is a framework for solving **control tasks** (also called decision problems) by building agents that learn from the environment by **interacting with it through trial and error** and **receiving rewards (positive or negative) as unique feedback**.

**Key contrast to keep in mind:** the reward is the agent's **only** feedback signal — there is no labeled "correct action" as in supervised learning.

---

## 2. The RL Framework

### The RL process (the loop)
The loop, at each time step:

1. The agent receives **state $S_0$** from the environment (e.g. the first frame of the game).
2. Based on $S_0$, the agent takes **action $A_0$** (e.g. move right).
3. The environment transitions to a **new state $S_1$** (the next frame).
4. The environment gives a **reward $R_1$** to the agent (e.g. not dead → $+1$).

This outputs a sequence of **state, action, reward, next state**:

$$
S_0,\; A_0,\; R_1,\; S_1,\; A_1,\; R_2,\; S_2,\; \dots
$$

The agent's goal is to **maximize its cumulative reward**, called the **expected return**.

### The reward hypothesis (the central idea)
Why maximize the expected return? Because RL is built on the **reward hypothesis**:

> **All goals can be described as the maximization of the expected return** (expected cumulative reward).

So to obtain the best behavior, we learn to take actions that **maximize the expected cumulative reward**.

### The Markov property
In papers the RL process is called a **Markov Decision Process (MDP)**. The one thing to remember for now:

> The Markov property implies the agent needs **only the current state** to decide its action — **not the history** of all previous states and actions.

---

## 3. Observations vs. States

Observations/states are the **information the agent gets from the environment** (a game frame, a stock price, …). The distinction:

| | **State $s$** | **Observation $o$** |
|---|---|---|
| Description | **Complete** description of the world | **Partial** description of the state |
| Environment | **Fully** observed | **Partially** observed |
| Example | **Chess** — we see the whole board | **Super Mario Bros** — we only see the part of the level near the player |

> ⚠️ In this course the term "state" is used to denote **both** state and observation, but the distinction is made in implementations.

---

## 4. Action Space

The **action space** is the set of **all possible actions** in an environment.

| Type | Size | Example |
|---|---|---|
| **Discrete** | Number of possible actions is **finite** | Super Mario Bros: 4 actions — left, right, up (jump), down (crouch) |
| **Continuous** | Number of possible actions is **infinite** | Self-driving car: turn left 20°, 21.1°, 21.2°, honk, turn right 20°… |

> This matters because it **determines which RL algorithm you can use**.

---

## 5. Rewards and Discounting

The reward is fundamental — it is **the only feedback** the agent gets.

### Cumulative reward (undiscounted)
The cumulative reward at time step $t$ is the sum of all rewards in the sequence:

$$
R(\tau) = r_{t+1} + r_{t+2} + r_{t+3} + \dots = \sum_{k=0}^{\infty} r_{t+k+1}
$$

### Why discount?
**We can't just add rewards like that.** Rewards that come **sooner are more likely to happen** — they are more predictable than long-term future rewards.

The course's *mouse and cat* example: the mouse moves one tile per step, the cat moves too, and the mouse wants to eat as much cheese as possible before being eaten.
- It's **more probable to eat the cheese near us** than cheese close to the cat.
- So the reward near the cat — **even if it is bigger** (more cheese) — **gets discounted more**, since we're not sure we'll survive to collect it.

### How to discount
1. Define a **discount rate $\gamma$ (gamma)**, with $\gamma \in [0, 1]$ — most often between **0.95 and 0.99**.
   - **Larger $\gamma$** → smaller discount → agent **cares more about long-term reward**.
   - **Smaller $\gamma$** → bigger discount → agent **cares more about short-term reward** (the nearest cheese).
2. Discount each reward by $\gamma$ raised to the **exponent of the time step** — as the time step grows, the cat gets closer, so the future reward is less and less likely.

The **discounted expected cumulative reward** is:

$$
R(\tau) = \sum_{k=0}^{\infty} \gamma^{k} \, r_{t+k+1}
= r_{t+1} + \gamma\, r_{t+2} + \gamma^{2} r_{t+3} + \dots
$$

> **Mnemonic:** $\gamma$ is *"how far-sighted"* the agent is. $\gamma \to 0$ = myopic/greedy-for-now; $\gamma \to 1$ = patient/long-horizon.

---

## 6. Types of Tasks

A **task** is an **instance** of an RL problem.

| | **Episodic** | **Continuing** |
|---|---|---|
| Structure | Has a **starting point and a terminal state** | Has a starting point but **no terminal state** |
| Unit of experience | Creates an **episode**: a list of States, Actions, Rewards, new States | Runs **forever** |
| Example | **Super Mario Bros** — begins at level launch, ends when you die or reach the end of the level | **Automated stock trading** — the agent keeps running until we stop it |

For continuing tasks the agent must **learn to choose the best actions and simultaneously interact with the environment**.

---

## 7. The Exploration / Exploitation Trade-off

- **Exploration**: exploring the environment by trying **random actions**, to **find more information** about it.
- **Exploitation**: **using known information** to maximize the reward.

**The trap.** In the maze example, the mouse can collect an **infinite amount of small cheese** ($+1$ each), but at the top of the maze sits a **gigantic pile of cheese** ($+1000$).
- **Pure exploitation** → the agent **never reaches the big reward**; it just farms the nearest small source.
- **A little exploration** → it can **discover the big reward**.

So we must **balance** how much we explore vs. exploit, and **define a rule** to handle the trade-off (covered in later units).

**Restaurant analogy** (Berkeley AI course):
- *Exploitation*: go to the restaurant you already know is good every day — and **risk missing a better one**.
- *Exploration*: try new restaurants — risk a bad meal, but **possibly discover something fantastic**.

---

## 8. Two Main Approaches to Solving RL Problems

### The Policy $\pi$: the agent's brain
The **policy $\pi$** is the function that tells us **what action to take given the state** we are in — it **defines the agent's behavior** at a given time.

The policy **is the function we want to learn**. The goal is the **optimal policy $\pi^{*}$**, the one that **maximizes the expected return**. We find $\pi^{*}$ **through training**.

Two ways to get there:
- **Directly** — teach the agent **which action to take** given the state → **policy-based methods**.
- **Indirectly** — teach the agent **which state is more valuable**, then take the action that **leads to more valuable states** → **value-based methods**.

### Policy-based methods
We learn the **policy function directly**: a mapping from each state to the best action, or a **probability distribution over actions** at that state.

Two types of policies:

- **Deterministic** — a given state **always returns the same action**:

$$
a = \pi(s)
$$

- **Stochastic** — outputs a **probability distribution over actions**:

$$
\pi(a \mid s) = P(A = a \mid S = s)
$$

### Value-based methods
Instead of learning a policy, we learn a **value function** mapping a state to the **expected value of being in that state**.

The **value of a state** is the **expected discounted return** the agent can get if it **starts in that state and then acts according to our policy**:

$$
V_{\pi}(s) = \mathbb{E}_{\pi}\!\left[\, R_{t+1} + \gamma R_{t+2} + \gamma^{2} R_{t+3} + \dots \mid S_t = s \,\right]
$$

Here *"act according to our policy"* just means the policy is **"go to the state with the highest value."** In the course's grid example, the policy follows the value function's numbers to the goal: $-7 \to -6 \to -5 \to \dots$

### Side-by-side

| | **Policy-based** | **Value-based** |
|---|---|---|
| What we learn | The **policy** $\pi$ directly | A **value function** $V$ (or $Q$) |
| How the action is chosen | Output of the policy | Pick the action leading to the **highest-value** state |
| Nature | Direct | Indirect (policy is *derived* from values) |

---

## 9. The "Deep" in Deep Reinforcement Learning

**Deep RL introduces deep neural networks to solve RL problems** — hence "deep".

The upcoming example (next unit) uses two **value-based** algorithms:

| | **Q-Learning** (classic RL) | **Deep Q-Learning** |
|---|---|---|
| Machinery | A **traditional algorithm** builds a **Q-table** giving the action for each state | A **neural network** **approximates** the Q value |
| Scales to large state spaces? | Poorly (table grows with states) | Yes — that's the point of function approximation |

So we say *Deep* RL because neural networks are introduced to **estimate the action to take** (policy-based) or **estimate the value of a state** (value-based).

> If you're new to deep learning, the course recommends [FastAI Practical Deep Learning for Coders](https://course.fast.ai) (free).

---

## 10. Summary (the one-screen recap)

- RL is a **computational approach of learning from actions**: an agent learns from the environment **by interacting through trial and error** and receiving rewards as feedback.
- The goal of any RL agent is to **maximize its expected cumulative reward** (expected return), because of the **reward hypothesis** — *all goals can be described as maximization of the expected cumulative reward*.
- The RL process is a **loop** outputting **state → action → reward → next state**.
- To compute the expected return we **discount** rewards: earlier rewards **are more probable** because they're more predictable.
- Solving an RL problem = **finding an optimal policy**. The policy is the agent's **"brain"**, telling us what action to take given a state. The optimal one maximizes expected return.
- Two ways to find it: **(1) policy-based** — train the policy directly; **(2) value-based** — train a value function that gives the expected return at each state, then derive the policy from it.
- It's **Deep** RL because **deep neural networks** estimate the action (policy-based) or the state value (value-based).

### The pipeline in one line

$$
\underbrace{S_t}_{\text{state}} \;\xrightarrow{\;\pi\;}\; \underbrace{A_t}_{\text{action}} \;\longrightarrow\; \underbrace{R_{t+1},\, S_{t+1}}_{\text{env. response}} \;\longrightarrow\; \text{maximize } \mathbb{E}\Big[\sum_k \gamma^k r_{t+k+1}\Big]
$$

---

## 11. Glossary

- **Agent** — learns to **make decisions by trial and error**, with rewards and punishments from the surroundings.
- **Environment** — a simulated world **where an agent can learn by interacting with it**.
- **Markov Property** — the action taken is **conditional solely on the present state**, independent of past states and actions.
- **State / Observation** — **State**: complete description of the world. **Observation**: partial description.
- **Actions** — **Discrete**: finite (left, right, up, down). **Continuous**: infinite possibilities (self-driving car).
- **Rewards & Discounting** — reward tells the agent whether the action was good/bad; algorithms maximize **cumulative reward**; the **reward hypothesis** says RL problems can be formulated as maximization of (cumulative) return; **discounting** is applied because early rewards are more likely.
- **Tasks** — **Episodic**: has a start and an end. **Continuous**: has a start but no end.
- **Exploration vs. Exploitation** — **Exploration**: trying random actions to get feedback about the environment. **Exploitation**: using what we know to gain maximum reward. The **trade-off** balances the two.
- **Policy** — the agent's **brain**; tells what action to take given the state. **Optimal policy**: maximizes expected return; learned through training.
- **Policy-based methods** — the policy is **learned directly**; maps each state to the best action (or a distribution over actions).
- **Value-based methods** — instead of a policy, train a **value function** mapping each state to the expected value of being in it.

---

## 12. Self-check (quick review questions)

1. Why is the reward the *only* feedback signal, and how does that differ from supervised learning?
2. State the **reward hypothesis** in one sentence.
3. What does the **Markov property** let us ignore?
4. Chess vs. Super Mario Bros — which gives a **state** and which an **observation**? Why?
5. Why can't we simply **sum** rewards? What does $\gamma$ control, and what happens as $\gamma \to 0$ vs. $\gamma \to 1$?
6. Write the discounted return $R(\tau)$ from memory.
7. Episodic vs. continuing — give one example of each.
8. In the maze, why does a **pure exploitation** agent never find the $+1000$ cheese?
9. Deterministic vs. stochastic policy — write the form of each.
10. In **value-based** methods, where does the policy come from if we never train one?
11. What exactly makes Deep RL *"deep"*? Where does the neural net sit in Q-Learning vs. Deep Q-Learning?

---

## 13. Additional Readings

These are **optional** if you want to go deeper.

**Deep Reinforcement Learning**
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2020.pdf) — Richard Sutton & Andrew G. Barto, **Chapters 1, 2 and 3**
- [Foundations of Deep RL Series, L1: MDPs, Exact Solution Methods, Max-ent RL](https://youtu.be/2GwBez0D20A) — Pieter Abbeel
- [Spinning Up RL by OpenAI — Part 1: Key Concepts of RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

**Gym**
- [Getting Started With OpenAI Gym: The Basic Building Blocks](https://blog.paperspace.com/getting-started-with-openai-gym/)
- [Make your own Gym custom environment](https://www.gymlibrary.dev/content/environment_creation/)

---

> **Next up (hands-on):** train a **lunar lander** to land correctly on the Moon with [Stable-Baselines3](https://stable-baselines3.readthedocs.io/), then push the trained agent to the Hugging Face Hub. Bonus unit: train **Huggy the Dog** 🐶 to fetch a stick.
