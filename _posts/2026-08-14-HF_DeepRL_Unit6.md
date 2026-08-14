---
layout: post
title:  "HF Deep RL Course — Unit 6 (Advantage Actor-Critic, A2C)"
date:   2026-08-14
desc: "HF Deep RL Course — Unit 6 (Advantage Actor-Critic, A2C)"
keywords: "Machine learning"
categories: [Machine learning]
tags: [Machine learning]
icon: icon-html
---

# HF Deep RL Course — Unit 6 (Advantage Actor-Critic, A2C)

> Source: [Hugging Face Deep RL Course, Unit 6](https://huggingface.co/learn/deep-rl-course/unit6/introduction) — notes from *Introduction* through *Additional Readings*, reorganized for review. Continues from [Unit 5]({{ '/machine learning/2026/08/14/HF_DeepRL_Unit5.html' | replace: ' ', '%20' | prepend: site.baseurl }}).

> **_Keypoints:_**

- Recap: REINFORCE optimizes the policy directly with Monte-Carlo returns
- **The variance problem**: $R(\tau)$ is **unbiased but high-variance**
- Mitigations: more trajectories (sample-inefficient) or a **baseline**
- The **bias–variance trade-off** in RL, stated precisely
- **Actor-Critic** = hybrid of policy-based (Actor $\pi_\theta$) and value-based (Critic $\hat{q}_w$)
- The Actor-Critic training loop, timestep by timestep
- **A2C**: use the **advantage** $A(s,a) = Q(s,a) - V(s)$ instead of raw $Q$
- The practical trick: the **TD error** estimates the advantage, so only $V$ must be learned
- A2C vs A3C (synchronous vs asynchronous)
- Hands-on: **panda-gym** robotics, parallel envs, **VecNormalize**
- Self-check + additional readings

---

## 0. What this unit is for

Unit 4 introduced our first **policy-based** algorithm, **REINFORCE** — a *policy-gradient* method that optimizes the policy **directly**, without a value function, by **estimating the weights of the optimal policy using gradient ascent**.

REINFORCE worked. But because it uses **Monte-Carlo sampling** to estimate the return (an *entire episode* is needed to compute it), **the policy-gradient estimate has significant variance**. High variance means we need **a lot of samples** to get a usable gradient direction, which means **slow training**.

So this unit studies **Actor-Critic methods**: a **hybrid architecture combining value-based and policy-based methods** that stabilizes training by reducing variance, using

- **an Actor** that controls **how the agent behaves** (policy-based), and
- **a Critic** that measures **how good the taken action is** (value-based).

We then study one specific hybrid, **Advantage Actor-Critic (A2C)**, and train it with **Stable-Baselines3** in a **robotics environment**: a robotic arm that must move its end-effector to the correct position.

---

## 1. Short recap: what REINFORCE actually does

In REINFORCE we want to **increase the probability of actions in a trajectory in proportion to how high the return is**:

- If the **return is high** → **push up** the probabilities of the $(s,a)$ pairs in that trajectory.
- If the **return is low** → **push down** the probabilities of those $(s,a)$ pairs.

The policy gradient is **the direction of the steepest increase in return**. In its Monte-Carlo form:

$$
\nabla_\theta J(\theta) \;=\; \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\, \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, R(\tau) \,\right]
$$

where the **return of the trajectory** is computed by Monte-Carlo sampling:

$$
R(\tau) = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots
$$

We collect a trajectory, compute the discounted return, and **use that one score to increase or decrease the probability of every action taken in that trajectory**. If the return is good, *all* actions in the episode get "reinforced" — including the bad ones that happened to be along the way.

> **Key insight:** the good news is this estimate is **unbiased**. We are not *estimating* the return with a learned approximation — we use the **true return we actually obtained**. In expectation, we are pushing in the right direction.

---

## 2. The problem of variance in REINFORCE

Being unbiased is not enough. Two sources of randomness fight us:

1. **Stochasticity of the environment** — random events during an episode.
2. **Stochasticity of the policy** — $\pi_\theta$ samples its actions.

Because of this, **trajectories can lead to very different returns**, and so **the return starting from the same state can vary significantly across episodes**. The estimator is centered on the right value, but any single sample can be far away from it.

Worse, the variance **compounds over the horizon**: the return is a sum of many random rewards along a long chain of random transitions and random action choices, so the spread of $R(\tau)$ grows with episode length. And since REINFORCE applies **one scalar $R(\tau)$ to every action in the episode**, the noise is smeared over all the gradient terms.

### The standard mitigations

**(a) Use a large number of trajectories.** Average over many episodes and hope the variance introduced in any one trajectory is **reduced in aggregate**, giving a "true" estimation of the return.

$$
\text{Var}\!\left[\bar{R}\right] = \frac{\text{Var}\!\left[R(\tau)\right]}{N}
$$

> ⚠️ **The catch:** increasing the batch size **significantly reduces sample efficiency**. Variance falls only as $1/N$, so halving the noise costs $4\times$ the environment interaction. We need **additional mechanisms**.

**(b) Subtract a baseline.** Instead of scaling by $R(\tau)$, scale by $R(\tau) - b(s)$ for some function $b$ that does not depend on the action. Because

$$
\mathbb{E}_{a \sim \pi_\theta}\!\left[\nabla_\theta \log \pi_\theta(a \mid s)\, b(s)\right] = b(s)\, \nabla_\theta \sum_a \pi_\theta(a\mid s) = b(s)\,\nabla_\theta 1 = 0,
$$

subtracting a baseline **does not change the expected gradient** (it stays unbiased) but **can dramatically reduce its variance**. The best natural choice of baseline is the **state-value function $V(s)$** — and $R(\tau) - V(s)$ is exactly the **advantage**, which is where this unit is heading.

---

## 3. The bias–variance trade-off, in RL terms

In supervised learning, bias/variance is about generalizing to held-out labeled data. **In RL there are no labels — only a reward signal.** So the trade-off is restated:

> **The bias–variance trade-off in RL reflects how well the reinforcement signal reflects the true reward the agent should get from the environment.**

| Property | Meaning in RL |
|---|---|
| **Unbiased** signal | Returns rewards **similar to the real / expected ones** from the environment |
| **Biased** signal | The reward signal we use **differs** from the real reward we should get |
| **High variance** signal | Lots of **noise**; strongly affected by **stochastic (non-constant) elements** of the environment |
| **Low variance** signal | Less affected by environment noise; produces **similar values** regardless of the random elements |

Monte-Carlo, as used in REINFORCE, is a **sampling mechanism**: we do not analyze all possible states/trajectories, only a **sample** of them. It is therefore **not resistant to stochasticity** — even the *same* trajectory can yield different reward values if stochastic elements are involved. The classic fix is to take $n$ samples and **average** them, reducing each one's individual impact.

The estimator ladder we now have:

| Estimator of the return | Bias | Variance | Needs full episode? |
|---|---|---|---|
| Monte-Carlo return $R(\tau)$ (REINFORCE) | **Unbiased** | **Very high** | **Yes** |
| MC return minus baseline, $R(\tau) - V(s)$ | Unbiased | High, but **lower** | Yes |
| $n$-step return $\sum_{k=0}^{n-1}\gamma^k r_{t+k+1} + \gamma^n V(s_{t+n})$ | Slightly biased | Medium | No |
| TD(0) target $r + \gamma V(s')$ | **Biased** (bootstraps) | **Low** | **No** |

> **Key insight:** Actor-Critic buys variance reduction **by paying a little bias**. The critic's $V$ or $Q$ is only an *approximation*, so bootstrapping from it introduces bias — but the payoff (a low-variance signal available at **every timestep**, no waiting for the episode to end) is usually a very good trade.

---

## 4. Actor-Critic: the hybrid

The solution to reducing REINFORCE's variance — and training faster and better — is to **combine policy-based and value-based methods**.

**The analogy from the course.** Imagine you're playing a video game with a friend who gives you feedback. **You are the Actor; your friend is the Critic.** At the beginning you don't know how to play, **so you try some actions randomly**. The Critic observes your action and **provides feedback**. Learning from that feedback, **you update your policy and get better at the game**. Meanwhile your friend (the Critic) **also updates how they give feedback**, so it's better next time.

So we learn **two function approximations (two neural networks)**:

- **The Actor** — a **policy** that controls **how our agent acts**, parameterized by $\theta$:

$$
\pi_\theta(s)
$$

- **The Critic** — a **value function** that assists the policy update by **measuring how good the action taken is**, parameterized by $w$:

$$
\hat{q}_w(s,a)
$$

**Why this reduces variance:** the critic **replaces the Monte-Carlo return**. Instead of waiting for the whole episode and using the noisy realized $R(\tau)$ as the scaling factor of the gradient, we use the critic's **learned estimate** of how good the action was. That estimate is a smooth function of $(s,a)$ rather than a sample of a long random sum — it is far less noisy, and it is available **immediately, at every timestep**.

### Where Actor-Critic sits

| | **Value-based** (Q-Learning, DQN) | **Policy-based** (REINFORCE) | **Actor-Critic** (A2C) |
|---|---|---|---|
| What is learned | A value function only | A policy only | **Both** — a policy *and* a value function |
| Policy | **Not** trained; derived by hand (e.g. $\arg\max_a Q$) | **Trained directly** | **Trained directly** (Actor) |
| Value function | Trained; **is** the whole method | **None** | Trained, but only to **help the policy update** (Critic) |
| Action spaces | Awkward for continuous actions | Handles **continuous** & stochastic policies | Handles **continuous** & stochastic policies |
| Update timing | Every step (TD) | **End of episode** (MC) | **Every step** (TD) |
| Bias / Variance | Biased, low variance | Unbiased, **high variance** | Slightly biased, **low variance** |

---

## 5. The Actor-Critic process, step by step

Two networks: the **Actor** $\pi_\theta(s)$ and the **Critic** $\hat{q}_w(s,a)$. At each timestep $t$:

**Step 1 — Observe the state.** Get the current state $S_t$ from the environment and **pass it as input through both the Actor and the Critic**.

**Step 2 — Actor picks an action.** The policy takes the state and **outputs an action** $A_t \sim \pi_\theta(\cdot \mid S_t)$.

**Step 3 — Critic evaluates it.** The Critic takes **that action as input too** and, using $S_t$ and $A_t$, **computes the value of taking that action at that state: the Q-value** $\hat{q}_w(S_t, A_t)$.

**Step 4 — Act in the environment.** Performing $A_t$ outputs a **new state $S_{t+1}$ and a reward $R_{t+1}$**.

**Step 5 — Update the Actor** using the Q-value from the Critic (gradient *ascent* on expected return):

$$
\Delta\theta = \alpha\, \nabla_\theta \log \pi_\theta(s,a)\, \hat{q}_w(s,a)
$$

Read it as: *move the policy in the direction that makes this action more likely, scaled by how good the Critic says it was.* If $\hat{q}_w$ is large and positive, push the log-probability up hard; if it's negative, push it down.

**Step 6 — Actor produces the next action.** Thanks to its updated parameters, the Actor produces the next action $A_{t+1}$ given the new state $S_{t+1}$.

**Step 7 — Update the Critic** using the **TD error** on its own prediction:

$$
\Delta w = \beta \Big(\underbrace{R_{t+1} + \gamma\, \hat{q}_w(S_{t+1}, A_{t+1})}_{\text{TD target}} - \hat{q}_w(S_t, A_t)\Big)\, \nabla_w \hat{q}_w(S_t, A_t)
$$

which is just gradient descent on the squared TD error

$$
L(w) = \Big(R_{t+1} + \gamma\, \hat{q}_w(S_{t+1}, A_{t+1}) - \hat{q}_w(S_t, A_t)\Big)^2
$$

with the target held fixed (not differentiated through).

### The whole loop in one line

$$
S_t \xrightarrow[\ \pi_\theta\ ]{} A_t \xrightarrow[\ \hat{q}_w\ ]{} \hat{q}_w(S_t,A_t) \longrightarrow (R_{t+1}, S_{t+1}) \longrightarrow \underbrace{\theta \mathrel{+}= \alpha \nabla_\theta \log\pi_\theta\, \hat{q}_w}_{\text{actor}} \longrightarrow \underbrace{w \mathrel{+}= \beta\, \delta_t\, \nabla_w \hat{q}_w}_{\text{critic}}
$$

> **Key insight:** both networks improve **together and continuously**. The Actor gets better because the Critic tells it which actions were good; the Critic gets better because the Actor keeps generating fresh experience to fit. Notice also that unlike REINFORCE, **nothing waits for the episode to end**.

---

## 6. A2C: adding the Advantage

We can **stabilize learning further** by **using the Advantage function as the Critic's signal instead of the raw action-value function**.

The **advantage function** computes the **relative advantage of an action compared to the others possible at that state** — i.e. **how much better taking this action at this state is compared to the average value of the state**. It subtracts the mean value of the state from the state-action pair:

$$
\boxed{\,A(s,a) = Q(s,a) - V(s)\,}
$$

In other words: **the extra reward we get if we take this action at that state, compared to the mean reward we get at that state.** The "extra" is whatever lies **beyond the expected value of that state**.

And since $V(s) = \mathbb{E}_{a\sim\pi}[Q(s,a)]$, the advantage is **centered on zero under the current policy**: some actions are above average, some below.

| Sign | Interpretation | Effect on the update |
|---|---|---|
| $A(s,a) > 0$ | This action is **better than average** at this state | Our gradient is **pushed in that direction** (make it more likely) |
| $A(s,a) < 0$ | This action does **worse than the average value** of that state | Our gradient is **pushed in the opposite direction** (make it less likely) |
| $A(s,a) \approx 0$ | Action is **unremarkable** here | Almost no update |

The Actor update becomes

$$
\Delta\theta = \alpha\, \nabla_\theta \log \pi_\theta(s,a)\, A(s,a)
$$

**Why this stabilizes learning.** Consider a state where *every* action yields a return around $+100$. With a raw $Q$ critic, every action gets a large positive push — the update is dominated by "how valuable is this *state*", which the policy cannot control, rather than "which *action* should I prefer here", which is the only thing the policy can act on. Subtracting $V(s)$ removes exactly that state-dependent offset. Concretely, the advantage:

- **Removes the state-value offset**, so the gradient magnitude reflects only *action quality*, not *state luckiness*.
- **Shrinks the scale** of the scaling factor (from possibly hundreds down to a small number around zero), keeping the gradient well-conditioned.
- Is a **baseline subtraction**, so by the identity in §2 it **does not introduce bias** into the policy gradient — pure variance reduction.
- Makes the sign meaningful: positive means "genuinely better than what I usually do here".

---

## 7. The practical trick: the TD error estimates the advantage

**The problem with implementing the advantage directly** is that it appears to require **two value functions** — $Q(s,a)$ *and* $V(s)$ — hence two networks to train, twice the parameters, twice the error.

**Fortunately, we can use the TD error as a good estimator of the advantage function:**

$$
\boxed{\,A(s,a) \;\approx\; \delta_t \;=\; r + \gamma V(s') - V(s)\,}
$$

**Why this works.** By definition $Q(s,a) = \mathbb{E}\!\left[\,r + \gamma V(s') \mid s, a\,\right]$. So

$$
\mathbb{E}\!\left[\, r + \gamma V(s') - V(s) \;\middle|\; s,a \,\right] \;=\; Q(s,a) - V(s) \;=\; A(s,a)
$$

The single-sample TD error is therefore an **unbiased estimator of the advantage** — *provided $V$ is the true value function*. In practice $V$ is a learned approximation $V_w$, so there is some bias from the approximation error; that is precisely the bias we accept in exchange for the variance reduction (§3).

> **Key insight:** **only the value function $V$ needs to be learned.** The reward $r$ comes free from the environment, and $V(s')$ is just the same network applied to the next state. One critic network, and we get advantages.

So the A2C update pair is:

**Actor (policy gradient with advantage):**

$$
\Delta\theta = \alpha\, \nabla_\theta \log \pi_\theta(a \mid s)\,\big(r + \gamma V_w(s') - V_w(s)\big)
$$

**Critic (regression on the TD target):**

$$
L(w) = \big(r + \gamma V_w(s') - V_w(s)\big)^2, \qquad \Delta w = -\beta \nabla_w L(w)
$$

with $r + \gamma V_w(s')$ treated as a **constant target**.

In real implementations the two losses are combined into one objective, plus an **entropy bonus** on the policy to keep exploring:

$$
L = \underbrace{-\log \pi_\theta(a\mid s)\,\hat{A}}_{\text{policy loss}} \;+\; c_v \underbrace{\big(\hat{A} \text{-based TD error}\big)^2}_{\text{value loss}} \;-\; c_e \underbrace{\mathcal{H}\!\left[\pi_\theta(\cdot\mid s)\right]}_{\text{entropy bonus}}
$$

Advantages are usually computed over a short rollout of $n$ steps (an $n$-step return, or GAE) rather than a single step, to tune the bias/variance dial further.

---

## 8. A2C vs A3C: synchronous vs asynchronous

The algorithm was introduced in **"Asynchronous Methods for Deep Reinforcement Learning"** (Mnih et al., 2016) — the **A3C** paper. The "2" vs "3" is just the number of leading A's:

| | **A3C** — *Asynchronous* Advantage Actor-Critic | **A2C** — *(synchronous)* Advantage Actor-Critic |
|---|---|---|
| Workers | Many workers, each with **its own copy** of the network | Many parallel **environments**, **one** network |
| Coordination | Workers push gradients to a shared model **asynchronously**, whenever they finish | A coordinator **waits for all envs** to step, then does **one batched update** |
| Consequence | Workers can be using **slightly stale** parameters | All actors always use the **same, current** parameters |
| Practical notes | Historically CPU-friendly | **Simpler, more reproducible, GPU-friendly** (one big batch), and empirically **as good or better** |

A2C is the version used in Stable-Baselines3 and in this unit's hands-on. The **parallel environments** are what supplies the decorrelated batch of transitions that the asynchronous workers used to provide.

---

## 9. Hands-on context: robotics with panda-gym

The hands-on trains **A2C with Stable-Baselines3** on **[panda-gym](https://github.com/qgallouedec/panda-gym)** — continuous-control robotics simulations. Details worth remembering:

**The environment: `PandaReachDense-v3`.** A robotic arm must place its **end-effector** (the device at the end of the arm designed to interact with the environment) at a **target position** (a green ball).

- **Dense** reward function → **a reward at every timestep** (the closer to completing the task, the higher the reward). Contrast with a **sparse** reward function, which returns a reward **if and only if** the task is completed. Dense rewards make learning far easier here.
- **End-effector displacement control** → the action **is the displacement of the end-effector**, not the individual motion of each joint (joint control). Again, easier training.
- **Observation space is a dictionary** with three elements: `achieved_goal` (x,y,z position of the goal), `desired_goal` (distance between goal position and current object position), and `observation` (position x,y,z and velocity vx,vy,vz of the end-effector). Because it's a dict, we must use **`MultiInputPolicy`**, not `MlpPolicy`.
- **Action space** is a 3-vector: control of x, y, z movement.
- Certification target: **`PandaReachDense-v3` result $\geq -3.5$**, where result $=$ mean_reward $-$ std of reward.

**Multiple parallel environments.** A2C batches experience across envs:

```python
env = make_vec_env(env_id, n_envs=4)
```

**Observation and reward normalization (`VecNormalize`).** A good practice in RL is to **normalize input features**. The wrapper computes a **running average and standard deviation** of the input features, and normalizes rewards too via `norm_reward=True`:

```python
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
```

> ⚠️ **Why normalization matters here:** the observations mix positions and velocities on different scales. Un-normalized inputs make the value function hard to fit, and since the *advantage* is a difference of value estimates, a badly-scaled critic directly corrupts the policy gradient. This is why normalization is close to mandatory for continuous-control A2C.

**The rest of the recipe:**

```python
model = A2C(policy="MultiInputPolicy", env=env, verbose=1)
model.learn(1_000_000)             # ~25-40 min on a Colab GPU

model.save("a2c-PandaReachDense-v3")
env.save("vec_normalize.pkl")      # SAVE THE NORMALIZATION STATS TOO
```

> ⚠️ **The classic footgun:** the `VecNormalize` statistics are **part of the model**. You must save them alongside the weights and reload them for evaluation, then set `eval_env.training = False` (don't keep updating the running stats at test time) and `eval_env.norm_reward = False` (reward normalization is not needed at test time — and would make the reported score meaningless).

Optional extra challenge from the course: try **`PandaPickAndPlace-v3`**.

---

## 10. Everything on one screen

### REINFORCE vs Actor-Critic vs A2C

| | **REINFORCE** | **Actor-Critic** | **A2C** |
|---|---|---|---|
| Family | Policy-based (policy-gradient) | **Hybrid** | **Hybrid** |
| Networks learned | Policy $\pi_\theta$ only | $\pi_\theta$ **and** $\hat{q}_w(s,a)$ | $\pi_\theta$ **and** $V_w(s)$ |
| What scales the gradient | MC return $R(\tau)$ | Critic's Q-value $\hat{q}_w(s,a)$ | **Advantage** $\hat{A}(s,a)$ |
| Estimates the return by | **Sampling a full episode** | **Bootstrapping** (TD) | **Bootstrapping** (TD) |
| Bias | **None** (unbiased) | Biased (critic is approximate) | Biased (critic is approximate) |
| Variance | **Very high** | **Much lower** | **Lowest of the three** |
| When it updates | **End of the episode** | **Every timestep** | **Every timestep / short rollout** |
| Needs episode termination? | **Yes** | No | No |
| Gradient scale | Can be huge and state-dependent | State-dependent offset remains | **Centered near 0** — offset removed |
| Sample efficiency | Poor (needs many trajectories) | Better | Better, plus **parallel envs** |

### The one-line summaries

$$
\text{REINFORCE:}\quad \Delta\theta = \alpha\, \nabla_\theta \log \pi_\theta(a\mid s)\; R(\tau)
$$

$$
\text{Actor-Critic:}\quad \Delta\theta = \alpha\, \nabla_\theta \log \pi_\theta(a\mid s)\; \hat{q}_w(s,a)
$$

$$
\text{A2C:}\quad \Delta\theta = \alpha\, \nabla_\theta \log \pi_\theta(a\mid s)\; \big(r + \gamma V_w(s') - V_w(s)\big)
$$

Same skeleton every time — **only the scaling factor changes**, and each change trades a little bias for a lot less variance.

### Mnemonic

**Actor acts, Critic critiques, Advantage asks "compared to what?"** — and the **TD error answers it for free**.

---

## 11. Self-check (quick review questions)

1. Why is the Monte-Carlo return $R(\tau)$ used in REINFORCE **unbiased**? What exactly makes it **high-variance**?
2. Name the **two sources of stochasticity** that make the same starting state produce very different returns.
3. If you quadruple the number of trajectories per update, by how much does the variance of the mean return fall — and what is the cost?
4. State the **bias–variance trade-off** as it applies to RL (not to supervised learning).
5. Which of these is true: "a high-variance reward signal has much noise and is affected by stochastic elements of the environment"? Why?
6. What are the **two function approximations** learned by an Actor-Critic method, and what is each parameterized by?
7. Walk through the Actor-Critic loop at timestep $t$ from $S_t$ to the critic update, naming every quantity.
8. Write the Actor update using the Critic's Q-value from memory. Write the Critic's TD-based update from memory.
9. Define the **advantage function**. In words, what does $A(s,a) > 0$ mean, and what does it do to the gradient?
10. Why does subtracting $V(s)$ **not** bias the policy gradient? (Sketch the one-line proof.)
11. Give a concrete scenario where a raw-$Q$ critic gives a badly-behaved gradient but the advantage does not.
12. What is the implementation problem with $A(s,a) = Q(s,a) - V(s)$, and how does the **TD error** solve it?
13. Show that $\mathbb{E}[r + \gamma V(s') - V(s) \mid s,a] = A(s,a)$. Where does bias sneak in for a *learned* $V_w$?
14. What is the difference between **A2C** and **A3C**? What plays the role of the asynchronous workers in A2C?
15. In `PandaReachDense-v3`, why do we need `MultiInputPolicy` rather than `MlpPolicy`? What is a *dense* vs *sparse* reward?
16. What does `VecNormalize` compute, and why must you **save and reload** it? What two attributes must you change at evaluation time?

---

## 12. Additional Readings

These are **optional** if you want to go deeper.

**Bias-variance trade-off in Reinforcement Learning**
- [Making Sense of the Bias / Variance Trade-off in (Deep) Reinforcement Learning](https://blog.mlreview.com/making-sense-of-the-bias-variance-trade-off-in-deep-reinforcement-learning-79cf1e83d565)
- [Bias-variance Tradeoff in Reinforcement Learning](https://www.endtoend.ai/blog/bias-variance-tradeoff-in-reinforcement-learning/)
- [High Variance in Policy Gradients](https://balajiai.github.io/high_variance_in_policy_gradients)

**Advantage Functions**
- [Advantage Functions, Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html?highlight=advantage%20functio#advantage-functions) — OpenAI

**Actor-Critic**
- [Foundations of Deep RL Series, L3: Policy Gradients and Advantage Estimation](https://www.youtube.com/watch?v=AKbX1Zvo7r8) — Pieter Abbeel
- [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783v2) — Mnih et al., the A3C/A2C paper

**Libraries & environments used in the hands-on**
- [panda-gym](https://github.com/qgallouedec/panda-gym) — the robotic-arm environments
- [Stable-Baselines3: A2C](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html#notes)
- [Stable-Baselines3: VecNormalize](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecnormalize)
- [Stable-Baselines3: RL Tips and Tricks](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)

---

> **Next up:** **Multi-Agent Reinforcement Learning (MARL)** — what changes when several agents share one environment (cooperative, competitive, and mixed settings), how to design such systems, **self-play** for training against ever-stronger versions of yourself, and the AI vs AI leaderboard where our soccer team plays against everyone else's.
