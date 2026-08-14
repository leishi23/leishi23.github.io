---
layout: post
title:  "HF Deep RL Course — Unit 4 (Policy Gradient & REINFORCE)"
date:   2026-08-14
desc: "HF Deep RL Course — Unit 4 (Policy Gradient & REINFORCE)"
keywords: "Machine learning"
categories: [Machine learning]
tags: [Machine learning]
icon: icon-html
---

# HF Deep RL Course — Unit 4 (Policy Gradient & REINFORCE)

> Source: [Hugging Face Deep RL Course, Unit 4](https://huggingface.co/learn/deep-rl-course/unit4/introduction) — notes from *Introduction* through *Additional Readings*, reorganized for review. Continues from [Bonus Unit 2]({{ '/machine learning/2026/08/14/HF_DeepRL_BonusUnit2.html' | replace: ' ', '%20' | prepend: site.baseurl }}).

> **_Keypoints:_**

- The first **non**-value-based family: optimize the policy **directly**, no intermediate value function
- **Policy-based** ⊃ **policy-gradient**: the latter optimizes $\theta$ *by gradient ascent*, the former may not use the gradient at all
- Parameterized **stochastic** policy $\pi_\theta(a \mid s)$; objective $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$
- Gradient **ascent**: $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$
- Advantages: simplicity, stochastic policies (free exploration, no perceptual aliasing), continuous/high-dim actions, smoother convergence
- Disadvantages: local optima, slow/sample-inefficient, **high variance**
- The **Policy Gradient Theorem** and its derivation (the log-derivative / likelihood-ratio trick)
- **REINFORCE** (Monte Carlo policy gradient): collect episode → compute return → update
- The MC return makes $\hat{g}$ **unbiased but high-variance** → motivates Actor-Critic (Unit 6)
- Hands-on: REINFORCE from scratch in PyTorch on **CartPole-v1** and **Pixelcopter**
- Glossary + self-check + additional readings

---

## 0. What this unit is for

Everything so far has been **value-based**: in Units 2 and 3 we estimated a value function ($Q$-table, then a deep $Q$-network) **as an intermediate step toward finding an optimal policy**. In those methods the policy $\pi$ **only exists because of the action-value estimates** — it's just a fixed function (e.g. greedy / epsilon-greedy) that reads off the highest-valued action.

This unit flips that around. With **policy-based methods** we **optimize the policy directly, without the intermediate step of learning a value function.**

The plan:

1. Study **policy-based methods** in general.
2. Study **policy gradient**, a subset of them.
3. Implement our first policy-gradient algorithm, **Monte Carlo REINFORCE**, from scratch in **PyTorch**.
4. Test its robustness on **CartPole-v1** and **Pixelcopter**.

---

## 1. What are policy-based methods?

The goal of RL is always the same: **find the optimal policy $\pi^{*}$ that maximizes the expected cumulative reward**, because RL rests on the *reward hypothesis* — **all goals can be described as the maximization of the expected cumulative reward**.

*(Course example: in a soccer game the goal "win" becomes "maximize goals scored into the opponent's net, minimize goals in your own net".)*

### Value-based, policy-based, and actor-critic

- **Value-based methods** — learn a **value function**. An optimal value function leads to an optimal policy $\pi^{*}$. The objective is to **minimize the loss between predicted and target value** so as to approximate the true action-value function. There *is* a policy, but it's **implicit**, generated directly from the value function (in Q-Learning: an (epsilon-)greedy policy).
- **Policy-based methods** — directly learn to approximate $\pi^{*}$ **without learning a value function**. The idea is to **parameterize the policy**, e.g. with a neural network $\pi_\theta$ that outputs a **probability distribution over actions** (a *stochastic* policy). The objective is to **maximize the performance of the parameterized policy using gradient ascent**, by controlling the parameter $\theta$ that shapes the action distribution at each state.
- **Actor-critic methods** — a **combination** of the two. That's Unit 6.

So the whole game becomes: define an objective function $J(\theta)$ (the expected cumulative reward) and **find the $\theta$ that maximizes it**.

$$
\pi_\theta(a \mid s) \;=\; \text{NN}_\theta(s) \quad\text{(softmax over actions)}
$$

### ⭐ Policy-based vs policy-gradient — the distinction that matters

**Policy-gradient methods are a *subclass* of policy-based methods.** Both search directly for the optimal policy. The difference **lies in *how* we optimize the parameter $\theta$**:

| | **Policy-based (general)** | **Policy-gradient (this unit)** |
|---|---|---|
| Search target | The optimal policy, directly | The optimal policy, directly |
| How $\theta$ is optimized | **Indirectly** — maximize a *local approximation* of $J(\theta)$ | **Directly** — **gradient ascent** on the performance of $J(\theta)$ |
| Uses $\nabla_\theta J(\theta)$? | **No** (gradient-free) | **Yes** |
| Typical techniques | **Hill climbing**, **simulated annealing**, **evolution strategies / genetic algorithms** | REINFORCE, A2C, PPO, … |

> **Key insight:** "policy-based" does **not** imply "gradient". Hill climbing and evolution strategies are policy-based but never compute a gradient — they perturb $\theta$ and keep what works. **Policy-gradient** is the specialization that differentiates the objective.

Also note: in policy-based methods the optimization is **most of the time *on-policy***, since for each update we only use trajectories collected **by the most recent version of $\pi_\theta$**. (Contrast with Q-Learning/DQN, which are off-policy and can reuse a replay buffer.)

### Three families side by side

| | **Value-based** | **Policy-based / policy-gradient** | **Actor-critic** (Unit 6) |
|---|---|---|---|
| What is learned | $Q_\theta(s,a)$ or $V_\theta(s)$ | $\pi_\theta(a \mid s)$ | **Both** |
| Policy | Implicit, hand-specified (greedy / $\varepsilon$-greedy) | Learned, **stochastic** | Learned |
| Objective | **Minimize** TD loss to the target | **Maximize** $J(\theta)$ by gradient **ascent** | Mixed |
| On/off-policy | Typically **off**-policy (replay buffer) | Typically **on**-policy | Typically on-policy |
| Continuous actions | Painful ($\arg\max$ over a continuum) | **Natural** | Natural |
| Variance / bias | Lower variance, **biased** (bootstrapping) | **Unbiased**, high variance | Middle ground |

---

## 2. Advantages and disadvantages of policy-gradient methods

> "But Deep Q-Learning is excellent! Why use policy-gradient methods?"

### Advantages

**(a) The simplicity of integration.** We can estimate the policy **directly, without storing additional data** (action values). No Q-table, no replay buffer of value targets.

**(b) They can learn a *stochastic* policy** — value functions can't. Two consequences:

1. **No hand-built exploration/exploitation trade-off.** Since the output is a *probability distribution* over actions, the agent **explores the state space without always taking the same trajectory**. No $\varepsilon$ schedule to tune.
2. **No perceptual aliasing problem.** *Perceptual aliasing* is when **two states seem (or are) the same but require different actions.**

> **The vacuum-cleaner / hamster example.** An intelligent vacuum cleaner must suck the dust and avoid killing the hamsters, but it **can only perceive where the walls are**. Two states are **aliased** — in each of them the agent perceives an upper and a lower wall, so they look identical.
> - Under a **deterministic** policy, the agent will *always* move right in that percept, or *always* move left. **Either way it gets stuck and never sucks the dust.**
> - Under a **value-based** algorithm we learn a *quasi*-deterministic policy ($\varepsilon$-greedy), so the agent can **spend a lot of time** before finding the dust.
> - Under an **optimal stochastic** policy it will **randomly move left or right** in the aliased states, so **it does not get stuck** and reaches the goal with high probability.

**(c) More effective in high-dimensional and continuous action spaces.** Deep Q-Learning **assigns a score (maximum expected future reward) to each possible action** at each timestep. But with a self-driving car, at each state there is a (near) **infinite** choice of actions — turn the wheel 15°, 17.2°, 19.4°, honk, … We'd need to output a Q-value **for every possible action**, and **taking the max of a continuous output is itself an optimization problem**. Policy gradient just outputs a **distribution** over actions instead.

**(d) Better convergence properties.** Value-based methods use an **aggressive operator**: $\max$ over Q-estimates. So **action probabilities can change dramatically for an arbitrarily small change in the estimated action values**, if that change flips which action is maximal.

> **Course example:** during training the best action is *left* with $Q = 0.22$. One step later *right* becomes $0.23$. Tiny value change — but the policy **flips completely** and now takes *right* almost always.

By contrast, in policy-gradient methods the stochastic policy's **action preferences change smoothly over time**.

### Disadvantages

- **Frequently converges to a local maximum** instead of the global optimum.
- Goes **slower, step by step** — it can take longer to train (**sample-inefficient**).
- Can have **high variance** — the *why* and the *fix* come in the actor-critic unit.

### Advantages vs disadvantages at a glance

| **Advantages** | **Disadvantages** |
|---|---|
| Simple integration — no action values to store | Frequently converges to a **local** optimum, not global |
| Learns a **stochastic** policy ⇒ automatic exploration | **Slow**, step-by-step, sample-inefficient |
| No **perceptual aliasing** problem | **High variance** in the gradient estimate |
| Effective in **high-dimensional / continuous** action spaces | On-policy ⇒ data can't easily be reused |
| **Smooth** policy changes ⇒ better convergence behavior | Needs full episodes (for the Monte Carlo variant) |

📺 Deeper dive suggested by the course: [this video on the advantages and disadvantages of policy-gradient methods](https://youtu.be/y3oqOjHilio).

---

## 3. Diving deeper into policy-gradient methods

### The big picture

We have a **parameterized stochastic policy**: a neural network that outputs a probability distribution over actions. The probability of taking each action is also called the **action preference**.

For **CartPole-v1**:
- **Input:** a state (4 numbers).
- **Output:** a probability distribution over the 2 actions at that state.

The goal is to **control the probability distribution of actions** by tuning the policy so that **good actions (the ones that maximize the return) are sampled more frequently in the future**. Each time the agent interacts, we tweak $\theta$ in that direction.

**How do we optimize the weights using the expected return?** The trick is simple and slightly crude:

> Let the agent **interact for an entire episode**. If we *win* the episode, we consider **every action taken was good and should be sampled more** in the future — because they led to the win. So for each state-action pair we **increase $P(a \mid s)$**. If we lost, we **decrease** it.

Simplified policy-gradient loop:

$$
\text{collect episode with } \pi_\theta \;\longrightarrow\; \text{compute } R(\tau) \;\longrightarrow\; \text{if } R \text{ high: push } \uparrow P(a_t \mid s_t);\ \text{else push } \downarrow \;\longrightarrow\; \text{repeat}
$$

### The stochastic policy

We have a stochastic policy $\pi$ with parameter $\theta$. Given a state it **outputs a probability distribution over actions**, and

$$
\pi_\theta(a_t \mid s_t) = \text{probability of selecting action } a_t \text{ from state } s_t \text{ under our policy.}
$$

### The objective function

**But how do we know if our policy is good?** We need a way to measure it — a score/objective function $J(\theta)$.

The objective function gives the **performance of the agent** given a **trajectory** (a state-action sequence, *without* considering reward — contrary to an *episode*), and outputs the **expected cumulative reward**:

$$
J(\theta) \;=\; \mathbb{E}_{\tau \sim \pi_\theta}\big[R(\tau)\big] \;=\; \sum_{\tau} P(\tau; \theta)\, R(\tau)
$$

Reading the formula piece by piece:

| Symbol | Meaning |
|---|---|
| $R(\tau)$ | **Return** from an arbitrary trajectory $\tau$ (sum of, possibly discounted, rewards). Not a function of $\theta$. |
| $P(\tau; \theta)$ | **Probability of trajectory $\tau$** — depends on $\theta$, because $\theta$ defines the policy that selects the actions, which in turn affects which states get visited. |
| $J(\theta)$ | **Expected return**: sum over all trajectories of (probability of that trajectory given $\theta$) × (return of that trajectory). |

The **expected return** is thus a **weighted average of all values $R(\tau)$ can take**, with weights $P(\tau;\theta)$.

The trajectory probability factorizes as

$$
P(\tau; \theta) \;=\; \mu(s_0) \prod_{t=0}^{H} \underbrace{P(s_{t+1} \mid s_t, a_t)}_{\text{MDP dynamics (environment)}} \; \underbrace{\pi_\theta(a_t \mid s_t)}_{\text{our policy}}
$$

where $\mu(s_0)$ is the **initial state distribution**.

And our objective is

$$
\boxed{\;\theta^{*} = \arg\max_{\theta} J(\theta) = \arg\max_{\theta}\; \mathbb{E}_{\tau \sim \pi_\theta}\big[R(\tau)\big]\;}
$$

### Gradient ascent

Policy gradient is an **optimization problem**: find the $\theta$ maximizing $J(\theta)$. So we use **gradient ascent** — the inverse of gradient *descent*, since it gives the direction of **steepest increase** of $J(\theta)$:

$$
\boxed{\;\theta \leftarrow \theta + \alpha \, \nabla_\theta J(\theta)\;}
$$

Apply this repeatedly, hoping $\theta$ converges to the maximizer of $J(\theta)$.

### ⚠️ Two problems with computing $\nabla_\theta J(\theta)$

1. **We can't compute the true gradient.** It requires the probability of **each possible trajectory** — computationally astronomical. So we must **estimate the gradient from samples** (collect some trajectories).
2. **We can't differentiate the state distribution.** Differentiating $J(\theta)$ naively means differentiating $P(s_{t+1} \mid s_t, a_t)$ — the **Markov Decision Process dynamics**, which belong to the **environment**. It gives the probability of transitioning to the next state given the current state and action. **We can't differentiate it because we might not even know it.**

> **Key insight:** the escape hatch is the **Policy Gradient Theorem**, which reformulates the objective's gradient into a **differentiable expression that does not involve differentiating the state distribution** — only $\nabla_\theta \log \pi_\theta(a \mid s)$, which is *our own network* and therefore fully differentiable.

---

## 4. The Policy Gradient Theorem

$$
\boxed{\;\nabla_\theta J(\theta) \;=\; \mathbb{E}_{\tau \sim \pi_\theta}\Big[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, R(\tau)\Big]\;}
$$

### The intuition

Two factors multiply together:

- $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ is the direction of **steepest increase of the (log-)probability** of selecting action $a_t$ from state $s_t$. It tells us **how to change the weights** if we want to increase/decrease the log-probability of that action at that state.
- $R(\tau)$ is the **scoring function** (the "how good was this trajectory?" multiplier):
  - If the return is **high**, it **pushes up** the probabilities of those (state, action) combinations.
  - If the return is **low**, it **pushes down** the probabilities of those (state, action) combinations.

> **One-line intuition:** *increase the log-probability of the actions that appeared in high-return trajectories, in proportion to how good those trajectories were.*

Note the elegance: the environment dynamics have **vanished**. The only $\theta$-dependence left is inside our own softmax policy.

### Sketch of the derivation

Start from the definition and push the gradient inside the sum:

$$
\nabla_\theta J(\theta) = \nabla_\theta \sum_{\tau} P(\tau;\theta) R(\tau) = \sum_{\tau} \nabla_\theta P(\tau;\theta)\, R(\tau)
$$

(the $R(\tau)$ stays put because **it does not depend on $\theta$**).

**Step 2 — multiply by 1.** Multiply every term by $\dfrac{P(\tau;\theta)}{P(\tau;\theta)} = 1$:

$$
= \sum_{\tau} P(\tau;\theta)\, \frac{\nabla_\theta P(\tau;\theta)}{P(\tau;\theta)}\, R(\tau)
$$

**Step 3 — the log-derivative trick** (a.k.a. the **likelihood-ratio trick** or **REINFORCE trick**), a simple calculus identity:

$$
\nabla_x \log f(x) = \frac{\nabla_x f(x)}{f(x)}
$$

So $\dfrac{\nabla_\theta P(\tau;\theta)}{P(\tau;\theta)} = \nabla_\theta \log P(\tau;\theta)$, giving the **likelihood policy gradient**:

$$
\nabla_\theta J(\theta) = \sum_{\tau} P(\tau;\theta)\, \nabla_\theta \log P(\tau;\theta)\, R(\tau)
$$

**Step 4 — sample-based estimate.** Because the outer sum is now an expectation under $P(\tau;\theta)$, we can approximate it with $m$ sampled trajectories:

$$
\nabla_\theta J(\theta) \approx \frac{1}{m} \sum_{i=1}^{m} \nabla_\theta \log P(\tau^{(i)};\theta)\, R(\tau^{(i)})
$$

**Step 5 — simplify $\nabla_\theta \log P(\tau;\theta)$.** Expand the trajectory probability:

$$
\nabla_\theta \log P(\tau^{(i)};\theta) = \nabla_\theta \log\Big[\mu(s_0) \prod_{t=0}^{H} P\big(s_{t+1}^{(i)} \mid s_t^{(i)}, a_t^{(i)}\big)\, \pi_\theta\big(a_t^{(i)} \mid s_t^{(i)}\big)\Big]
$$

The **log of a product is the sum of the logs**:

$$
= \nabla_\theta \Big[\log \mu(s_0) + \sum_{t=0}^{H} \log P\big(s_{t+1}^{(i)} \mid s_t^{(i)}, a_t^{(i)}\big) + \sum_{t=0}^{H} \log \pi_\theta\big(a_t^{(i)} \mid s_t^{(i)}\big)\Big]
$$

and the **gradient of a sum is the sum of the gradients**. Now the crucial step: **neither the initial state distribution nor the MDP transition dynamics depend on $\theta$**, so their derivatives are **zero**:

$$
\nabla_\theta \log \mu(s_0) = 0, \qquad \nabla_\theta \sum_{t=0}^{H} \log P\big(s_{t+1}^{(i)} \mid s_t^{(i)}, a_t^{(i)}\big) = 0
$$

Drop them:

$$
\nabla_\theta \log P(\tau^{(i)};\theta) = \sum_{t=0}^{H} \nabla_\theta \log \pi_\theta\big(a_t^{(i)} \mid s_t^{(i)}\big)
$$

**Final formula for estimating the policy gradient:**

$$
\boxed{\;\nabla_\theta J(\theta) = \hat{g} = \frac{1}{m} \sum_{i=1}^{m} \sum_{t=0}^{H} \nabla_\theta \log \pi_\theta\big(a_t^{(i)} \mid s_t^{(i)}\big)\, R\big(\tau^{(i)}\big)\;}
$$

### Derivation in five lines (memory aid)

| Step | Move | Why it's legal |
|---|---|---|
| 1 | $\nabla \sum_\tau P R \to \sum_\tau (\nabla P) R$ | $R(\tau)$ is independent of $\theta$ |
| 2 | Multiply by $P/P$ | Equals 1 |
| 3 | $\dfrac{\nabla P}{P} \to \nabla \log P$ | **Log-derivative trick** |
| 4 | $\sum_\tau P(\cdot) \to \frac{1}{m}\sum_{i=1}^{m}(\cdot)$ | It's an expectation ⇒ Monte Carlo sampling |
| 5 | $\nabla \log P \to \sum_t \nabla \log \pi_\theta$ | log of product = sum of logs; **dynamics and $\mu(s_0)$ don't depend on $\theta$** |

---

## 5. REINFORCE (Monte Carlo policy gradient)

> **REINFORCE**, also called **Monte Carlo policy-gradient**, is a policy-gradient algorithm that **uses an estimated return from an entire episode to update the policy parameter $\theta$.**

### The loop

In a loop:
1. Use the policy $\pi_\theta$ to **collect an episode** $\tau$.
2. Use the episode to **estimate the gradient** $\hat{g} = \nabla_\theta J(\theta)$:

$$
\hat{g} = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, R(\tau)
$$

3. **Update the weights** of the policy:

$$
\theta \leftarrow \theta + \alpha\, \hat{g}
$$

Collecting **multiple episodes (trajectories)** gives a lower-variance estimate:

$$
\hat{g} = \frac{1}{m} \sum_{i=1}^{m} \sum_{t=0}^{H} \nabla_\theta \log \pi_\theta\big(a_t^{(i)} \mid s_t^{(i)}\big)\, R\big(\tau^{(i)}\big)
$$

### Pseudocode

```
1  Initialize the policy parameters θ (a neural network with softmax output)
2  loop forever:
3      for each episode:
4          Generate an episode  s0, a0, r1, s1, a1, r2, …, s_{T-1}, a_{T-1}, r_T   following π_θ
5          for each step t = 0 … T-1:
6              G_t ← Σ_{k=t+1}^{T} γ^{k-t-1} r_k          # return-to-go from t
7          L(θ) ← − Σ_t  log π_θ(a_t | s_t) · G_t          # negated: frameworks minimize
8          θ ← θ − α ∇_θ L(θ)     (≡ gradient ASCENT on J)
```

**Reward-to-go, not total return.** When computing $G_t$ at line 6 we sum the discounted rewards **starting at timestep $t$**. Why? Because the policy should only **reinforce actions on the basis of their consequences**: rewards obtained **before** an action were not caused by it, so they're useless as credit — **only what comes after matters**. (Spinning Up calls this "[don't let the past distract you](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#don-t-let-the-past-distract-you)".)

$$
G_t = r_{t+1} + \gamma G_{t+1}
$$

computed **backwards from the last timestep** — an $O(N)$ dynamic-programming pass instead of the naive $O(N^2)$.

### ⚠️ Why do we *minimize* a loss if we said gradient *ascent*?

We want to **maximize** $J(\theta)$, but PyTorch/TensorFlow optimizers **minimize**. So we minimize the negated objective:

$$
L(\theta) = -\sum_{t} \log \pi_\theta(a_t \mid s_t)\, G_t \qquad\Longrightarrow\qquad \nabla_\theta L = -\nabla_\theta J
$$

The course's version of the intuition: suppose we want to reinforce action $a_3$ whose probability is currently $0.25$. We want $\pi_\theta(a_3 \mid s) > 0.25$. Because all probabilities sum to 1, **maximizing $\pi_\theta(a_3 \mid s)$ minimizes the other action probabilities**. So we tell PyTorch to **minimize $1 - \pi_\theta(a_3 \mid s)$** — a loss that approaches 0 as the probability nears 1. Same gradient direction, framework-friendly sign.

### ⚠️ Unbiased but high variance

$R(\tau)$ (or $G_t$) is a **Monte Carlo estimate** of the return: it's the *actual* sampled return, not a bootstrapped estimate. Hence:

| | REINFORCE's gradient estimate |
|---|---|
| Bias | **Unbiased** — in expectation it *is* $\nabla_\theta J(\theta)$ |
| Variance | **High** — a single trajectory's return can differ wildly between episodes (stochastic policy × stochastic environment) |
| Consequence | Noisy, unstable updates; needs **many episodes** to average out |

Mitigations used already in the hands-on: **standardize the returns** (subtract the mean, divide by the std + $\varepsilon$) and **average over several trajectories**. The real fix — subtracting a learned **baseline / value function**, i.e. using the **advantage** — is **Actor-Critic**, Unit 6.

> **Key insight:** the MC/TD trade-off from Unit 2 reappears one level up. Value-based TD methods were *biased but low-variance*; REINFORCE is *unbiased but high-variance*. Actor-Critic is the deliberate compromise.

### REINFORCE vs Deep Q-Learning

| | **Deep Q-Learning (Unit 3)** | **REINFORCE (Unit 4)** |
|---|---|---|
| Learns | $Q_\theta(s,a)$ | $\pi_\theta(a \mid s)$ |
| Output layer | One value per action (linear) | **Softmax** over actions |
| Direction | Gradient **descent** on TD loss | Gradient **ascent** on $J(\theta)$ |
| Update timing | Every step, from a **replay buffer** | End of **episode** (Monte Carlo) |
| On/off-policy | **Off**-policy | **On**-policy |
| Exploration | Explicit $\varepsilon$-greedy schedule | **Built in** — sampling from the distribution |
| Estimate quality | Biased, low variance | **Unbiased, high variance** |
| Continuous actions | Awkward | Natural |

---

## 6. Hands-on: REINFORCE in PyTorch

Environments and libraries worth remembering:

- **[CartPole-v1](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)** — a pole attached by an un-actuated joint to a cart on a frictionless track; push the cart **left or right** so the pole stays in equilibrium. Reward **+1 per timestep** the pole stays up. Episode ends if the **pole angle exceeds ±12°**, the **cart position exceeds ±2.4**, or the **episode length exceeds 500**.
- **[Pixelcopter](https://pygame-learning-environment.readthedocs.io/en/latest/user/games/pixelcopter.html)** (from `gym-games`, built on PyGame) — observation space of **7** values (player y position, player velocity, distance to floor, distance to ceiling, next block x distance, next block top y, next block bottom y); action space of **2** (up = press accelerator, or do nothing). Reward **+1** per vertical block passed, **−1** on reaching a terminal state.
- Libraries: plain **Python + PyTorch** (no Stable-Baselines3 here — we write the algorithm ourselves), `gym` (not `gymnasium`, because `gym-games` isn't migrated: `env.step()` returns `state, reward, done, info` with a single `done` instead of `terminated`/`truncated`), `huggingface_hub`, `imageio`.

**Why start with CartPole?** From *RL Tips and Tricks*: when implementing an agent from scratch you must **be sure it works and find bugs on easy environments first**. "Try to have some *sign of life* on toy problems", then "validate by running on harder and harder envs". If CartPole fails, the bug is in **your integration**, not in the environment.

### The policy network

```python
class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)          # probability distribution over actions

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()                 # NOT argmax!
        return action.item(), m.log_prob(action)
```

> ⚠️ **The deliberate bug in the course.** The first version writes `action = np.argmax(m)`, which raises `ValueError: The value argument to log_prob must be a Tensor`. Two things are wrong: `argmax` of a distribution object isn't a tensor, **and more importantly it would always take the highest-probability action**. We need to **sample** from $P(\cdot \mid s)$ — `action = m.sample()` — otherwise the policy is no longer stochastic and we lose exploration entirely.

For Pixelcopter a **deeper** network is needed (`fc1: s→h`, `fc2: h→2h`, `fc3: 2h→a`, ReLU between, softmax out).

### The update, in code

```python
returns = torch.tensor(returns)
returns = (returns - returns.mean()) / (returns.std() + eps)   # variance reduction

policy_loss = [-log_prob * disc_return
               for log_prob, disc_return in zip(saved_log_probs, returns)]
policy_loss = torch.cat(policy_loss).sum()

optimizer.zero_grad()
policy_loss.backward()
optimizer.step()
```

### Hyperparameters worth remembering

| | **CartPole-v1** | **Pixelcopter** |
|---|---|---|
| `h_size` (hidden units) | **16** | **64** (net is 3 layers) |
| `n_training_episodes` | **1000** | **50000** |
| `max_t` | **1000** | **10000** |
| `gamma` | **1.0** | **0.99** |
| `lr` (Adam) | **1e-2** | **1e-4** |
| `n_evaluation_episodes` | 10 | 10 |
| Certification threshold | **≥ 350** | **≥ 5** |

The leaderboard score is **`mean_reward − std_reward`** — so a high-variance agent is penalized, which is itself a nice reminder of REINFORCE's core weakness.

---

## 7. In one screen

$$
\underbrace{\pi_\theta(a \mid s)}_{\text{softmax NN}} \;\longrightarrow\; \underbrace{J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]}_{\text{what we maximize}} \;\longrightarrow\; \underbrace{\hat g = \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) G_t}_{\text{policy gradient theorem + MC estimate}} \;\longrightarrow\; \underbrace{\theta \leftarrow \theta + \alpha \hat g}_{\text{gradient ascent}}
$$

- **Value-based** ⇒ learn values, derive the policy. **Policy-based** ⇒ learn the policy directly. **Policy-gradient** ⇒ do it with the gradient.
- The gradient of the objective **only needs our own policy's log-probabilities**; the unknown environment dynamics differentiate to zero.
- The return acts as a **scalar credit multiplier**: good trajectory ⇒ raise those action probabilities; bad ⇒ lower them.
- REINFORCE = policy gradient + **Monte Carlo** return ⇒ **unbiased, high variance**.

---

## 8. Glossary

- **Deep Q-Learning** — a **value-based** deep RL algorithm that uses a deep neural network to approximate Q-values for actions in a given state. Its goal is to find the optimal policy maximizing the expected cumulative reward **by learning the action-values**.
- **Value-based methods** — RL methods that estimate a **value function as an intermediate step** toward finding an optimal policy.
- **Policy-based methods** — RL methods that **directly learn to approximate the optimal policy without learning a value function**. In practice they output a **probability distribution over actions**. Benefits over value-based methods:
  - **simplicity of integration** — no need to store action values;
  - **ability to learn a stochastic policy** — the agent explores the state space without always taking the same trajectory, and avoids the problem of **perceptual aliasing**;
  - **effectiveness in high-dimensional and continuous action spaces**;
  - **improved convergence properties**.
- **Policy Gradient** — a **subset of policy-based methods** where the objective is to **maximize the performance of a parameterized policy using gradient ascent**. Its goal is to **control the probability distribution of actions** by tuning the policy such that **good actions (that maximize the return) are sampled more frequently in the future**.
- **Monte Carlo Reinforce** — a policy-gradient algorithm that **uses an estimated return from an entire episode to update the policy parameter** $\theta$.

**Extra terms from this unit's prose, worth adding to the glossary:**

- **Objective function $J(\theta)$** — the expected cumulative reward of the parameterized policy; the thing we maximize.
- **Trajectory $\tau$** — a state-action sequence, *without* considering rewards (contrary to an *episode*).
- **Action preference** — the probability the policy assigns to each action.
- **Gradient ascent** — the inverse of gradient descent; steps in the direction of steepest **increase** of $J(\theta)$.
- **Perceptual aliasing** — when two states seem (or are) the same but require **different** actions.
- **Log-derivative / likelihood-ratio / REINFORCE trick** — the identity $\nabla_x \log f(x) = \frac{\nabla_x f(x)}{f(x)}$, which converts $\frac{\nabla_\theta P}{P}$ into $\nabla_\theta \log P$.
- **MDP dynamics / state distribution** — $P(s_{t+1} \mid s_t, a_t)$; belongs to the environment, may be unknown, and **does not depend on $\theta$** (which is why it drops out of the policy gradient).
- **Reward-to-go** — using $G_t$ (rewards *after* time $t$) rather than the full-episode return, since earlier rewards weren't caused by $a_t$.

---

## 9. Self-check (quick review questions)

1. In value-based methods, why does the course say the policy **only exists because of** the action-value estimates?
2. State the **difference between policy-based and policy-gradient** methods in one sentence. Name two policy-based methods that are **not** policy-gradient.
3. Why are policy-based methods usually **on-policy**?
4. Write $J(\theta)$ from memory, both as an expectation and as an explicit sum. What are the two factors in the sum?
5. Write the **gradient ascent** update. How does it differ in sign from what you'd write in PyTorch, and why?
6. Name the **four advantages** and **three disadvantages** of policy-gradient methods.
7. What is **perceptual aliasing**? Walk through the vacuum-cleaner/hamster example and explain why a **deterministic** policy fails and a **stochastic** one succeeds.
8. Why is Deep Q-Learning awkward for a **continuous** action space? Name the two specific problems.
9. Explain the "$Q = 0.22$ vs $Q = 0.23$" example — what convergence property of value-based methods does it illustrate?
10. Write the **Policy Gradient Theorem** from memory. What do the two multiplied factors each contribute?
11. What are the **two problems** with differentiating $J(\theta)$ directly, and which one does the theorem solve?
12. Reproduce the derivation: what is the **log-derivative trick**, and at which step is it used?
13. **Why do the MDP dynamics vanish** from the final gradient formula? (Name the exact reason.)
14. Write out the REINFORCE loop in four steps. What exactly is $\hat{g}$?
15. Why use $G_t$ (**reward-to-go**) rather than $R(\tau)$ at each timestep?
16. Is REINFORCE's gradient estimate **biased**? Is it **low-variance**? Which unit fixes the problem, and how?
17. In the hands-on `act()` method, why is `m.sample()` correct and `np.argmax(m)` wrong — for **two** separate reasons?
18. Why does standardizing the returns help, and what is `eps` for?
19. Why start on **CartPole-v1** rather than Pixelcopter?
20. From memory: CartPole's `gamma` and `lr` vs Pixelcopter's. Why does Pixelcopter need a bigger network and 50× more episodes?

---

## 10. Additional Readings

These are **optional** if you want to go deeper.

**Introduction to Policy Optimization**
- [Part 3: Intro to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) — OpenAI *Spinning Up* documentation

**Policy Gradient**
- [Policy Gradients](https://johnwlambert.github.io/policy-gradients/) — John Lambert
- [RL — Policy Gradient Explained](https://jonathan-hui.medium.com/rl-policy-gradients-explained-9b13b688b146) — Jonathan Hui
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2020.pdf) — Richard Sutton & Andrew G. Barto, **Chapter 13, Policy Gradient Methods**

**Implementation**
- [PyTorch REINFORCE implementation](https://github.com/pytorch/examples/blob/main/reinforcement_learning/reinforce.py) — official PyTorch examples
- [Implementations from DDPG to PPO](https://github.com/MrSyee/pg-is-all-you-need) — *pg-is-all-you-need*, MrSyee
- [Udacity REINFORCE notebook](https://github.com/udacity/deep-reinforcement-learning/blob/master/reinforce/REINFORCE.ipynb) — one of the three implementations the hands-on is based on

---

> **Next up:** **Unit 5 — Introduction to Unity ML-Agents**, where we leave Gym behind and train agents inside a real game engine (Pyramids, SnowballTarget). Then **Unit 6 — Actor-Critic** finally solves REINFORCE's high-variance problem by combining the policy gradient with a learned value function (a baseline / advantage estimate).
