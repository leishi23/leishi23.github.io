---
layout: post
title:  "HF Deep RL Course — Unit 8 (Proximal Policy Optimization, PPO)"
date:   2026-08-14
desc: "HF Deep RL Course — Unit 8 (Proximal Policy Optimization, PPO)"
keywords: "Machine learning"
categories: [Machine learning]
tags: [Machine learning]
icon: icon-html
---

# HF Deep RL Course — Unit 8 (Proximal Policy Optimization, PPO)

> Source: [Hugging Face Deep RL Course, Unit 8](https://huggingface.co/learn/deep-rl-course/unit8/introduction) — notes from *Introduction* through *Additional Readings*, reorganized for review. Continues from [Unit 7]({{ '/machine learning/2026/08/14/HF_DeepRL_Unit7.html' | replace: ' ', '%20' | prepend: site.baseurl }}).

> **_Keypoints:_**

- Why **too-large policy updates** are dangerous: "falling off the cliff"
- The **ratio function** $r_t(\theta)$ as a cheap divergence estimate (vs TRPO's KL constraint)
- The **unclipped surrogate objective** and why it's unbounded
- The **clipped surrogate objective** $L^{CLIP}$, with $\epsilon = 0.2$
- The **six cases** of the clip (the crux of the unit) — where the gradient is zero and why
- The **asymmetry**: no reward for going too far in the good direction, still pushed back from the bad direction
- The **final combined loss**: clipped policy term − value loss + entropy bonus
- The **PPO loop**: rollout → GAE advantages → **K epochs of minibatch updates** on the same batch
- Two hands-on tracks: **CleanRL** (single-file PPO, LunarLander-v2) and **Sample Factory** (async PPO, VizDoom)
- Self-check + additional readings

---

## 0. What this unit is for

Unit 6 covered **Advantage Actor-Critic (A2C)**, a hybrid of value-based and policy-based methods that stabilizes training by reducing variance with:

- **An Actor** that controls **how the agent behaves** (policy-based).
- **A Critic** that measures **how good the action taken is** (value-based).

This unit covers **Proximal Policy Optimization (PPO)** — an architecture that **improves the agent's training stability by avoiding policy updates that are too large**. It does this with a **ratio** indicating the difference between the current and old policy, and **clips** that ratio to the range $[1-\epsilon,\, 1+\epsilon]$.

The unit is in **two parts**:

1. **Part 1** — the theory, then code PPO **from scratch** using the [CleanRL](https://github.com/vwxyzjn/cleanrl) implementation as a model, tested on **LunarLander-v2**. (Nice symmetry: LunarLander-v2 was the very first environment of the course — back then you didn't know how PPO worked; now you can write it yourself.)
2. **Part 2** — deeper PPO optimization with [Sample Factory](https://samplefactory.dev/), an **asynchronous implementation of PPO**, training an agent to play **VizDoom** (an open-source version of Doom).

---

## 1. The intuition behind PPO

The idea: **improve training stability by limiting how much the policy changes at each training epoch** — i.e. **avoid too large of a policy update**.

Two reasons:

1. **Empirically, smaller policy updates are more likely to converge** to an optimal solution.
2. **A too-big step can make you fall "off the cliff"** — you land on a bad policy, and it takes **a long time, or is outright impossible, to recover.**

The course's picture is a hiker on a ridge: taking huge strides toward "higher reward" can send you over the edge, whereas short, cautious strides climb reliably.

> **Key insight:** the collapse is not just "a bad update." A bad policy collects **bad data**, and since a policy-gradient method learns only from the data its own policy generates, a bad policy can be **self-reinforcing** — there is no replay buffer of good experience to fall back on.

**So with PPO we update the policy conservatively.** To do that we need to *measure* how much the current policy has changed relative to the former one — via a **ratio** between current and former policy — and then **clip that ratio into $[1-\epsilon, 1+\epsilon]$**, which **removes the incentive for the current policy to go too far from the old one** (hence *proximal* policy optimization).

---

## 2. Recap: the policy objective function, and the step-size problem

In REINFORCE the objective was

$$
L^{PG}(\theta) = \hat{\mathbb{E}}_t\Big[\, \log \pi_\theta(a_t \mid s_t)\, \hat{A}_t \,\Big]
$$

Taking a **gradient ascent** step on this (equivalently gradient descent on its negative) pushes the agent to **take actions leading to higher rewards and avoid harmful actions**.

But the problem is **the step size**:

| Step size | Consequence |
|---|---|
| **Too small** | the training process is **too slow** |
| **Too high** | **too much variability** in training — possible collapse |

And the step size in *parameter* space is not the same thing as the step size in *policy* space: a small change in $\theta$ can produce a huge change in $\pi_\theta$. That's exactly what we want to control.

PPO's answer is a new objective, the **Clipped Surrogate Objective function**, which **constrains the policy change to a small range using a clip** and is **designed to avoid destructively large weight updates**.

---

## 3. The ratio function

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}
$$

It is the **probability of taking action $a_t$ at state $s_t$ under the current policy, divided by the same probability under the previous policy**. So $r_t(\theta)$ denotes the **probability ratio between the current and old policy**:

| Value of $r_t(\theta)$ | Meaning |
|---|---|
| $r_t(\theta) > 1$ | the action $a_t$ at state $s_t$ is **more likely** under the current policy than the old one |
| $0 < r_t(\theta) < 1$ | the action is **less likely** under the current policy than the old one |
| $r_t(\theta) = 1$ | the two policies agree on this action — **no change** |

So the probability ratio is an **easy way to estimate the divergence between old and current policy**. Note $r_t(\theta_{old}) = 1$ by construction: at the start of every update pass, before any gradient step, the ratio is exactly 1 everywhere.

> **Key insight:** the ratio is the *importance-sampling weight*. It's what lets us reuse a batch collected under $\pi_{\theta_{old}}$ to estimate the gradient of $\pi_\theta$ — which is why PPO can do **several epochs** on the same data. It's also the *cheap* stand-in for an explicit KL divergence: no second-order optimization, just a division.

---

## 4. The unclipped surrogate objective — and why it isn't enough

The ratio **can replace the log-probability** used in the policy objective, giving the left half of PPO's new objective — the ratio times the advantage:

$$
L^{CPI}(\theta) = \hat{\mathbb{E}}_t\Big[\, r_t(\theta)\, \hat{A}_t \,\Big] = \hat{\mathbb{E}}_t\!\left[\, \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}\, \hat{A}_t \,\right]
$$

(from [*Proximal Policy Optimization Algorithms*](https://arxiv.org/pdf/1707.06347.pdf), Schulman et al.)

**However, without a constraint**, if an action is much more probable under the current policy than under the former one, **this leads to a significant policy-gradient step** and therefore **an excessive policy update**. Maximizing $r_t \hat{A}_t$ for a positive advantage is unbounded: the optimizer would happily drive $r_t \to \infty$, pushing that action's probability toward 1 based on a **single noisy advantage estimate**.

So we need to **penalize changes that lead to a ratio far away from 1**. Two ways to do that:

| | **TRPO (Trust Region Policy Optimization)** | **PPO** |
|---|---|---|
| Mechanism | **KL-divergence constraint** placed **outside** the objective function | **Clips the probability ratio directly inside** the objective |
| Enforcement | A hard trust region, solved with a constrained (second-order / conjugate-gradient) optimization | A soft, first-order penalty — plain SGD/Adam |
| Cost | **Complicated to implement** and **takes more computation time** | Cheap; a few extra lines of code |
| Divergence measure | KL between old and new policy | The ratio $r_t(\theta)$ as a proxy |
| Practical use | Rare outside research | The default policy-gradient method |

---

## 5. The clipped surrogate objective

$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\Big[\min\big(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon)\,\hat{A}_t\big)\Big]
$$

**By clipping the ratio, we ensure we do not have a too-large policy update, because the current policy can't be too different from the old one.**

Two pieces:

- **`clip`** — the clipped ratio is $r_t(\theta)$ **forced into the interval $[1-\epsilon,\, 1+\epsilon]$**: anything below $1-\epsilon$ becomes $1-\epsilon$, anything above $1+\epsilon$ becomes $1+\epsilon$. $\epsilon$ is the hyperparameter defining the clip range — **in the paper $\epsilon = 0.2$**, so **the ratio can only vary from 0.8 to 1.2**.
- **`min`** — we take the **minimum of the clipped and non-clipped objective**, so the final objective is a **lower bound (a pessimistic bound) of the unclipped objective**. Taking the minimum means **we select either the clipped or the non-clipped objective depending on the ratio-and-advantage situation**.

> ⚠️ Do not confuse the two operations. **`clip` alone would not work.** Clipping only makes the objective *flat* outside the interval; the `min` is what decides *when* that flatness applies. Without the `min`, a ratio pushed far past $1+\epsilon$ with a **negative** advantage would sit on a flat line and never be pulled back.

---

## 6. Visualizing the clipped objective: the six cases

This is the crux of the unit. There are **six situations**, from the crossing of two binary/ternary choices: the sign of the advantage $\hat{A}_t$ (positive or negative) × where the ratio sits (below, inside, or above the interval). Remember throughout: **we take the minimum of the clipped and unclipped objectives.**

*(The course reproduces the table from Daniel Bick's thesis; below is the same content typeset.)*

### The six cases in one table

| # | $\hat{A}_t$ | Ratio $r_t(\theta)$ | Which term is the `min`? | Gradient | What happens |
|---|---|---|---|---|---|
| **1** | $> 0$ (good action) | inside $[1-\epsilon,\, 1+\epsilon]$ | unclipped | **non-zero** | **Increase** $\pi(a_t \mid s_t)$ — clipping does not apply |
| **2** | $< 0$ (bad action) | inside $[1-\epsilon,\, 1+\epsilon]$ | unclipped | **non-zero** | **Decrease** $\pi(a_t \mid s_t)$ — clipping does not apply |
| **3** | $> 0$ | **below** $1-\epsilon$ | unclipped | **non-zero** | We moved too far *down* on a good action ⇒ **increase** the probability (moving back toward the range) |
| **4** | $< 0$ | **below** $1-\epsilon$ | **clipped** | **= 0** | Already much less likely than the old policy, and the action is bad — **we don't want to decrease further**. Flat line ⇒ **no weight update** |
| **5** | $> 0$ | **above** $1+\epsilon$ | **clipped** | **= 0** | Already much more likely than the old policy — **don't get too greedy**. Flat line ⇒ **no weight update** |
| **6** | $< 0$ | **above** $1+\epsilon$ | unclipped | **non-zero** | We moved too far *up* on a bad action ⇒ **decrease** the probability (moving back toward the range) |

### Case 1 and 2 — the ratio is inside the range

**Clipping does not apply**, since $r_t(\theta) \in [1-\epsilon, 1+\epsilon]$.

- **Situation 1** — **positive advantage**: the action is **better than the average** of all actions in that state, so we should **encourage** the current policy to **increase** the probability of taking it. Since the ratio is between the intervals, **we can increase it.**
- **Situation 2** — **negative advantage**: the action is **worse than the average** of all actions at that state, so we should **discourage** it. Since the ratio is between the intervals, **we can decrease it.**

This is the "business as usual" regime: PPO behaves like an ordinary advantage-weighted policy gradient.

### Case 3 and 4 — the ratio is below the range

If $r_t(\theta) < 1-\epsilon$, the probability of taking that action at that state is **much lower than under the old policy**.

- **Situation 3** — the advantage estimate is **positive** ($\hat{A}_t > 0$): **you want to increase** the probability of taking that action at that state. The update is allowed, because it moves us **back toward** the trust region.
- **Situation 4** — the advantage estimate is **negative**: **we don't want to decrease further** the probability of that action. Therefore the **gradient is 0** (we're on a flat line), so **we don't update our weights.**

### Case 5 and 6 — the ratio is above the range

If $r_t(\theta) > 1+\epsilon$, the probability of taking that action at that state under the current policy is **much higher than under the former policy**.

- **Situation 5** — the advantage is **positive**: **we don't want to get too greedy.** We already have a higher probability of taking that action than the former policy did. Therefore the **gradient is 0** (flat line), so **we don't update our weights.**
- **Situation 6** — the advantage is **negative**: **we want to decrease** the probability of taking that action at that state. The update is allowed — again, it moves us **back toward** the range.

### The rule, distilled

> **We only update the policy with the unclipped objective part.** When the minimum picks the *clipped* part, we don't update the policy weights, because the gradient will equal 0.

So **we update our policy only if**:

- the ratio is **in the range** $[1-\epsilon,\, 1+\epsilon]$, **or**
- the ratio is **outside** the range, **but the advantage leads to getting closer to the range**:
  - **below** the range but the advantage is $> 0$ (case 3),
  - **above** the range but the advantage is $< 0$ (case 6).

### Why is the gradient exactly zero when the clipped term wins?

Because when the ratio is clipped, the derivative is **not** the derivative of $r_t(\theta)\hat{A}_t$, but the derivative of either $(1-\epsilon)\hat{A}_t$ or $(1+\epsilon)\hat{A}_t$ — **both of which are 0 with respect to $\theta$**, since neither expression contains $\theta$ anymore. The clip has cut the gradient path.

### The asymmetry — the single most important takeaway

| Direction of the move | What the objective does |
|---|---|
| Too far in the **"good" direction** (case 5: $\hat{A}_t > 0$, $r_t > 1+\epsilon$) | **No further reward.** Gradient 0 — the objective refuses to pay you for over-committing. We are **not** pulled back either; we simply stop. |
| Too far in the **"bad" direction** (case 6: $\hat{A}_t < 0$, $r_t > 1+\epsilon$) | **Still penalized.** Gradient non-zero — we **are** pushed back toward the range. |
| Too far **down** on a **good** action (case 3) | **Still pushed back up** toward the range. |
| Too far **down** on a **bad** action (case 4) | **No further reward** for going even lower. Gradient 0. |

The `min` is what creates this asymmetry: it makes the objective a **pessimistic lower bound**, so **exceeding the trust region never buys you anything**, while **corrections that bring you back inside are always permitted**. To summarize: thanks to the clipped surrogate objective, **we restrict the range that the current policy can vary from the old one**, because we **remove the incentive for the ratio to move outside the interval** — the clip forces the gradient to zero out there.

---

## 7. The final combined PPO loss (Actor-Critic style)

The full loss PPO actually optimizes is a **combination of the clipped surrogate objective, a value loss, and an entropy bonus**:

$$
L_t(\theta) = \hat{\mathbb{E}}_t\Big[\, L^{CLIP}_t(\theta) \;-\; c_1\, L^{VF}_t(\theta) \;+\; c_2\, S[\pi_\theta](s_t) \,\Big]
$$

| Term | What it is | Why it's there |
|---|---|---|
| $L^{CLIP}_t(\theta)$ | The **clipped surrogate policy objective** from §5 | Improves the policy, conservatively (the Actor) |
| $-\,c_1 L^{VF}_t(\theta)$ | The **value-function loss**, a squared error $L^{VF}_t = (V_\theta(s_t) - V_t^{\text{targ}})^2$; **subtracted** because we *maximize* $L_t$ but want to *minimize* this error | Trains the Critic, so the advantage estimates get better |
| $+\,c_2 S[\pi_\theta](s_t)$ | The **entropy bonus** — the entropy of the policy's action distribution | **Encourages exploration**: rewards keeping the distribution spread out, delaying premature collapse to a deterministic policy |

$c_1$ and $c_2$ are coefficients (in the CleanRL defaults, `vf-coef = 0.5` and `ent-coef = 0.01`).

> **Key insight:** the sign convention matters and is a classic source of bugs. In code you *minimize* `pg_loss + c1 * v_loss - c2 * entropy`, which is exactly $-L_t(\theta)$. Also note the actor and critic here typically **share nothing but the optimizer** (or share an encoder), which is why the coefficients are needed to balance the two gradients.

---

## 8. The PPO algorithm loop

Putting it together, one PPO iteration looks like:

1. **Collect a rollout** with the current policy $\pi_{\theta_{old}}$: run $N$ parallel environments for $T$ steps each, storing observations, actions, log-probs, rewards, dones, and value estimates. Batch size $= N \times T$.
2. **Compute the advantages** $\hat{A}_t$ — CleanRL uses **GAE (Generalized Advantage Estimation)**, computed backwards through the rollout:

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \qquad
\hat{A}_t = \delta_t + \gamma\lambda\, \hat{A}_{t+1}
$$

   and the value targets are $\text{returns}_t = \hat{A}_t + V(s_t)$. The $\lambda$ knob (default **0.95**) trades bias against variance: $\lambda = 0$ gives one-step TD, $\lambda = 1$ gives Monte-Carlo.
3. **Freeze the old log-probs** — these define $\pi_{\theta_{old}}$ for the whole update phase.
4. **Do $K$ epochs of minibatch updates** on that *same* batch: shuffle indices, split into minibatches, recompute $\log \pi_\theta$, form the ratio $r_t = \exp(\log\pi_\theta - \log\pi_{\theta_{old}})$, and take a gradient step on $L_t(\theta)$.
5. **Discard the batch**, set $\theta_{old} \leftarrow \theta$, and go back to step 1.

> **Key insight:** step 4 is *why PPO exists.* Vanilla policy gradient must throw away each batch after a single gradient step, because the data is only valid for the policy that generated it. The ratio + clip let PPO squeeze **$K$ epochs** out of one batch while keeping the policy provably close to the one that collected the data — **much better sample efficiency while remaining essentially on-policy**. As soon as the ratio drifts past $1\pm\epsilon$, the clip switches those samples off.

In the CleanRL code this is literally:

```python
ratio = (newlogprob - b_logprobs[mb_inds]).exp()
pg_loss1 = -mb_advantages * ratio
pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
pg_loss  = torch.max(pg_loss1, pg_loss2).mean()   # max of negatives == min of the objective
```

Note `torch.max` of the two **negated** terms is exactly the `min` of the objective.

---

## 9. Hands-on track 1: CleanRL (PPO from scratch, LunarLander-v2)

The best way to understand an architecture is to **implement it from scratch** — done already for a value-based method (Q-Learning) and a policy-based one (REINFORCE).

Resources the course uses:
- A tutorial by **[Costa Huang](https://costa.sh/)**, author of **[CleanRL](https://github.com/vwxyzjn/cleanrl)** — a Deep RL library with **high-quality single-file implementations** and research-friendly features. Video: [https://youtu.be/MEt6rrxH8W4](https://youtu.be/MEt6rrxH8W4).
- **[The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)** (of which 13 are "core") — PPO's reported performance depends heavily on these details, not just the equations.

Environment: **[LunarLander-v2](https://www.gymlibrary.dev/environments/box2d/lunar_lander/)** — the very first environment in the course. Finally, push the trained model to the Hub with a replay video and an evaluation score.

### Hyperparameters worth remembering (CleanRL PPO defaults)

| Argument | Default | Meaning |
|---|---|---|
| `--learning-rate` | `2.5e-4` | Adam LR (with `eps=1e-5`) |
| `--anneal-lr` | `True` | **Linearly decay** the LR to 0 over training |
| `--num-envs` | `4` | Parallel (vectorized) environments |
| `--num-steps` | `128` | Rollout length per env ⇒ batch size $= 4 \times 128 = 512$ |
| `--gamma` | `0.99` | Discount factor |
| `--gae` / `--gae-lambda` | `True` / `0.95` | Use GAE, with $\lambda = 0.95$ |
| `--num-minibatches` | `4` | Minibatch size $=$ batch size $/\,4$ |
| `--update-epochs` | `4` | **$K = 4$ epochs** over each collected batch |
| `--norm-adv` | `True` | Normalize advantages **per minibatch** |
| `--clip-coef` | `0.2` | This is **$\epsilon$** |
| `--clip-vloss` | `True` | Also clip the **value** loss, as in the paper |
| `--ent-coef` | `0.01` | $c_2$, the entropy bonus coefficient |
| `--vf-coef` | `0.5` | $c_1$, the value-loss coefficient |
| `--max-grad-norm` | `0.5` | **Global gradient-norm clipping** |
| `--target-kl` | `None` | Optional **early stop** of the epoch loop if approx-KL exceeds it |

Diagnostics the implementation logs that are worth knowing: **`approx_kl`** (an estimate $\mathbb{E}[(r-1) - \log r]$, from [Schulman's KL-approximation note](http://joschu.net/blog/kl-approx.html)) and **`clipfrac`** (fraction of samples with $\lvert r - 1 \rvert > \epsilon$). A `clipfrac` near 0 means the updates are timid; a very high one means the policy is straining against the trust region.

---

## 10. Hands-on track 2: Sample Factory (async PPO, VizDoom)

**[Sample Factory](https://www.samplefactory.dev/)** is one of the **fastest RL libraries**, focused on very efficient **synchronous and asynchronous** implementations of policy gradients (PPO). It reaches **SOTA performance in a variety of domains while minimizing training time and hardware requirements**.

**How it works:** it **spawns multiple processes** running **rollout workers, inference workers, and a learner worker**. The workers **communicate through shared memory**, which lowers inter-process communication cost.

- **Rollout workers** interact with the environment and send observations to the inference workers.
- **Inference workers** query a **fixed version of the policy** and send actions back to the rollout workers.
- After **$k$ steps**, rollout workers send a **trajectory of experience** to the **learner worker**, which uses it to update the policy network.

> ⚠️ Because the inference workers use a *slightly stale* copy of the policy, the data arriving at the learner is **off-policy by a few updates**. This is exactly the regime the clipped ratio is built to tolerate — asynchronous PPO is only sane *because* of the trust region.

**Actor-Critic models in Sample Factory** have three components:

| Component | Role |
|---|---|
| **Encoder** | Processes input observations (images, vectors) into a vector — the part you'd most likely customize |
| **Core** | Integrates vectors from one or more encoders; can optionally include a single/multi-layer **LSTM/GRU** for a memory-based agent |
| **Decoder** | Extra layers on the core's output before computing the policy and value heads |

Other key features: **serial (single-process) mode** for easy debugging; CPU-based and **GPU-accelerated** environments; single- and multi-agent training, **self-play**, multiple policies at once on one or many GPUs; **Population-Based Training (PBT)**; discrete, continuous and hybrid action spaces; vector, image and dictionary observation spaces; automatic model construction by parsing the action/observation spaces; WandB/TensorBoard summaries; and Hugging Face Hub integration.

**[ViZDoom](https://vizdoom.cs.put.edu.pl/)** is an **open-source Python interface to the Doom engine**, created in 2016 by Marek Wydmuch and Michał Kempka at Poznań University of Technology. It allows **training agents directly from screen pixels** across many scenarios (up to team deathmatch). Being a 90s game engine, it runs at **accelerated speeds on modern hardware**, so complex behaviors can be learned fairly quickly.

**The scenario trained in the notebook: `doom_health_gathering_supreme`.** The objective is to **teach the agent to survive without knowing what makes it survive**. The agent knows only that **life is precious and death is bad**, so it must **learn what prolongs its existence and that its health is connected to survival**. The map is a rectangle with walls and a **green, acidic floor that periodically hurts the player**; **medkits** are spread over the map and new ones fall from the sky. Medkits heal part of the player's health, so **to survive the agent must pick them up**. The episode ends on death or timeout. Certification target: **mean reward $\geq 5$**.

Notable run configuration: `--num_workers=8 --num_envs_per_worker=4 --train_for_env_steps=4000000`. The optional harder challenge is **`doom_deathmatch_bots`**, which **takes many hours on a machine beefier than Colab** — the course instead has you download a pretrained checkpoint (`edbeeching/doom_deathmatch_bots_2222`) and watch it play.

The Part-2 hands-on was written by **[Edward Beeching](https://twitter.com/edwardbeeching)**, ML Research Scientist at Hugging Face (also behind Godot RL Agents).

---

## 11. Putting PPO in context

### PPO vs A2C vs REINFORCE

| | **REINFORCE** | **A2C** | **PPO** |
|---|---|---|---|
| Family | Pure policy gradient (Monte Carlo) | Actor-Critic (hybrid) | Actor-Critic (hybrid) |
| Objective weight on $\log \pi$ | Return $G_t$ | Advantage $A_t = Q(s,a) - V(s)$ | Advantage $\hat{A}_t$ **× the clipped ratio** |
| Needs full episodes? | **Yes** | No (bootstraps with the Critic) | No (GAE over a fixed-length rollout) |
| Variance | **Very high** | Reduced by the Critic baseline | Reduced by the Critic + advantage normalization |
| Gradient steps per batch | **1** | **1** | **$K$ epochs of minibatches** |
| Guards against huge updates? | No | No | **Yes — the clip** |
| Sample efficiency | Lowest | Middle | **Best of the three** (data reused $K$ times) |
| On-policy? | Yes | Yes | Yes, but tolerates **slightly stale** data |

### TRPO's KL constraint vs PPO's clipping

| | **TRPO** | **PPO** |
|---|---|---|
| Constraint form | $\hat{\mathbb{E}}_t[\,D_{KL}(\pi_{\theta_{old}} \Vert \pi_\theta)\,] \leq \delta$, a **hard constraint outside the objective** | $\min(\cdot,\, \text{clip}(\cdot))$, **inside the objective** |
| Optimization | Constrained / second-order (Fisher-vector products, conjugate gradient, line search) | **Unconstrained first-order** — Adam, as usual |
| Guarantee | A genuine monotonic-improvement bound | Heuristic, but empirically strong |
| Per-state vs per-batch | Constrains the **average** KL over the batch | Clips **per-sample** ratios |
| Implementation cost | **Complicated, more computation time** | A handful of lines |
| Verdict | Elegant theory, painful engineering | The practical default |

### One-line summary

$$
\theta \leftarrow \arg\max_{\theta} \; \hat{\mathbb{E}}_t\Big[\min\big(r_t\hat{A}_t,\ \text{clip}(r_t, 1\!-\!\epsilon, 1\!+\!\epsilon)\hat{A}_t\big) - c_1 (V_\theta - V^{\text{targ}})^2 + c_2 S[\pi_\theta]\Big]
$$

**Mnemonic for the clip:** *"Never pay for going too far; always allow coming back."*

---

## 12. Self-check (quick review questions)

1. Give the **two reasons** the course gives for avoiding too-large policy updates. What is the "cliff"?
2. Write $r_t(\theta)$ from memory. What is its value **before** any gradient step of an update phase, and why?
3. What does $r_t > 1$ mean? $0 < r_t < 1$?
4. What does the ratio **replace** in the original REINFORCE objective?
5. Write the **unclipped** surrogate objective. Precisely why is it dangerous?
6. Write $L^{CLIP}(\theta)$ from memory. What is $\epsilon$ in the paper, and what range does the ratio end up confined to?
7. Explain the **separate** jobs of `clip` and `min`. What would break if you kept only `clip`?
8. Fill in the **six cases** table from memory: for each combination of $\text{sign}(\hat{A}_t)$ and (below / inside / above the range), is the gradient zero or not, and which way does the probability move?
9. In cases **4** and **5** the gradient is exactly 0. Show *algebraically* why.
10. State the **asymmetry** in one sentence: what do we not get rewarded for, and what do we still get pushed back from?
11. Under what **two conditions** do we actually update the policy?
12. Write the **full** PPO loss with its three terms. What does each do, and why is the value term **subtracted**?
13. Why does the entropy bonus help, and what failure mode does it delay?
14. Describe the PPO loop in five steps. What exactly makes $K$ epochs on one batch legitimate?
15. What is **GAE**, and what does $\lambda$ control? What do $\lambda = 0$ and $\lambda = 1$ correspond to?
16. Contrast **TRPO's KL constraint** with **PPO's clipping** on: mechanism, optimization order, and implementation cost.
17. What do **`approx_kl`** and **`clipfrac`** tell you during training? What does a `clipfrac` of ~0 suggest?
18. Why is *asynchronous* PPO (Sample Factory) able to learn from slightly stale trajectories at all?

---

## 13. Additional Readings

These are **optional readings** if you want to go deeper.

**PPO Explained**
- [Towards Delivering a Coherent Self-Contained Explanation of Proximal Policy Optimization](https://fse.studenttheses.ub.rug.nl/25709/1/mAI_2021_BickD.pdf) — Daniel Bick (**especially part 3.4** — the source of the six-case table, and the course's top recommendation)
- [What is the way to understand Proximal Policy Optimization Algorithm in RL?](https://stackoverflow.com/questions/46422845/what-is-the-way-to-understand-proximal-policy-optimization-algorithm-in-rl) — Stack Overflow
- [Foundations of Deep RL Series, L4: TRPO and PPO](https://youtu.be/KjWF8VIMGiY) — Pieter Abbeel
- [OpenAI PPO Blogpost](https://openai.com/blog/openai-baselines-ppo/)
- [Spinning Up RL: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — OpenAI
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — Schulman, Wolski, Dhariwal, Radford, Klimov (2017) — **the PPO paper**
- [RL — Proximal Policy Optimization (PPO) Explained](https://jonathan-hui.medium.com/rl-proximal-policy-optimization-ppo-explained-77f014ec3f12) — Jonathan Hui (source of the "cliff" figure)

**PPO Implementation details**
- [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) — Huang et al., ICLR Blog Track
- [Part 1 of 3 — PPO Implementation: 11 Core Implementation Details](https://www.youtube.com/watch?v=MEt6rrxH8W4) — Costa Huang

**Importance Sampling**
- [Importance Sampling Explained](https://youtu.be/C3p2wI4RAi8)

**Libraries used in the hands-on**
- [CleanRL](https://github.com/vwxyzjn/cleanrl) — single-file, research-friendly Deep RL implementations
- [Sample Factory](https://samplefactory.dev/) — high-throughput synchronous & asynchronous PPO
- [ViZDoom](https://vizdoom.cs.put.edu.pl/) — Doom engine as an RL environment

---

> **Next up:** the **bonus units and advanced topics** — model-based RL, **offline vs online RL** and **Decision Transformers** (treating RL as sequence modeling), generalization in RL, curriculum learning, and **RLHF (Reinforcement Learning from Human Feedback)**, where the PPO you just learned becomes the workhorse for aligning language models.
