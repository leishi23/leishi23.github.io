---
layout: post
title:  "Theory and Methods for Reinforcement Learning — Lec.4 & 5"
date:   2025-09-23
desc: "Theory and Methods for Reinforcement Learning — Lec.4 & 5"
keywords: "Machine learning"
categories: [Machine learning]
tags: [Machine learning]
icon: icon-html
---

# Lecture Notes: Policy Gradient (PG) I & II — From PG Theorem to NPG, TRPO, PPO, and OPPO

> Goal: Convert the two slide decks (Lecture 4: PG-I and Lecture 5: PG-II) into structured Markdown notes that comprehensively summarize the content, explain the math in depth, and clearly show how the ideas connect.

---

## Knowledge Map (at a glance)

**MDP & Value Functions**  
→ define $V^\pi, Q^\pi$, **advantage** $A^\pi = Q^\pi - V^\pi$; **discounted state visitation** $\lambda_\mu^\pi$ (a.k.a. discounted occupancy measure)  
→ **Performance Difference Lemma (PDL)**: compare two policies’ returns via advantages  
→ **Policy Gradient Theorem**: multiple equivalent forms (REINFORCE/Q-form, advantage with baselines, state-visitation form)  
→ **Mirror Descent view**: KL-regularized policy mirror ascent (PMA), exponential-weights update  
→ **Natural Policy Gradient (NPG)**: Fisher information, KL quadratic approx, trust region equivalence; least-squares solution  
→ **Convergence & complexity**: vanilla PG vs NPG rates; exploration hardness constant $\kappa$, statistical error $\epsilon_{\text{stat}}$  
→ **Trust-region & proximal methods**: TRPO, PPO (clipped), OPPO (optimistic PPO for finite horizon)

---

## Prerequisites (what you need before this lecture)

- **Finite discounted MDP** with discount $\gamma \in (0,1)$.
- **Value functions**:
$$
V^\pi(s)=\mathbb{E}\!\left[\sum_{t\ge0}\gamma^t r_t \,\middle|\, s_0=s,\pi\right],\quad
Q^\pi(s,a)=\mathbb{E}\!\left[\sum_{t\ge0}\gamma^t r_t \,\middle|\, s_0=s,a_0=a,\pi\right],\quad
A^\pi=Q^\pi-V^\pi.
$$

- **Discounted state visitation** (from start distribution $\mu$):
$$
\lambda_\mu^\pi(s)=(1-\gamma)\,\mathbb{E}\!\left[\sum_{t\ge0}\gamma^t \mathbf{1}\{s_t=s\}\,\middle|\, s_0\sim\mu,\,\pi\right].
$$

- **KL divergence, Bregman divergence, Mirror Descent**: used to interpret KL-regularized updates and NPG geometry.

---

# Part I (PG-I): Foundations of Policy Gradient

## 1. Policy Gradient Theorem — forms and derivations

### 1.1 REINFORCE / Q-form
**Statement.** For a differentiable stochastic policy $\pi_\theta$,
$$
\nabla_\theta J(\pi_\theta)
=\mathbb{E}_{\tau\sim p_\theta}\!\left[\sum_{t=0}^{\infty}\gamma^t\, Q^{\pi_\theta}(s_t,a_t)\,\nabla_\theta \log \pi_\theta(a_t\mid s_t)\right].
$$

**Sketch of derivation (key ideas).**
1. Differentiate trajectory probability $p_\theta(\tau)$ via the log-likelihood trick:
$$
\nabla_\theta \mathbb{E}_{\tau}[R(\tau)] = \mathbb{E}_{\tau}[R(\tau)\nabla_\theta\log p_\theta(\tau)].
$$
2. Only policy terms depend on $\theta$: $\log p_\theta(\tau)=\sum_t \log \pi_\theta(a_t\mid s_t)$.
3. Move from cumulative return to $Q^\pi$ using the tower property.

**Meaning.** Weight the score function $\nabla\log\pi$ by the long-term payoff $Q^\pi$ at the visited state-action.

---

### 1.2 Advantage form and baselines
Replace $Q^\pi$ by the **advantage** $A^\pi$ without bias:
$$
\nabla_\theta J(\pi_\theta)
=\frac{1}{1-\gamma}\,\mathbb{E}_{s\sim \lambda_\mu^{\pi_\theta},\,a\sim\pi_\theta}
\!\left[A^{\pi_\theta}(s,a)\,\nabla_\theta\log\pi_\theta(a\mid s)\right].
$$

**Baselines.** For any function $b(s)$,
$$
\mathbb{E}_{a\sim\pi(\cdot|s)}\!\big[\nabla_\theta\log\pi_\theta(a|s)\,b(s)\big]=0.
$$
Thus replacing $Q^\pi$ by $A^\pi=Q^\pi-b$ reduces variance while keeping the gradient **unbiased**.

---

### 1.3 State-visitation (direct policy) form
For **tabular direct parameterization**,
$$
\frac{\partial J(\pi)}{\partial \pi(a|s)}=\frac{1}{1-\gamma}\,\lambda_\mu^\pi(s)\,Q^\pi(s,a).
$$
For **softmax parameterization** $\pi_\theta(a|s)\propto \exp(\theta_{s,a})$,
$$
\frac{\partial J(\theta)}{\partial \theta_{s,a}}
=\frac{1}{1-\gamma}\,\lambda_\mu^{\pi_\theta}(s)\,\pi_\theta(a|s)\,A^{\pi_\theta}(s,a).
$$

**Meaning.** Improvement comes from (i) visiting helpful states and (ii) increasing probability of high-advantage actions there.

---

### 1.4 Performance Difference Lemma (PDL)
For any two policies $\pi,\pi'$,
$$
J(\pi)-J(\pi')
=\frac{1}{1-\gamma}\,
\mathbb{E}_{s\sim \lambda_\mu^\pi,\,a\sim\pi(\cdot|s)}\!\left[A^{\pi'}(s,a)\right].
$$

**Interpretation.** Relative performance is the new policy’s **advantage under the old critic**, reweighted by the **new** occupancy.

---

## 2. Mirror Descent view & Policy Mirror Ascent (PMA)

**Mirror Descent** with a strongly convex potential $\omega$ and Bregman $D_\omega$:
$$
x_{t+1}=\arg\min_{x\in\mathcal{X}}
\left\{\langle\nabla f(x_t),x-x_t\rangle+\frac{1}{\eta_t}D_\omega(x,x_t)\right\}.
$$

**On the policy simplex** with negative entropy (so $D_\omega$ is **KL**), we get **Policy Mirror Ascent**:
$$
\pi_{t+1}=\arg\max_{\pi}\Big\{
\langle\nabla J(\pi_t),\pi\rangle
-\frac{1}{\eta}\sum_s \lambda_{\mu}^{\pi_t}(s)\,\mathrm{KL}\big(\pi(\cdot|s)\,\|\,\pi_t(\cdot|s)\big)\Big\}.
$$

This yields a **per-state exponential-weights update**:
$$
\pi_{t+1}(a|s)\;\propto\;\pi_t(a|s)\,
\exp\!\left(\frac{\eta}{1-\gamma}\,Q_t(s,a)\right).
$$

**Takeaway.** Mirror ascent with a KL proximity term naturally leads to logit/softmax-style updates and connects to NPG.

---

## 3. Vanilla PG under softmax & basic convergence

**Update.**
$$
\theta_{t+1}=\theta_t+\eta\,\nabla_\theta J(\pi_{\theta_t}).
$$

**Gradient component.**
$$
\frac{\partial J}{\partial \theta_{s,a}}=\frac{1}{1-\gamma}\,\lambda^\pi_\mu(s)\,\pi(a|s)\,A^\pi(s,a).
$$

**Convergence (clean setting).** With exact gradients and a small enough step size, vanilla PG attains an $O(1/T)$ suboptimality bound in tabular softmax, but with **large constants** and sensitivity to **exploration** and **logit saturation**.

---

# Part II (PG-II): Natural Policy Gradient & Trust-Region Methods

## 4. Fisher information, KL, and Natural Gradient

### 4.1 Fisher matrix and KL quadratic approximation
For a distribution family $p_\theta$,
$$
F_\theta=\mathbb{E}_{z\sim p_\theta}\!\left[\nabla_\theta\log p_\theta(z)\nabla_\theta\log p_\theta(z)^\top\right]
=\nabla_\theta^2\,\mathrm{KL}\!\left(p_{\theta_0}\,\|\,p_\theta\right)\Big|_{\theta=\theta_0}.
$$

Second-order expansion near $\theta_0$:
$$
\mathrm{KL}(p_{\theta_0}\,\|\,p_\theta)\approx \tfrac12(\theta-\theta_0)^\top F_{\theta_0}(\theta-\theta_0).
$$

**Meaning.** Fisher defines the **local geometry**; equal-KL moves correspond to equal “statistical distance.”

---

### 4.2 Natural gradient (Amari) and equivalent views
Natural gradient update for minimizing $f$:
$$
\theta_{t+1}=\theta_t-\eta\,F_{\theta_t}^{\dagger}\nabla_\theta f(\theta_t).
$$

Equivalent characterizations:
- **Quadratic regularization**: minimize first-order model plus $\tfrac{1}{2\eta}\|\theta-\theta_t\|_{F_{\theta_t}}^2$.
- **Trust region**: minimize first-order model subject to $\tfrac12\|\theta-\theta_t\|_{F_{\theta_t}}^2\le\delta$.

**Takeaway.** Natural gradient is steepest descent **under the KL metric**.

---

### 4.3 Natural Policy Gradient (NPG)
In RL with policy $\pi_\theta$,
$$
\theta_{t+1}=\theta_t+\eta\,F_{\theta_t}^{\dagger}\,\nabla_\theta J(\pi_{\theta_t}),
$$
with
$$
F_\theta=\mathbb{E}_{s\sim \lambda_\mu^{\pi_\theta},\,a\sim\pi_\theta}
\!\left[\nabla_\theta\log\pi_\theta(a|s)\nabla_\theta\log\pi_\theta(a|s)^\top\right],
\quad
\nabla_\theta J(\pi_\theta)=\frac{1}{1-\gamma}\,
\mathbb{E}\!\left[A^{\pi_\theta}(s,a)\,\nabla_\theta\log\pi_\theta(a|s)\right].
$$

**Interpretation.** Scale the vanilla gradient by the inverse Fisher to respect policy geometry and control step size in KL.

---

### 4.4 Least-squares view (efficient computation)
Define $x(s,a)=\nabla_\theta\log\pi_\theta(a|s)$ and solve
$$
w^\star=\arg\min_{w}\;
\mathbb{E}_{s\sim\lambda_\mu^{\pi_\theta},\,a\sim\pi_\theta}
\big[\big(w^\top x(s,a)-A^{\pi_\theta}(s,a)\big)^2\big].
$$
Then
$$
(1-\gamma)\,F_\theta^{\dagger}\nabla_\theta J(\pi_\theta)=w^\star(\theta),
\qquad
\theta_{t+1}=\theta_t+\frac{\eta}{1-\gamma}\,w^\star(\theta_t).
$$

**Meaning.** NPG direction = least-squares fit of advantages by policy scores; compute via regression without explicit matrix inversion.

---

### 4.5 Softmax parameterization: NPG = PMA
Under tabular softmax logits,
$$
\pi_{t+1}(a|s)\;\propto\;\pi_t(a|s)\,
\exp\!\left(\frac{\eta}{1-\gamma}\,A^{\pi_t}(s,a)\right),
$$
which matches the **policy mirror ascent** exponential-weights update.

**Takeaway.** NPG and KL-regularized PMA are **equivalent** in softmax/tabular settings.

---

## 5. Variance, baselines, and exploration

- **Baselines** reduce variance but do not remove all instability; practical choices (value function, GAE) matter.
- **Importance sampling** can suffer **unbounded variance** if behavior/target policies are too far apart; trust-region control helps.
- **Exploration hardness** summarized by a constant $\kappa$ (how well the data distribution covers states/actions needed for improvement). Larger $\kappa$ ⇒ harder exploration, larger sample complexity.
- **Statistical error** $\epsilon_{\text{stat}}$ (from estimating $A^\pi$) directly impacts sample-based NPG performance.

---

## 6. Convergence/comparison snapshot

| Algorithm | Typical (tabular) rate | Comments |
|---|---|---|
| **Vanilla PG (softmax)** | $O(1/T)$ with **large constants** depending on $|S|$, $(1-\gamma)^{-1}$, minimal action mass $c^{-1}$ | Sensitive to exploration and logit saturation |
| **Tabular NPG** | $O(1/((1-\gamma)^2 T))$ with **better constants** | Geometry-aware steps; monotonic improvements under TR constraints |
| **Sample-based NPG** | $\tilde O(1/\sqrt{T})+\sqrt{\kappa}\,\epsilon_{\text{stat}}$ | Regression view; quality depends on advantage estimation |
| **OPPO** | $\tilde O(|S||A|/\sqrt{(1-\gamma)^3T})$ (finite horizon analogs) | Uses optimism (bonus) + exponential-weights updates |

*Symbols:* $c^{-1}=\min_{s,t}\pi_t(a^\star(s)\mid s)$ lower-bounds optimal-action probability; $\kappa$ is an exploration hardness constant.

---

## 7. Trust-region and proximal methods: TRPO / PPO / OPPO

### 7.1 TRPO (Trust Region Policy Optimization)
**Idea.** Improve a **surrogate objective** while constraining the **mean KL** from the old policy:
$$
\max_\theta\; \mathbb{E}_{s\sim\lambda^{\pi_{\theta_t}}_\mu,\,a\sim\pi_{\theta_t}}
\!\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_t}(a|s)}\,A^{\pi_{\theta_t}}(s,a)\right]
\quad\text{s.t.}\quad
\mathbb{E}_{s\sim\lambda^{\pi_{\theta_t}}_\mu}\!\left[\mathrm{KL}\!\big(\pi_\theta(\cdot|s)\,\|\,\pi_{\theta_t}(\cdot|s)\big)\right]\le\delta.
$$

**Implementation sketch.**
- Linearize the objective at $\theta_t$; second-order expand KL to get direction $F(\theta_t)^{-1}\nabla J$ (i.e., **NPG direction**).
- **Line search** for step size $\eta = \sqrt{2\delta / (\nabla J^\top F^{-1}\nabla J)}$.
- **Monotonic improvement** guarantees under modeling assumptions.

**Relation.** TRPO is a **trust-region realization** of NPG; both embody the KL geometry.

---

### 7.2 PPO (Proximal Policy Optimization)
**Clipped surrogate** avoids solving a constrained problem each iteration:
$$
\max_\theta\; \mathbb{E}\,\min\!\Big\{
\rho_t(\theta)\,A_t,\;
\mathrm{clip}\!\big(\rho_t(\theta),1-\epsilon,1+\epsilon\big)\,A_t
\Big\},
\quad
\rho_t(\theta)=\frac{\pi_\theta(a|s)}{\pi_{\theta_t}(a|s)}.
$$

**Intuition.** The clip forms an implicit **trust region** on the likelihood ratio; easy to optimize with SGD; strong empirical performance.

**Tuning.** $\epsilon\in[0.1,0.3]$ common; combine with value loss and entropy bonus.

---

### 7.3 OPPO (Optimistic PPO; finite horizon $H$)
**Setup.** Non-stationary policies $\{\pi_h\}_{h=1}^H$ and values $V_h,Q_h$ per stage.  
**Key idea.** Add **optimistic bonuses** (e.g., count-based) to build optimistic $Q_h$ estimates, then do **exponential-weights/NPG-style** policy updates:
$$
\pi_{t+1,h}(a|s)\;\propto\;\pi_{t,h}(a|s)\,\exp\!\big(\eta\,Q_{t,h}^{\text{optimistic}}(s,a)\big).
$$
**Benefit.** Encourages exploration with finite-horizon structure and yields improved theoretical sample complexity.

---

## 8. Assumptions, applicability, and practical notes

- **Policy gradient theorem** assumes differentiable **stochastic** policies; gradients are expectations over **on-policy** state-action distributions.
- **Baselines** (e.g., learned $V_\phi$, GAE) are essential in practice to control variance; bias-variance trade-off matters.
- **NPG/TRPO** rely on **KL/Fisher** geometry; trust-region controls distribution shift and stabilizes updates.
- **PPO** approximates a trust region via clipping; simple and scalable with minibatch SGD.
- **Exploration** is a real bottleneck: choose entropy regularization, bonuses, or optimism to mitigate large $\kappa$.
- **Estimation error** $\epsilon_{\text{stat}}$ from critics/advantages directly affects sample efficiency.

---

## 9. How the pieces connect (logic flow)

**PG Theorem**  
→ switch to **advantage** for variance reduction  
→ view updates as **Mirror Descent** with **KL** (policy mirror ascent)  
→ understand the **geometry** via **Fisher/KL** → **Natural Gradient**  
→ implement as **NPG** (steepest ascent under KL)  
⇄ realize as **TRPO** (explicit trust region)  
→ simplify to **PPO** (clipped surrogate)  
→ extend to finite-horizon **OPPO** (optimism + exponential weights)

**PDL** ties surrogate improvement to true return improvement, justifying trust-region constraints and step-size selection.

---

## 10. Selected derivations and interpretations

### 10.1 Baseline correctness
For any $b(s)$,
$$
\mathbb{E}_{a\sim\pi(\cdot|s)}\!\left[\nabla_\theta\log\pi_\theta(a|s)\,b(s)\right]
=b(s)\,\nabla_\theta \sum_a \pi_\theta(a|s)=b(s)\,\nabla_\theta 1=0.
$$
Hence replacing $Q$ by $A=Q-b(s)$ keeps $\nabla_\theta J$ unchanged (unbiasedness).

---

### 10.2 PDL (sketch)
Using Bellman equations and telescoping sums,
$$
J(\pi)-J(\pi')
=\frac{1}{1-\gamma}\,
\mathbb{E}_{s\sim \lambda_\mu^\pi,\,a\sim \pi}\!\left[
Q^{\pi'}(s,a)-V^{\pi'}(s)\right]
=\frac{1}{1-\gamma}\,\mathbb{E}_{\lambda_\mu^\pi,\pi}\!\left[A^{\pi'}(s,a)\right].
$$
This motivates evaluating a new policy with advantages under a reference policy.

---

### 10.3 NPG direction via least squares
Let $x(s,a)=\nabla_\theta\log\pi_\theta(a|s)$. The regression normal equations give
$$
\mathbb{E}[x\,x^\top]\,w=\mathbb{E}[x\,A]
\;\Rightarrow\;
w=F_\theta^{-1}\,\mathbb{E}[x\,A]=(1-\gamma)\,F_\theta^{-1}\nabla_\theta J.
$$
Thus $w$ solves the least-squares fit of $A$ by score features and equals the NPG direction up to $(1-\gamma)$.

---

## 11. Where this leads (next steps)

- **Estimator design:** GAE($\lambda$), learned critics, bias-variance calibration.
- **Exploration:** entropy regularization, count-based/optimistic bonuses, intrinsic motivation.
- **Theory ↔ practice:** tighter monotonic improvement bounds, sample-efficient trust-region approximations, finite-horizon analyses.

---

## 12. Formula cheat-sheet

- **PDL:** $J(\pi)-J(\pi')=\tfrac{1}{1-\gamma}\,\mathbb{E}_{\lambda_\mu^\pi,\pi}[A^{\pi'}(s,a)]$.
- **PG (Q-form):** $\nabla_\theta J=\mathbb{E}\big[\sum_t \gamma^t Q\,\nabla_\theta\log\pi\big]$.
- **PG (softmax):** $\partial J/\partial \theta_{s,a}=\tfrac{1}{1-\gamma}\lambda^\pi_\mu(s)\pi(a|s)A(s,a)$.
- **NPG:** $\theta_{t+1}=\theta_t+\eta\,F_\theta^{-1}\nabla_\theta J$.
- **TRPO:** maximize surrogate s.t. mean KL $\le\delta$.
- **PPO:** clipped ratio objective.
- **OPPO:** exponential weights with optimistic bonuses.

---
