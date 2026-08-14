---
layout: post
title:  "HF Deep RL Course — Bonus Unit 2 (Hyperparameter Tuning with Optuna)"
date:   2026-08-14
desc: "HF Deep RL Course — Bonus Unit 2 (Hyperparameter Tuning with Optuna)"
keywords: "Machine learning"
categories: [Machine learning]
tags: [Machine learning]
icon: icon-html
---

# HF Deep RL Course — Bonus Unit 2 (Hyperparameter Tuning with Optuna)

> Source: [Hugging Face Deep RL Course, Bonus Unit 2](https://huggingface.co/learn/deep-rl-course/unitbonus2/introduction) — notes from *Introduction* through *Additional Readings*, reorganized for review. Continues from [Unit 3]({{ '/machine learning/2026/08/14/HF_DeepRL_Unit3.html' | replace: ' ', '%20' | prepend: site.baseurl }}).

> **_Keypoints:_**

- Why "one of the most critical tasks in Deep RL is finding a good set of training hyperparameters"
- Algorithm/theory vs **implementation details + hyperparameters** — the second half is what usually decides success
- Manual tuning ("grad student descent") and why it doesn't scale
- Optuna vocabulary: **study**, **trial**, **objective function**, `study.optimize()`, `trial.suggest_*`, `study.best_params`
- **Samplers / search strategies**: grid search, random search, Bayesian optimization (**TPE**), CMA-ES
- Why **random search beats grid search** in high dimensions
- **Pruning** unpromising trials early (`MedianPruner`, `trial.report()`, `trial.should_prune()`)
- A full worked Optuna + Stable-Baselines3 script (A2C on `CartPole-v1`)
- RL-specific practice: which hyperparameters matter, budget, **separate eval env**, multiple seeds, noisy objectives
- Self-check + additional readings (Optuna docs, Antonin Raffin's ICRA 2022 talk, RL Baselines3 Zoo)

---

## 0. What this bonus unit is for

The course states it plainly:

> One of the most critical tasks in Deep Reinforcement Learning is to **find a good set of training hyperparameters**.

[Optuna](https://optuna.org/) is the library that **automates the search**. The unit's plan is:

1. A little bit of the **theory** behind automatic hyperparameter tuning.
2. First **tune the DQN of the previous unit manually** — to feel the pain.
3. Then **automate the search using Optuna**.

The teaching material is Antonin Raffin's [ICRA 2022 presentation](https://araffin.github.io/tools-for-robotic-rl-icra2022/) — he is one of the founders of **Stable-Baselines** and **RL-Baselines3-Zoo**, so the advice here is exactly what produced the tuned hyperparameters shipped in the Zoo.

This unit is **light on math** and heavy on tooling and practice. So the things worth memorizing are the **vocabulary**, the **API shape**, and the **practical rules of thumb**.

---

## 1. Why hyperparameters matter *so much* in RL

Compared with supervised learning, **Deep RL is far more sensitive** to the choice of hyperparameters — learning rate, number of neurons, number of layers, optimizer, discount factor, rollout length, … A poor choice leads to **poor or unstable convergence**, and the whole problem is **compounded by variance across random seeds** (which initialize both the network weights *and* the environment).

### The theory / implementation split

It's tempting to think that reading the PPO or DQN paper is the hard part. In practice a Deep RL result is two things stacked:

| | **Theory / algorithm** | **Implementation details + hyperparameters** |
|---|---|---|
| Content | The loss, the objective, the update rule (e.g. the DQN target, the clipped PPO objective) | Optimizer, learning-rate schedule, network size/activation, $\gamma$, $n_{\text{steps}}$, buffer size, target-net sync period, observation/reward normalization, gradient clipping, seeds |
| Where you read it | The paper | The reference codebase — often *not* in the paper at all |
| Effect if wrong | Wrong algorithm | **Correct algorithm that simply doesn't learn** |
| How you fix it | Understand the derivation | **Search** |

> **Key insight:** the same algorithm, same code, same environment, with two different hyperparameter sets, is routinely the difference between a **working agent and a broken one**. Tuning is not polish — it is often the *whole* result.

### The course's own demonstration

The notebook makes the point on **`Pendulum-v1`**, a classic swing-up task:

- **PPO** with default hyperparameters and a small budget of **4000 timesteps** (~20 episodes) — mean reward around **−200**, nowhere near solving it. **A2C** does no better.
- **Training 10× longer doesn't fix it.** "In the case of A2C/PPO, training longer won't help much, **finding better hyperparameters is needed instead**."
- With the **tuned hyperparameters from the RL Zoo** — `gamma = 0.9`, `use_sde = True`, `sde_sample_freq = 4`, `learning_rate = 1e-3` — PPO trained for 50 000 steps actually solves the task.

Two lessons hidden in there:

1. **More compute is not a substitute for better hyperparameters.**
2. **Algorithm choice is itself a hyperparameter.** On `Pendulum-v1` an off-policy algorithm (SAC/TD3) works out of the box where on-policy ones struggle. Conversely, SAC on the *simple* `MountainCarContinuous-v0` fails without tuning: **simple environments can be challenging even for SOTA algorithms.**

---

## 2. Manual tuning: "grad student descent"

The unit's tongue-in-cheek name for tuning by hand. The notebook poses it as a timed challenge:

> Find the best hyperparameters for **A2C** on **`CartPole-v1`** with a budget of **20 000 training steps**. Maximum reward is **500**. The hyperparameters should work **for different random seeds**.

The knobs it exposes for A2C:

```python
policy_kwargs = dict(
    net_arch=[dict(vf=[64, 64], pi=[64, 64])],  # critic / actor architectures
    activation_fn=nn.Tanh,
)

hyperparams = dict(
    n_steps=5,            # steps of data collected before each policy update
    learning_rate=7e-4,
    gamma=0.99,           # discount factor
    max_grad_norm=0.5,    # gradient-clipping threshold
    ent_coef=0.0,         # entropy coefficient in the loss
)
```

Why this doesn't scale:

- **Humans are bad multi-dimensional optimizers.** With 6+ interacting knobs, intuition about one knob is conditioned on the current value of the others.
- **The objective is noisy.** A "better" score may just be a lucky seed, so you keep chasing noise.
- **It's serial and slow.** Each evaluation costs a full training run; a human tries maybe 10 configurations and then gets bored.
- **It isn't reproducible or reportable.** No record of the search space or the budget.

Automatic tuning fixes exactly these four things: it searches a **declared** space, on a **fixed** budget, with a **logged** history, and it can run **in parallel**.

---

## 3. Optuna vocabulary

This is the part to memorize. Optuna's design is "**define-by-run**": you don't hand it a static grid, you hand it a **function** that *asks* for parameter values as it executes.

| Term | What it is | In code |
|---|---|---|
| **Study** | The whole optimization **campaign**: a search space + a sampler + a pruner + a direction + the history of everything tried | `study = optuna.create_study(direction="maximize")` |
| **Trial** | **One** evaluation of **one** hyperparameter configuration (here: train an agent once and score it) | the `trial` argument of `objective` |
| **Objective function** | Takes a `trial`, samples hyperparameters from it, runs the experiment, **returns a single float** to be maximized/minimized | `def objective(trial) -> float:` |
| **Direction** | Whether bigger is better | `direction="maximize"` (mean episodic reward) |
| **Sampler** | The **search strategy** deciding the next configuration | `TPESampler`, `RandomSampler`, `GridSampler`, `CmaEsSampler` |
| **Pruner** | The **early-stopping** rule that kills hopeless trials mid-training | `MedianPruner` |
| **`study.optimize()`** | Runs the loop: repeatedly call `objective` until `n_trials` or `timeout` is hit | `study.optimize(objective, n_trials=100, n_jobs=1, timeout=900)` |
| **`study.best_params`** | Dict of the best hyperparameters found | also `study.best_trial`, `study.best_value` |
| **`study.trials`** | Every trial with its params, value and state (`COMPLETE` / `PRUNED` / `FAIL`) | `study.trials_dataframe().to_csv(...)` |

### The `trial.suggest_*` family

Each call **declares one dimension of the search space** and returns a concrete value for this trial.

| Call | Use it for | Example from the unit |
|---|---|---|
| `trial.suggest_categorical(name, choices)` | Unordered discrete choices | `suggest_categorical("activation_fn", ["tanh", "relu"])` |
| `trial.suggest_int(name, low, high)` | Integers (layer counts, exponents) | `2 ** trial.suggest_int("exponent_n_steps", 3, 10)` → $n_{\text{steps}} \in \{8, 16, \dots, 1024\}$ |
| `trial.suggest_float(name, low, high)` | Continuous, uniform | `suggest_float("max_grad_norm", 0.3, 5.0, log=True)` |
| `trial.suggest_float(name, low, high, log=True)` | Continuous, **log-uniform** | `suggest_float("lr", 1e-5, 1, log=True)` |
| `trial.set_user_attr(name, value)` | Record a *derived* value for logging (not searched) | `trial.set_user_attr("gamma_", gamma)` |

> ⚠️ **Always use `log=True` for scale-like parameters** — learning rate, entropy coefficient, $1-\gamma$. Sampling the learning rate uniformly on $[10^{-5}, 1]$ puts ~99.999% of the probability mass above $10^{-2}$, so you would essentially never try a small learning rate. Log-uniform spreads the samples evenly across **orders of magnitude**.

> **Trick worth stealing:** the discount factor is sampled as
> `gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)`,
> i.e. search $1-\gamma$ log-uniformly to get $\gamma \in [0.9,\ 0.9999]$. What matters about $\gamma$ is the **effective horizon** $\tfrac{1}{1-\gamma}$, not $\gamma$ itself — the difference between $0.99$ and $0.999$ is a 10× horizon change, while $0.5$ vs $0.6$ is almost nothing. Same idea for `n_steps`: search the **exponent**, not the value.

---

## 4. Search strategies (samplers)

### Grid search
Enumerate a Cartesian product of hand-picked values. With $d$ hyperparameters and $k$ values each, the cost is

$$
N_{\text{grid}} = k^{d}
$$

which explodes: $5$ values for $6$ hyperparameters is $5^6 = 15\,625$ training runs. Worse, **the resolution per dimension is terrible**: $k^d$ trials still only ever test $k$ distinct values of the learning rate.

### Random search
Sample each hyperparameter independently from its declared distribution. Two properties make it strictly better than grid in practice:

1. **Every trial tests a new value of every dimension.** $N$ random trials give $N$ distinct learning rates; $N$ grid trials give $k \ll N$.
2. **It degrades gracefully with useless dimensions.** RL objectives are usually dominated by 2–3 hyperparameters (learning rate first). Random search spends its full budget resolving those, while grid search wastes most of its trials re-testing them at fixed values while varying parameters that don't matter.

A useful probabilistic sanity check: if the top $5\%$ of the space is "good", then $N$ random trials miss it with probability $0.95^{N}$ — so $N = 60$ gives you better than a $95\%$ chance of landing in it, **regardless of the dimension** $d$.

### Bayesian optimization — TPE
**TPE (Tree-structured Parzen Estimator)** is Optuna's default sampler and the one the unit uses. It is **sequential model-based**: it *learns from the trials already run* instead of sampling blindly.

The idea: split the finished trials at some quantile of the objective into a **good** group and a **bad** group, fit a density over hyperparameters for each — $\ell(x)$ for good, $g(x)$ for bad — and propose the next configuration by maximizing the ratio

$$
x_{\text{next}} = \arg\max_{x} \frac{\ell(x)}{g(x)}
$$

i.e. **"where were the good configurations dense and the bad ones sparse?"** This trades off exploitation (near known-good regions) against exploration (where little is known). Note the plumbing detail from the config:

```python
sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)  # N_STARTUP_TRIALS = 5
```

`n_startup_trials` is the number of **purely random** trials run first — the model needs data before it can be trusted. **Bayesian optimization always starts as random search.**

Optuna also ships **`CmaEsSampler`** (CMA-ES — an evolution strategy, strong on purely continuous, moderately-dimensional spaces) and **`GridSampler`** / **`RandomSampler`** for the baselines.

### Side-by-side

| | **Grid search** | **Random search** | **Bayesian / TPE** |
|---|---|---|---|
| Chooses next point | Enumerates a fixed lattice | i.i.d. from the priors | From a **model of past trials** |
| Uses past results? | **No** | **No** | **Yes** |
| Cost in $d$ dims | $k^{d}$ — exponential | Any budget $N$ you like | Any budget $N$ you like |
| Distinct values per dim (in $N$ trials) | $k$ | $N$ | $N$ |
| Handles irrelevant dims | **Badly** (wastes budget) | Well | Well |
| Parallelizable | Embarrassingly | Embarrassingly | **Partly** (inherently sequential; `n_jobs` weakens the model) |
| Handles categorical/conditional spaces | Yes | Yes | **Yes** (TPE is fine with them; CMA-ES is not) |
| Needs warm-up trials | No | No | **Yes** (`n_startup_trials`) |
| Best when | ≤2 hyperparameters, need a full map | Cheap baseline, large budget, many dims | **Expensive** trials, small budget — i.e. RL |
| Cost of one "unit" of insight | Highest | Medium | Lowest |

> **Key insight:** the ordering is *grid ≪ random < TPE*, and the reason is the same in both steps — **use your budget to gain information rather than to fill a lattice.** Random search stops wasting trials on redundant coordinates; Bayesian optimization additionally stops wasting trials on regions already known to be bad.

---

## 5. Pruning: stop wasting compute on losers

A trial in RL costs a **full training run**. But you usually know within the first fraction of it that a configuration is hopeless — a diverged learning rate shows up immediately. **Pruning** = evaluate periodically *during* training, report the intermediate score to Optuna, and let it kill the trial early.

The mechanism is three lines:

```python
self.trial.report(self.last_mean_reward, self.eval_idx)  # intermediate value
if self.trial.should_prune():                            # pruner's verdict
    self.is_pruned = True
    return False                                         # stop training
```

and then in the objective:

```python
if eval_callback.is_pruned:
    raise optuna.exceptions.TrialPruned()
```

Raising `TrialPruned` is important: it marks the trial `PRUNED` rather than `FAIL`, so its partial information is kept and it isn't mistaken for a legitimately bad *final* score.

### MedianPruner
The pruner used in the unit. At evaluation step $k$, prune the current trial if its intermediate value is **worse than the median of the intermediate values of all previous trials at that same step $k$**.

```python
pruner = MedianPruner(
    n_startup_trials=N_STARTUP_TRIALS,        # need some trials before a median means anything
    n_warmup_steps=N_EVALUATIONS // 3,        # "do not prune before 1/3 of the max budget is used"
)
```

Both guards matter:

| Guard | Failure it prevents |
|---|---|
| `n_startup_trials` | Pruning against a median computed from 1–2 trials (meaningless) |
| `n_warmup_steps` | Killing **slow starters** — configurations that are behind early but win at the end. Very real in RL (e.g. a longer exploration schedule, or a low learning rate that is stable but slow) |

**Why it saves compute:** if the median pruner kills roughly half the trials at the first checkpoint, you buy nearly **2× the number of trials** for the same wall-clock. That effectively converts compute from "carefully measuring bad configurations" into "trying more configurations".

> ⚠️ Pruning interacts with the objective: a pruned trial's reported value is *partial*. Compare final scores only among `COMPLETE` trials, and beware that aggressive pruning biases the search toward **fast learners** rather than **best final performers**. That's precisely the trade-off `n_warmup_steps` controls.

---

## 6. The whole thing in code

This is the unit's script (A2C on `CartPole-v1`), completed and condensed into one block. The shape — **config → search space sampler → reporting callback → objective → optimization loop** — is the reusable part.

```python
import gym
import torch as th
import torch.nn as nn
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

# ---------------- Config ----------------
N_TRIALS = 100              # maximum number of trials
N_JOBS = 1                  # trials run in parallel
N_STARTUP_TRIALS = 5        # random trials before TPE / before pruning kicks in
N_EVALUATIONS = 2           # evaluations during a single training run
N_TIMESTEPS = int(2e4)      # training budget per trial
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_ENVS = 5             # parallel envs used ONLY for evaluation
N_EVAL_EPISODES = 10
TIMEOUT = int(60 * 15)      # 15 minutes, whichever comes first

ENV_ID = "CartPole-v1"
DEFAULT_HYPERPARAMS = {"policy": "MlpPolicy", "env": ENV_ID}


# ---------------- 1. The search space ----------------
def sample_a2c_params(trial: optuna.Trial) -> dict:
    """Sample one A2C hyperparameter configuration from the trial."""
    # Search 1 - gamma log-uniformly  =>  gamma in [0.9, 0.9999]
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
    # Search the exponent: n_steps in {8, 16, 32, ..., 1024}
    n_steps = 2 ** trial.suggest_int("exponent_n_steps", 3, 10)
    # Learning rate spans 5 orders of magnitude -> log scale
    learning_rate = trial.suggest_float("lr", 1e-5, 1, log=True)
    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small"])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    # Log the derived (true) values so the study report is readable
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("n_steps", n_steps)

    net_arch = [
        {"pi": [64], "vf": [64]} if net_arch == "tiny"
        else {"pi": [64, 64], "vf": [64, 64]}
    ]
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "max_grad_norm": max_grad_norm,
        "policy_kwargs": {"net_arch": net_arch, "activation_fn": activation_fn},
    }


# ---------------- 2. Report intermediate scores for pruning ----------------
class TrialEvalCallback(EvalCallback):
    """Periodically evaluate, report to Optuna, and prune if hopeless."""

    def __init__(self, eval_env, trial, n_eval_episodes=5, eval_freq=10000,
                 deterministic=True, verbose=0):
        super().__init__(eval_env=eval_env, n_eval_episodes=n_eval_episodes,
                         eval_freq=eval_freq, deterministic=deterministic,
                         verbose=verbose)
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()                 # parent runs the evaluation
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            if self.trial.should_prune():
                self.is_pruned = True
                return False                   # stop this training run
        return True


# ---------------- 3. The objective function ----------------
def objective(trial: optuna.Trial) -> float:
    """Train one agent with sampled hyperparameters, return mean episodic reward."""
    kwargs = DEFAULT_HYPERPARAMS.copy()
    kwargs.update(sample_a2c_params(trial))

    model = A2C(**kwargs)

    # A SEPARATE set of envs, used only to score the agent
    eval_envs = make_vec_env(ENV_ID, n_envs=N_EVAL_ENVS)
    eval_callback = TrialEvalCallback(
        eval_envs, trial,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=EVAL_FREQ,
        deterministic=True,
    )

    nan_encountered = False
    try:
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        # Random hyperparameters can blow up into NaNs
        print(e)
        nan_encountered = True
    finally:
        model.env.close()
        eval_envs.close()

    if nan_encountered:
        return float("nan")        # tell the optimizer the trial failed
    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward


# ---------------- 4. The optimization loop ----------------
th.set_num_threads(1)   # 1 thread per trial trains faster on small nets

sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS,
                      n_warmup_steps=N_EVALUATIONS // 3)

study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")

try:
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS, timeout=TIMEOUT)
except KeyboardInterrupt:
    pass   # keep whatever we have

print("Number of finished trials:", len(study.trials))
trial = study.best_trial
print(f"Best value: {trial.value}")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
for key, value in trial.user_attrs.items():
    print(f"    {key}: {value}")

study.trials_dataframe().to_csv("study_results_a2c_cartpole.csv")
plot_optimization_history(study).show()
plot_param_importances(study).show()
```

### Details in that script worth remembering

| Line | Why it's there |
|---|---|
| `direction="maximize"` | The objective is **mean episodic reward**; bigger is better |
| `return eval_callback.last_mean_reward` | The objective must return **one float** — the score of the whole configuration |
| `make_vec_env(ENV_ID, n_envs=N_EVAL_ENVS)` | A **separate evaluation environment**, never the training env |
| `deterministic=True` in evaluation | Score the **greedy/mean** policy, not the exploring one |
| `timeout=TIMEOUT` | Bound the search by **wall-clock**, not just trial count |
| `except AssertionError → return nan` | Sampled hyperparameters *will* sometimes produce NaNs; report the failure instead of crashing the study |
| `try/except KeyboardInterrupt` | The study is still usable after you stop it early |
| `th.set_num_threads(1)` | Small MLPs are slower with thread contention; matters a lot across 100 trials |
| `plot_param_importances(study)` | Tells you **which hyperparameters actually mattered** — use it to shrink the space and re-run |

`plot_optimization_history` and `plot_param_importances` are the two plots to always look at: the first shows whether the search was still improving when the budget ran out (if yes: increase the budget), the second shows which dimensions to keep searching and which to freeze.

---

## 7. Practical, RL-specific advice

### 7.1 Which hyperparameters to tune first
Not all knobs are equal. Rough priority for Deep RL:

| Priority | Hyperparameter | Note |
|---|---|---|
| 1 | **Learning rate** | Almost always the single most important; search log-uniformly over several orders of magnitude |
| 2 | **Algorithm choice** | On/off-policy is a bigger lever than any single knob (PPO vs SAC on `Pendulum-v1`) |
| 3 | **$\gamma$ (as $1-\gamma$)** | Sets the effective horizon $\tfrac{1}{1-\gamma}$ |
| 4 | **Rollout / batch size** (`n_steps`, `batch_size`, `train_freq`) | Controls the gradient noise vs data-freshness trade-off; search the exponent |
| 5 | **Exploration** (`ent_coef`, $\varepsilon$-schedule, `use_sde`) | Log scale for coefficients |
| 6 | **Network architecture & activation** | Categorical over a few sane presets (`tiny` / `small`), not free-form |
| 7 | **Clipping / regularization** (`max_grad_norm`, `clip_range`) | Usually a smaller effect, but stabilizing |

> ⚠️ **Never make the search space bigger than your budget can resolve.** Every extra dimension dilutes the information per trial. Start with 4–6 dimensions and *narrow ranges you can defend*, look at parameter importances, then re-run on a smaller space.

### 7.2 Budget
Three budgets have to be chosen together, and they trade off against each other:

$$
\text{total compute} \;\approx\; N_{\text{trials}} \times N_{\text{timesteps}} \times N_{\text{seeds}}
$$

- **`N_TIMESTEPS` too small** → you select for *fast* learners, not *good* ones. The tuned config may be worthless at full training length.
- **`N_TRIALS` too small** → TPE never leaves its random warm-up (`n_startup_trials = 5` out of 100 is fine; out of 8 it means you only did random search).
- **`N_SEEDS` too small** → you select noise (see below).

Pruning is what makes this affordable: it reallocates timesteps from bad trials to more trials.

### 7.3 Use a separate evaluation environment
The unit does this everywhere: `eval_envs = make_vec_env(env_id, n_envs=10)`, "env used only for evaluation".

- The **training** env is being explored in, has exploration noise, and (in SB3) reports episode statistics through wrappers that training itself perturbs.
- The **evaluation** env should be a clean copy, run with `deterministic=True`, with **normalization statistics frozen** if you use `VecNormalize`.
- Tuning against the *training* reward is a form of leakage: you'd select hyperparameters that make the *logging* look good.

### 7.4 Noisy evaluation and multiple seeds
This is the RL-specific trap. The objective is a **random variable**, not a number:

$$
J(\theta_{\text{hp}}) \;=\; \mathbb{E}_{\text{seed}}\big[\, \text{mean episodic reward}\,\big]
$$

and you only ever get a noisy sample of it. Two independent noise sources: the **training seed** (weight init, env resets, exploration) and the **evaluation episodes** themselves.

If you run one seed and 5 episodes per trial, then across 100 trials the winner is very likely the **luckiest** trial, not the best configuration — Optuna will happily optimize your noise.

Mitigations, cheapest first:

| Mitigation | How |
|---|---|
| Many evaluation episodes | `n_eval_episodes=10` … `100`; average over parallel eval envs (`n_envs=5`–`10`) |
| Deterministic evaluation | `deterministic=True` removes the action-sampling noise |
| Multiple training seeds per trial | Average (or take the **minimum**, for robustness) over 2–3 seeds; costs a linear factor in compute |
| Re-run the top-$k$ | Cheap and effective: tune with 1 seed, then **re-evaluate the best 5–10 trials on fresh seeds** and pick the winner there |
| Sanity-check the "best" | Retrain `study.best_params` from scratch on an unseen seed before believing it |

> **Key insight:** hyperparameter search **overfits** — to the seed, to the evaluation episodes, and to the shortened training budget. The final number reported by a study is an **optimistically biased** estimate. Always re-validate `best_params` with a fresh, full-length, multi-seed run.

### 7.5 Don't start from scratch
[RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) already ships **tuned hyperparameters** for many (algorithm, environment) pairs, plus scripts for training, evaluating, **tuning**, and recording videos. Two good uses:

- Take the Zoo config for a *similar* environment as your **default**, and tune a narrow band around it.
- Use the Zoo's `train.py --optimize` path instead of rewriting the objective/callback plumbing.

---

## 8. Recap (one screen)

1. Deep RL is **far more sensitive** to hyperparameters than supervised learning, and **variance across seeds** compounds the problem.
2. Correct algorithm + wrong hyperparameters = **broken agent**. Training longer does **not** fix bad hyperparameters.
3. A **study** is the whole search; a **trial** is one configuration; the **objective function** samples hyperparameters via `trial.suggest_*`, trains, and returns **one float** (mean episodic reward).
4. `study.optimize(objective, n_trials=..., timeout=...)` runs the loop; `study.best_params` / `study.best_trial` reads out the answer.
5. **Grid ≪ random < TPE.** Grid is exponential in $d$ and wastes budget; random resolves every dimension; **TPE** models past trials and maximizes $\ell(x)/g(x)$ after `n_startup_trials` random warm-ups.
6. **Pruning** (MedianPruner + `trial.report` + `should_prune` + `raise TrialPruned`) buys roughly 2× the trials; guard it with `n_warmup_steps` so slow starters aren't killed.
7. Use **log scale** for learning rate / entropy coefficient, search $1-\gamma$ and **exponents** for `n_steps`.
8. Always a **separate eval env**, `deterministic=True`, many episodes, and **re-validate the winner on fresh seeds** — the search overfits.
9. Read `plot_param_importances` and `plot_optimization_history`, then shrink the space and re-run.

### The loop in one line

$$
\text{study} \;\to\; \big[\underbrace{\text{trial} \to \text{suggest} \to \text{train} \to \text{eval} \to \text{report}}_{\text{prune early if below median}}\big]^{\,N_{\text{trials}}} \;\to\; \text{best\_params}
$$

---

## 9. Glossary

The unit has no `glossary.mdx`; this is the vocabulary it actually uses.

**Optuna**
- **Study** — one optimization campaign: direction, sampler, pruner, and the full history of trials. Created with `optuna.create_study(...)`.
- **Trial** — a single evaluation of a single hyperparameter configuration. States: `COMPLETE`, `PRUNED`, `FAIL`.
- **Objective function** — `objective(trial) -> float`. Samples hyperparameters from the trial, runs the experiment, returns the scalar to optimize.
- **Direction** — `"maximize"` or `"minimize"`.
- **Search space** — implicitly defined by the `trial.suggest_*` calls executed inside the objective ("define-by-run").
- **Sampler** — the strategy for choosing the next configuration: `GridSampler`, `RandomSampler`, `TPESampler` (default), `CmaEsSampler`.
- **TPE (Tree-structured Parzen Estimator)** — Bayesian optimization that fits densities $\ell(x)$ over good trials and $g(x)$ over bad ones, and proposes $\arg\max_x \ell(x)/g(x)$.
- **`n_startup_trials`** — number of random trials before the model-based sampler (or the pruner) becomes active.
- **Pruner** — early-stopping rule for trials. `MedianPruner` prunes a trial whose intermediate value is below the median of previous trials at the same step.
- **`n_warmup_steps`** — number of intermediate steps that must pass before a trial may be pruned; protects slow starters.
- **Intermediate value** — a mid-training score sent with `trial.report(value, step)`.
- **`TrialPruned`** — the exception raised to mark a trial as pruned rather than failed.
- **`best_params` / `best_value` / `best_trial`** — the outcome of the study.
- **User attributes** — `trial.set_user_attr(name, value)`, for logging derived quantities (like the true $\gamma$) that aren't search dimensions.

**Search strategies**
- **Grid search** — exhaustive Cartesian product; cost $k^d$.
- **Random search** — independent samples from the priors; budget-agnostic, resolves every dimension.
- **Bayesian optimization** — build a surrogate model of the objective from past trials and use it to choose the next point.
- **CMA-ES** — an evolution strategy sampler; good for continuous spaces, not for categorical/conditional ones.

**RL side**
- **"Grad student descent"** — tuning hyperparameters by hand.
- **Evaluation environment** — a separate env instance used only to score the agent, with deterministic actions and frozen normalization.
- **Budget** — training timesteps per trial (`N_TIMESTEPS`), number of trials (`N_TRIALS`), and wall-clock (`timeout`).
- **RL Baselines3 Zoo** — collection of pre-trained SB3 agents plus **tuned hyperparameters** and tuning scripts.

---

## 10. Self-check (quick review questions)

1. Why is Deep RL **more sensitive** to hyperparameters than supervised learning? Name the two independent noise sources.
2. On `Pendulum-v1`, PPO with default hyperparameters fails. Why doesn't **training 10× longer** fix it?
3. Distinguish a **study** from a **trial** in one sentence each.
4. What is the **type signature** of an Optuna objective function, and what does it return in an RL setting?
5. Write, from memory, the four lines that create a study with TPE + median pruning and run 100 trials with a 15-minute timeout.
6. Why does `suggest_float("lr", 1e-5, 1, log=True)` need `log=True`? What goes wrong without it?
7. Explain the trick `gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)`. What quantity are you really searching over?
8. Why is `2 ** trial.suggest_int("exponent_n_steps", 3, 10)` preferred over `suggest_int("n_steps", 8, 1024)`?
9. Give **two distinct reasons** why random search beats grid search when $d$ is large.
10. In one formula, what does TPE maximize to pick its next configuration? What do $\ell$ and $g$ represent?
11. Why does `TPESampler` need `n_startup_trials`? What is the search doing during those trials?
12. Describe the **MedianPruner** rule. What do `n_startup_trials` and `n_warmup_steps` each protect against?
13. What are the three calls that implement pruning inside a training callback, and why must the objective `raise optuna.exceptions.TrialPruned()` instead of returning a low value?
14. Why must the evaluation env be **separate** from the training env, and why `deterministic=True`?
15. You ran 100 trials with 1 seed and 5 eval episodes each. Why is `study.best_value` an **optimistic** estimate, and what is the cheapest fix?
16. Which two Optuna plots should you always inspect after a study, and what decision does each drive?

---

## 11. Additional Readings

**Optuna**
- [Optuna](https://optuna.org/) — official site and documentation (samplers, pruners, visualization, distributed studies)
- [Optuna on GitHub](https://github.com/optuna/optuna)

**The unit's source material — Antonin Raffin, ICRA 2022**
- [Tools for Robotic Reinforcement Learning (ICRA 2022 tutorial)](https://araffin.github.io/tools-for-robotic-rl-icra2022/) — slides and materials
- **"The theory behind Hyperparameter tuning"** — video: <https://www.youtube.com/watch?v=AidFTOdGNFQ>
- **"Optuna Tutorial"** — video: <https://www.youtube.com/watch?v=ihP7E76KGOI>
- [The Optuna lab notebook (Colab)](https://colab.research.google.com/github/araffin/tools-for-robotic-rl-icra2022/blob/main/notebooks/optuna_lab.ipynb) — the code in §6

**Libraries used**
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) and its [documentation](https://stable-baselines3.readthedocs.io/en/master/)
- [SB3-Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) — additional algorithms (QR-DQN, TQC, …)
- [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) — pre-trained agents, **tuned hyperparameters**, and scripts for training / evaluating / **tuning** / recording videos. See e.g. [`hyperparams/ppo.yml`](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml) for the tuned `Pendulum-v1` config used in §1.

**Hands-on suggestions from the unit**
- Beat your **LunarLander-v2** agent (Unit 1) with Optuna-tuned hyperparameters; also try `MountainCar-v0` and `CartPole-v1`.
- Beat your **SpaceInvaders** agent (Unit 3).

---

> **Next up:** **Unit 4 — Policy Gradient with PyTorch.** We leave value-based methods behind and optimize the policy *directly*: policy-based vs value-based methods, the advantages and disadvantages of policy-gradient, the Monte-Carlo REINFORCE algorithm, and the Policy Gradient Theorem.
