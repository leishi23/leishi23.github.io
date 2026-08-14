---
layout: post
title:  "HF Deep RL Course — Unit 7 (Multi-Agent RL & Self-Play)"
date:   2026-08-14
desc: "HF Deep RL Course — Unit 7 (Multi-Agent RL & Self-Play)"
keywords: "Machine learning"
categories: [Machine learning]
tags: [Machine learning]
icon: icon-html
---

# HF Deep RL Course — Unit 7 (Multi-Agent RL & Self-Play)

> Source: [Hugging Face Deep RL Course, Unit 7](https://huggingface.co/learn/deep-rl-course/unit7/introduction) — notes from *Introduction* through *Additional Readings*, reorganized for review. Continues from [Unit 6]({{ '/machine learning/2026/08/14/HF_DeepRL_Unit6.html' | replace: ' ', '%20' | prepend: site.baseurl }}).

> **_Keypoints:_**

- Single-agent → **multi-agent** (MARL): several agents sharing and interacting in one environment
- Three environment types: **cooperative**, **competitive/adversarial**, and **mixed**
- Two design paradigms: **decentralized** (independent learners, **non-stationary** environment) vs **centralized** (shared experience buffer, one joint policy, global reward)
- **CTDE** — centralized training, decentralized execution — the practical middle ground (MA-POCA's centralized critic)
- **Self-play**: use *former copies of your own policy* as opponents to get a matched, self-improving adversary
- ML-Agents self-play hyperparameters: `save_steps`, `team_change`, `swap_steps`, `window`, `play_against_latest_model_ratio` — all trading **stability vs diversity**
- **ELO rating** instead of cumulative reward for adversarial evaluation: expected score + linear K-factor update, zero-sum
- Hands-on: **SoccerTwos** 2v2 with the `poca` trainer + self-play, evaluated on the **AI vs AI** ELO leaderboard

---

## 0. What this unit is for

Everything so far has been a **single-agent system**: our agent was alone in its environment, **not cooperating or collaborating with other agents**. That works great, and covers many applications (all the CartPole / LunarLander / Pyramids / Doom environments trained so far).

But **as humans we live in a multi-agent world** — our intelligence comes from interaction with other agents. So the goal becomes: build agents that can **interact with other humans and other agents**, and are robust enough to **adapt, collaborate, or compete**.

This unit is the basics of **multi-agent reinforcement learning (MARL)**, and the hands-on is the first genuinely multi-agent training run of the course: **a 2v2 soccer team that must beat the opponent team** (SoccerTwos, built by the Unity ML-Agents team).

> ⚠️ **Maintenance note from the course itself:** the Deep RL course is now in a low-maintenance state, and the **AI vs AI leaderboard for SoccerTwos has been shut down**. You can still train the agent and watch it play — just don't expect the ladder to rank you.

---

## 1. From single agent to multiple agents

In **MARL** we have **multiple agents that share and interact in a common environment** — and interact *with each other*, not just with the world.

Course examples:

- A **warehouse** where **multiple robots need to navigate to load and unload packages**.
- A **road with several autonomous vehicles**.
- A **team of vacuum cleaners** that must cover a floor between them (Brian Douglas's running example in the linked video).
- **SoccerTwos**: 4 agents, two teams of two.

The new difficulty is that the other agents are **learning too**. In single-agent RL the environment's dynamics $P(s' \mid s,a)$ are fixed; the moment other learners are in the loop, the thing you are adapting to is *itself adapting to you*.

> **Key insight:** MARL is not "single-agent RL with more bodies". The hard part is that the *other agents' policies are part of your transition function*, and they change while you learn.

---

## 2. Different types of multi-agent environments

Because agents now interact with each other, the environment can be one of three kinds:

- **Cooperative environments** — agents need to **maximize the common benefits**. In a warehouse, **robots must collaborate to load and unload packages efficiently (as fast as possible)**.
- **Competitive / adversarial environments** — an agent **wants to maximize its benefits by minimizing the opponent's**. In a game of **tennis**, **each agent wants to beat the other agent**.
- **Mixed (adversarial *and* cooperative)** — as in **SoccerTwos**: two agents are on a team (blue or purple), so they must **cooperate with each other and beat the opponent team**.

| | **Cooperative** | **Competitive / adversarial** | **Mixed** |
|---|---|---|---|
| Objective | Maximize the **common** benefit | Maximize own benefit, **minimize the opponent's** | Cooperate **within** a team, compete **across** teams |
| Reward structure | Shared / team reward | Roughly **zero-sum** | Team reward, opposed across teams |
| Course example | Warehouse robots loading packages | Tennis (1v1) | **SoccerTwos** (2v2) |
| Typical tooling | Centralized critic / team reward (MA-POCA) | **Self-play** + ELO | **Self-play + MA-POCA** together |
| Evaluation metric | Cumulative team reward | **ELO** (reward is opponent-dependent) | ELO for the team |

Mnemonic: **cooperative = maximize *ours*; competitive = maximize mine *minus* theirs; mixed = both, at two different levels.**

---

## 3. Designing multi-agent systems: decentralized vs centralized

There are **two solutions** for designing a MARL system.

### 3.1 Decentralized system

In **decentralized learning**, **each agent is trained independently from the others**. Each vacuum learns to clean as many places as it can **without caring about what other vacuums (agents) are doing**.

- **The benefit:** **since no information is shared between agents, these agents can be designed and trained exactly like we train single agents** — you can reuse DQN, A2C, PPO unchanged.
- **The idea:** the training agent **considers other agents as part of the environment dynamics. Not as agents.**
- **The big drawback:** it makes the environment **non-stationary**, since the **underlying Markov decision process changes over time** as the other agents also learn and change their behavior. This is **problematic for many RL algorithms, which can't reach a global optimum in a non-stationary environment.**

Why non-stationarity is fatal to the usual guarantees: from agent $i$'s viewpoint the transition and reward it experiences are really marginals over everyone else's policies $\pi_{-i}$,

$$
P_i\!\left(s' \mid s, a_i\right) \;=\; \sum_{a_{-i}} P\!\left(s' \mid s, a_i, a_{-i}\right) \prod_{j \neq i} \pi_j\!\left(a_j \mid s\right)
$$

and since $\pi_{-i}$ is **changing during training**, $P_i$ is a **moving target**. The convergence proofs behind Q-Learning and friends assume a *fixed* MDP, so they simply don't apply: the agent can end up chasing its own tail, and old replay-buffer transitions become stale/off-distribution rather than merely off-policy.

### 3.2 Centralized approach

Here **a high-level process collects all the agents' experiences** — a shared **experience buffer** — and those experiences are used **to learn a common policy**.

In the vacuum cleaner example, the (joint) observation would be:
- the **coverage map** of the vacuums,
- the **position of all the vacuums**.

We use that collective experience **to train a policy that moves all three robots in the most beneficial way as a whole**, so each robot learns from the **common experience**. Crucially, we now have a **stationary environment**, because all the agents are treated as one larger entity and the changes in other agents' policies are known (they *are* the same policy).

### 3.3 Recap table

| | **Decentralized** | **Centralized** |
|---|---|---|
| Who learns | **Each agent independently** | **One policy** learned from all agents |
| What other agents are | Just **part of the environment** | Part of a **single larger entity** |
| Input / output | Local observation → own action | Present state of the environment → **joint actions** |
| Reward | Individual / local | **Global** |
| Stationarity | **Non-stationary** ⇒ **no convergence guarantee** | **Stationary** |
| Scaling | Scales easily, no communication needed | Joint action space grows **exponentially** in the number of agents; needs shared info at execution time |
| Reuse of single-agent algos | **Direct** | Needs multi-agent machinery |

> **Key insight:** the decentralized/centralized choice is a trade between **convenience and correctness**. Decentralized is trivially easy to implement but breaks the stationarity assumption; centralized restores stationarity but needs global information (which a real robot fleet may not have at run time).

### 3.4 CTDE — centralized training, decentralized execution

The practical compromise, and the one the hands-on actually uses: **centralized training, decentralized execution (CTDE)**. Train with privileged global information, then throw it away at deployment.

That's exactly what **MA-POCA (Multi-Agent POsthumous Credit Assignment)**, the ML-Agents multi-agent trainer, does. The problem it solves: agents typically **receive a reward as a group** (+1 minus penalty) when the team scores, so **every agent is rewarded even if it didn't contribute equally to the win** — which makes it hard to learn what to do individually (the credit-assignment problem).

The idea is simple but powerful: a **centralized critic processes the states of all agents in the team to estimate how well each agent is doing** — *think of this critic as a coach*. This lets each agent

- **make decisions based only on what it perceives locally** (decentralized execution), while
- **simultaneously being evaluated in the context of the whole group** (centralized training).

The "posthumous" part is the neat trick: the critic can still assign credit to an agent whose action mattered even if that agent is **removed from the episode before the reward arrives** (absorbing states) — see the paper in §9.

---

## 4. Self-play

### 4.1 The problem it solves

Training agents correctly in an adversarial game is **quite complex**, and it's a chicken-and-egg problem:

1. You need a **well-trained opponent** to play against your training agent.
2. But if you find a **very good** opponent, **how will your agent improve its policy when the opponent is too strong?**

The course's analogy: **a child who just started learning soccer**. Playing against a very good player is useless — it's too hard to win, or even to get the ball from time to time. The child **continuously loses without ever having time to learn a good policy**. The symmetric failure also holds:

- opponent **too strong** ⇒ **we learn nothing** (no reward signal ever; all trajectories look equally bad),
- opponent **too weak** (e.g. a **fixed random opponent**) ⇒ we **overlearn useless behavior** that collapses against a stronger opponent later.

So a fixed random opponent is not a shortcut: you will get an agent that is excellent at beating noise and hopeless against anyone with a strategy.

> **The best solution is to have an opponent that is on the same level as the agent, and that upgrades its level as the agent upgrades its own.**

### 4.2 The mechanism

That solution is **self-play**: **the agent uses former copies of itself (of its policy) as an opponent.**

This way the agent
- plays against an agent of the **same level** (challenging, but not too much),
- has opportunities to **gradually improve its policy**,
- and then **updates its opponent as it becomes better**.

It is a way to **bootstrap an opponent and progressively increase the opponent's complexity** — an automatic curriculum that you never have to hand-design. It mirrors how humans learn in competition: start against a similar-level opponent, learn from them, then move on to stronger opponents.

The loop, in one line:

$$
\pi_{\theta} \;\text{trains against}\; \{\pi_{\theta_{t_1}}, \pi_{\theta_{t_2}}, \dots\} \;\subset\; \text{snapshot pool} \quad\Longrightarrow\quad \text{snapshot } \pi_{\theta} \text{ every } \texttt{save\_steps} \;\Longrightarrow\; \text{pool improves with } \pi_{\theta}
$$

**The theory is not new.** It was already used by **Arthur Samuel's checkers player in the fifties** and by **Gerald Tesauro's TD-Gammon in 1995**.

---

## 5. Self-play in ML-Agents: the hyperparameters

Self-play is built into ML-Agents and controlled by several hyperparameters. The documentation states the central tension explicitly:

> **The tradeoff is between the skill level and generality of the final policy, and the stability of learning.**

Training against a set of **slowly changing or unchanging adversaries with low diversity results in more stable training — but there is a risk of overfitting if the change is too slow.** Conversely, quickly changing, highly diverse adversaries give a more general policy but a noisier, less stable learning process.

So there are four things to control: **how often we change opponents**, **how many opponents we keep**, **how often we face the newest one**, and **how often we snapshot**.

| Hyperparameter | Unit | What it does | Larger value ⇒ | Trade-off it controls | SoccerTwos value |
|---|---|---|---|---|---|
| `save_steps` | trainer steps | **Number of training steps before saving a new opponent** (a snapshot of the current policy) | Snapshots are **further apart in skill**, so the pool **covers a wider range of skill levels and play styles** (each snapshot got more training). Harder problem ⇒ may need more total steps, but a **more general, robust** final policy | Snapshot **granularity**: fine-grained-and-similar vs coarse-and-varied opponents | `50000` |
| `team_change` | trainer steps | **Number of steps between switching which team is the learning team** | The agent **trains longer against the same set of opponents**, so it gets better at defeating *them* — but **too long risks overfitting to those particular strategies**, and it may fail against the next batch | Depth vs breadth of adaptation; also determines **how many snapshots get saved per switch** (docs recommend setting it as a multiple of `save_steps`, typically 4×–10×) | `200000` |
| `swap_steps` | **ghost** steps | **Number of steps between swapping the opponent's policy for a different snapshot** from the pool. (A *ghost step* is a step by an agent **following a fixed policy and not learning** — the distinction matters in asymmetric games such as 2v1, where the two teams collect agent-steps at different rates) | The agent **plays the same fixed opponent for longer** ⇒ **more stable** training, but **risks overfitting its behavior to that particular opponent**, so it may lose more than expected right after a swap | Within-session opponent **churn rate** (stability vs not overfitting) | `2000` |
| `window` | snapshots | **Size of the sliding window of past snapshots** from which opponents are sampled; when a new snapshot is taken, the **oldest is discarded** | The pool contains **a larger diversity of behaviors**, since it keeps policies **from earlier in the training run** — more general policy, harder problem | Opponent-pool **diversity** (and protection against cycling/forgetting how to beat old strategies) | `10` |
| `play_against_latest_model_ratio` | probability | **Probability of playing against the latest opponent policy**; with probability $1-\rho$ the agent plays a **snapshot from a past iteration** | Plays the **current** opponent more often. Since that opponent keeps updating, this is an **unstable learning environment** — but it poses an **auto-curriculum of increasingly challenging situations**, which may give a **stronger final policy** | **Freshness vs stability**: chase the moving target, or grind against a frozen history | `0.5` |
| `initial_elo` | rating | Starting ELO for the self-play bookkeeping | — | — | `1200.0` |

Typical ranges from the docs: `save_steps` 10k–100k, `swap_steps` 10k–100k, `window` 5–30, `play_against_latest_model_ratio` 0.0–1.0, `team_change` 4×–10× `save_steps`.

**`swap_steps` formula for asymmetric teams.** If you want $x$ swaps for a team of `num_agents` against a team of `num_opponent_agents` over `team_change` total steps:

$$
\texttt{swap\_steps} \;=\; \frac{\texttt{num\_agents}}{\texttt{num\_opponent\_agents}} \times \frac{\texttt{team\_change}}{x}
$$

Worked example from the docs (2v1, $x=4$ swaps over `team_change` $=200000$): the team of **one** agent gets $\tfrac{1}{2}\times\tfrac{200000}{4} = 25000$; the team of **two** gets $\tfrac{2}{1}\times\tfrac{200000}{4} = 100000$. With **equal team sizes** the first factor is 1, so `swap_steps` is just total steps ÷ desired number of swaps.

> **Key insight:** every one of these knobs is the *same* dial in disguise — **stability of learning vs diversity/generality of opponents**. Slow, static, latest-only opponents ⇒ stable but overfit. Fast, wide-window, snapshot-sampled opponents ⇒ general but noisy.

**Note on reward signals for self-play.** ML-Agents assumes the **final reward in a trajectory encodes the episode outcome**: **+1 = win, −1 = loss, 0 = draw** (the ELO calculation depends on this). The docs advise being **conservative when shaping** rewards in adversarial games, because of the instability and non-stationarity — start with the **simplest possible reward function (+1 win, −1 loss)** and allow more training iterations to compensate for its sparsity.

---

## 6. Evaluating competitive agents: the ELO rating

### 6.1 Why not cumulative reward?

In adversarial games, **tracking cumulative reward is not always a meaningful metric of learning progress, because it depends only on the skill of the opponent.** An agent scoring +0.9 per episode against a weak opponent has told you nothing about how good it is; and in self-play the opponent is changing under you, so the reward curve is measuring *two* moving things at once.

Instead we use the **ELO rating system** (named after **Arpad Elo**), which computes the **relative skill level** between 2 players from a given population in a **zero-sum game**.

| | **Cumulative reward** | **ELO rating** |
|---|---|---|
| Measures | Absolute performance in a fixed environment | **Relative** skill against a population |
| Depends on the opponent? | **Yes — entirely** (that's the problem) | Yes, **and it corrects for it** (opponent's rating is in the formula) |
| Interpretable across training? | Not in self-play — the opponent improves too | Yes — should rise as the agent gets stronger |
| Good for | Cooperative / single-agent tasks | **Adversarial** tasks, ladders, leaderboards |
| Team games | Team reward is fine | Works — **average the team's ratings** and use that |
| Failure mode | Meaningless if opponent strength varies | Deflation; ignores individual contribution; not comparable across eras |

**Zero-sum** means one agent wins and the other loses: each participant's gain or loss of utility **is exactly balanced by the gain or loss of the other participants**, so the sum of utility equals zero.

### 6.2 How it works

- ELO starts at a specific score — **frequently 1200** (`initial_elo: 1200.0` in the config). It **can decrease initially** but **should increase progressively during training**.
- The system is **inferred from the wins, losses and draws against other players**: ratings depend on **the ratings of the opponents and the results scored against them**.
- The central idea is to treat a player's performance as a **random variable that is normally distributed**, so the **difference in rating between two players predicts the outcome of a match**.
- If a player wins but the win **was highly probable**, it **only takes a few points**, because it was already known to be much stronger.

After every game, **the winner takes points from the loser**, and the number of points is set by **the difference in the two ratings** (hence *relative*):

| Outcome | Points transferred |
|---|---|
| **Higher**-rated player wins | **Few** points taken from the lower-rated player |
| **Lower**-rated player wins | **A lot** of points taken from the higher-rated player |
| **Draw** | The **lower**-rated player gains a few points from the higher |

### 6.3 The two formulas

For players $A$ and $B$ with ratings $R_A$ and $R_B$, the **expected scores** are

$$
E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}, \qquad E_B = \frac{1}{1 + 10^{(R_A - R_B)/400}}
$$

so that $E_A + E_B = 1$ (a 400-point gap corresponds to 10:1 expected odds). Then, at the end of the game, we update the actual rating with a **linear adjustment proportional to the amount by which the player over- or under-performed**:

$$
\boxed{\,R_A' = R_A + K\,\bigl(S_A - E_A\bigr)\,}
$$

- $S_A$ is the **actual score**: $\;S_A = 1$ for a **win**, $\;0.5$ for a **draw**, $\;0$ for a **loss**.
- $E_A \in (0,1)$ is the **expected score** above.
- $K$ is the **K-factor**: the **maximum adjustment rating per game**. The course's values: **$K = 16$ for masters**, **$K = 32$ for weaker players**. Large $K$ ⇒ ratings react fast but jitter; small $K$ ⇒ stable but slow to reflect real improvement.

Because $E_A + E_B = 1$ and $S_A + S_B = 1$, the two updates are exactly opposite: $K(S_A - E_A) = -K(S_B - E_B)$. **The total rating in the pool is conserved** — that's the zero-sum property showing up in the bookkeeping.

### 6.4 Worked example (verbatim from the course)

Player **A** rated **2600**, player **B** rated **2300**, and $K = 16$.

Expected scores:

$$
E_A = \frac{1}{1 + 10^{(2300 - 2600)/400}} = 0.849, \qquad E_B = \frac{1}{1 + 10^{(2600 - 2300)/400}} = 0.151
$$

**If A wins** ($S_A = 1$, $S_B = 0$):

$$
\text{ELO}_A = 2600 + 16 \times (1 - 0.849) = 2602, \qquad \text{ELO}_B = 2300 + 16 \times (0 - 0.151) = 2298
$$

**If B wins** ($S_A = 0$, $S_B = 1$):

$$
\text{ELO}_A = 2600 + 16 \times (0 - 0.849) = 2586, \qquad \text{ELO}_B = 2300 + 16 \times (1 - 0.151) = 2314
$$

Note the asymmetry, which is the whole point: the favourite winning moves the ratings by **≈2 points**, the underdog winning moves them by **≈14**.

### 6.5 Advantages and disadvantages

**Advantages**
- Points are **always balanced** — more points are exchanged on an unexpected outcome, but **the sum is always the same**.
- It's a **self-correcting system**: beating a weak player only wins you a few points, so you can't farm rating off easy opponents.
- It **works with team games**: **compute the average rating for each team** and use it in the ELO formula.

**Disadvantages**
- It **does not take into account the individual contribution** of each member of the team.
- **Rating deflation**: **keeping the same rating requires sustained skill over time** — it's not "easy to keep a high score".
- You **can't compare ratings across history** (a 2600 in one era isn't a 2600 in another).

---

## 7. Hands-on context: SoccerTwos, `poca`, and AI vs AI

### 7.1 The environment

**`SoccerTwos`** (Unity ML-Agents team): **four agents compete in a 2v2 toy soccer game**. The goal is **to get the ball into the opponent's goal while preventing the ball from entering your own goal** — i.e. the **mixed** cooperative/competitive setting. There are **two Multi-Agent Groups with two agents each**, and the provided executable contains **8 copies of SoccerTwos** running in parallel.

| Piece | Spec |
|---|---|
| **Observation space** | **Vector of size 336**: **11 ray-casts forward** over **120°** (**264** dims) + **3 ray-casts backward** over **90°** (**72** dims), each detecting **6 object types** — Ball, Blue Goal, Purple Goal, Wall, Blue Agent, Purple Agent — plus the object's distance |
| **Action space** | **3 discrete branches** — forward/backward movement, sideways movement, and rotation |
| **Visual observations** | None |
| **Reward (team-dependent)** | **$(1 - \text{accumulated time penalty})$ when the ball enters the opponent's goal**, where the penalty is incremented by $1/\texttt{MaxStep}$ every fixed update and reset to 0 at episode start; **$-1$ when the ball enters your own goal** |

So the reward is a **team reward** — it lands on both teammates regardless of who actually did the work. That's precisely the credit-assignment problem MA-POCA exists for (§3.4). **The recipe is therefore: self-play (to beat the other team) + the MA-POCA trainer (to learn cooperation inside your team).** In the config that trainer is called `poca`.

### 7.2 The config file

`./config/poca/SoccerTwos.yaml` — the parts worth remembering:

```yaml
behaviors:
  SoccerTwos:
    trainer_type: poca
    hyperparameters:
      batch_size: 2048
      buffer_size: 20480
      learning_rate: 0.0003
      beta: 0.005            # entropy regularization
      epsilon: 0.2           # PPO-style clipping
      lambd: 0.95            # GAE lambda
      num_epoch: 3
      learning_rate_schedule: constant
    network_settings:
      normalize: false
      hidden_units: 512
      num_layers: 2
      vis_encode_type: simple
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
    keep_checkpoints: 5
    max_steps: 5000000
    time_horizon: 1000
    summary_freq: 10000
    self_play:
      save_steps: 50000
      team_change: 200000
      swap_steps: 2000
      window: 10
      play_against_latest_model_ratio: 0.5
      initial_elo: 1200.0
```

Training is launched with `mlagents-learn <config> --env=<executable> --run-id="SoccerTwos" --no-graphics`, and pushed with `mlagents-push-to-hf --run-id=... --local-dir=./results/<run_id> --repo-id=<user>/<repo> --commit-message=...`.

Practical notes from the course:
- **5M timesteps (recommended; 10M also worth trying) takes ~4–8 hours** — do it locally, not in Colab (timeout risk). A laptop is enough.
- ⚠️ **It's normal not to see the ELO rise — and even to see it fall below 1200 — before ~2M timesteps**, because the agents spend that time moving essentially at random on the field before they can score.
- ⚠️ **Press `Ctrl-C` only once** to stop training: ML-Agents needs to write the final `.onnx` file before closing.
- **What makes the difference in the challenge are the hyperparameters** — the observation/action spaces and the trainer must not be changed, or the model won't evaluate.

### 7.3 AI vs AI: the ladder

**AI vs AI** is the open-source tool built for this unit to **compete agents on the Hub against one another in a multi-agent setting** and rank them on a leaderboard. The point is **robust evaluation: by evaluating your agent against many others, you get a good idea of the quality of your policy** — exactly the argument for ELO over reward.

It is three tools:
1. A **matchmaking process** that defines the matches (which model against which) and runs the fights as a background task in a Space.
2. A **leaderboard** collecting match history and displaying **ELO ratings** — [AIvsAI-SoccerTwos](https://huggingface.co/spaces/huggingface-projects/AIvsAI-SoccerTwos).
3. A **Space demo** to watch your agent play against others — [ML-Agents-SoccerTwos](https://huggingface.co/spaces/unity/ML-Agents-SoccerTwos). Plus a community-built [SoccerTwos Challenge Analytics](https://huggingface.co/spaces/cyllum/soccertwos-analytics) for detailed per-model results.

The loop: **every four hours** the algorithm **fetches all available models for the environment** (selected by the `ML-Agents-SoccerTwos` tag, with a `SoccerTwos.onnx` file in the repo), **builds a match queue**, **simulates each match in a headless Unity process** recording **1 / 0.5 / 0** for win / draw / loss, and when the queue is done **updates every model's ELO and the leaderboard**. Note that those three result values are exactly the $S_A$ of §6.3 — the ladder is literally running the ELO update above.

> ⚠️ Matches you watch live in the demo Space are **not** counted toward your rating; they're only for visualization. And per the maintenance notice, the leaderboard itself is now offline.

---

## 8. Unit 7 in one screen

- **MARL** = multiple agents **sharing and interacting in a common environment**, and interacting with each other.
- Environments are **cooperative** (maximize common benefit), **competitive** (maximize mine, minimize yours), or **mixed** (SoccerTwos).
- **Decentralized**: train each agent independently, others are just environment ⇒ easy, reuses single-agent algorithms, but the environment is **non-stationary** ⇒ **no convergence guarantee**.
- **Centralized**: one shared experience buffer, **one joint policy**, **global reward** ⇒ **stationary**, but needs global information and scales badly.
- **CTDE / MA-POCA**: **centralized critic during training** ("the coach"), **local observations at execution** — solves team-reward credit assignment.
- **Self-play**: opponents are **former copies of your own policy** ⇒ always matched in skill, improves as you improve, an automatic curriculum. A fixed random or fixed expert opponent both fail.
- **Five knobs**, one trade-off (stability ↔ diversity): `save_steps` (snapshot spacing), `team_change` (which team learns), `swap_steps` (opponent churn), `window` (pool size), `play_against_latest_model_ratio` (latest vs historical).
- **ELO**, not reward, for adversarial evaluation: $E_A = 1/\bigl(1+10^{(R_B-R_A)/400}\bigr)$ and $R_A' = R_A + K(S_A - E_A)$, zero-sum, self-correcting, team-averageable — but blind to individual contribution and subject to deflation.

---

## 9. Self-check (quick review questions)

1. What exactly changes when you move from a single-agent to a multi-agent system? Name three course examples.
2. Define **cooperative**, **competitive** and **mixed** environments in one line each, with the course's example for each. Which one is SoccerTwos?
3. In **decentralized** learning, what does the training agent treat the other agents as? Why is that convenient?
4. Why does that convenience make the environment **non-stationary**, and why does non-stationarity break standard RL convergence arguments?
5. In the **centralized** approach, what does the high-level process collect, what is learned from it, and what is the reward? Why is the environment stationary again?
6. What is **CTDE**, and what specifically is centralized in **MA-POCA**? What problem does its "coach" critic solve?
7. Why can't you train a competitive agent against a **fixed random** opponent? And why not against a **fixed very strong** one? (Use the child-learning-soccer analogy.)
8. State the core idea of **self-play** in one sentence. What plays the role of the opponent?
9. Match each hyperparameter to its definition: *probability of playing the current self vs a pooled opponent* / *diversity of training levels among the opponents you face* / *number of training steps before spawning a new opponent* / *opponent change rate* → `window`, `play_against_latest_model_ratio`, `save_steps`, `swap_steps` + `team_change`.
10. What single trade-off do **all** the self-play hyperparameters control? Which direction risks **overfitting to one opponent**, and which risks **unstable learning**?
11. What is a **ghost step**, and why does ML-Agents distinguish it from a trainer step? (Think 2v1.)
12. Why is **cumulative reward** a poor progress metric in adversarial games?
13. Write both ELO formulas from memory. What are $K$ and $S_A$, and what are the three possible values of $S_A$?
14. Redo the worked example: A = 2600, B = 2300, $K=16$. What is $E_A$? What are the new ratings if A wins? If B wins? Why is the second case a much bigger swing?
15. Show that the ELO update is **zero-sum** (the pool's total rating is conserved).
16. Give the three **advantages** and three **disadvantages** of ELO that the course lists. How is ELO applied to a *team*?
17. SoccerTwos: what is the size of the observation vector, how is it built up, and what are the three action branches? What is the reward when the ball enters the opponent's goal?
18. In the AI vs AI ladder, what three values can a match result take, and where do they appear in the ELO formula?
19. Why is it normal for your SoccerTwos ELO to drop **below** its `initial_elo` of 1200 in the first ~2M steps?

---

## 10. Additional Readings

These are **optional** if you want to go deeper.

**An introduction to multi-agents**
- [Multi-agent reinforcement learning: An overview](https://www.dcsc.tudelft.nl/~bdeschutter/pub/rep/10_003.pdf) — L. Buşoniu, R. Babuška, B. De Schutter
- [Multiagent Reinforcement Learning](https://rlss.inria.fr/files/2019/07/RLSS_Multiagent.pdf) — Marc Lanctot (RLSS 2019 lecture slides)
- [Example of a multi-agent environment](https://www.mathworks.com/help/reinforcement-learning/ug/train-3-agents-for-area-coverage.html?s_eid=PSM_15028) — MathWorks, train 3 agents for area coverage
- [A list of different multi-agent environments](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/) — Edinburgh Autonomous Agents Group
- [Multi-Agent Reinforcement Learning: Independent vs. Cooperative Agents](https://bit.ly/3nVK7My) — Ming Tan (1993), the classic independent-vs-cooperative learners paper
- [Dealing with Non-Stationarity in Multi-Agent Deep Reinforcement Learning](https://bit.ly/3v7LxaT) — Papoudakis, Christianos, Rahman, Albrecht

**Self-Play and MA-POCA**
- [Training intelligent adversaries using self-play with ML-Agents](https://blog.unity.com/technology/training-intelligent-adversaries-using-self-play-with-ml-agents) — Andrew Cohen, Unity (also the recommended read on the *history* of self-play: Samuel's checkers, TD-Gammon)
- [ML-Agents v2.0 release: now supports training complex cooperative behaviors](https://blog.unity.com/technology/ml-agents-v20-release-now-supports-training-complex-cooperative-behaviors) — Unity
- [ML-Agents plays dodgeball](https://blog.unity.com/technology/ml-agents-plays-dodgeball) — Unity
- [On the Use and Misuse of Absorbing States in Multi-agent Reinforcement Learning (MA-POCA)](https://arxiv.org/pdf/2111.05992.pdf) — Cohen, Teng, Berges, Dong, Henry, Mattar, Zook, Ganguly (arXiv:2111.05992)

**Reference docs used in the hands-on**
- [ML-Agents self-play configuration reference](https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Training-Configuration-File.md#self-play) — every hyperparameter of §5 with typical ranges
- [SoccerTwos environment documentation](https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Learning-Environment-Examples.md#soccer-twos) — Unity ML-Agents
- [Introducing AI vs. AI](https://huggingface.co/blog/aivsai) — Hugging Face blog post explaining the ladder in detail
- [Introduction to Multi-Agent Reinforcement Learning](https://www.youtube.com/watch?v=qgb0gyrpiGk) — Brian Douglas (the vacuum-cleaner video behind §3)

---

> **Next up:** **Proximal Policy Optimization (PPO)** — deriving the clipped surrogate objective that keeps policy updates inside a trust region, the workhorse behind (among many other things) the `epsilon: 0.2` we just copied into the SoccerTwos config.
