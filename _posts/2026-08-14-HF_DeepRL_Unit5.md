---
layout: post
title:  "HF Deep RL Course — Unit 5 (Unity ML-Agents)"
date:   2026-08-14
desc: "HF Deep RL Course — Unit 5 (Unity ML-Agents)"
keywords: "Machine learning"
categories: [Machine learning]
tags: [Machine learning]
icon: icon-html
---

# HF Deep RL Course — Unit 5 (Unity ML-Agents)

> Source: [Hugging Face Deep RL Course, Unit 5](https://huggingface.co/learn/deep-rl-course/unit5/introduction) — notes from *Introduction* through the *Bonus* page, reorganized for review. Continues from [Unit 4]({{ '/machine learning/2026/08/14/HF_DeepRL_Unit4.html' | replace: ' ', '%20' | prepend: site.baseurl }}).

> **_Keypoints:_**

- Game engines as environment builders; what the **Unity ML-Agents Toolkit** is
- The **six components** of ML-Agents (Learning Environment, Python Low-level API, External Communicator, Python trainers, Gym & PettingZoo wrappers)
- Inside the Learning Environment: the **Agent** (whose policy is called the *Brain*) and the **Academy** that orchestrates the simulation loop
- Observation options: **vector** vs **visual/camera** vs **raycasts**
- **SnowballTarget**: raycast observations, discrete actions, dense `+1`-per-hit reward, and the *reward engineering* trap
- **Pyramids**: a **sparse-reward** task (switch → pyramid → knock over → gold brick)
- **Curiosity** as an **intrinsic reward**; the **ICM** forward + inverse models; prediction error as reward
- The training workflow: `config.yaml` hyperparameters + `reward_signals`, `mlagents-learn`, TensorBoard, `mlagents-push-to-hf`
- Bonus: building your own Unity environments; self-check + further resources

---

## 0. What this unit is for

One of the real challenges in RL is **creating environments**. Game engines — [Unity](https://unity.com/), [Godot](https://godotengine.org/), [Unreal](https://www.unrealengine.com/) — are already built for exactly the things an environment needs: **physics systems, 2D/3D rendering**, collision, spawning, cameras.

Unity shipped the [**Unity ML-Agents Toolkit**](https://github.com/Unity-Technologies/ml-agents): a **plugin for the Unity game engine that lets you use Unity as an environment builder to train agents**. It also ships many excellent **pre-made environments** (football/soccer, learning to walk, jumping over big walls, …). This is the same toolkit used in Bonus Unit 1 to train Huggy to catch a stick.

> **You do not need to know Unity to do this unit.** You only need Unity if you want to *author* new environments (see §9). Training uses a pre-built executable + a Python CLI.

Two agents get trained in this unit:

1. **SnowballTarget** — learn to **shoot snowballs onto a spawning target**.
2. **Pyramids** — **press a button to spawn a pyramid, navigate to it, knock it over, and move to the gold brick on top**. This one requires real *exploration*, done with a technique called **curiosity**.

Afterwards the trained models get **pushed to the Hugging Face Hub** and can be **watched playing in the browser**, with no Unity Editor involved. This unit is the setup for the next challenge: **AI vs. AI** in multi-agent environments.

> This unit is **engineering-heavy, not math-heavy**. The things worth memorizing are the *components*, the *config file*, and the *curiosity* idea.

---

## 1. What is Unity ML-Agents?

> **Unity ML-Agents** is a toolkit for the Unity game engine that **allows us to create environments using Unity, or to use pre-made environments, to train our agents**.

- Developed by **Unity Technologies** — the makers of Unity, the engine behind *Firewatch*, *Cuphead*, and *Cities: Skylines*.
- The **environment lives in C#/Unity**; the **learning algorithms live in Python/PyTorch**. The whole design is about getting those two worlds to talk to each other cleanly.

That split is the single most important architectural fact about ML-Agents, and it's why there's an explicit *communicator* component.

---

## 2. How ML-Agents works — the six components

| # | Component | Language / side | Role |
|---|---|---|---|
| 1 | **Learning Environment** | C# / Unity | Contains **the Unity scene (the environment) and its elements** (game characters). This is the simulation. |
| 2 | **Python Low-level API** | Python | The **low-level Python interface for interacting with and manipulating the environment**. *This is the API we use to launch training* — not the Unity/C# side. |
| 3 | **External Communicator** | bridge | **Connects the Learning Environment (C#) to the low-level Python API (Python).** |
| 4 | **Python trainers** | Python / PyTorch | The **RL algorithms implemented in PyTorch** (PPO, SAC, …). |
| 5 | **Gym wrapper** | Python | Encapsulates the RL environment in a **Gym** wrapper (so standard single-agent tooling works). |
| 6 | **PettingZoo wrapper** | Python | **PettingZoo is the multi-agent version of the Gym wrapper.** |

> **Key insight:** you write/inspect the *world* in Unity, but you **launch and drive training from Python**. The External Communicator is the only thing that makes those two halves one system.

### 2.1 Inside the Learning Environment

Two important elements live inside it:

| Element | What it is | What it does |
|---|---|---|
| **Agent component** | The **actor of the scene** | The thing we train. **We train the agent by optimizing its policy** (which tells us what action to take in each state). In ML-Agents the policy is called the **Brain**. |
| **Academy** | The orchestrator | **Orchestrates agents and their decision-making processes.** Think of it as a **teacher who handles the Python API's requests** and keeps all agents **in sync**. |

### 2.2 The Academy's role in the simulation loop

Recall the RL loop from Unit 1 — a loop of state, action, reward, next state. For an agent learning a platform game:

- The Agent receives **state $S_0$** from the Environment (the first frame of the game).
- Based on $S_0$ the Agent takes **action $A_0$** (e.g. move right).
- The environment moves to a **new state $S_1$** (new frame).
- The environment gives a **reward $R_1$** (we're not dead → $+1$).

The loop outputs a sequence of **state, action, reward, next state**, and the agent's goal is to **maximize the expected cumulative reward**.

The **Academy is what sends the orders to the Agents and guarantees they stay in sync**, stepping them through:

1. **Collect observations**
2. **Select an action using your policy**
3. **Take the action**
4. **Reset** if you reached the max step or if you're `done`

That's it — the Academy is essentially the environment-side scheduler for the RL loop, batched over however many agent instances the scene contains (ML-Agents scenes usually clone the same training area many times to collect experience in parallel).

---

## 3. How the Agent senses, acts, and is rewarded

Every ML-Agents Agent is defined by three things you must design:

| Design choice | Question it answers |
|---|---|
| **Observations** | What does the agent *see*? |
| **Actions** | What can the agent *do*? |
| **Reward** | What does the agent *want*? |

### 3.1 The three observation options

| Observation type | What it is | When to use it | Cost |
|---|---|---|---|
| **Vector observations** | A flat float vector of hand-picked state values (positions, velocities, booleans, distances) | You know what matters and can enumerate it; fastest to learn from | **Cheapest** |
| **Visual / camera observations** | Rendered camera frames fed to a CNN encoder (`vis_encode_type` in the config) | The state is genuinely visual / hard to enumerate | **Most expensive**; slow to render and to train |
| **Raycasts** | "**Lasers**" cast from the agent that report **whether they pass through an object**, which *tag* of object they hit, and at what distance | Navigation / detection tasks — a huge amount of spatial information at a tiny fraction of the cost of pixels | **Cheap**, and the sweet spot |

> **Both environments in this unit use raycasts, not frames.** That is deliberate: raycasts give the agent the *geometry* it needs (where's the target, where's the wall) without paying for rendering + a convolutional encoder. Raycast hits are ultimately delivered to the network as part of the vector observation.

### 3.2 Actions and rewards

- **Actions** are either **discrete** (one or more *branches*, each branch being a mutually exclusive choice — e.g. a "rotate" branch with `{left, none, right}`) or **continuous** (a vector of floats). Both unit-5 environments are **discrete**.
- The **reward** is emitted by the C# `Agent` script via `AddReward` / `SetReward`, and the episode is ended with `EndEpisode`. The Python side never sees the C# code — only the numbers.

---

## 4. The SnowballTarget environment

SnowballTarget is an environment **created at Hugging Face** using assets from [Kay Lousberg](https://kaylousberg.com/).

### 4.1 The agent's goal

The agent is **Julien the bear 🐻**, trained **to hit targets with snowballs**.

- **Goal:** hit **as many targets as possible in the limited time** — **1000 timesteps** per episode.
- To do that it must **place itself correctly in relation to the target and shoot**.
- **Cool-off system:** to avoid "snowball spamming" (shooting every single timestep), Julien must **wait 0.5 seconds after a shot before shooting again**.

### 4.2 The reward function, and the reward engineering problem

The reward function is deliberately simple:

$$
r_t = \begin{cases} +1 & \text{if the agent's snowball hits a target} \\ 0 & \text{otherwise} \end{cases}
$$

Since the agent maximizes expected cumulative reward, **maximizing return = hitting as many targets as possible**. Nothing else needs to be said.

> ⚠️ **The reward engineering problem.** We *could* write something fancier (a time penalty to push the agent to go faster, an aiming bonus, …). But over-engineering the reward to **force the agent to behave the way you imagined** is a trap: **you might miss interesting strategies the agent would have found on its own with a simpler reward function**. Prefer the simplest reward that actually encodes the goal.

Note also that this reward is **dense enough** to learn from: a randomly flailing bear will hit *something* eventually. Contrast with Pyramids (§5).

### 4.3 Observation space

Not frames — **raycasts**, in **multiple sets** (different fans/angles, detecting targets and walls), **plus a `can I shoot?` boolean** so the agent can learn around the cool-off timer.

> That boolean is a nice illustration of a general rule: **if part of the environment's state is hidden from the agent's sensors, you must hand it in explicitly**, or the problem stops being Markovian. Without "can I shoot", the agent cannot tell a wasted trigger-pull from a real one.

### 4.4 Action space

**Discrete.** The published `SnowballTarget.onnx` model exposes **three discrete branches with 3, 3 and 2 options** (8 logits total) — i.e. *move forward/back/idle*, *rotate left/right/idle*, and *shoot / don't shoot*. Its observation inputs are five separate tensors (sizes 27, 99, 99, 3 and 132) — the several raycast sensors plus the small extra vector.

**Benchmark to aim for:** **mean reward ≈ 15**, i.e. roughly **30 targets shot in an episode**.

---

## 5. The Pyramids environment

Made by **the Unity team** (it's one of the toolkit's built-in example environments).

### 5.1 The task

Train the agent to **get the gold brick on the top of the pyramid**. To do that it must:

1. **Press a button** (the switch) to **spawn a pyramid**,
2. **navigate to the pyramid**,
3. **knock it over**,
4. and **move to the gold brick at the top**.

### 5.2 Reward function

| Event | Reward |
|---|---|
| Moving to the **golden brick** | **+2** |
| Every step | **−0.001** (a small time penalty) |

**Benchmark mean reward:** **1.75**.

### 5.3 Why this is a *sparse reward* problem

Look at the reward table again. **The only positive reward in the entire environment is at the very end of a four-stage chain.** A randomly-acting agent must, by luck: find the switch, press it, find the newly spawned pyramid, ram it over, and then climb to the brick — before any signal at all arrives. Until that happens, every reward it sees is the tiny constant $-0.001$, which contains **no information about which action was good**.

So Pyramids is trained with **two reward signals combined**:

- the **extrinsic** reward given by the environment (the table above), and
- an **intrinsic** one called **curiosity**, which **pushes the agent to be curious — i.e. to better explore its environment**.

### 5.4 Observation & action space

- **Observation:** **148 raycasts**, each able to **detect objects — switch, bricks, golden brick, and walls** — plus a **boolean indicating the switch state** (has the switch been turned on to spawn the pyramid?) and a **vector containing the agent's speed**.
- **Action:** **discrete**, **1 branch with 4 possible actions**, corresponding to **agent rotation and forward/backward movement**.

The switch boolean is the same trick as SnowballTarget's "can I shoot": the *stage* of the task is not visible from geometry alone, so it's handed to the agent directly.

---

## 6. Curiosity: intrinsic reward for exploration

*(The course marks this section optional. It is the one genuinely conceptual piece of Unit 5 — do not skip it.)*

### 6.1 Two major problems in modern RL

**Problem 1 — the sparse rewards problem.** **Most rewards contain no information, and hence are set to zero.** RL rests on the *reward hypothesis*: every goal can be described as the maximization of rewards, so **rewards are the agent's feedback**. If the agent **receives no reward, its knowledge of which action is appropriate cannot change** — there is literally nothing to learn from.

The course's example: in [**ViZDoom**](https://vizdoom.cs.put.edu.pl/)'s `DoomMyWayHome`, the agent is **only rewarded if it finds the vest**. The vest is far from the start, so **almost all rewards are zero**, and the agent can **spend all its time turning around without ever finding the goal**.

**Problem 2 — the extrinsic reward function is handmade.** In every environment, **a human has to implement a reward function**. That does not scale to big, complex worlds.

### 6.2 The idea

> A solution to both problems is **a reward function intrinsic to the agent — generated by the agent itself.** The agent becomes a **self-learner: both the student and its own feedback master.**

**This intrinsic reward mechanism is known as Curiosity**, because the reward **pushes the agent to explore states that are novel/unfamiliar**: the agent **receives a high reward when exploring new trajectories**. It's modelled on humans, who **naturally have an intrinsic desire to explore and discover new things**.

The total reward used for learning is then the sum of the two signals, each with its own weight (`strength` in the config):

$$
r_t \;=\; \underbrace{r_t^{e}}_{\text{extrinsic, from the environment}} \;+\; \beta \,\underbrace{r_t^{i}}_{\text{intrinsic, from curiosity}}
$$

> ⚠️ $\beta$ (i.e. `curiosity -> strength`) must be **large enough not to be drowned out by the extrinsic reward, and small enough not to drown it out** — typical range `0.001`–`0.1`. Get it wrong in one direction and exploration never kicks in; wrong in the other and you've trained a tourist that never collects the gold brick.

### 6.3 Curiosity through next-state prediction

The classical approach computes curiosity as **the agent's error in predicting the next state, given the current state and the action taken**:

$$
r_t^{i} \;\propto\; \big\lVert\, \hat{s}_{t+1} - s_{t+1} \,\big\rVert^2, \qquad \hat{s}_{t+1} = f(s_t, a_t)
$$

The logic:

- Curiosity **encourages actions that reduce the uncertainty in the agent's ability to predict the consequences of its actions**. Uncertainty is higher **in areas where the agent has spent less time, or in areas with complex dynamics**.
- If the agent has **spent a lot of time in a state, it predicts the next state well → low curiosity reward**.
- If the state is **new and unexplored, the next state is hard to predict → high curiosity reward**.

So curiosity **makes the agent favour high-prediction-error transitions, and consequently explore the environment better**.

### 6.4 The Intrinsic Curiosity Module (ICM): two networks

Predicting raw next states is a bad idea (predicting every leaf and pixel is hopeless and irrelevant). The **ICM** of Pathak et al. therefore does the prediction in a **learned feature space $\phi(\cdot)$**, using **two networks**:

| Network | Input | Output | Loss | Purpose |
|---|---|---|---|---|
| **Forward model** | $\phi(s_t)$ and $a_t$ | predicted next-state features $\hat{\phi}(s_{t+1})$ | squared error vs. the true $\phi(s_{t+1})$ | **Its loss *is* the curiosity reward** — "how surprised was I?" |
| **Inverse model** | $\phi(s_t)$ and $\phi(s_{t+1})$ | predicted action $\hat{a}_t$ taken between them | cross-entropy (discrete) vs. the true $a_t$ | **Trains the encoder $\phi$** so features encode only **what the agent can influence** |

Formally:

$$
\hat{\phi}(s_{t+1}) = f\big(\phi(s_t), a_t\big), \qquad
L_{F} = \tfrac{1}{2}\big\lVert \hat{\phi}(s_{t+1}) - \phi(s_{t+1}) \big\rVert_2^2
$$

$$
\boxed{\; r_t^{i} \;=\; \tfrac{\eta}{2}\big\lVert \hat{\phi}(s_{t+1}) - \phi(s_{t+1}) \big\rVert_2^2 \;}
$$

$$
\hat{a}_t = g\big(\phi(s_t), \phi(s_{t+1})\big), \qquad L_{I} = \text{CE}\big(\hat{a}_t,\, a_t\big)
$$

> **Why the inverse model is the clever part.** If $\phi$ were trained *only* by the forward model, the encoder could cheat — collapse to a constant (prediction error zero, no curiosity ever) — and, worse, a plain next-state predictor rewards the agent for staring at **unpredictable things it cannot control** (the classic "noisy TV": random flicker is forever surprising, so a naive curious agent watches TV forever). The **inverse model can only recover $a_t$ from $\phi(s_t), \phi(s_{t+1})$ if $\phi$ retains exactly the parts of the state the agent's actions affect** — so it filters out uncontrollable noise. The forward model then measures surprise *in that filtered space*.

**In ML-Agents this is one line of config.** The `curiosity` reward signal **enables the Intrinsic Curiosity Module**, an implementation of *Curiosity-driven Exploration by Self-supervised Prediction* (Pathak et al.), training exactly those two networks; **the forward model's loss is used as the intrinsic reward — the more surprised the model is, the larger the reward.**

### 6.5 Other ways to compute curiosity: RND

There are **other curiosity calculation methods**. ML-Agents also ships **`rnd` — Random Network Distillation**, and the toolkit's shipped Pyramids config (`PyramidsRND.yaml`) is the one the hands-on actually uses.

| | **`curiosity` (ICM)** | **`rnd` (Random Network Distillation)** |
|---|---|---|
| Networks | **Forward** + **inverse** model over learned features | A **fixed random-weight** encoder + a **trained predictor** of its output |
| Intrinsic reward | Forward-model prediction error | Squared error between predictor and the frozen random network |
| Handles uncontrollable noise | Yes — the inverse model filters $\phi$ | Yes — there's no dynamics model to be fooled by stochastic transitions |
| Novelty signal decays because… | The agent gets good at predicting *visited* dynamics | **The more an agent visits a state, the more accurate the prediction and the lower the reward** |

Both push the agent toward states with **high prediction error**, i.e. states it hasn't seen. Also worth knowing: `gail` is a third intrinsic-style signal in ML-Agents, but that one comes from **demonstrations**, not novelty.

---

## 7. SnowballTarget vs Pyramids, side by side

| | **SnowballTarget** | **Pyramids** |
|---|---|---|
| Author | Hugging Face (Kay Lousberg assets) | The Unity team (built-in example) |
| Agent | Julien the bear 🐻 | A generic cube agent |
| Goal | **Hit as many targets as possible** in 1000 timesteps | **Reach the gold brick** on top of the pyramid |
| Sub-tasks | Aim + shoot (with a 0.5 s cool-off) | Press switch → find pyramid → knock over → climb to brick |
| Observations | **Multiple raycast sets** + a **`can I shoot` bool** | **148 raycasts** (switch, bricks, golden brick, walls) + **switch-state bool** + **agent speed vector** |
| Action space | **Discrete**, 3 branches (3 / 3 / 2) | **Discrete**, 1 branch, **4 actions** (rotate + forward/back) |
| Reward | **+1 per target hit** | **+2** for the golden brick, **−0.001 per step** |
| Reward density | **Dense-ish** — random shooting scores sometimes | **Sparse** — one payout at the end of a 4-stage chain |
| Reward signals used | `extrinsic` only | `extrinsic` **+ curiosity/`rnd`** |
| Trainer | PPO | PPO |
| `max_steps` used | 200 000 | 1 000 000 (lowered from the shipped 3 000 000) |
| Benchmark mean reward | **15** (≈ 30 targets) | **1.75** |
| Wall-clock (Colab) | ~10–35 min | ~30–45 min |

> **The one-line takeaway of this unit:** *dense reward → plain PPO is fine; sparse reward → you need an intrinsic reward signal to get exploration off the ground.*

---

## 8. The training workflow

### 8.1 The configuration YAML

> In ML-Agents, you define the **training hyperparameters in `config.yaml` files** — one block per *behavior name*, and the behavior name must match the one set on the Agent in the Unity scene.

The SnowballTarget config from the hands-on:

```yaml
behaviors:
  SnowballTarget:
    trainer_type: ppo
    summary_freq: 10000
    keep_checkpoints: 10
    checkpoint_interval: 50000
    max_steps: 200000
    time_horizon: 64
    threaded: true
    hyperparameters:
      learning_rate: 0.0003
      learning_rate_schedule: linear
      batch_size: 128
      buffer_size: 2048
      beta: 0.005
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3
    network_settings:
      normalize: false
      hidden_units: 256
      num_layers: 2
      vis_encode_type: simple
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
```

What each knob means (this is the table to actually remember):

| Setting | Meaning |
|---|---|
| `trainer_type: ppo` | Which algorithm. ML-Agents implements **PPO** (default, general-purpose and stable) and **SAC** (off-policy, replay buffer, 5–10× more sample-efficient but more model updates — good when the *simulation* is the bottleneck). |
| `max_steps` | Total environment steps to train for. |
| `time_horizon` | How many steps of experience to collect per agent before adding it to the buffer (bootstraps the value estimate if the episode hasn't ended). |
| `batch_size` | Number of experiences per gradient descent update. |
| `buffer_size` | Experiences to collect before updating the policy (must be a multiple of `batch_size`); larger = more stable, slower. |
| `learning_rate` (+ `_schedule: linear`) | Gradient-descent step size, decayed linearly to 0 over `max_steps`. |
| `beta` | Strength of the **entropy regularization** — keeps the policy stochastic, i.e. keeps it exploring. Too high → the agent never commits. |
| `epsilon` | The **PPO clipping** threshold on the probability ratio — how far the new policy may move from the old one per update. Small = stable but slow. |
| `lambd` | The $\lambda$ of **GAE** — how much the value estimate is trusted vs. the observed rewards (bias/variance dial). |
| `num_epoch` | Passes over the experience buffer per update. Lowering it makes updates more stable. |
| `network_settings` | `hidden_units`, `num_layers`, `normalize` (observation normalization), `vis_encode_type` (the CNN used *only* for visual observations). |
| `reward_signals` | One block per signal, each with its own `gamma` and `strength`. This is where **`extrinsic`**, **`curiosity`**, **`rnd`** and **`gail`** are enabled/weighted. |
| `summary_freq` / `checkpoint_interval` / `keep_checkpoints` | How often to write TensorBoard summaries; how often to snapshot the model; how many snapshots to keep. |

**Adding curiosity** is purely additive — a second entry under `reward_signals`:

```yaml
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
      curiosity:              # enables the Intrinsic Curiosity Module (ICM)
        gamma: 0.99
        strength: 0.02        # typical range 0.001 - 0.1
        learning_rate: 0.0003 # typical range 1e-5 - 1e-3
        network_settings:
          hidden_units: 256   # small enough to compress the observation
```

And the shipped `config/ppo/PyramidsRND.yaml` (the hands-on only edits `max_steps` down to `1000000`) uses the RND variant:

```yaml
behaviors:
  Pyramids:
    trainer_type: ppo
    hyperparameters:
      batch_size: 128
      buffer_size: 2048
      learning_rate: 0.0003
      beta: 0.01
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3
      learning_rate_schedule: linear
    network_settings:
      normalize: false
      hidden_units: 512
      num_layers: 2
      vis_encode_type: simple
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
      rnd:                     # <- the intrinsic/curiosity signal
        gamma: 0.99
        strength: 0.01
        network_settings:
          hidden_units: 64
          num_layers: 3
        learning_rate: 0.0001
    keep_checkpoints: 5
    max_steps: 3000000         # <- change to 1000000
    time_horizon: 128
    summary_freq: 30000
```

Note the contrasts with SnowballTarget: **bigger network** (512 vs 256), **longer `time_horizon`** (128 vs 64, because credit must travel further), **higher `beta`** (0.01 vs 0.005, more entropy → more exploration), and the extra **intrinsic reward signal**.

### 8.2 Commands

Training is `mlagents-learn <config>` plus the environment executable:

```bash
# get the toolkit
git clone --depth 1 https://github.com/Unity-Technologies/ml-agents
cd ml-agents
pip3 install -e ./ml-agents-envs
pip3 install -e ./ml-agents

# fetch + unpack a prebuilt Linux environment executable
mkdir -p ./training-envs-executables/linux
wget "https://github.com/huggingface/Snowball-Target/raw/main/SnowballTarget.zip" \
  -O ./training-envs-executables/linux/SnowballTarget.zip
unzip -d ./training-envs-executables/linux/ ./training-envs-executables/linux/SnowballTarget.zip
chmod -R 755 ./training-envs-executables/linux/SnowballTarget

# train
mlagents-learn ./config/ppo/SnowballTarget.yaml \
  --env=./training-envs-executables/linux/SnowballTarget/SnowballTarget \
  --run-id="SnowballTarget1" \
  --no-graphics
```

The four things you pass:

| Flag | Meaning |
|---|---|
| `mlagents-learn <config>` | Path to the **hyperparameter config file**. |
| `--env` | Path to the **environment executable**. |
| `--run-id` | The **name of this training run** (results land in `results/<run-id>/`). |
| `--no-graphics` | **Don't render** during training (much faster; required on headless machines). |
| `--resume` | Continue an interrupted run. *(It fails the first time you use it — just rerun.)* |
| `--force` | Overwrite an existing run with the same `run-id`. |

Pyramids is the identical shape:

```bash
mlagents-learn ./config/ppo/PyramidsRND.yaml \
  --env=./training-envs-executables/linux/Pyramids/Pyramids \
  --run-id="Pyramids Training" \
  --no-graphics
```

### 8.3 Monitoring with TensorBoard

`mlagents-learn` writes TensorBoard event files into `results/<run-id>/`:

```bash
tensorboard --logdir results
```

Things to watch:

| Metric | Read it as |
|---|---|
| `Environment/Cumulative Reward` | **The** curve. Should climb toward the benchmark (15 for SnowballTarget, 1.75 for Pyramids). |
| `Environment/Episode Length` | Task-dependent; in Pyramids, shorter usually means the agent found the brick faster. |
| `Policy/Entropy` | Should **decrease slowly**. Collapses instantly → no exploration; stays flat and high → nothing is being learned. |
| `Policy/Learning Rate` | Confirms the linear decay is happening. |
| `Losses/Policy Loss`, `Losses/Value Loss` | Value loss typically rises while reward is being discovered, then falls. |
| `Policy/Curiosity Reward` (or RND) | The intrinsic reward. Expect it to be **high early and decay** as the world becomes predictable. If it never decays, exploration isn't converging. |

### 8.4 Push to the Hub and watch it play

```bash
huggingface-cli login    # or notebook_login() in Colab; token needs the write role

mlagents-push-to-hf \
  --run-id="SnowballTarget1" \
  --local-dir="./results/SnowballTarget1" \
  --repo-id="<your-username>/ppo-SnowballTarget" \
  --commit-message="First Push"
```

| Flag | Meaning |
|---|---|
| `--run-id` | The training run's id. |
| `--local-dir` | Where the agent was saved — `results/<run-id>`. |
| `--repo-id` | `<your HF username>/<repo name>`. **Created automatically if it doesn't exist.** |
| `--commit-message` | HF repos are git repos, so a commit message is required. |

The resulting repo contains the **`.onnx` model(s), a model card, the TensorBoard logs, and the config file** — and because it's a git repo you can keep pushing new commits.

Then **watch the agent in the browser**, no Unity Editor needed:

- SnowballTarget → <https://huggingface.co/spaces/ThomasSimonini/ML-Agents-SnowballTarget>
- Pyramids → <https://huggingface.co/spaces/unity/ML-Agents-Pyramids>

In the Space: type your **case-sensitive** username, pick your model repo, then choose **which checkpoint to replay** (`SnowballTarget.onnx` is the final one; the intermediate `SnowballTarget-<steps>.onnx` files let you literally watch the agent get better over training).

---

## 9. Bonus: create your own environments with Unity and ML-Agents

**You can build your own RL environments with Unity + ML-Agents.** A game engine is intimidating at first, so the course's suggested ramp is:

| Step | What | Resource |
|---|---|---|
| **1. Learn Unity** | Build **5 small games** as a beginner video series | ["Create with Code"](https://learn.unity.com/course/create-with-code) |
| **2. Build the simplest RL environment** | Your **first basic RL environment**, end to end | [Learning-Environment-Create-New.md](https://github.com/Unity-Technologies/ml-agents/blob/release_20_docs/docs/Learning-Environment-Create-New.md) |
| **3. Iterate to nicer environments** | Read especially the **Designing Agents / Agent** parts of the docs | [ML-Agents docs](https://github.com/Unity-Technologies/ml-agents/blob/release_20_docs/docs/) |
| **3b. A worked custom environment** | Free course: **"Create a hummingbird environment"** by Adam Kelly | [ml-agents-hummingbirds](https://learn.unity.com/course/ml-agents-hummingbirds) |

Design tips carried over from the two environments studied above:

- **Keep the reward function as simple as the goal allows** (§4.2) — don't reward-engineer the strategy, reward the *outcome*.
- **Prefer raycasts to camera observations** unless the task is genuinely visual.
- **Explicitly expose hidden state** the agent's sensors can't see (cool-off timer, switch state).
- **If your reward is sparse, plan on an intrinsic signal** (`curiosity` / `rnd`) from the start.
- **Clone the training area many times** in the scene to collect experience in parallel.
- ML-Agents provides **17 different environments** — the fastest way to sanity-check your understanding of the config file is to retrain one of them.

If you build something, share it in the course's `#rl-i-made-this` Discord channel.

---

## 10. Self-check (quick review questions)

1. Why are **game engines** a natural fit for building RL environments? Name one thing they give you for free.
2. List the **six components** of ML-Agents and say which side (C# or Python) each lives on.
3. Which component **connects Unity to Python**, and why is it needed at all?
4. What is the **Academy**, and what four things does it order the agents to do each step?
5. In ML-Agents, what is the **Brain**?
6. Compare **vector observations**, **visual/camera observations** and **raycasts**. Which do both unit-5 environments use, and why?
7. What is SnowballTarget's **reward function**, **observation space**, and **action space**? What is the **cool-off system** and why does the agent get a `can I shoot` boolean?
8. State the **reward engineering problem** in one sentence. What do you risk by over-designing the reward?
9. Describe the **four stages** of the Pyramids task. Write down its reward function.
10. Why is Pyramids a **sparse reward** problem, and why does that break naive RL? (Tie it back to the reward hypothesis.)
11. What are the **two major problems in modern RL** that curiosity addresses?
12. Define **intrinsic reward**. Write the total reward as a formula and say what $\beta$ / `strength` controls — and what goes wrong if it's too big or too small.
13. In the **next-state-prediction** formulation, why is prediction error a sensible *novelty* signal? What happens to the curiosity reward in a state the agent has visited many times?
14. Name the **two networks in the ICM**, their inputs and outputs, and say **which one's loss becomes the reward**.
15. What is the **inverse model** actually for? Explain the **noisy-TV** failure it prevents.
16. What's the difference between the **`curiosity`** and **`rnd`** reward signals in ML-Agents?
17. In the config file, what do **`beta`**, **`epsilon`**, **`lambd`** and **`buffer_size`** control?
18. Why does the Pyramids config use a **longer `time_horizon`** and a **bigger network** than SnowballTarget?
19. Write the `mlagents-learn` command from memory, including what each of the four arguments does.
20. Which TensorBoard curves would tell you that (a) the agent is learning and (b) exploration has collapsed?
21. What ends up in the Hub repo after `mlagents-push-to-hf`, and why can you watch the agent without Unity?

---

## 11. Further resources

*(Unit 5 has no glossary or additional-readings page — these are the references worth keeping.)*

**Unity ML-Agents**
- [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents) — the repository (Unity Technologies).
- [ML-Agents documentation](https://unity-technologies.github.io/ml-agents/) — start with **ML-Agents Overview**, **Designing a Learning Environment**, and **Learning Environment Examples**.
- [Training Configuration File reference](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Training-Configuration-File.md) — every hyperparameter in §8.1, with typical ranges.
- [Making a New Learning Environment](https://github.com/Unity-Technologies/ml-agents/blob/release_20_docs/docs/Learning-Environment-Create-New.md) — the step-2 tutorial from the bonus page.

**Curiosity / exploration**
- [Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/abs/1705.05363) — Deepak Pathak, Pulkit Agrawal, Alexei A. Efros, Trevor Darrell (2017). **The ICM paper** — the forward + inverse models of §6.4; the ML-Agents `curiosity` signal implements this.
- [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894) — Burda et al. (2018). The `rnd` signal used by `PyramidsRND.yaml`.
- [Solving sparse-reward tasks with Curiosity](https://blogs.unity3d.com/2018/06/26/solving-sparse-reward-tasks-with-curiosity/) — Unity's own blog post on the module.
- [Curiosity-Driven Learning through Next State Prediction](https://medium.com/data-from-the-trenches/curiosity-driven-learning-through-next-state-prediction-f7f4e2f592fa) — the course author's write-up with the math details.
- [Random Network Distillation: a new take on Curiosity-Driven Learning](https://medium.com/data-from-the-trenches/curiosity-driven-learning-through-random-network-distillation-488ffd8e5938) — same author, on RND.

**Algorithms behind the trainers**
- [Proximal Policy Optimization](https://blog.openai.com/openai-baselines-ppo/) — OpenAI (the default `trainer_type`; covered in depth in Unit 8).
- [Soft Actor-Critic](https://bair.berkeley.edu/blog/2018/12/14/sac/) — BAIR (the off-policy alternative in ML-Agents).

---

> **Next up:** **Advantage Actor-Critic (A2C)** — combining the value-based and policy-based worlds by using a *critic* to reduce the high variance of policy-gradient updates, replacing the raw return with an **advantage** estimate.
