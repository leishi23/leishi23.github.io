---
layout: post
title:  "HF Deep RL Course — Unit 3 (Deep Q-Learning)"
date:   2026-08-14
desc: "HF Deep RL Course — Unit 3 (Deep Q-Learning)"
keywords: "Machine learning"
categories: [Machine learning]
tags: [Machine learning]
icon: icon-html
---

# HF Deep RL Course — Unit 3 (Deep Q-Learning)

> Source: [Hugging Face Deep RL Course, Unit 3](https://huggingface.co/learn/deep-rl-course/unit3/introduction) — notes from *Introduction* through *Additional Readings*, reorganized for review. Continues from [Unit 2]({{ '/machine learning/2026/08/14/HF_DeepRL_Unit2.html' | replace: ' ', '%20' | prepend: site.baseurl }}).

> **_Keypoints:_**

- Why tabular Q-Learning doesn't scale (Atari state space is astronomically large)
- From a **Q-table** to a **parametrized Q-function** $Q_{\theta}(s,a)$ — our first *Deep* RL agent
- Input preprocessing: grayscale, crop, downsample to 84×84, **stack 4 frames** (temporal limitation)
- The DQN architecture: 3 conv layers + fully connected head, one Q-value per action
- The Deep Q-Learning **loss** and its two phases: **sampling** and **training**
- Why training is unstable: non-linear function approximation **+** bootstrapping
- The three stabilizers: **Experience Replay**, **Fixed Q-Target**, **Double DQN**
- Full algorithm as pseudocode, plus the hands-on (Space Invaders + RL-Baselines3-Zoo)
- Glossary + self-check + additional readings

---

## 0. What this unit is for

Unit 2 gave us Q-Learning from scratch, trained on **FrozenLake-v1** (16 states) and **Taxi-v3** (500 states). It worked great — *because those state spaces were discrete and tiny*.

This unit studies our **first Deep Reinforcement Learning agent: Deep Q-Learning**. Instead of a Q-table, we use a **neural network that takes a state and approximates the Q-values for each action** at that state.

And we train it to play **Space Invaders** (and other Atari games) with **[RL-Zoo / RL-Baselines3-Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)** — a training framework on top of Stable-Baselines3 that gives us scripts for training, evaluating, hyperparameter tuning, plotting and video recording.

---

## 1. From Q-Learning to Deep Q-Learning

Recall: **Q-Learning** trains a **Q-function**, an **action-value function** that gives the value of being in a state and taking a specific action there. The **Q** stands for **"the Quality"** of that action at that state. Internally it's encoded by a **Q-table** — each cell is a state-action pair value — the **memory / cheat sheet** of the Q-function.

The problem: **Q-Learning is a *tabular method***. That breaks as soon as the state and action spaces **are not small enough to be represented efficiently by arrays and tables**. In one word: it is **not scalable**.

### How bad is Atari?

An Atari frame is a **210×160 pixel** image, in color, so **3 channels** ⇒ observation shape **(210, 160, 3)**, with each pixel value in **0–255**. Counting distinct observations:

$$
256^{\,210 \times 160 \times 3} = 256^{100800}
$$

For comparison there are roughly $10^{80}$ atoms in the observable universe. (The unit intro quotes the *effective* Atari state-space size as around $10^{9}$ to $10^{11}$ states — either way, hopeless for a table.)

| | **FrozenLake-v1** | **Taxi-v3** | **Atari (from pixels)** |
|---|---|---|---|
| States | 16 | 500 | $\sim 10^{9}$–$10^{11}$ (up to $256^{100800}$ raw observations) |
| Q-table feasible? | Yes | Yes | **No** |

### The fix: parametrize the Q-function

Instead of storing values, **approximate** them with a **parametrized Q-function $Q_{\theta}(s,a)$** — a neural network with weights $\theta$ that, **given a state, outputs the different Q-values for each possible action** at that state.

$$
\text{Q-table } Q(s,a) \;\;\longrightarrow\;\; \text{Q-network } Q(s,a;\theta)
$$

> **Key insight:** we didn't change *what* we're learning (still the action-value function, still trained with a TD target and $\varepsilon$-greedy behavior). We changed *how it is stored*: **lookup → function approximation**. That single change buys generalization to unseen states — and costs us stability (see §4).

| | **Q-Learning** | **Deep Q-Learning** |
|---|---|---|
| Representation of $Q$ | **Q-table** (array lookup) | **Neural network** $Q(s,a;\theta)$ |
| Output for a state | one cell per (state, action) | a **vector** of Q-values, one per action |
| Update | write **directly** into the cell: $Q \mathrel{+}= \alpha[\,\text{TD error}\,]$ | **gradient descent** on a loss between prediction and TD target |
| Generalizes to unseen states? | **No** | **Yes** (that's the point) |
| Scales to huge/continuous state spaces? | No | **Yes** |
| Stable? | Yes (tabular Q-Learning converges) | **Not by default** — needs replay, fixed target, Double DQN |
| Sample handling | one transition, then discarded | transitions **stored** in a replay buffer and reused |

---

## 2. The Deep Q-Network (DQN)

The network takes as input a **stack of 4 frames** (that *is* the state) and outputs a **vector of Q-values, one per possible action**. Then, exactly like Q-Learning, we use the **epsilon-greedy policy** to pick the action:

$$
a_t =
\begin{cases}
\text{random action} & \text{with probability } \varepsilon \quad (\textbf{exploration})\\[4pt]
\arg\max_{a} Q(s_t, a; \theta) & \text{with probability } 1-\varepsilon \quad (\textbf{exploitation})
\end{cases}
$$

When the network is initialized, **the Q-value estimation is terrible**. During training the agent learns to associate a situation with the appropriate action, and **learns to play the game well**.

### 2.1 Preprocessing the input

We **preprocess the input** to **reduce the complexity of the state** and therefore the computation time needed for training:

| Step | What it does | Why |
|---|---|---|
| **Grayscale** | RGB (3 channels) ⇒ **1 channel** | Colors in Atari **don't add important information** — a 3× reduction for free |
| **Crop** | drop parts of the screen (score bars, borders) in some games | those regions **contain no useful information** |
| **Downsample** | resize the frame to **84×84** | far fewer pixels ⇒ far less computation |
| **Stack 4 frames** | state = last **4** preprocessed frames | gives the network **temporal information** (see below) |

So a single state fed to the network is a **84×84×4** tensor rather than a 210×160×3 image.

### 2.2 Why stack four frames? The temporal limitation

**A single frame carries no motion information.** The course's Pong example: look at one frame — *where is the ball going?* You can't say. Add three more frames and **you can now see the ball is going to the right**.

> **Key insight:** this is a **Markov problem**. The RL framework assumes the state is sufficient to decide optimally, but a lone image is **not** a sufficient statistic of the history — position without velocity. Stacking 4 frames makes the observation (approximately) Markovian again: it encodes position **and** direction **and** speed.

This is what the glossary calls **temporal limitation**: *"a difficulty presented when the environment state is represented by frames. A frame by itself does not provide temporal information. In order to obtain temporal information, we need to **stack** a number of frames together."*

### 2.3 The architecture

1. The stacked frames go through **three convolutional layers**. These **capture and exploit spatial relationships in images** — and because the frames are stacked along the channel dimension, they also **exploit temporal properties across those frames**.
2. Then **a couple of fully connected layers** that **output a Q-value for each possible action** at that state.

$$
\underbrace{84 \times 84 \times 4}_{\text{stacked frames}} \;\to\; \text{Conv} \to \text{Conv} \to \text{Conv} \;\to\; \text{FC} \to \text{FC} \;\to\; \underbrace{\big[\,Q(s,a_1),\, \dots,\, Q(s,a_n)\,\big]}_{\text{one Q-value per action}}
$$

> **Note the efficiency trick:** the network is $Q(s;\theta) \in \mathbb{R}^{|\mathcal{A}|}$, **not** $Q(s,a;\theta) \in \mathbb{R}$. One forward pass gives **all** action values at once — which is exactly what you need for both $\arg\max_a$ (acting) and $\max_a$ (the TD target).

(If convolutional layers are new to you, the course points to [Lesson 4 of Udacity's free Deep Learning with PyTorch course](https://www.udacity.com/course/deep-learning-pytorch--ud188).)

---

## 3. The Deep Q-Learning loss and its two phases

In tabular Q-Learning we updated the Q-value of a state-action pair **directly**:

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \Big[\, \underbrace{R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a)}_{\text{TD target}} - Q(S_t, A_t) \,\Big]
$$

In Deep Q-Learning we can't "write into a cell" — the values live in weights. So instead we **create a loss function that compares our Q-value prediction with the Q-target, and use gradient descent to update the weights** of the Deep Q-Network so it approximates the Q-values better:

$$
L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D}\Big[\big(\underbrace{r + \gamma \max_{a'} Q(s',a';\theta^{-})}_{\text{TD target (Q-target)}} - \underbrace{Q(s,a;\theta)}_{\text{prediction}}\big)^2\Big]
$$

with the gradient step

$$
\theta \leftarrow \theta - \alpha \nabla_{\theta} L(\theta),
$$

and for terminal transitions the target collapses to just $r$ (there is no next state):

$$
y =
\begin{cases}
r & \text{if } s' \text{ is terminal}\\[3pt]
r + \gamma \max_{a'} Q(s',a';\theta^{-}) & \text{otherwise}
\end{cases}
$$

> ⚠️ The TD target is treated as a **constant** during the gradient step — we do **not** backpropagate through it. (In code: `.detach()` / `stop_gradient`, and it's computed with the *target* network $\theta^{-}$; see §4.2.)

### The two phases

The Deep Q-Learning training algorithm has **two phases**, and they are **interleaved but decoupled**:

| Phase | What happens | Policy used |
|---|---|---|
| **Sampling** | we **perform actions** and **store the observed experience tuples $(s,a,r,s')$ in a replay memory** | $\varepsilon$-greedy on $Q(\cdot;\theta)$ |
| **Training** | **select a small batch of tuples randomly** and learn from it with **a gradient descent update step** | greedy $\max_{a'}$ inside the target |

Because acting is $\varepsilon$-greedy and the target is greedy, **Deep Q-Learning is off-policy** — just like tabular Q-Learning. That's also *why* replaying old, stale experience is legitimate at all.

---

## 4. Three solutions to instability

Deep Q-Learning training **might suffer from instability**, mainly because of combining:

1. a **non-linear Q-value function** (a neural network), and
2. **bootstrapping** (updating targets with *existing estimates* instead of an actual complete return).

To stabilize training we implement **three** solutions.

| # | Solution | Problem it fixes | Mechanism |
|---|---|---|---|
| 1 | **Experience Replay** | wasteful sample use; **correlated** consecutive samples; **catastrophic forgetting**; oscillating/diverging action values | replay buffer $D$ of capacity $N$; train on **random minibatches** |
| 2 | **Fixed Q-Target** | the target **moves** with the weights we're updating ("chasing your own tail") | a **separate target network** $\theta^{-}$, copied from $\theta$ **every C steps** |
| 3 | **Double DQN** | **overestimation** of Q-values from the noisy $\max$ | **online net selects** the action, **target net evaluates** it |

### 4.1 Experience Replay — more efficient use of experiences

Experience Replay has **two functions**:

**(1) Make more efficient use of the experiences during training.** Usually in online RL the agent interacts with the environment, gets an experience $(s, a, r, s')$, learns from it (updates the network), and **discards it**. That is not efficient. A **replay buffer** saves experience samples **that we can reuse during training** ⇒ this allows the agent to **learn from the same experiences multiple times**.

**(2) Avoid forgetting previous experiences and reduce the correlation between experiences.**

- **[Catastrophic forgetting](https://en.wikipedia.org/wiki/Catastrophic_interference)** (a.k.a. catastrophic interference): if we feed **sequential** samples to the neural network, it tends to **forget the previous experiences as it gets new ones**. For instance, if the agent is on level 1 and then on level 2 (which is different), it can **forget how to behave and play level 1**.
- Sampling a **small random batch** prevents the network from **only learning about what it has done immediately before**.
- **Randomly sampling** removes the **correlation in the observation sequences**, avoiding **action values oscillating or diverging catastrophically**.

Concretely, in the pseudocode we **initialize a replay memory buffer $D$ with capacity $N$** ($N$ is a hyperparameter), store experiences in it, and sample a batch to feed the Deep Q-Network during the training phase.

> **Why correlation hurts:** gradient descent assumes roughly i.i.d. samples. Consecutive Atari frames are almost identical, so a sequential minibatch is effectively **one** sample with a huge gradient in one direction. Shuffled replay restores something much closer to i.i.d.

### 4.2 Fixed Q-Target — stabilize the training

To compute the TD error (the loss) we take the **difference between the TD target (Q-target) and the current Q-value (the estimation of Q)**. But **we have no idea what the real TD target is** — we have to *estimate* it, and by the Bellman equation the estimate is *the reward of taking that action at that state plus the discounted highest Q-value for the next state*.

**The problem:** we use the **same parameters (weights) for estimating the TD target *and* the Q-value**. So there's a significant **correlation between the TD target and the parameters we're changing**: at every training step **both our Q-values and the target values shift**. We get closer to the target, but **the target is also moving**. Significant oscillation follows.

The course's image: you're a **cowboy** (the Q estimation) trying to catch a **cow** (the Q-target). At each timestep you approach the cow — but **the cow also moves at each timestep**, because it's driven by the same parameters. The result is a bizarre chasing path, i.e. large oscillation in training.

**The fix:**
- Use a **separate network with fixed parameters $\theta^{-}$** for estimating the TD target.
- **Copy the parameters from the Deep Q-Network every $C$ steps** to update the target network: $\theta^{-} \leftarrow \theta$.

$$
L(\theta) = \Big(\, r + \gamma \max_{a'} Q(s',a';\underbrace{\theta^{-}}_{\text{frozen, updated every } C \text{ steps}}) - Q(s,a;\theta) \,\Big)^{2}
$$

> **Key insight:** freezing $\theta^{-}$ turns a moving-target problem into a sequence of **stationary supervised-regression problems**, each lasting $C$ steps. Small $C$ ⇒ fast propagation of value information but more instability; large $C$ ⇒ very stable but stale targets and slow learning.

### 4.3 Double DQN — handle overestimation of Q-values

**Double DQNs** (Double Deep Q-Learning networks) were introduced [by Hado van Hasselt](https://papers.nips.cc/paper/3964-double-q-learning). This method **handles the problem of the overestimation of Q-values**.

Look again at the TD target: it contains $\max_{a'} Q(s',a')$. The question is: **how are we sure the best action for the next state is the action with the highest Q-value?**

We know the **accuracy of Q-values depends on what action we tried and what neighboring states we explored**. At the beginning of training **we don't have enough information about the best action to take**. So taking the maximum Q-value — **which is noisy** — as the best action **can lead to false positives**. If non-optimal actions are regularly **given a higher Q-value than the optimal action, learning will be complicated**.

> **Why the $\max$ is biased upward (the one-line reason):** for noisy estimates, $\mathbb{E}[\max_a \hat{Q}] \ge \max_a \mathbb{E}[\hat{Q}]$ — **taking a maximum over noisy numbers systematically picks up the noise**. The same network both *chooses* and *scores* the action, so its own errors are rewarded and then bootstrapped into every earlier state.

**The solution:** when computing the Q-target, use **two networks to decouple action selection from target Q-value generation**:
- Use the **DQN network** ($\theta$) to **select** the best action for the next state (the action with the highest Q-value).
- Use the **Target network** ($\theta^{-}$) to **calculate the target Q-value** of taking that action at the next state.

| | **Target used** |
|---|---|
| **DQN (vanilla)** | $y = r + \gamma\, \max_{a'} Q(s', a'; \theta^{-})$ |
| **Double DQN** | $y = r + \gamma\, Q\big(s',\; \arg\max_{a'} Q(s',a';\theta)\;;\; \theta^{-}\big)$ |

Read the Double-DQN target as: **"select with $\theta$, evaluate with $\theta^{-}$."** In vanilla DQN, $\theta^{-}$ does *both*, so its own errors get amplified.

$$
\boxed{\;y^{\text{DoubleDQN}} = r + \gamma\, Q\Big(s',\; \underbrace{\arg\max_{a'} Q(s',a';\theta)}_{\text{SELECT with online net}}\;;\; \underbrace{\theta^{-}}_{\text{EVALUATE with target net}}\Big)\;}
$$

Therefore Double DQN **reduces the overestimation of Q-values** and, as a consequence, helps us **train faster and with more stable learning**.

> Since these three improvements, many more have been added — such as **Prioritized Experience Replay** and **Dueling Deep Q-Learning**. They're out of scope for this course (see §7 for the papers).

---

## 5. The full Deep Q-Learning algorithm

Putting all of it together:

1. **Initialize** the replay memory $D$ to capacity $N$.
2. **Initialize** the action-value function $Q$ with random weights $\theta$.
3. **Initialize** the target action-value function $\hat{Q}$ with weights $\theta^{-} \leftarrow \theta$.
4. **For each episode:**
   1. **Reset** the environment; preprocess the first frame and build the initial **stack of 4 frames** $s_1$.
   2. **For each timestep $t$:**
      1. **[Sampling] Select an action** with the **epsilon-greedy** policy: random with probability $\varepsilon$, otherwise $a_t = \arg\max_a Q(s_t, a; \theta)$.
      2. **Execute $a_t$**, observe reward $r_t$ and the next frame; preprocess it and roll it into the frame stack to form $s_{t+1}$.
      3. **Store the transition** $(s_t, a_t, r_t, s_{t+1}, \text{done})$ in $D$ (evicting the oldest if $D$ is full).
      4. **[Training] Sample a random minibatch** of transitions $(s, a, r, s')$ from $D$.
      5. **Compute the target** for each sample:
         $y = r$ if $s'$ is terminal, else
         $y = r + \gamma \max_{a'} \hat{Q}(s',a';\theta^{-})$ — or, with **Double DQN**, $y = r + \gamma\, \hat{Q}\big(s', \arg\max_{a'} Q(s',a';\theta); \theta^{-}\big)$.
      6. **Perform a gradient descent step** on $\big(y - Q(s,a;\theta)\big)^{2}$ with respect to $\theta$ (treat $y$ as a constant).
      7. **Every $C$ steps, reset the target network:** $\theta^{-} \leftarrow \theta$.
      8. **Decay $\varepsilon$** (from ~1.0 toward a small final value).

### One-line mental model

$$
\underbrace{s \xrightarrow[\ \varepsilon\text{-greedy}\ ]{} a \to (r, s') \to D}_{\textbf{sampling}} \qquad\Big|\qquad \underbrace{D \to \text{minibatch} \to \nabla_{\theta}\big(y - Q(s,a;\theta)\big)^{2}}_{\textbf{training}} \qquad\Big|\qquad \underbrace{\theta^{-} \leftarrow \theta \text{ every } C}_{\textbf{stability}}
$$

---

## 6. The hands-on: Space Invaders with RL-Baselines3-Zoo

Worth remembering from the practical side:

- **Environment:** `SpaceInvadersNoFrameskip-v4` (Atari, via Gymnasium). Any Atari game works.
- **Library:** [**RL-Baselines3-Zoo**](https://github.com/DLR-RM/rl-baselines3-zoo) — a training framework on **Stable-Baselines3** with scripts for training, evaluation, hyperparameter tuning, plotting and video recording. Note: the Zoo integration is a **vanilla Deep Q-Learning** — **no** Double-DQN, Dueling-DQN or Prioritized Experience Replay.
- To implement DQN yourself afterwards, the course recommends the [CleanRL `dqn_atari.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py) single-file implementation.
- Certification threshold: push the model to the Hub with a result **≥ 200** (result = mean_reward − std of reward).

Hyperparameters from the reference `dqn.yml` — these are the numbers worth internalizing:

| Hyperparameter | Value | Maps to which concept |
|---|---|---|
| `env_wrapper` | `AtariWrapper` | preprocessing: frame reduction, grayscale, stack frames |
| `frame_stack` | **4** | temporal limitation (§2.2) |
| `policy` | `CnnPolicy` | conv layers to process frames (§2.3) |
| `n_timesteps` | **1e7** (10M) | ~9 h on a P100; the course suggests 1M (~90 min) to start |
| `buffer_size` | **100 000** | Experience Replay capacity $N$ (§4.1) |
| `learning_rate` | **1e-4** | the $\alpha$ of the gradient step (§3) |
| `batch_size` | **32** | minibatch sampled from $D$ (§3) |
| `learning_starts` | **100 000** | fill the buffer before training begins |
| `target_update_interval` | **1000** | the **$C$** of Fixed Q-Target (§4.2) |
| `train_freq` / `gradient_steps` | **4** / **1** | 1 gradient step per 4 environment steps |
| `exploration_fraction` | **0.1** | decay $\varepsilon$ over the first 10% of training |
| `exploration_final_eps` | **0.01** | final $\varepsilon$ — still 1% exploration forever |

The course's advice: if you tune anything, tune **`learning_rate`, `buffer_size`, and `batch_size`**.

Commands, for the record:

```bash
python -m rl_zoo3.train      --algo dqn --env SpaceInvadersNoFrameskip-v4 -f logs/ -c dqn.yml
python -m rl_zoo3.enjoy      --algo dqn --env SpaceInvadersNoFrameskip-v4 --no-render --n-timesteps 5000 --folder logs/
python -m rl_zoo3.push_to_hub --algo dqn --env SpaceInvadersNoFrameskip-v4 --repo-name dqn-SpaceInvadersNoFrameskip-v4 -orga <user> -f logs/
```

---

## 7. Glossary

- **Tabular Method** — a type of problem in which the state and action spaces are **small enough** to approximate value functions represented as **arrays and tables**. **Q-Learning** is an example of a tabular method, since a table is used to represent the value for different state-action pairs.
- **Deep Q-Learning** — a method that trains a **neural network** to approximate, given a state, the different **Q-values** for each possible action at that state. It is used to solve problems when the **observation space is too big** to apply a tabular Q-Learning approach.
- **Temporal Limitation** — a difficulty presented when the environment state is represented by **frames**. A frame by itself **does not provide temporal information**; to obtain temporal information we need to **stack** a number of frames together.

**Phases of Deep Q-Learning**
- **Sampling** — actions are performed, and the observed experience tuples are **stored in a replay memory**.
- **Training** — batches of tuples are **selected randomly** and the neural network **updates its weights using gradient descent**.

**Solutions to stabilize Deep Q-Learning**
- **Experience Replay** — a replay memory is created to save experience samples that can be **reused during training**. This allows the agent to **learn from the same experiences multiple times**, and helps it **avoid forgetting** previous experiences as it gets new ones.
- **Random sampling** (from the replay buffer) — allows us to **remove correlation in the observation sequences** and prevents action values from **oscillating or diverging catastrophically**.
- **Fixed Q-Target** — to calculate the **Q-Target** we must estimate the discounted optimal **Q-value** of the next state using the Bellman equation. The problem is that **the same network weights** are used to compute the **Q-Target** and the **Q-value** — so every time we modify the Q-value, the **Q-Target moves with it**. To avoid this, a **separate network with fixed parameters** is used for estimating the TD target; the target network is updated by **copying parameters from the Deep Q-Network after C steps**.
- **Double DQN** — a method to handle **overestimation** of Q-values. It uses **two networks to decouple the action selection from the target value generation**:
  - the **DQN Network** selects the best action to take for the next state (the action with the highest Q-value);
  - the **Target Network** calculates the target Q-value of taking that action at the next state.
  
  This approach reduces Q-value overestimation, and helps train **faster** with **more stable** learning.

---

## 8. Self-check (quick review questions)

1. Why exactly does tabular Q-Learning fail on Atari? Give the observation shape and the resulting count of possible observations.
2. What does "parametrized Q-function $Q_{\theta}(s,a)$" mean, and what does the network output — a scalar or a vector?
3. Name the four preprocessing steps applied to Atari frames, and the reason for each. What is the final input tensor shape?
4. What is the **temporal limitation**, and how does frame stacking fix it? Use the Pong example.
5. Describe the DQN architecture layer by layer. Why conv layers *and* why do they also capture temporal structure?
6. Write the Deep Q-Learning **loss** from memory. Which term is the TD target, and which weights compute it?
7. What are the **two phases** of Deep Q-Learning training, and which policy does each use? Why is DQN off-policy?
8. What two ingredients combine to make Deep Q-Learning unstable?
9. Give the **two functions** of Experience Replay. What is catastrophic forgetting, and how does replay avoid it?
10. Explain the "cowboy chasing a cow" analogy. What is the moving-target problem and how does the target network fix it?
11. What does $C$ control in Fixed Q-Target? What goes wrong if $C$ is far too small? Far too large?
12. Why does $\max_{a'} Q(s',a')$ **overestimate**? Where does the noise come from?
13. Write both the vanilla DQN target and the Double DQN target. Which network **selects** and which **evaluates**?
14. Recite the full Deep Q-Learning algorithm as a numbered list, including where $\theta^{-}$ gets updated and where $\varepsilon$ gets decayed.
15. In the hands-on config, which hyperparameter is the replay buffer capacity $N$, and which is the $C$ of the fixed Q-target?

---

## 9. Additional Readings

These are **optional readings** if you want to go deeper.

- [Foundations of Deep RL Series, L2: Deep Q-Learning](https://youtu.be/Psrhxy88zww) — Pieter Abbeel
- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — Mnih et al., the original DQN paper
- [Double Deep Q-Learning](https://papers.nips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — Hado van Hasselt
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) — Schaul et al.
- [Dueling Deep Q-Learning](https://arxiv.org/abs/1511.06581) — Wang et al.

Also useful for the prerequisites and the hands-on:
- [Lesson 4 of Udacity's free Deep Learning with PyTorch course](https://www.udacity.com/course/deep-learning-pytorch--ud188) — convolutional layers
- [RL-Baselines3-Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) and the [Stable-Baselines3 DQN parameters](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html#parameters)
- [CleanRL `dqn_atari.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py) — a readable single-file DQN implementation

---

> **Next up:** **Policy-Gradient methods** — instead of learning a value function and *deriving* a policy from it, we'll parametrize and optimize the **policy directly** (Policy Gradient / REINFORCE), which handles stochastic policies and continuous action spaces naturally.
