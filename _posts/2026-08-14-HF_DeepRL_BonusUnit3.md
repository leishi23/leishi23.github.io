---
layout: post
title:  "HF Deep RL Course — Bonus Unit 3 (Advanced Topics in RL)"
date:   2026-08-14
desc: "HF Deep RL Course — Bonus Unit 3 (Advanced Topics in RL)"
keywords: "Machine learning"
categories: [Machine learning]
tags: [Machine learning]
icon: icon-html
---

# HF Deep RL Course — Bonus Unit 3 (Advanced Topics in RL)

> Source: [Hugging Face Deep RL Course, Bonus Unit 3](https://huggingface.co/learn/deep-rl-course/unitbonus3/introduction) — notes from *Introduction* through *Brief introduction to RL documentation*, reorganized for review. Continues from [Unit 8]({{ '/machine learning/2026/08/14/HF_DeepRL_Unit8.html' | replace: ' ', '%20' | prepend: site.baseurl }}).

> **_Keypoints:_**

- This unit is a **survey**, not an algorithm: a map of research directions past PPO
- **Offline vs Online RL** — learn from a fixed logged dataset vs collect your own data
- **Generalisation** — RL agents overfit their training environment; robustness & transfer
- **Model-based RL (MBRL)** — learn the dynamics $f_\theta(s_t,a_t)$ and plan with it
- **(Automatic) Curriculum Learning** — order tasks easy→hard so skills accumulate
- **Decision Transformers** — RL as conditional **sequence modeling** on return-to-go
- **Language models + RL** — LMs as priors about the world; grounding them with PPO
- **RLHF** — preference data ⇒ reward model ⇒ RL fine-tune with a KL leash
- **Game-engine toolkits** — Godot RL Agents, Unreal Learning Agents
- **Environments to try** (DIAMBRA, MineRL, DonkeyCar, StarCraft II) + **Reward Reports** documentation
- Unlike other units, each section here has a **different Hugging Face author**

---

## 0. What this bonus unit is for

You've finished the main course, so you have a solid background in Deep RL: Q-Learning → Deep Q-Learning → policy gradients / REINFORCE → A2C → PPO → multi-agent. **But the course was just the beginning** — there are many subfields left to discover.

This optional unit gives you **resources to explore multiple concepts and research topics in RL**. Contrary to other units, it's a **collective work of multiple people from Hugging Face**, and the author is credited per section:

| Section | Author |
|---|---|
| Offline vs. Online RL, Environments to try | Thomas Simonini |
| Model-Based RL, RLHF, RL documentation | Nathan Lambert |
| Curriculum Learning, Language models in RL | Clément Romac |
| Decision Transformers, Godot RL Agents | Edward Beeching |

So: read this unit as an **index of directions**, and treat the "Further reading" links as the real content.

---

## 0.1 Map of the territory (the main review aid)

Every advanced topic below exists because **some assumption of vanilla online deep RL breaks**. That's the cleanest way to memorize the unit:

| Topic | Assumption it attacks | Problem it addresses | Core idea in one line |
|---|---|---|---|
| **Offline RL** | "I can freely interact with the env" | Real-world interaction is expensive/unsafe; you already have logged data | Learn a policy **purely from a fixed dataset**, no environment interaction |
| **Generalisation** | "Train env == test env" | Agents overfit a single level/layout; the real world is non-stationary and open-ended | Train over **variations** (procedural generation, domain randomization); measure the **generalisation gap** |
| **Model-based RL** | "Model-free is enough" | **Sample inefficiency** — millions of steps per task | Learn a **dynamics model** and **plan** with it (e.g. MPC) |
| **Curriculum Learning** | "One fixed task, learn it" | Hard / sparse-reward tasks needing **incremental skill acquisition**; robustness to env variations | **Order tasks easy→hard**; ACL *learns* that ordering automatically |
| **Decision Transformers** | "RL = maximize a value function" | Bootstrapping/value fitting is unstable, and offline data is plentiful | Recast RL as **conditional sequence modeling**: condition on desired return, predict actions |
| **LMs in RL** | "Tabula rasa learning" | Learning from scratch ⇒ sample inefficiency + behaviors humans find bizarre | Use an LM's **world knowledge** as prior; RL **grounds/corrects** it |
| **RLHF** | "A reward function exists and I can write it" | Human preferences are hard to specify as an equation | Learn a **reward model** from human comparisons, then RL fine-tune |
| **Godot RL / Learning Agents** | "Gym envs are enough" | You want agents in **your own game** | Bridge a game engine (Godot / Unreal) to Python RL frameworks |
| **Reward Reports** | "Ship it and forget it" | RL systems **change over time** once deployed with humans in the loop | **Living documentation** with a change-log + update triggers |

---

## 1. Offline vs. Online Reinforcement Learning

Deep RL agents **learn from batches of experience**. The whole distinction here is: *how is that experience collected?*

### Online RL (everything in this course so far)
The agent **gathers data directly**: it collects a batch of experience by **interacting with the environment**, then uses that experience **immediately (or via a replay buffer)** to update its policy.

But this implies you either **train the agent directly in the real world, or you have a simulator**. If you don't have one you must build it, which is:
- **complex** — how do you reflect the messy reality of the real world in an environment?
- **expensive**,
- and **insecure** — if the simulator has flaws that give a competitive advantage, **the agent will exploit them** (reward hacking / sim exploits).

### Offline RL
The agent **only uses data collected from other agents or human demonstrations**. It **does not interact with the environment at all**. The process:

1. **Create a dataset** using one or more policies and/or human interactions.
2. **Run offline RL on that dataset** to learn a policy.

Why this is attractive: **safety** (no dangerous exploration in the real world), **cost** (no simulator to build, no robot to break), and **data reuse** — you can learn from logs you already have, the way supervised learning reuses static datasets.

### The central difficulty

> ⚠️ **The counterfactual queries problem** (the course's name for it; in the literature this is **distributional shift** / **out-of-distribution actions**): what do we do if our agent **decides to do something for which we don't have data?** For instance, it wants to turn right at an intersection, but no trajectory in the dataset ever turned right there.

Because there is **no exploration**, the learned $Q$ or policy can confidently extrapolate garbage on actions the dataset never took, and **nothing in the pipeline can correct it** — there is no environment to disprove the optimistic estimate. This is why offline RL algorithms add conservatism/constraint machinery (staying near the behavior policy, penalizing OOD actions) rather than just running DQN on a static buffer.

Solutions exist; the course points you to a [video overview of offline RL](https://www.youtube.com/watch?v=k08N5a0gG0A).

| | **Online RL** | **Offline RL** |
|---|---|---|
| Data source | The agent's **own** interaction | A **fixed, pre-collected** dataset (other agents / humans) |
| Env interaction during training | **Yes** | **No** |
| Needs simulator or real-world access | **Yes** | **No** |
| Can explore to fix its own mistakes | **Yes** | **No** |
| Main failure mode | Sample cost, unsafe exploration, sim exploitation | **Counterfactual queries / distributional shift** |
| Feels like | Trial and error | **Supervised learning on trajectories** |

> **Key insight:** offline RL turns RL into something that looks like the rest of deep learning (a dataset + a loss), which is exactly why the **Decision Transformer** (§5) fits the offline setting so naturally.

**Further reading (from the page):**
- [Offline Reinforcement Learning — talk by Sergey Levine](https://www.youtube.com/watch?v=qgZPZREor5I)
- [Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/abs/2005.01643)
- The figure comparing the two settings comes from [this post](https://offline-rl.github.io/)

*(The course page names no dataset suite or library here — if you want to actually run offline RL, the standard benchmark in the wider literature is D4RL, and 🤗 Transformers ships the Decision Transformer, see §5.)*

---

## 2. Generalisation in Reinforcement Learning

**Generalisation plays a pivotal role in RL.** RL algorithms demonstrate **good performance in controlled environments**, but the real world presents a unique challenge **due to its non-stationary and open-ended nature**.

So the goal for real-world RL is algorithms that are:
- **robust to environmental variations**, and
- able to **transfer and adapt to uncharted yet analogous tasks and settings**.

The terminology worth carrying away (the survey below is built around it):

| Term | Meaning |
|---|---|
| **Overfitting in RL** | The agent memorizes **one environment instance / one level** — including its exact layout, textures, seed — instead of learning the task |
| **Procedural generation** | Generate a **distribution of levels** programmatically, so train and test levels are different draws from the same distribution |
| **Generalisation gap** | Performance on **training levels** minus performance on **held-out test levels** — the RL analogue of train/test error gap |
| **Domain randomization** | Apply **random variations** to environment parameters (physics, visuals, dynamics) during training so the policy is invariant to them — the classic sim-to-real trick |
| **Zero-shot vs few-shot transfer** | Deploy on a new-but-related environment with no adaptation, vs with a small amount of fine-tuning |

> **Key insight:** in supervised learning you'd never evaluate on the training set — but a huge amount of classic RL does exactly that (train and report on one fixed environment). Generalisation research is largely about **fixing the evaluation protocol** as much as the algorithm.

**Further reading (from the page):**
- [Generalization in Reinforcement Learning — Robert Kirk](https://robertkirk.github.io/2022/01/17/generalisation-in-reinforcement-learning-survey.html) — a comprehensive survey; the recommended starting point
- [Improving Generalization in Reinforcement Learning using Policy Similarity Embeddings](https://blog.research.google/2021/09/improving-generalization-in.html?m=1)

Note the natural link to §4: **domain randomization is also a curriculum tool** (OpenAI's Rubik's-cube hand), so generalisation and curriculum learning are two views of the same "train over a distribution of tasks" idea.

---

## 3. Model-Based Reinforcement Learning (MBRL)

> MBRL differs from its model-free counterpart **only in learning a *dynamics model*** — but that has **substantial downstream effects on how decisions are made**.

The dynamics model usually models the environment transition dynamics

$$
s_{t+1} = f_\theta(s_t, a_t)
$$

but the same framework accommodates **inverse dynamics models** (mapping states to actions) or **reward models** (predicting rewards).

### Simple definition (the loop)
1. An agent repeatedly tries to solve a problem, **accumulating state and action data**.
2. With that data, it creates a structured learning tool — **a dynamics model** — to reason about the world.
3. With the dynamics model, the agent **decides how to act by predicting the future**.
4. With those actions, it **collects more data, improves the model, and hopefully improves future actions**.

### Academic definition
The agent acts in an MDP governed by a transition function $s_{t+1} = f(s_t, a_t)$ that returns a reward $r(s_t,a_t)$ at each step. From a collected dataset

$$
\mathcal{D} := \{\, s_i,\, a_i,\, s_{i+1},\, r_i \,\}
$$

the agent learns a model $s_{t+1} = f_\theta(s_t,a_t)$ **to minimize the negative log-likelihood of the transitions**:

$$
\mathcal{L}(\theta) \;=\; -\sum_{i \in \mathcal{D}} \log p_\theta\!\left(s_{i+1} \mid s_i, a_i\right)
$$

Then control is done with **sample-based model-predictive control (MPC)** using the learned model: optimize the **expected reward over a finite, recursively predicted horizon $\tau$**, from a set of actions sampled from a uniform distribution $U(a)$:

$$
a_{t:t+\tau}^{*} \;=\; \arg\max_{a_{t:t+\tau} \sim U(a)} \; \sum_{k=0}^{\tau-1} r\!\left(\hat{s}_{t+k},\, a_{t+k}\right), \qquad \hat{s}_{t+k+1} = f_\theta(\hat{s}_{t+k},\, a_{t+k})
$$

Only the **first action** of the winning sequence is executed; then you re-plan (that's the "predictive control" part).

### Model-based vs model-free

| | **Model-free** (this course) | **Model-based** |
|---|---|---|
| What is learned | Policy $\pi_\theta$ and/or value $Q_\theta$ | A **dynamics model** $f_\theta$ (plus optionally reward model) |
| How actions are chosen | Read them off the policy/value function | **Plan** by predicting the future (e.g. MPC) |
| Sample efficiency | **Low** — needs many env steps | **High** — the model amortizes experience |
| Compute at decision time | Cheap (one forward pass) | **Expensive** (rollouts / search per step) |
| Main failure mode | Sample cost, instability | **Model error**, which **compounds** over the horizon; the planner exploits model bugs |

> ⚠️ **Compounding error** is the defining MBRL problem: each predicted step feeds the next, so small one-step errors grow along $\tau$. Worse, the planner **actively searches for** the action sequence with the highest predicted reward — which is precisely where an optimistically-wrong model lives. Hence short horizons, model ensembles, and uncertainty penalties.

**Further reading (from the page):**
- [A blog post on debugging MBRL](https://www.natolambert.com/writing/debugging-mbrl)
- [A recent review paper on MBRL](https://arxiv.org/abs/2006.16712)
- MPC papers cited by the page: [1](https://arxiv.org/pdf/2002.04523), [2](https://arxiv.org/pdf/2012.09156.pdf), [3](https://arxiv.org/pdf/2009.01221.pdf)

---

## 4. (Automatic) Curriculum Learning for RL

Most methods in this course work well in practice, **but there are cases where using them alone fails** — for instance when:

- the task is hard and requires an **incremental acquisition of skills**. To make a **bipedal agent go through hard obstacles**, it must first learn to **stand**, then **walk**, then maybe **jump**…
- there are **variations in the environment** that affect difficulty, and you want the agent to be **robust** to them.

In such cases you need to **propose different tasks to the agent and organize them so that skills are acquired progressively**. This is **Curriculum Learning**, and it usually implies a **hand-designed curriculum** (a set of tasks in a specific order). In practice you can control:
- the **generation of the environment** (terrain difficulty, obstacle density),
- the **initial states** (start the agent near the goal, then further away),
- or use **Self-Play** and control the **level of the opponents** proposed to the agent.

### Why it helps sparse-reward / hard-exploration problems
With a sparse reward, a randomly-initialized policy in the hard task **essentially never sees a reward**, so the gradient carries no signal. An easy task variant is one where **random behavior sometimes succeeds** — that produces reward, which produces learning signal, which produces a policy good enough to occasionally succeed in the next-hardest variant. The curriculum is a **ladder of non-zero-gradient tasks** up to a task you could not have learned directly.

> This is exactly the **Curriculum Learning feature you met in Unit 5 with ML-Agents** (e.g. progressively raising the wall height in WallJump / progressively enlarging the target): same idea, exposed as a config file where a lesson's threshold on the reward triggers the next lesson.

### Automatic Curriculum Learning (ACL)
Designing such a curriculum is **not always trivial**, so **ACL proposes approaches that learn to create the organization of tasks in order to maximize the RL agent's performance**. Portelas et al. define ACL as:

> … a family of mechanisms that automatically adapt the distribution of training data by learning to adjust the selection of learning situations to the capabilities of RL agents.

So ACL is a **teacher** whose action is "which task to serve next" and whose objective is the **student's learning progress** — a nice second-order RL problem.

**Concrete example:** OpenAI used **Domain Randomization** (random variations applied to the environment) to make a **robot hand solve Rubik's Cubes** — see [OpenAI: Solving Rubik's Cube with a Robot Hand](https://openai.com/blog/solving-rubiks-cube/).

You can also play with the robustness of trained agents in the **[TeachMyAgent](https://developmentalsystems.org/TeachMyAgent/)** benchmark by controlling environment variations — or even **drawing the terrain yourself** — in the interactive demo: [Interactive DeepRL Demo (flowers-team)](https://huggingface.co/spaces/flowers-team/Interactive_DeepRL_Demo).

**Further reading (from the page):**

*Overview of the field*
- [Automatic Curriculum Learning For Deep RL: A Short Survey](https://arxiv.org/pdf/2003.04664.pdf)
- [Curriculum for Reinforcement Learning](https://lilianweng.github.io/posts/2020-01-29-curriculum-rl/) — Lilian Weng

*Recent methods*
- [Evolving Curricula with Regret-Based Environment Design](https://arxiv.org/abs/2203.01302)
- [Curriculum Reinforcement Learning via Constrained Optimal Transport](https://proceedings.mlr.press/v162/klink22a.html)
- [Prioritized Level Replay](https://arxiv.org/abs/2010.03934)

---

## 5. Decision Transformers (RL as sequence modeling)

The **Decision Transformer** was introduced by [*Decision Transformer: Reinforcement Learning via Sequence Modeling*, Chen L. et al.](https://arxiv.org/abs/2106.01345). It **abstracts RL as a conditional-sequence-modeling problem**.

### The main idea
Instead of training a policy with RL methods — such as **fitting a value function** that tells us which action maximizes the return — we use a **sequence modeling algorithm (a Transformer) that, given a desired return, past states, and past actions, generates future actions to achieve that desired return**.

It is an **autoregressive model conditioned on the desired return, past states and past actions**, generating future actions that achieve the desired return.

Concretely, the trajectory is fed as a **flat token sequence of three modalities**, with the return replaced by the **return-to-go** $\hat{R}_t$:

$$
\tau \;=\; \big(\hat{R}_1,\, s_1,\, a_1,\; \hat{R}_2,\, s_2,\, a_2,\; \dots,\; \hat{R}_T,\, s_T,\, a_T \big), \qquad \hat{R}_t = \sum_{t'=t}^{T} r_{t'}
$$

and the model is trained with a plain **supervised** loss to predict $a_t$ from everything before it. At test time you **specify the return you want** (e.g. "get 6000") as $\hat{R}_1$, feed the current state, read off the action, execute it, then **decrement** the return-to-go by the reward received, and repeat.

> **Key insight:** this is **a complete shift in the RL paradigm** — we use **generative trajectory modeling** (modeling the joint distribution of the sequence of states, actions and rewards) to replace conventional RL algorithms. In Decision Transformers **we don't maximize the return; we generate a series of future actions that achieve the *desired* return.**

### Why it fits the offline setting
- No bootstrapping, no TD target, no $\max_a Q$ ⇒ **none of the instability or OOD-action blowup** of §1.
- The objective is **behavior cloning conditioned on outcome**, so a fixed dataset is all you need — exactly the offline RL data assumption.
- Mediocre and failed trajectories are still **useful training data**: they teach the model what a *low* return-to-go looks like. Conditioning is what lets you ask for the good behavior at inference.
- It inherits the Transformer's strengths: long context (partial observability handled by attention over history) and scaling.

The 🤗 Transformers team **integrated the Decision Transformer — an Offline RL method — into the library and onto the Hugging Face Hub**.

**Learn / do:**
- Read [Introducing Decision Transformers on Hugging Face](https://huggingface.co/blog/decision-transformers)
- Then train your first offline Decision Transformer from scratch to **make a half-cheetah run**: [Train your first Decision Transformer](https://huggingface.co/blog/train-decision-transformers)

**Further reading (from the page):**
- [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345)
- [Online Decision Transformer](https://arxiv.org/abs/2202.05607)

---

## 6. Language models in RL

### LMs encode useful knowledge for agents
**Language models (LMs)** show impressive abilities when manipulating text — question answering, even step-by-step reasoning. Additionally, training on **massive text corpora** let them **encode various types of knowledge, including abstract knowledge about the physical rules of our world** (what is possible to do with an object, what happens when you rotate an object, …).

The natural question: can this knowledge **benefit agents such as robots** solving everyday tasks? Early works showed interesting results, **but the proposed agents lacked any learning method** — which **prevents them from adapting to the environment** (e.g. fixing wrong knowledge) **or learning new skills**.

### LMs and RL: the synergy
There's a potential synergy: **LMs bring knowledge about the world, RL aligns and corrects that knowledge by interacting with an environment.**

This is especially interesting from the RL side, because RL mostly relies on the **tabula rasa** setup where **everything is learned from scratch**, leading to:

1. **Sample inefficiency**
2. **Unexpected behaviors from humans' eyes**

Two directions studied:

| Direction | Paper | What it does |
|---|---|---|
| **Adapt / align the LM itself** | [Grounding Large Language Models with Online Reinforcement Learning](https://arxiv.org/abs/2302.02662v1) | Aligns an LM to a **textual environment using PPO**. The knowledge in the LM gave **fast adaptation** to the environment (a path to sample-efficient RL agents) and **better generalisation to new tasks** once aligned |
| **Keep the LM frozen, use it as a guide** | [Guiding Pretraining in Reinforcement Learning with Large Language Models](https://arxiv.org/abs/2302.06692) | Leverages LM knowledge to **guide the RL agent's exploration** toward **human-meaningful and plausibly useful behaviors**, without needing a human in the loop during training |

> ⚠️ Several limitations make this work **still very preliminary**: the agent's observations must be **converted to text** before being given to the LM, and **interacting with very large LMs is computationally expensive**.

**Further reading (from the page):**
- [Google Research, 2022 & beyond: Robotics](https://ai.googleblog.com/2023/02/google-research-2022-beyond-robotics.html)
- [Towards Helpful Robots: Grounding Language in Robotic Affordances](https://ai.googleblog.com/2022/08/towards-helpful-robots-grounding.html)
- [Pre-Trained Language Models for Interactive Decision-Making](https://arxiv.org/abs/2202.01771)
- [Grounding Large Language Models with Online Reinforcement Learning](https://arxiv.org/abs/2302.02662v1)
- [Guiding Pretraining in Reinforcement Learning with Large Language Models](https://arxiv.org/abs/2302.06692)

---

## 7. RLHF — Reinforcement Learning from Human Feedback

This is the section worth the most attention: it's the bridge from this course to modern LLM training.

> **RLHF is a methodology for integrating human data labels into an RL-based optimization process.**

**Motivation: the challenge of modeling human preferences.** For many questions, **even if you could write down an equation for one ideal, humans differ on their preferences**. Updating models **based on measured data is an avenue to alleviate these inherently human ML problems** — i.e. instead of hand-writing a reward function, **learn it from data about what people actually prefer**.

Notice this is the same failure mode as reward hacking in §1, approached from the other side: rather than fixing the simulator, we **fix the reward**.

### The pipeline (three stages)

**Stage 1 — Pretraining / supervised fine-tuning.** Start from a pretrained LM. Optionally supervised-fine-tune it on demonstration data of the desired behavior. Call the result the **reference model** $\pi_{\text{ref}}$; it is also the initialization of the policy $\pi_\theta$.

**Stage 2 — Collect human preferences and train a reward model.** Sample several completions $y_1, y_2, \dots$ for the same prompt $x$, and ask humans **which one they prefer** — *comparisons*, not absolute scores, because relative judgments are far more consistent between annotators. Train a **reward model** $r_\phi(x,y)$ (a scalar-head LM) so that the preferred (*winning*) completion $y_w$ scores above the rejected (*losing*) one $y_l$:

$$
\mathcal{L}(\phi) \;=\; -\,\mathbb{E}_{(x,\,y_w,\,y_l)\sim \mathcal{D}}\Big[\log \sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)\Big]
$$

The reward model is thus a **learned stand-in for the human**, cheap enough to query millions of times inside an RL loop.

**Stage 3 — RL fine-tuning (typically PPO).** Treat the LM as a policy: the **state** is the prompt plus tokens generated so far, an **action** is the next token, and the **reward** is $r_\phi(x,y)$ at the end of the generation. Then run the **PPO** you learned in Unit 8 — with one crucial addition, a **KL penalty** that keeps the policy near the reference model:

$$
\max_{\theta}\; \mathbb{E}_{x\sim\mathcal{D},\; y\sim\pi_\theta(\cdot\mid x)}\Big[\, r_\phi(x,y) \,\Big] \;-\; \beta\, \mathbb{D}_{\mathrm{KL}}\!\Big[\pi_\theta(y\mid x)\,\big\|\,\pi_{\text{ref}}(y\mid x)\Big]
$$

equivalently, RL is run on the **shaped per-sample reward**

$$
r_{\text{total}}(x,y) \;=\; r_\phi(x,y) \;-\; \beta \log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}
$$

> **Key insight — why the KL term is not optional.** The reward model is only accurate **on the distribution of text it was trained on**. Unconstrained PPO will find high-scoring gibberish: it drifts off-distribution and **over-optimizes the reward model** rather than the humans it proxies. The KL penalty is a leash to $\pi_{\text{ref}}$, so the policy improves **without leaving the region where $r_\phi$ is trustworthy**. The scaling behavior of exactly this failure is studied in [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760).

Note the **two distinct clipping/constraint mechanisms** in play, which are easy to confuse:

| Mechanism | Between what | Purpose |
|---|---|---|
| **PPO clipped objective / ratio clip** | current policy vs the policy that **collected this batch** | Keep each *update step* small — optimization stability (Unit 8) |
| **KL penalty $\beta$** | current policy vs the **frozen reference model** | Keep the *final model* near the pretrained LM — prevent reward-model over-optimization and loss of language quality |

### Learning path the course gives you
1. Read the introduction: [Illustrating Reinforcement Learning from Human Feedback (RLHF)](https://huggingface.co/blog/rlhf).
2. Watch the recorded live where **Nathan Lambert** covers the basics of RLHF and how it enables state-of-the-art ML tools like ChatGPT — mostly an overview of the **interconnected ML models**, the basics of NLP and RL, how RLHF is used on large language models, and **open questions in RLHF**. (YouTube id `2MBJOuVq380`.)
3. Read other blogs, e.g. [Closed-API vs Open-source continues: RLHF, ChatGPT, data moats](https://robotic.substack.com/p/rlhf-chatgpt-data-moats).

### Additional readings — RLHF papers

*(copied from the Illustrating RLHF blog post; the field was popularized with the emergence of DeepRL around 2017 and grew into a broader study of LLM applications)*

**Pre-dating the LM focus:**
- [TAMER: Training an Agent Manually via Evaluative Reinforcement](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/ICDL08-knox.pdf) (Knox & Stone, 2008) — humans provide **scores on actions** iteratively to learn a reward model.
- [Interactive Learning from Policy-Dependent Human Feedback](http://proceedings.mlr.press/v70/macglashan17a/macglashan17a.pdf) (MacGlashan et al., 2017) — **COACH**, an actor-critic where human feedback (positive **and** negative) tunes the **advantage function**.
- [Deep Reinforcement Learning from Human Preferences](https://proceedings.neurips.cc/paper/2017/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html) (Christiano et al., 2017) — RLHF applied to **preferences between Atari trajectories**.
- [Deep TAMER: Interactive Agent Shaping in High-Dimensional State Spaces](https://ojs.aaai.org/index.php/AAAI/article/view/11485) (Warnell et al., 2018) — extends TAMER with a **deep network** for reward prediction.

**RLHF for language models:**
- [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593) (Ziegler et al., 2019) — early study of reward learning on four specific tasks.
- [Learning to summarize with human feedback](https://proceedings.neurips.cc/paper/2020/hash/1f89885d556929e98d3ef9b86448f951-Abstract.html) (Stiennon et al., 2020) — RLHF for summarization; follow-up [Recursively Summarizing Books with Human Feedback](https://arxiv.org/abs/2109.10862) (OpenAI Alignment Team, 2021).
- [WebGPT: Browser-assisted question-answering with human feedback](https://arxiv.org/abs/2112.09332) (OpenAI, 2021) — RLHF to train an agent to **navigate the web**.
- **InstructGPT**: [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (OpenAI Alignment Team, 2022) — RLHF on a general LM ([blog post](https://openai.com/blog/instruction-following/)).
- **GopherCite**: [Teaching language models to support answers with verified quotes](https://www.deepmind.com/publications/gophercite-teaching-language-models-to-support-answers-with-verified-quotes) (Menick et al., 2022) — answers **with citations**.
- **Sparrow**: [Improving alignment of dialogue agents via targeted human judgements](https://arxiv.org/abs/2209.14375) (Glaese et al., 2022) — fine-tuning a **dialogue agent** with RLHF.
- [ChatGPT: Optimizing Language Models for Dialogue](https://openai.com/blog/chatgpt/) (OpenAI, 2022).
- [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760) (Gao et al., 2022) — scaling properties of the **learned preference model**.
- [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862) (Anthropic, 2022) — detailed documentation of training an LM assistant with RLHF.
- [Red Teaming Language Models to Reduce Harms](https://arxiv.org/abs/2209.07858) (Ganguli et al., 2022) — efforts to "discover, measure, and attempt to reduce [language models'] potentially harmful outputs".
- [Dynamic Planning in Open-Ended Dialogue using Reinforcement Learning](https://arxiv.org/abs/2208.02294) (Cohen et al., 2022).
- [Is Reinforcement Learning (Not) for Natural Language Processing?](https://arxiv.org/abs/2210.01241) (Ramamurthy & Ammanabrolu et al., 2022) — discusses the design space of open-source RLHF tooling and proposes **NLPO (Natural Language Policy Optimization)** as an alternative to PPO.

---

## 8. Godot RL Agents

[**Godot RL Agents**](https://github.com/edbeeching/godot_rl_agents) is an **open source package that lets video game creators, AI researchers and hobbyists learn complex behaviors for their Non-Player Characters or agents** inside the [Godot Engine](https://godotengine.org/). It provides an interface between Godot games and ML algorithms running in Python, with **wrappers for four well-known RL frameworks** — [StableBaselines3](https://stable-baselines3.readthedocs.io/en/master/), [CleanRL](https://docs.cleanrl.dev/), [Sample Factory](https://www.samplefactory.dev/) and [Ray RLlib](https://docs.ray.io/en/latest/rllib-algorithms.html) — plus support for **memory-based agents (LSTM / attention)**, **2D and 3D games**, and a suite of **AI sensors** to augment what the agent can observe. Godot and Godot RL Agents are **completely free and open source under a permissive MIT license**. Installation is just `pip install godot-rl`. The library's creator, [Ed Beeching](https://edbeeching.github.io/), is a Research Scientist at Hugging Face; see the [GitHub page](https://github.com/edbeeching/godot_rl_agents) and the AAAI-2022 workshop [paper](https://arxiv.org/abs/2112.03636).

**Worth remembering from the hands-on** (building *Ring Pong* — pong on a ring, where the objective is to keep the ball bouncing inside the ring — with **Godot 4.0**):

- You add an **`AIController3D`** node as a child of the player, then **"Extend Script"** it (`controller.gd`) and implement four methods: **`get_obs()`**, **`get_reward()`**, **`get_action_space()`**, **`set_action(action)`**. GDScript is syntactically similar to Python.
- In Ring Pong the observation is **ball position and velocity in the paddle's local frame** (`[ball_pos.x, ball_pos.z, ball_vel.x/10.0, ball_vel.z/10.0]`) and the action space is **a single continuous value clamped to $[-1, 1]$**.
- A **Godot RL Agents Sync** node (added to `train.tscn`) handles **Python↔Godot communication over TCP**; it exposes a **"Speed Up"** property (try 8) to accelerate training. Multiple game instances run in parallel for the same reason.
- Launch training with `gdrl` (live in the editor) — a reasonable Ring Pong policy is learned **in several minutes**.
- **ONNX export** works with SB3, RLlib and CleanRL: train via the [SB3 example](https://github.com/edbeeching/godot_rl_agents/blob/main/examples/stable_baselines3_example.py) with `--onnx_export_path=model.onnx`, then set the Sync node's control mode to **`Onnx Inference`** and point `Onnx Model Path` at the file — that runs the agent **inside Godot with no Python server**. Inference requires the **mono (.NET) build** of Godot plus [.NET](https://dotnet.microsoft.com/en-us/download).
- More: the [examples repo](https://github.com/edbeeching/godot_rl_agents_examples), the [advanced SB3 tutorial](https://github.com/edbeeching/godot_rl_agents/blob/main/docs/ADV_STABLE_BASELINES_3.md), [troubleshooting](https://github.com/edbeeching/godot_rl_agents/blob/main/docs/TROUBLESHOOTING.md), and Godot's own [documentation](https://docs.godotengine.org/en/latest/index.html) (plus [GDQuest](https://www.gdquest.com/), [KidsCanCode](https://kidscancode.org/godot_recipes/4.x/), [Bramwell](https://www.youtube.com/channel/UCczi7Aq_dTKrQPF5ZV5J3gg) for learning Godot).

---

## 9. Unreal Learning Agents

[**Learning Agents**](https://dev.epicgames.com/community/learning/tutorials/8OWY/unreal-engine-learning-agents-introduction) is an **Unreal Engine (UE) plugin that allows you to train AI characters using machine learning inside Unreal**. It's the Unreal counterpart of Godot RL Agents / ML-Agents: you build a unique environment with Unreal's tooling and train agents in it — the course's worked example is **teaching a car to drive in an Unreal environment**.

**If you're new to Unreal**, the course prescribes a warm-up before touching the plugin:
1. [Your first hour in Unreal Engine 5](https://dev.epicgames.com/community/learning/courses/ZpX/your-first-hour-in-unreal-engine-5/E7L/introduction-to-your-first-hour-in-unreal-engine-5) — foundational knowledge.
2. [Blueprints video course](https://youtu.be/W0brCeJNMqk?si=zy4t4t1l6FMIzbpz) — Unreal's **visual scripting** system.

**Then (and this is the whole path if you already know Unreal):**
1. [Learning Agents overview](https://dev.epicgames.com/community/learning/tutorials/8OWY/unreal-engine-learning-agents-introduction) — the big picture.
2. [Teach a Car to Drive using Reinforcement Learning in Learning Agents](https://dev.epicgames.com/community/learning/tutorials/qj2O/unreal-engine-learning-to-drive).
3. [Imitation Learning with the Unreal Engine 5.3 Learning Agents Plugin](https://www.youtube.com/watch?v=NwYUNlFvajQ) — note this is **imitation learning**, the theme picked up again in Bonus Unit 5.

---

## 10. Interesting environments to try

Four environments the course recommends once you want problems harder than the Gym classics:

### DIAMBRA Arena
A software package featuring **high-quality environments for RL research**, giving a **standard interface to popular arcade emulated video games** with a Python API **fully compliant with the OpenAI Gym/Gymnasium format**. Supports **Linux, Windows and macOS**, installable via [pip](https://pypi.org/project/diambra-arena/); **free to use**, you only need to register on the [official website](https://diambra.ai/register/).

- **Main features:** all environments are **episodic** RL tasks with **discrete actions** (gamepad buttons) and observations made of **screen pixels plus additional numerical data** (RAM values such as character health bars or stage side). All support **single player (1P) and two players (2P)**, which makes them ideal for **standard RL, competitive multi-agent, competitive human–agent, self-play, imitation learning and human-in-the-loop**. [Interfaced games](https://docs.diambra.ai/envs/games/) are popular fighting retro-games sharing the same fundamentals but with different challenges (number/type of characters, combos, health-bar recharging…).
- Native interfaces to [Stable Baselines 3](https://stable-baselines3.readthedocs.io/en/master/) and [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html) (Stable Baselines is available but **deprecated**); see the [DIAMBRA Agents examples repo](https://github.com/diambra/agents).
- **Competition platform** fully integrated with the **Hugging Face Hub**: submit trained agents, compete in tournaments, climb a **public leaderboard**, unlock achievements — submitted agents' episodes are streamed on the [DIAMBRA Twitch channel](https://www.twitch.tv/diambra_ai).
- Resources: [Official docs](https://docs.diambra.ai/) · [Competition platform](https://diambra.ai) · [GitHub](https://github.com/diambra/) · [Discord](https://diambra.ai/discord)

### MineRL
A Python library providing a **Gym interface to Minecraft**, accompanied by **datasets of human gameplay** (so it's also an offline-RL / imitation playground). **Yearly challenges** — see the [website](https://minerl.io/).
- [What is MineRL?](https://www.youtube.com/watch?v=z6PTrGifupU) · [First steps in MineRL](https://www.youtube.com/watch?v=8yIrWcyWGek) · [Docs and tutorials](https://minerl.readthedocs.io/en/latest/)

### DonkeyCar Simulator
**Donkey** is a **self-driving-car platform for hobby remote-control cars**. This simulator version is built on **Unity**, using its physics and graphics, and connects to a **Donkey Python process** so your trained model can control the simulated car.
- [Simulator documentation](https://docs.donkeycar.com/guide/deep_learning/simulator/)
- Antonin Raffin's *Learn to Drive Smoothly*: [Part 1](https://www.youtube.com/watch?v=ngK33h00iBE) · [Part 2](https://www.youtube.com/watch?v=DUqssFvcSOY) · [Part 3](https://www.youtube.com/watch?v=v8j2bpcE4Rg)
- Pretrained agents: [tqc-donkey-mountain-track-v0](https://huggingface.co/araffin/tqc-donkey-mountain-track-v0) · [tqc-donkey-avc-sparkfun-v0](https://huggingface.co/araffin/tqc-donkey-avc-sparkfun-v0) · [tqc-donkey-minimonaco-track-v0](https://huggingface.co/araffin/tqc-donkey-minimonaco-track-v0)

### StarCraft II
A famous **real-time strategy game**; DeepMind used it for Deep RL research with [AlphaStar](https://www.deepmind.com/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii).
- [StarCraft gym](http://starcraftgym.com/) · [A.I. Learns to Play StarCraft 2 tutorial](https://www.youtube.com/watch?v=q59wap1ELQ4)

*(The unit also has a [**Student Works**](https://huggingface.co/learn/deep-rl-course/unitbonus3/student-works) page collecting projects built by course participants — games and agents in Unity/Pygame/etc., arranged by publication date. You can [add your own by opening a PR](https://github.com/huggingface/deep-rl-class).)*

---

## 11. Brief introduction to RL documentation (Reward Reports)

The question this section addresses: **how should we monitor and keep track of powerful RL agents that we are training in the real world and interfacing with humans?**

As ML systems have increasingly impacted modern life, **the call for documentation of these systems has grown**. Such documentation can cover the **training data** (where it is stored, when it was collected, who was involved) or the **model optimization framework** (architecture, evaluation metrics, relevant papers) and more. **Model cards and datasheets** are increasingly available — e.g. on the Hub ([model cards documentation](https://huggingface.co/docs/hub/model-cards)); click any [popular model on the Hub](https://huggingface.co/models) and you can learn about its creation process.

> ⚠️ **The gap:** these model- and data-specific logs are designed to be **completed when the model or dataset is created**, which leaves them **un-updated when those models are built into evolving systems** later.

### Motivating Reward Reports
RL systems are **fundamentally designed to optimize based on measurements of reward and time**. A reward function maps nicely onto well-understood supervised learning (via a loss function), but **our understanding of how ML systems evolve over time is limited**.

Hence **Reward Reports for Reinforcement Learning** — the name deliberately mirroring *Model Cards for Model Reporting* and *Datasheets for Datasets*. The goal is documentation focused on the **human factors of reward** and on **time-varying feedback systems**. Building on the documentation frameworks of [model cards](https://arxiv.org/abs/1810.03993) (Mitchell et al.) and [datasheets](https://arxiv.org/abs/1803.09010) (Gebru et al.), the authors argue the need for Reward Reports for AI systems:

> **Reward Reports are living documents for proposed RL deployments that demarcate design choices.**

Many questions remain — applicability across different RL applications, roadblocks to **system interpretability**, and the resonances between deployed supervised ML systems and the **sequential decision-making** used in RL. At a minimum, Reward Reports are **an opportunity for RL practitioners to deliberate on these questions**.

### Capturing temporal behavior with documentation
The piece **specific to RL and feedback-driven ML** is a **change-log**. It records:
- updates **from the designer** — changed training parameters, data, etc.,
- alongside noticed changes **from the user** — harmful behavior, unexpected responses, etc.

The change-log is accompanied by **update triggers** that encourage monitoring these effects.

### Contributing
Some of the most impactful RL-driven systems are **multi-stakeholder and behind the closed doors of private corporations**, which are largely **unregulated** — so **the burden of documentation falls on the public**. Reward Reports for popular ML systems are being built as a public record on [GitHub](https://github.com/RewardReports/reward-reports); see [an example report](https://github.com/RewardReports/reward-reports/tree/main/examples) and the [paper](https://arxiv.org/abs/2204.10817).

---

## 12. Terminology recap (one screen)

| Term | One-line definition |
|---|---|
| **Online RL** | The agent collects its own experience by interacting with the env (directly or via a replay buffer) |
| **Offline RL** | The policy is learned **only** from a fixed dataset of other agents'/humans' trajectories; no interaction |
| **Counterfactual queries problem** | Offline RL's core failure: the agent wants to do something the dataset never did (a.k.a. **distributional shift** / OOD actions) |
| **Generalisation gap** | Train-level performance minus held-out-level performance |
| **Domain randomization** | Randomizing env parameters during training to gain robustness / sim-to-real transfer |
| **Dynamics model** | $s_{t+1} = f_\theta(s_t,a_t)$, learned by minimizing the negative log-likelihood of transitions |
| **MPC** | Plan over a finite horizon $\tau$ with the model, execute the first action, re-plan |
| **Compounding error** | Model error accumulating over a multi-step predicted rollout |
| **Curriculum Learning** | A hand-designed ordering of tasks so skills are acquired incrementally |
| **ACL** | Mechanisms that **automatically adapt the training-task distribution** to the agent's current capabilities |
| **Return-to-go** | $\hat R_t = \sum_{t'=t}^{T} r_{t'}$ — the conditioning signal of a Decision Transformer |
| **Decision Transformer** | Autoregressive model over $(\hat R, s, a)$ tokens that generates actions achieving a **desired** return |
| **Tabula rasa** | The standard RL setup where the agent learns everything from scratch (no priors) |
| **Reward model** | $r_\phi(x,y)$, trained on human **pairwise comparisons**, used as a proxy human inside RL |
| **KL penalty ($\beta$)** | Term keeping the RL-tuned policy close to the frozen reference model $\pi_{\text{ref}}$ |
| **Reward-model over-optimization** | Gaining reward-model score while getting **worse** by real human judgment |
| **Reward Report** | A **living document** for an RL deployment, centered on a change-log + update triggers |

---

## 13. Self-check (quick review questions)

1. Define **online** and **offline** RL in one sentence each. Which one has this course been doing?
2. Give **three** reasons building a simulator for online RL can be a bad option — and explain what "the agent will exploit the simulator's flaws" means.
3. What is the **counterfactual queries problem**, and why can't offline RL simply "explore a bit" to resolve it?
4. Why is offline RL often described as "supervised learning on trajectories"? Where does that analogy break?
5. What is the **generalisation gap**, and how do **procedural generation** and **domain randomization** attack it?
6. Write down what a **dynamics model** predicts, and the loss used to fit it. What are two other kinds of models that fit the MBRL framework?
7. In sample-based **MPC**, what exactly is optimized, over what horizon, and how many of the planned actions do you actually execute?
8. Why does MBRL buy **sample efficiency**, and what does it pay for it with? Explain **compounding error** and why the planner makes it worse.
9. Give the bipedal-walker example of why a **curriculum** is needed. Then explain, in terms of gradients, why a curriculum helps a **sparse-reward** task.
10. Quote (roughly) Portelas et al.'s definition of **ACL**. What is the "teacher's" objective?
11. What did OpenAI use domain randomization for, and what is **TeachMyAgent**?
12. Write the **token sequence** a Decision Transformer consumes, and define **return-to-go**. At inference, how does $\hat R_t$ change from step to step?
13. Decision Transformers "don't maximize the return." So what *do* they do — and why does that make them a natural fit for **offline** data?
14. Why are LMs interesting as priors for agents? Name the **two** problems of tabula-rasa RL the section lists.
15. Contrast the two LM+RL directions: **aligning the LM with PPO** vs **keeping it frozen to guide exploration**. What are the two limitations that make this work preliminary?
16. Describe the **three stages** of RLHF. Why are humans asked for **comparisons** instead of absolute scores?
17. Write the **reward-model loss** and the **RLHF objective with the KL penalty** from memory. What does $\beta$ control?
18. Distinguish the **PPO ratio clip** from the **KL-to-reference penalty** — what is each protecting against?
19. What is **reward-model over-optimization**, and which paper studies its scaling behavior?
20. What are the four RL framework wrappers Godot RL Agents provides, and which four methods must you implement on an `AIController3D`?
21. What does the **Sync** node do in Godot RL Agents, and what does exporting to **ONNX** buy you?
22. Which observations and action space did the *Ring Pong* agent use?
23. What makes **DIAMBRA Arena** suited to self-play and imitation learning, and what does MineRL ship besides an environment?
24. Why are model cards and datasheets **insufficient** for RL systems, and what is the piece of a **Reward Report** that specifically addresses this?

---

## 14. Additional Readings (consolidated)

Everything below appears in the Bonus Unit 3 pages; grouped for convenience.

**Offline RL**
- [Offline Reinforcement Learning — talk by Sergey Levine](https://www.youtube.com/watch?v=qgZPZREor5I)
- [Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/abs/2005.01643)
- [Offline RL blog/post the course's figure is from](https://offline-rl.github.io/) · [intro video](https://www.youtube.com/watch?v=k08N5a0gG0A)

**Generalisation**
- [Generalization in Reinforcement Learning — Robert Kirk (survey)](https://robertkirk.github.io/2022/01/17/generalisation-in-reinforcement-learning-survey.html)
- [Improving Generalization in RL using Policy Similarity Embeddings](https://blog.research.google/2021/09/improving-generalization-in.html?m=1)

**Model-based RL**
- [Debugging MBRL (blog)](https://www.natolambert.com/writing/debugging-mbrl) · [Review paper on MBRL](https://arxiv.org/abs/2006.16712)

**Curriculum learning**
- [Automatic Curriculum Learning For Deep RL: A Short Survey](https://arxiv.org/pdf/2003.04664.pdf) · [Curriculum for RL (Lilian Weng)](https://lilianweng.github.io/posts/2020-01-29-curriculum-rl/)
- [Evolving Curricula with Regret-Based Environment Design](https://arxiv.org/abs/2203.01302) · [Curriculum RL via Constrained Optimal Transport](https://proceedings.mlr.press/v162/klink22a.html) · [Prioritized Level Replay](https://arxiv.org/abs/2010.03934)
- [TeachMyAgent](https://developmentalsystems.org/TeachMyAgent/) · [Interactive demo](https://huggingface.co/spaces/flowers-team/Interactive_DeepRL_Demo) · [OpenAI: Solving Rubik's Cube with a Robot Hand](https://openai.com/blog/solving-rubiks-cube/)

**Decision Transformers**
- [Decision Transformer: RL via Sequence Modeling](https://arxiv.org/abs/2106.01345) · [Online Decision Transformer](https://arxiv.org/abs/2202.05607)
- [Introducing Decision Transformers on Hugging Face](https://huggingface.co/blog/decision-transformers) · [Train your first Decision Transformer](https://huggingface.co/blog/train-decision-transformers)

**LMs in RL**
- [Google Research, 2022 & beyond: Robotics](https://ai.googleblog.com/2023/02/google-research-2022-beyond-robotics.html) · [Towards Helpful Robots: Grounding Language in Robotic Affordances](https://ai.googleblog.com/2022/08/towards-helpful-robots-grounding.html)
- [Pre-Trained Language Models for Interactive Decision-Making](https://arxiv.org/abs/2202.01771) · [Grounding LLMs with Online RL](https://arxiv.org/abs/2302.02662v1) · [Guiding Pretraining in RL with LLMs](https://arxiv.org/abs/2302.06692)

**RLHF** — see the full paper list in §7, starting from [Illustrating RLHF](https://huggingface.co/blog/rlhf).

**Game engines**
- [Godot RL Agents](https://github.com/edbeeching/godot_rl_agents) · [paper](https://arxiv.org/abs/2112.03636) · [examples](https://github.com/edbeeching/godot_rl_agents_examples)
- [Unreal Learning Agents overview](https://dev.epicgames.com/community/learning/tutorials/8OWY/unreal-engine-learning-agents-introduction) · [Learning to drive](https://dev.epicgames.com/community/learning/tutorials/qj2O/unreal-engine-learning-to-drive)

**RL documentation**
- [Reward Reports paper](https://arxiv.org/abs/2204.10817) · [GitHub](https://github.com/RewardReports/reward-reports) · [example report](https://github.com/RewardReports/reward-reports/tree/main/examples)
- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) · [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) · [Hub model cards docs](https://huggingface.co/docs/hub/model-cards)

---

> **Next up:** **Bonus Unit 5 — Imitation Learning with Godot RL Agents**, where instead of shaping a reward you *record your own demonstrations* and train a robot to copy them inside a Godot environment — and then the **certification** page to claim your certificate of completion.
