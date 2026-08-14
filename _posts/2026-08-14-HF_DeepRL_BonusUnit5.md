---
layout: post
title:  "HF Deep RL Course — Bonus Unit 5 (Imitation Learning with Godot)"
date:   2026-08-14
desc: "HF Deep RL Course — Bonus Unit 5 (Imitation Learning with Godot)"
keywords: "Machine learning"
categories: [Machine learning]
tags: [Machine learning]
icon: icon-html
---

# HF Deep RL Course — Bonus Unit 5 (Imitation Learning with Godot)

> Source: [Hugging Face Deep RL Course, Bonus Unit 5](https://huggingface.co/learn/deep-rl-course/unitbonus5/introduction) — notes from *Introduction* through *Conclusion*, reorganized for review. Continues from [Bonus Unit 3]({{ '/machine learning/2026/08/14/HF_DeepRL_BonusUnit3.html' | replace: ' ', '%20' | prepend: site.baseurl }}).

> **_Keypoints:_**

- **Imitation Learning (IL)**: learn from **expert demonstrations**, not from a reward signal
- **Behavioral Cloning (BC)**: plain supervised learning on (observation → expert action) pairs
- BC's classic failure mode: **compounding errors / distributional shift**
- **GAIL**: adversarial IL — a discriminator supplies the reward, so the learner trains **in the env**
- The unit's recipe: **BC pre-training → GAIL fine-tuning** with PPO as the generator
- Concrete pipeline: Godot 4.3 .NET + `godot-rl` + `imitation`, record demos → export game → train → `.onnx`
- Robot level: pull lever → raise stairs → collect key → return and open chest
- Practical numbers: **22–24 successful episodes**, `--bc_epochs=100`, `--gail_timesteps≈1.45e6`, stop when `ep_rew_mean → 3`
- Gotchas: demos only save on window close, `action_repeat` must match in all 3 scenes, `R` deletes a bad episode
- Self-check + further resources

---

## 0. What this unit is for

Every unit so far has trained an agent from a **reward**. This bonus unit does the opposite: **train a robot agent to complete a mini-game level using imitation learning**, where the only supervision is **human-recorded expert demonstrations**.

The end result is a trained agent that solves a 3-room Godot level: pull a lever, collect a key, come back and open a chest — exported as an `.onnx` file and run natively inside the Godot game.

**Prerequisites the unit states explicitly:**

| Requirement | Detail |
|---|---|
| Previous chapter | [Godot RL Agents (Bonus Unit 3)](https://huggingface.co/learn/deep-rl-course/unitbonus3/godotrl) recommended |
| Godot familiarity | Recommended, but **no GDScript coding knowledge required** |
| Godot version | Godot **with .NET support** — tested on [4.3.dev5 .NET](https://godotengine.org/article/dev-snapshot-godot-4-3-dev-5/), newer may work |
| Godot RL Agents | `pip install godot-rl` in your venv/conda env |
| Imitation library | `pip install imitation` in the **same** env |
| Time | **~1–2 hours** for the project + training (hardware dependent) |

```bash
# both packages must live in the SAME environment
pip install godot-rl
pip install imitation
```

---

## 1. What Imitation Learning is (and how it differs from RL)

In RL, the agent has **no idea what good behavior looks like**; it discovers it by trial and error, guided only by a **scalar reward** you designed. In IL, you **skip the reward entirely** and instead hand the agent a dataset of what an expert did:

$$
\mathcal{D} = \{(s_1, a_1), (s_2, a_2), \dots, (s_N, a_N)\} \quad\text{sampled from the expert policy } \pi_E
$$

**Why you'd want this:**

- **Reward design is hard.** "Pull the lever, then get the key, then come back to the chest" is a long multi-stage task. A naive sparse reward gives ≈0 signal for millions of steps.
- **Sparse-reward exploration is hopeless** at this horizon — random exploration almost never stumbles onto the full lever→key→chest sequence.
- **You already know what you want.** If a human can play the level in 30 seconds, demonstrating is far cheaper than shaping a reward function.
- **The behavior style is part of the goal.** IL reproduces *how* the expert plays, not just the outcome.

> **Key insight:** RL asks *"what maximizes my return?"*. IL asks *"what would the expert have done here?"*. The unit's own framing: IL "can be an alternative to training in-game agents with RL in some cases."

### RL vs BC vs GAIL

| | **RL (e.g. PPO)** | **Behavioral Cloning (BC)** | **GAIL** |
|---|---|---|---|
| Learns from | **Reward** $r(s,a)$ from the env | **Expert (obs, action) pairs** | **Expert trajectories** + a learned discriminator |
| Problem type | Sequential decision making | **Plain supervised learning** | Adversarial (GAN-like) + RL inner loop |
| Needs env interaction during training? | **Yes** | **No** — offline, one pass over the dataset | **Yes** — the generator rolls out in the env |
| Needs a reward function? | **Yes** | No | No (the discriminator *is* the reward) |
| Main strength | Can exceed the expert; optimizes the true objective | Extremely cheap, fast, stable | Fixes BC's drift; learns from **its own** states |
| Main weakness | Reward design + exploration | **Compounding errors / distributional shift** | Slower, adversarial instability, needs the env |
| Role in this unit | Optional extra `--rl_timesteps` | **Pre-training** stage (`--bc_epochs`) | **Main** training stage (`--gail_timesteps`) |

### Behavioral Cloning, precisely

BC treats control as regression/classification. Fit a policy $\pi_\theta$ to the demonstration dataset by maximum likelihood:

$$
\theta^{*} = \arg\max_{\theta} \sum_{(s,a)\in\mathcal{D}} \log \pi_\theta(a \mid s)
$$

For the continuous actions used here, that's essentially a squared-error fit of the action vector:

$$
\mathcal{L}_{\text{BC}}(\theta) = \mathbb{E}_{(s,a)\sim\mathcal{D}}\big[\; \lVert \pi_\theta(s) - a \rVert^2 \;\big]
$$

No environment, no rollouts, no reward — just gradient descent on a fixed file. This is why BC pre-training in the unit takes **~5.5 minutes on CPU** while GAIL takes **~41 minutes**.

### BC's classic weakness: compounding errors

Supervised learning assumes train and test data come from the **same distribution**. Control violates that assumption, because **the policy's own actions determine the states it will see next**.

The failure chain:

1. BC is trained only on states the **expert** visited: the distribution $d_{\pi_E}$.
2. At run time the learner makes a small error and lands slightly **off** the expert's path.
3. That new state is **out of distribution** — BC never saw it, so its output there is arbitrary.
4. The arbitrary action moves it even further off-distribution. Repeat.

$$
d_{\pi_\theta} \neq d_{\pi_E} \quad\Longrightarrow\quad \text{errors accumulate over the episode, not per-step}
$$

> ⚠️ **The intuition:** BC's mistakes are not independent. A per-step error rate $\epsilon$ over a horizon $T$ compounds — regret grows roughly like $O(\epsilon T^2)$ rather than $O(\epsilon T)$, because each mistake also *changes which states you have to be correct on*. Concretely: the human demonstrator never fell in the water, so BC has **zero data** on how to recover once the robot is near the edge.

### How GAIL fixes it

**GAIL (Generative Adversarial Imitation Learning)** turns imitation into a two-player game, exactly like a GAN:

- A **discriminator** $D_\psi(s,a)$ learns to tell **expert** transitions from **learner** transitions.
- The **generator** is the policy $\pi_\theta$, trained with normal RL (PPO here) on a **reward derived from the discriminator** — high reward for looking expert-like.

$$
\min_{\theta}\;\max_{\psi}\;\; \mathbb{E}_{\pi_\theta}\!\big[\log D_\psi(s,a)\big] \;+\; \mathbb{E}_{\pi_E}\!\big[\log\big(1 - D_\psi(s,a)\big)\big] \;-\; \lambda H(\pi_\theta)
$$

The crucial difference from BC: **the policy is rolled out in the environment**, so it is trained on **its own** state distribution — including the off-expert states BC has no answer for. That's what directly attacks distributional shift.

| | **BC** | **GAIL** |
|---|---|---|
| Supervision | Direct action labels | A **learned reward** (discriminator score) |
| Trained on which states? | Only the **expert's** states | The **learner's own** states |
| Distributional shift | Unaddressed | **Directly addressed** |
| Cost | Minutes, offline | Hours, needs env rollouts |

> **Key insight — why the unit uses both:** BC gives you a policy that is already roughly right, for almost free. GAIL then repairs the drift. Starting GAIL from a BC-initialized policy (the script passes `policy=learner.policy` into `bc.BC`, so **BC writes into the very same PPO network**) means GAIL's expensive env interaction is spent on refinement, not on discovering the task from scratch.

---

## 2. The environment

The tutorial level features a **robot** in a 3-room level that must:

1. **Pull a lever** to raise the stairs leading to the second room,
2. **Navigate to the key 🔑 and collect it**, while avoiding falling into **traps**, **water**, or **outside the map**,
3. **Navigate back to the treasure chest** in the first room and **open it**. Victory! 🏆

> ⚠️ The **key has 2 alternating spawn positions** (left platform or right platform). This matters for demo recording: you want your demonstrations to cover **both** positions roughly equally, or the agent will only learn one.

### Observations

Built in `get_obs()` on the `AIController3D` node. Two sources:

**(a) Two Raycast sensors** (under a `RaycastSensors` node) that let the agent "sense" walls, floors and other level geometry:

```python
var observations: Array[float] = []
for raycast_sensor in raycast_sensors:
    observations.append_array(raycast_sensor.get_observation())
```

**(b) 24 hand-crafted floats** appended after them:

| Group | Values | Count |
|---|---|---|
| Chest | local direction $(x,y,z)$ + distance | 4 |
| Lever | local direction $(x,y,z)$ + distance | 4 |
| Key | local direction $(x,y,z)$ + distance | 4 |
| Raft | local direction $(x,y,z)$ + distance | 4 |
| Raft state | `raft.movement_direction_multiplier` | 1 |
| Game flags | `_is_lever_pulled`, `_is_chest_opened`, `_is_key_collected`, `is_on_floor()` | 4 |
| Player speed | normalized local velocity $(x,y,z)$ | 3 |

Three design details worth remembering:

- **Relative, not absolute.** Positions are converted with `to_local(...)`, then split into a **normalized direction** and a **clamped distance**:

```python
var chest_local = to_local(chest.global_position)
var chest_direction = chest_local.normalized()
var chest_distance = clampf(chest_local.length(), 0.0, level_size)   # level_size = 16.0
```

  Splitting direction from distance means the network gets a clean unit vector plus a bounded scalar, instead of one large-magnitude vector.

- **Booleans become floats.** `float(player._is_lever_pulled)` → `0.0`/`1.0`. `is_on_floor()` doubles as "can I jump?".

- **Velocity is normalized and local:** `player.global_basis.inverse() * player.velocity.limit_length(5.0) / 5.0` — expressed in the robot's own frame and scaled into roughly $[-1,1]$.

### Actions

All **continuous**, 5 floats total:

```python
func get_action_space() -> Dictionary:
	return {
		"movement":   {"size": 2, "action_type": "continuous"},
		"rotation":   {"size": 1, "action_type": "continuous"},
		"jump":       {"size": 1, "action_type": "continuous"},
		"use_action": {"size": 1, "action_type": "continuous"}
	}
```

| Action | Size | Meaning |
|---|---|---|
| `movement` | 2 | planar move (WASD equivalent) |
| `rotation` | 1 | **Y-axis only** — the agent does not rotate the camera on X, it isn't needed for the task |
| `jump` | 1 | thresholded: `bool(action.jump[0] > 0)` |
| `use_action` | 1 | thresholded: activate lever / open chest |

> **Key insight:** the two boolean-ish actions are still **continuous**, encoded as $-1$ / $+1$ via `-1.0 + 2.0 * float(flag)` when recording, and decoded with `> 0` when acting. This keeps the whole action space continuous (one Gaussian policy head), and keeps recorded expert values inside the required $[-1, 1]$ range.

### Reward (used only for monitoring, not for training)

The unit is explicit that setting env rewards "is not necessary and not used for the training here" — a simple **sparse** reward exists only to **track success**:

| Event | Reward |
|---|---|
| Activating the lever | `+1` |
| Collecting the key | `+1` |
| Opening the chest | `+1` |
| Falling outside the map, in water, or in traps | `−1` |

So **`ep_rew_mean` approaching 3 means the agent solves the level.**

### Episode termination

Two timeouts, both implemented in `_physics_process()`:

```python
n_steps += 1
if n_steps > reset_after:
    player.game_over()

steps_without_lever_pulled += 1
if steps_without_lever_pulled > 200 and (not player._is_lever_pulled):
    player.game_over()
```

The second one is a nice practical trick: **if the lever isn't pulled within 200 steps the episode is cut short**, because nothing useful can happen afterwards (the stairs are still down). `reset()` must then clear that counter:

```python
func reset():
	super.reset()
	steps_without_lever_pulled = 0
```

---

## 3. The pipeline, step by step

| # | Step | Where / command |
|---|---|---|
| 1 | Install deps | `pip install godot-rl imitation` |
| 2 | Download the project | [ivan267/imitation-learning-tutorial-godot-project](https://huggingface.co/ivan267/imitation-learning-tutorial-godot-project/tree/main) → `GDRL-IL-Project.zip` |
| 3 | Open the **Starter** project | Godot → *Import* → `Starter\Godot` |
| 4 | Implement the `AIController3D` script | `get_obs`, `get_reward`, `_physics_process`, `reset`, `get_action_space`, `get_action`, `set_action`, `_input` |
| 5 | **Record expert demos** | `demo_record_scene` → **F6** → play → **close the window** |
| 6 | **Export the game** | Godot → `Project > Export` |
| 7 | **Train** (BC → GAIL) | `sb3_imitation.py ...` |
| 8 | **Test the agent** | copy `model.onnx` into the project → `onnx_inference_scene` → **F6** |

The zip ships **both** a "Starter" and a "Complete" project, so you can diff against the finished version (and reuse its pre-recorded `expert_demos.json` if you'd rather skip recording).

### Step 4 in a bit more detail

The game code and node setup are already done; you only write the `AIController3D` glue. The key structural point is that **one script serves four modes**, and `set_action()` branches on whether it was handed an action:

```python
func set_action(action = null) -> void:
	if not action:
		# HUMAN / demo_record: read the actual keyboard + mouse
		...
		player.requested_movement = Input.get_vector("move_left", "move_right", "move_forward", "move_back")
	else:
		# TRAINING / onnx inference: apply the model's action
		player.requested_movement = Vector2(action.movement[0], action.movement[1])
```

And `_physics_process()` compensates for the fact that the `Sync` node calls `set_action()` for you in training/record modes, but **not** in pure human mode:

```python
if control_mode == ControlModes.HUMAN:
    set_action()
```

`get_action()` is the mirror of `set_action()` and is **only needed for demo record mode** — it reports "here is the action the human just took in this state", which is exactly the label BC will regress on. Order must match `get_action_space()` exactly, and values must stay in $[-1, 1]$.

Mouse input is buffered in `_input()` rather than applied immediately, so that **frame skipping (`action_repeat`) applies identically** to the human and to the agent:

```python
func _input(event):
	if not (heuristic == "human" or heuristic == "demo_record"):
		return
	if event is InputEventMouseMotion:
		var movement_scale: float = 0.005
		mouse_movement.y = clampf(event.relative.y * movement_scale, -1.0, 1.0)
		mouse_movement.x = clampf(event.relative.x * movement_scale, -1.0, 1.0)
```

Then **save the script**.

Also note the `robot` scene contains a `CameraXRotation` node used for mouse look in human modes — **the AI agent does not control it**, since vertical camera aim isn't needed for the task.

---

## 4. Recording expert demonstrations

Open the **demo record scene** and select `Level > Robot > AIController3D`. Everything is preset, but these are the properties you'd set in your own environment:

| Property (on `AIController3D`) | Value / purpose |
|---|---|
| `Control Mode` | **`Record Expert Demos`** |
| `Expert Demo Save Path` | output file, e.g. `expert_demos.json` |
| `Action Repeat` | **3** — must match the `Sync` node in `training_scene` **and** `onnx_inference_scene` |
| `Remove Last Episode` key | **`R`** — deletes a failed episode mid-session |

And on the **`Sync`** node:

| Property | Purpose |
|---|---|
| `Speed Up` | default **1**; **lower it to slow the env down** and make hard episodes easier to record |

### Why `Action Repeat` matters

Every action is repeated for **3 physics frames**. Setting the same value on the `AIController` during recording deliberately **adds the same lag to human input**, so the human's demonstrations are generated under the same control dynamics the agent will face. It's a low enough value that the lag is barely noticeable.

> ⚠️ If you change `action_repeat`, **change it in all three places** (demo record `AIController`, `training_scene` `Sync`, `onnx_inference_scene` `Sync`). A mismatch means you trained on a different control problem than you demonstrated.

### Recording controls

Press **F6** in `demo_record_scene` to start.

| Input | Action |
|---|---|
| **Mouse** | camera (adjust via `Robot` node → `Rotation Speed`) |
| **WASD** | movement |
| **SPACE** | jump |
| **E** | activate lever / open chest |
| **R** | remove the previously recorded (failed) episode |

> ⚠️ Keep `Rotation Speed` **identical** across recording, training and inference.

### ⚠️ The big gotcha: demos only save on window close

> ⚠️ Demos are written **only** if you have recorded **at least one complete episode** *and* you close the game window with **"X"** or **ALT+F4**. **Pressing Stop in the Godot editor will NOT save the demos.** Record a single episode first and verify `expert_demos.json` appears in the FileSystem / project folder before investing 10 minutes.

### How many, and how good?

The unit's concrete advice:

- **At least 22–24 complete *successful* episodes.**
- **22 or 24** gives an even split over the key's **2 alternating spawn positions** (23 is close enough).
- **Multiple demo files are allowed** — you don't have to record in one sitting; just change `Expert Demo Save Path` between sessions and pass all files to the trainer.
- The author recorded **23 episodes in ~10 minutes**.
- **Press and hold `E` slightly longer** near the lever and chest, so the `use_action` is recorded across **multiple steps**. Otherwise the label is a single frame and the agent barely learns to press it.
- **Delete failures.** The author removed a couple of unsuccessful episodes by pressing `R` during the following episode.
- Take a few **practice runs** first to get familiar with the env.

> **Key insight:** the whole quality of the result is bounded by these ~23 episodes. There is no reward to correct a sloppy demonstrator — **the agent will faithfully imitate your mistakes**. This is IL's fundamental trade-off: you removed reward engineering and replaced it with *demonstration* engineering.

### Export the game

```
Godot → Project > Export
```

You need the exported executable for `--env_path`, which is what lets you run `--n_parallel 4` copies at `--speedup 20`.

---

## 5. Training (BC → GAIL)

Grab [`sb3_imitation.py`](https://github.com/edbeeching/godot_rl_agents/blob/main/examples/sb3_imitation.py) from the Godot RL Agents repo, then:

```bash
python sb3_imitation.py \
  --env_path="path_to_ILTutorial_executable" \
  --bc_epochs=100 \
  --gail_timesteps=1450000 \
  --demo_files "path_to_expert_demos.json" \
  --n_parallel=4 \
  --speedup=20 \
  --onnx_export_path=model.onnx \
  --experiment_name=ILTutorial
```

Multiple demo files are space-separated:

```bash
--demo_files demos.json demos2.json
```

Add `--viz` to watch the env during training.

### Arguments

| Argument | Meaning |
|---|---|
| `--env_path` | the exported Godot binary (omit for in-editor training) |
| `--demo_files` | one or more recorded demo JSON files |
| `--bc_epochs` | epochs of **BC pre-training** (here **100**) |
| `--gail_timesteps` | env timesteps of **GAIL** (here **1,450,000**) |
| `--rl_timesteps` | *optional* extra plain-RL timesteps **after** IL |
| `--n_parallel` | env executable instances (**4**) — requires `--env_path` |
| `--speedup` | physics speed multiplier (**20**) |
| `--onnx_export_path` | where to write `model.onnx` |
| `--experiment_name` | logs go to `logs/<experiment_name>` |
| `--eval_episode_count` | *optional* post-training evaluation episodes |
| `--viz` | render the sim while training |

The three stages run in order: **BC → GAIL → (optional) RL**, each skipped if its budget is 0.

### The hyperparameters in the script

The PPO **generator**:

| Hyperparameter | Value |
|---|---|
| policy | `MlpPolicy` |
| `batch_size` | 256 |
| `ent_coef` | 0.007 |
| `learning_rate` | 0.0002 |
| `n_steps` | 64 |
| `target_kl` | 0.02 |
| `n_epochs` | 5 |
| `log_std_init` | $\log(1.0) = 0$ |
| `device` | `cpu` |

The **GAIL** trainer:

| Hyperparameter | Value |
|---|---|
| `demo_batch_size` | 256 |
| `n_disc_updates_per_round` | 16 |
| reward net | `BasicRewardNet` |
| `allow_variable_horizon` | `True` |
| generator | the **same** PPO `learner` |

BC uses `policy=learner.policy`, i.e. **BC and GAIL share one network** — that's the mechanism by which BC acts as pre-training rather than as a separate model. Each recorded episode becomes **one trajectory** (`Trajectory(obs, acts, terminal=True)`).

> ⚠️ `device="cpu"` is deliberate: these are small MLPs over ~30-dim observations, and CPU is faster than GPU for that. The author's whole run was **~5.5 min BC + ~41 min GAIL on CPU**.

### Reading the logs

| Metric | What it tells you |
|---|---|
| `ep_rew_mean` | the **real** sparse env reward — **approaching 3 = solving the level** |
| `ep_rew_wrapped_mean` | the reward from the **GAIL discriminator** — does **not** directly tell you how successful the agent is |

In the unit's run the two closely matched. Logs live in `logs/ILTutorial` (relative to where you launched training) and are viewable with [tensorboard](https://github.com/edbeeching/godot_rl_agents/blob/main/docs/TRAINING_STATISTICS.md).

> **Key insight:** always judge progress by `ep_rew_mean`. The discriminator reward is a **moving target** — it measures "how expert-like do I look to a discriminator that is itself still learning", so it can go up while task performance stagnates.

**Practical stopping strategy:** set a large `--gail_timesteps` and interrupt with **CTRL+C** when `ep_rew_mean` approaches 3. The author stopped at `total_timesteps | 1.38e+06`. The script catches `KeyboardInterrupt` and still exports the `.onnx` / saves the model in its `finally` block.

> ⚠️ If you make multiple runs, **change `--experiment_name`** each time or tensorboard won't display properly.

Two `--viz` quirks:

- During **BC**, the env appears **frozen** — that stage doesn't use the env except to query the observation and action spaces.
- During **GAIL**, rendering updates normally, since the generator is actually rolling out.

---

## 6. Evaluating: ONNX inference in Godot

After training you get `model.onnx` in the folder you launched from (the full path is also printed near the end of the console log).

1. **Copy `model.onnx` into the Godot game project folder.**
2. Open the **onnx inference scene** — like the demo record scene it uses **one** copy of the level, and its `Sync` node mode is set to **`Onnx Inference`**.
3. Click the `Sync` node and set **`Onnx Model Path` → `model.onnx`**.
4. Press **F6**.

Expected result: the agent **collects the key from both spawn positions** (left and right platform) and **replicates the recorded behavior well**. That means it learned a policy that generalizes over the level's one source of variation, rather than memorizing a single route.

If your results differ significantly:

| Lever to pull | Why |
|---|---|
| Record **more / cleaner** demos | Amount and quality of demos directly affects results |
| Adjust **BC epochs / GAIL timesteps** | Under- or over-training either stage |
| Modify **hyperparameters** in the Python script | PPO / GAIL settings are tuned for this env, not universal |
| Just **re-run** | There is genuine **run-to-run variation** — same settings can give slightly different results |

> **Key insight:** ONNX export is what makes this a *shippable* result. Training needs Python + `stable-baselines3` + `imitation`; inference needs only the ONNX runtime built into Godot RL Agents. The trained policy becomes an ordinary game asset.

---

## 7. (Optional) Customizing the environment

Open the level scene `res://scenes/level.tscn` and the modules folder `res://scenes/modules/`.

The level is **3 rooms built from modules**, plus the robot, plus **extra colliders whose only job is to block a shortcut** — without them you could climb a wall in the first room and reach the key directly, skipping the lever entirely. Adding modules to the scene adds new rooms and items.

### The three mechanisms you need to understand

**(a) Signals — wiring game logic.** Select the `Key` node (it lives in `Room3`), then `Node > Signals`: the **`collected`** signal is connected to **both the robot and the chest** — the robot tracks that the key was collected, and the chest unlocks. The lever→stairs link uses the same system. **If you add more levers / stairs / keys, connect them with signals.**

**(b) The `resetable` group — episode resets.** Switch to `Groups`: the key is a member of **`resetable`**, along with the **raft, lever, chest and player**. Any node needing per-episode reset goes in this group.

- Every `resetable` **must implement a `reset()` method** that resets itself.
- Because training runs **multiple instances of the level scene simultaneously**, you must **not** reset all resetables globally. `level_manager.gd` provides **`reset_all_resetables()`**, which resets only those **within the same scene**, and the robot script calls it when a reset is needed.

**(c) Observation bookkeeping — keeping the AIController in sync.**

| If you change… | You must also… |
|---|---|
| The level **size** | Update `level_size` in `robot_ai_controller.gd` — roughly measure the level's **longest dimension** (it clamps the distance observations; currently `16.0`) |
| The **number of tracked objects** (levers, rafts, …) | Update the script, add **export properties** for them, and connect them in the `AIController` inspector in the **level scene** |
| …and then | Update the same `AIController` properties in the **demo record scene** too |

> ⚠️ The last row is the easiest thing to forget: there are **two** `AIController` instances to configure (level scene and demo record scene). Miss one and your demos will be recorded with different observations than training uses.

---

## 8. Gotcha checklist

| Gotcha | Consequence if ignored |
|---|---|
| Close the game window with **X / ALT+F4**, not the editor Stop button | **Demos are silently not saved** |
| Need **≥1 complete episode** before demos save | Empty file |
| `action_repeat` identical in **3** places | You demonstrate a different control problem than you train |
| `Rotation Speed` identical across record/train/inference | Action scale mismatch |
| Action values must be in $[-1, 1]$ and in `get_action_space()` order | Silently wrong labels |
| Hold **`E`** longer near lever/chest | `use_action` appears for too few steps to learn |
| Cover **both** key spawn positions | Agent only solves one variant |
| Use **`R`** to drop failed episodes | You teach the agent to fail |
| `--experiment_name` unique per run | Tensorboard logs overlap |
| `imitation` in the **same** env as `godot-rl` | Import errors |
| Read **`ep_rew_mean`**, not `ep_rew_wrapped_mean` | You misjudge progress |
| Update **both** `AIController` instances after edits | Obs mismatch between recording and training |

---

## 9. Glossary

- **Imitation Learning (IL)** — training a policy from **expert demonstrations** instead of from a reward signal.
- **Expert demonstrations** — recorded (observation, action) trajectories from a human (or other expert) playing the task. Here: `expert_demos.json`, one **trajectory per episode**.
- **Behavioral Cloning (BC)** — supervised learning of $\pi_\theta(a \mid s)$ directly on the demonstration pairs. Offline, no env needed.
- **Distributional shift / compounding errors** — BC is trained on the *expert's* state distribution but evaluated on its *own*; small errors move it off-distribution, where it has no data, causing errors to compound over the episode.
- **GAIL (Generative Adversarial Imitation Learning)** — adversarial IL: a **discriminator** distinguishes expert from learner transitions and its output is used as the **reward** for an RL generator (PPO here), so the policy trains on its own state distribution.
- **Discriminator reward** (`ep_rew_wrapped_mean`) — GAIL's learned reward. Measures expert-likeness, **not** task success.
- **Godot RL Agents** — the `godot-rl` package + Godot plugin connecting Godot environments to Python RL libraries.
- **`AIController3D`** — the Godot node where you implement `get_obs()`, `get_reward()`, `get_action_space()`, `get_action()`, `set_action()`.
- **`Sync` node** — drives the env: sets its **mode** (training / onnx inference), `action_repeat`, and `Speed Up`.
- **Control mode / heuristic** — `human`, `demo_record`, training, `onnx inference`. `demo_record` is the one that writes demos.
- **`action_repeat`** — number of physics frames each action is held for (**3** here); applied to human input during recording too.
- **`resetable` group** — Godot group of nodes that implement `reset()` and are reset per episode, **per level instance**.
- **ONNX export** — serializing the trained policy to `model.onnx` so Godot can run inference without Python.

---

## 10. Self-check (quick review questions)

1. What is the **one sentence** difference between RL and IL in terms of the *supervision signal*?
2. Give two situations where IL is clearly preferable to RL, and one where it clearly isn't.
3. Write the **BC objective** from memory. What kind of learning problem is it, formally?
4. Explain **compounding errors** in BC. Why is per-step supervised accuracy not enough for control?
5. Why does BC have **no data** on how the robot should recover from nearly falling in the water?
6. How does **GAIL** differ from BC in (a) what it learns from, (b) whether it needs the env, (c) which state distribution it trains on?
7. In GAIL, **what plays the role of the reward function**, and why is its value a poor progress metric?
8. Why does this unit run **BC before GAIL** instead of just one of them? What line in the script makes BC act as pre-training?
9. Name the **three sub-goals** in the level and the sparse reward for each. What `ep_rew_mean` means "solved"?
10. In `get_obs()`, why are object positions split into a **normalized direction** and a **clamped distance** rather than kept as a raw vector?
11. Why is `jump` a **continuous** action of size 1, and how are booleans encoded/decoded?
12. What does `steps_without_lever_pulled > 200` accomplish, and what must `reset()` do because of it?
13. Why must `action_repeat` be applied to **human input** during demo recording? Which three places must agree?
14. How do you **save** recorded demos, and what happens if you use the editor's Stop button?
15. Why **22 or 24** successful episodes rather than any number, and why hold **`E`** longer near the lever?
16. What is the `resetable` group, and why does training use `reset_all_resetables()` instead of resetting everything?
17. You double the level's size. Which variable must you update, and where?
18. What exactly do you copy where to run the trained agent in Godot, and which `Sync` properties do you set?

---

## 11. Further resources

**The tooling**
- [Godot RL Agents repository](https://github.com/edbeeching/godot_rl_agents) — the plugin + `pip install godot-rl`
- [`sb3_imitation.py`](https://github.com/edbeeching/godot_rl_agents/blob/main/examples/sb3_imitation.py) — the exact BC + GAIL training script used here
- [Godot RL Agents training statistics docs](https://github.com/edbeeching/godot_rl_agents/blob/main/docs/TRAINING_STATISTICS.md) — how to read the tensorboard logs
- [`imitation` library docs / installation](https://imitation.readthedocs.io/en/latest/getting-started/installation.html) — the BC and GAIL implementations
- [Godot Engine](https://godotengine.org/) — and the tested build, [4.3.dev5 .NET](https://godotengine.org/article/dev-snapshot-godot-4-3-dev-5/)

**The project**
- [Tutorial Godot project (`GDRL-IL-Project.zip`)](https://huggingface.co/ivan267/imitation-learning-tutorial-godot-project/tree/main) — Starter + Complete, including pre-recorded `expert_demos.json`

**Course pages**
- [Bonus Unit 5 — Introduction](https://huggingface.co/learn/deep-rl-course/unitbonus5/introduction)
- [Bonus Unit 3 — Godot RL Agents](https://huggingface.co/learn/deep-rl-course/unitbonus3/godotrl) — the recommended prerequisite
- [Bonus Unit 5 — Train our robot](https://huggingface.co/learn/deep-rl-course/unitbonus5/train-our-robot)

**Credits (from the unit)**
- Tutorial written by [Ivan Dodic](https://github.com/Ivan-267), with reviews and feedback from [Edward Beeching](https://twitter.com/edwardbeeching) and [Thomas Simonini](https://twitter.com/thomassimonini).

---

> **Next up:** that's the end of the course's bonus material — from here it's **claiming your certification** (pass the units' hands-on requirements and grab the certificate of completion / excellence) and then continuing on your own: read the literature the units pointed at (Sutton & Barto, the PPO / DQN / A2C papers, and for this unit's topic the GAIL and DAgger lines of work), keep pushing models to the Hub, and pick a project where you get to design the environment yourself.
