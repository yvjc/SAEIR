

# SAEIR

Open-source code for [SAEIR: Sequentially Accumulated Entropy Intrinsic Reward for Cooperative Multi-Agent Reinforcement Learning with Sparse Reward].

The paper is submitted to IJCAI24 and the submission number is 3113. Our approach can guide multiple agents to find better states that decrease the disorder of system and facilitate the learning of successful strategies in challenging sparse reward environments.



## Installation instructions

Install Python packages

Set up Google Football:

```shell
bash install_gfootball.sh
```

## Command Line Tool

**Run an experiment**

```shell
# For Academy 3 vs.1 with Keeper scenario
bash  train_football_3v1_SAEIR.sh
```

