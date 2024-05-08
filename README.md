# pytorch-distributed-reinforcement-learning
## Architecture on Single Machine 
![architecture](https://github.com/ymg1114/pytorch-distributed-reinforcement-learning/assets/54105796/ec9f8c7f-7827-4452-bad6-ee26705c8f6d)

## Algo
PPO

PPO-Continuous

IMPALA

V-MPO

SAC

SAC-Continuous

## Caution!
Discrete Learning environment is configured to `CartPole-v1`.

Continuous Learning environment is configured to `MountainCarContinuous-v0`.

## How to run
`python main.py manager_sub_process`

`python main.py worker_sub_process`

`python main.py learner_sub_process`
