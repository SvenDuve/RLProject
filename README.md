Add untracked packages, ie. not neccessary to git add, to reflect changes


```julia
]dev ./Example
```

for a package in the Environment folder. 


Workflow, 

in the actual module folder, before ```using``` in environment, add all packages, so they are in the project toml. then potentially ]resolve


Model Free Learner

|Algorithm|Action space|
|DDPG|Continuous|
|DQN|Discrete|


Learning the environment

|Algorithm|Action space|
|DyNODE|Discrete/ Continuous|
|ODE-RNN|Discrete/ Continuous|

Environments

|Name|Type|
|Pendulum|Continuous|
|LunarLander|Continuous|
|LunarLander|Discrete|
|Acrobot|Discrete|

MBRL

|Pendulum|Continuous|DyNODE|DDPG|MPC|
|Pendulum|Continuous|ODE-RNN|DDPG|MPC|
|LunarLander|Continuous|DyNODE|DDPG|MPC|
|LunarLander|Continuous|ODE-RNN|DDPG|MPC|
|Acrobot|Discrete|DyNODE|DQN|MPC|
|Acrobot|Discrete|ODE-RNN|DQN|MPC|
|LunarLander|Discrete|DyNODE|DQN|MPC|
|LunarLander|Discrete|ODE-RNN|DQN|MPC|