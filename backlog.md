# To do

- implemente pure NODE MB - done, works
- add path variable to MBRL function -done, works
- remove envparams size, and gymansium state vector output
- check Git experimentally with RewardApproximation - done, failed
- Check DDPG learning from the new state vector, working for Lunar Lander, not yet for BipedalWalker, still struggling


- generate DQN project, done
- code up model free DQN algorithm, done
- code up model based DQN algorithm combining model free with MPC
- Changed input type of remember function of the replay buffer. All still working. Done until further notice. 
- Care for the action range, it is -1 from the argmax calculation, done. Using onehot batching.  
- The Q function should output values for the number of actions. it works such that the highest q output equals the number of the index of the action to be taken. done.


Checks:

- Check line 372 of MBRL NODE Dynamics learning, return function. I think this returns only one variable. fixed
- found a trainstep! in MBRL Node, with missing enviornment signature. perhaps more, fixed.
- Check new reward function correct.


# To do

- Check feasibility of update to 1.9, done, works
- Check all implementations working
  - DDPG, Code Execution works fine. Agent learning. 
  - DQN, Code Execution works fine. Agent learning.
  - NODEDynamics, code Execution works fine. Model learning.
  - ODERNNDynamics, code Execution works fine. Model learning.
  - MBRL, All working fine.
  - RLTypes, all fine
  - Rewards, all fine
- Introduce git repos for the packages
- Generate Analytics function
- Create Experiment schedule
    - Model Free
    - Model Based

- remove rewards from model learners
- Clean up git repository of gymnasium