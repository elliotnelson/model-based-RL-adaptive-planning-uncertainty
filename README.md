## Model-based Reinforcement Learning with Adaptive Planning under Uncertainty

This project aims to train an RL agent to plan adaptively with a perfect model by using its observations to evaluate its model's quality locally when making and using predictions in different regions of state space.

The current implementation augments a baseline model-free A2C (Advantage Actor Critic) algorithm with a model-based loop which trains the transition (and/or reward) model at each iteration, and uses it to train the policy on model rollouts.

<a href="https://www.codecogs.com/eqnedit.php?latex=||\vec{g}(s_{real})&space;-&space;\vec{g}(s_{sim})||" target="_blank"><img src="https://latex.codecogs.com/gif.latex?||\vec{g}(s_{real})&space;-&space;\vec{g}(s_{sim})||" title="||\vec{g}(s_{real}) - \vec{g}(s_{sim})||" /></a>
