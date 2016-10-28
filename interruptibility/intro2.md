We want to align AI with human interests. Reinforcement learning (RL) algorithms
are a class of current AI. Rafael Cosman has adapted several environments from
the OpenAI Gym to observe AI alignment-related properties of RL algorithms. One
such property is the response to interruptions. One environment to observe this
is the OffSwitchCartpole. This is an adaptation of the well-known CartPole
environment where the learning gets interrupted (reward $0$) everytime the cart
moves more than $1.0$ units to the right. In this notebook I observe in a
primitive experiment how Sarsa(λ) and Q-learning react to interruptions by
comparing their behaviour in the CartPole and the OffSwitchCartpole
environments.
