We want to align AI with human interests. Reinforcement learning (RL) algorithms
are a class of AI. The OpenAI Gym has [several
adaptations](https://gym.openai.com/envs#safety) of classic RL environments that
allow us to observe AI alignment-related properties of RL algorithms. One such
property is the response to interruptions. One environment to observe this is
the [`OffSwitchCartpole-v0`](https://gym.openai.com/envs/OffSwitchCartpole-v0).
This is an adaptation of the well-known
[`CartPole-v0`](https://gym.openai.com/envs/CartPole-v1) environment where the
learning gets interrupted (reward $0$) everytime the cart moves more than $1.0$
units to the right. In this notebook I observe in a primitive experiment how
Sarsa(λ) and Q-learning react to interruptions by comparing their behaviour in
the `CartPole-v0` and the `OffSwitchCartpole-v0` environments.

You can see this notebook as the second version of [(Non-)Interruptibility of
Sarsa(λ) and
Q-Learning](https://nbviewer.jupyter.org/github/rmoehn/cartpole/blob/master/notebooks/ProcessedOSCP.ipynb)
(referred to as the ‘old notebook’ throughout the text). In that old notebook I
wrote as if I was sure of what I'd found. I ignored that my experiments yielded
quite different numbers on every run. And there were some flaws in the method,
which Stuart Armstrong and Patrick LaVictoire pointed out. For this notebook I
eliminated those flaws (other flaws might be remaining), used ten times as much
data, more expressive measures and illustrations. And I mostly don't draw
conclusions, but ask questions, because questions are all I can confidently
derive from my experiments.

(Note: When you see the section headings in this notebook, you might think that
I was trying to produce a proper academic publication. This is not so. Such a
framework just makes writing easier.)
