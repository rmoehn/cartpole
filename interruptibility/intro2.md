We want to align AI with human interests. Reinforcement learning (RL) algorithms
are a class of current AI. The OpenAI Gym has [several
adaptations](https://gym.openai.com/envs#safety) of classic RL environments that
allow us to observe AI alignment-related properties of RL algorithms. One such
property is the response to interruptions. One environment to observe this is
the [`OffSwitchCartpole-v0`](https://gym.openai.com/envs/OffSwitchCartpole-v0).
This is an adaptation of the well-known
[`CartPole-v1`](https://gym.openai.com/envs/CartPole-v1) environment where the
learning gets interrupted (reward $0$) everytime the cart moves more than $1.0$
units to the right. In this notebook I observe in a primitive experiment how
Sarsa(λ) and Q-learning react to interruptions by comparing their behaviour in
the `CartPole-v1` and the `OffSwitchCartpole-v0` environments.

(Note: Don't be confused by the `v0` and `v1`. I'm just using them to be
consistent throughout the text and with the OpenAI Gym. Actually, `CartPole-v1`
is the same as `CartPole-v0`, only the way the evaluation is run in the Gym is
different: in `v0` an episode lasts for at most 200 timesteps, in `v1` for at
most 500. The `OffSwitchCartpole-v0` is also run for 200 timesteps. I'm writing
`CartPole-v1` everywhere, because in my experiments I also run the environments
for at most 500 steps. Since there is no `OffSwitchCartpole-v1`, though, I have
  to write `OffSwitchCartpole-v0`. Okay, now you are confused. Never mind. Just
  ignore the `vx` and you'll be fine.)

(Another note: When you see the section headings in this notebook, you might
think that I was trying to produce a proper academic publication. This is not
so. Such a framework just makes writing easier.)
