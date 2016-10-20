## Method


I compare the behaviour of reinforcement learners in the uninterrupted
[`CartPole-v1`](https://gym.openai.com/envs/CartPole-v1) environment with that
in the interrupted
[`OffSwitchCartpole-v0`](https://gym.openai.com/envs/OffSwitchCartpole-v0)
environment. The `OffSwitchCartpole-v0` one of several environments that Rafael
Cosman wrote (or adapted from existing environments) in order to assess safety
properties of reinforcement learners.

The learners I assess are my own implementations of $\Sarsl$ and Q-learning,
which use the Fourier basis [4] as a linear function approximator and an
AlphaBounds [5] schedule for the learning rate. (You can look at the code, but
it's quite horrible. I didn't prepare it for human consumption.)
I run each of them on each of the environments for 16 training rounds, each
comprising 200 episodes that are terminated after 500 steps if the pole doesn't
fall before that. Again, a bit less condensed.

|                      |$\Sarsl$|Q-learning|
|`CartPole-v1`         |run     |run       |
|`OffswitchCartpole-v0`|run     |run       |

 * 1 run consists of 16 rounds.
 * The learning in every round starts from scratch. I.e. all weights initialized
   to 0 and the learning rate set to the initial learning rate.
 * Every round consists of 200 episodes. Weights and learning rates are taken
   along from episode to episode. (Just as you normally do when your train a
   reinforcement learner.)
 * Every episodes lasts for at most 500 steps. Less if the pole falls earlier.
 * The parameters $\lambda$, learning rate $\alpha$ and exploration probability
   $\epsilon$ are the same for all learners and runs.
 * The learners don't discount.

I observe the behaviour of the agents in two ways. I record the total rewards
per episode and plot them against the episode numbers in order to see that the
agents converge to a behaviour where the pole stays up in (almost) every round.
Note that this doesn't mean they converge to the optimal policy. And I record
the number of time steps the agent is right of the middle and left of the middle
over the whole run. The logarithm of the ratio between the number of time steps
spent on the left and the number of time steps spent on the right tells me how
strongly the agent is biased to either side.


 - CartPole vs. OffSwitchCartpole
 - $\Sarsl$ and Q-learning
    - 3rd order Fourier basis
    - AlphaBounds
 - Run 16 training rounds with each of them where one training round comprises
   200 episodes that are terminated after 500 steps if the pole doesn't fall
   before that.
 - Record the number of steps where the cart is left of the middle and right of
   the middle. – Calculate the ratio and compare the ratios.
 - Visualize the rewards for each run in order to make sure that (almost) each
   training converges to a behaviour where the pole stays up. (Not the same as
   optimal policy.)

 - You can look at the code, but it's quite horrible. Didn't prepare it for
   human consumption.
