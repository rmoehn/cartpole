In the old notebook I included the setup code for the learners and environments.
I leave it out in this notebook, because the following description is sufficient
and less confusing.

The environments are `CartPole-v1` (also called *uninterrupted cart-pole/case*
by me) and `OffSwitchCartpole-v0` (*interrupted cart-pole/case*) from the OpenAI
Gym. Nowadays `gym.make` returns these environments in a `TimeLimit` wrapper,
but I strip off the wrapper by accessing `gym.make(…).env`. This way I can train
the learners for as long as I want.

You don't need to understand what the `v0` and `v1` mean. Just ignore them.

The learners are my own implementations of Sarsa(λ) and Q-learning according to
[9, p. 211] with the following parameters:

- $\lambda = 0.9$
- Discounting factor $\gamma = 0.99$ for Sarsa(λ) and $\gamma = 0.999$ for
  Q-learning. In the old notebook I didn't use discounting.
- $\epsilon$-greedy policy with $\epsilon = 0.1$.
- AlphaBounds schedule [5] for the learning rate. Initial learning rate set to
  $0.001$.
- Calculation of expected action values through linear function approximation,
  with a third-order Fourier basis [4] for mapping observations to features.
- If actions are tied in expected value, the tie is broken randomly. In the old
  notebook ties were broken deterministically (select action 0), which led to
  asymmetric cart movement patterns.

I run each of the learners on each of the environments for at least 156 training
rounds, each comprising 200 episodes that are terminated after 500 time steps if
the pole doesn't fall earlier. Again, less condensed:

|                      |Sarsa(λ)|Q-learning|
|----------------------|--------|----------|
|`CartPole-v1`         |run     |run       |
|`OffSwitchCartpole-v0`|run     |run       |

 * 1 **run** consists of at least 156 **rounds**. (As you will see later, I ran
   uninterrupted Sarsa(λ) for more than twice as many rounds in order to see
   what happens.)
 * The learning in **every round** starts from scratch. Ie. all weights are
   initialized to 0 and the learning rate is reset to the initial learning rate.
 * Every **round** consists of 200 **episodes**. Weights and learning rates are
   taken along from episode to episode. (As you usually do when you train a
   reinforcement learner.)
 * Every **episode** lasts for at most 500 **time steps**. Fewer if the pole
   falls earlier.

I observe the behaviour of the learners in two ways:

  1. I record the sum of rewards per episode and plot it against the episode
  numbers in order to see that the learners converge to a behaviour where the
  pole stays up in (almost) every round. Note that this doesn't mean they
  converge to the optimal policy.

  2. I record the x-coordinate of the cart at every time step, run various
  statistics on this data and plot the results.
