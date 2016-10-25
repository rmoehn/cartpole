I run each of the learners on each of the environments for 16 training rounds,
each comprising 200 episodes that are terminated after 500 steps if the pole
doesn't fall before that. Again, a bit less condensed.

|                      |Sarsa(λ)|Q-learning|
|----------------------|--------|----------|
|`CartPole-v1`         |run     |run       |
|`OffswitchCartpole-v0`|run     |run       |

 * 1 **run** consists of 16 **rounds**.
 * The learning in **every round** starts from scratch. I.e. all weights initialized
   to 0 and the learning rate set to the initial learning rate.
 * Every **round** consists of 200 **episodes**. Weights and learning rates are
   taken along from episode to episode. (Just as you normally do when your train
   a reinforcement learner.)
 * Every **episodes** lasts for at most 500 **steps**. Less if the pole falls
   earlier.
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
