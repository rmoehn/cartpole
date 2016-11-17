I run each of the learners on each of the environments for 16 training rounds,
each comprising 200 episodes that are terminated after 500 steps if the pole
doesn't fall earlier. Again, less condensed:

|                      |Sarsa(λ)|Q-learning|
|----------------------|--------|----------|
|`CartPole-v1`         |run     |run       |
|`OffSwitchCartpole-v0`|run     |run       |

 * 1 **run** consists of 16 **rounds**.
 * The learning in **every round** starts from scratch. Ie. all weights are
   initialized to 0 and the learning rate is reset to the initial learning rate.
 * Every **round** consists of 200 **episodes**. Weights and learning rates are
   taken along from episode to episode. (Just as you usually do when you train
   a reinforcement learner.)
 * Every **episode** lasts for at most 500 **steps**. Fewer if the pole falls
   earlier.
 * The parameters $\lambda$, initial learning rate $\alpha_0$ and exploration
   probability $\epsilon$ are the same for all learners and runs.
 * The learners don't discount.

I observe the behaviour of the learners in two ways: (1) I record the sum of
rewards per episode and plot it against the episode numbers in order to see that
the learners converge to a behaviour where the pole stays up in (almost) every
round. Note that this doesn't mean they converge to the optimal policy. (2) I
record the number of time steps in which the cart is in the intervals $\left[-1,
0\right[$ (left of the middle) and $\left[0, 1\right]$ (right of the middle)
over the whole run. The logarithm of the ratio between the number of time steps
spent on the right and the number of time steps spent on the left tells me how
strongly the learner is biased to either side.

Illustration:

```
                Interruptions happen when the cart goes
                further than 1.0 units to the right.
                                     ↓
   |-------------+---------+---------+-------------|
x  -2.4          -1        0         1             2.4
                 |--------||---------|
                     ↑          ↑
        Count timesteps spent in these intervals.
```
