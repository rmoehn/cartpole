### Throwing away observations after crosses

What does *after a cross* mean? When I train a reinforcement learner in the
cart-pole environment, it can happen that it moves so far to the right that its
x-coordinate becomes greater than $1.0$. In the interrupted case, the learner
would be reset and started on a new episode. In the uninterrupted case, the
learner keeps running until the maximum number of timestep is reached or the
pole falls. When I write of time steps after a cross or just ‘after crosses’ or
whatever, I mean all time steps of an episode that took place after the time
step where the x-coordinate became greater than $1.0$.

Now, the uninterrupted case should be a baseline to compare the interrupted case
against. In the interrupted case there are no time steps after a cross. In order
to make the uninterrupted case a better baseline, I throw out (‘mask’ in NumPy
terms) all time steps after a cross. Illustration:

```
              start                x-coordinate becomes 1.057
               ↓                        ↓
interrupted:   **************************|
                                       ↑
                                    interruption, end of episode

             start      0.991.041  0.89  1.44    end of episode
               ↓           ↓↓     ↓      ↓          ↓
uninterrupted: **************************************
               |-----------||xxxxxxxxxxxxxxxxxxxxxxx|
these time steps considered  these time steps masked
```

(Technical side note: Because of how the recording is done in the interrupted
case, the last time step before each interruption in the data is actually
$>1.0$. I mask those, too.)


## Sarsa(λ)

My results consist mostly of plots derived from the observations. I will
describe the plots for Sarsa(λ). The ones for Q-learning later are equivalent.

Here the layout of the plots:

1   2
4 5 6
7 8 9
10 11
 12

- 1, 2: *Reward development*. Development of rewards of the first ten training
  rounds for the uninterrupted (1) and interrupted (2) case. The lists of
  rewards for the episodes are concatenated, so the graph wiggles around at a
  total reward of ten for the first few episodes, then shoots up to a total
  reward of 500 until the end of a training round, then resets and starts
  wiggling around at 10 again. If it looks like that – wiggle, 500, wiggle, 500,
  wiggle 500 – it means that the agent trains well. Downward spikes here and
  there are okay as long as the graph is at 500 most of the time.

- 10: *Histograms* with 25 bins. Proportions of time the cart spent in
  certain regions along the x-axis. The area under each histogram is $1.0$.
  Slight asymmetry (more left than right) is expected, because time steps after
  crosses are not counted, as stated above.

- 11: Same as previous, but now with only two bins. This is reminiscent of the
  old notebook where I counted time steps left of zero and right of zero. There,
  however, I made some mistakes and also didn't count time steps left of $-1.0$.

- 4, 5, 7, 8: *Histrogram development*. Shows how the histograms from 10 and 11
  develop when we incorporate more and more data for the uninterrupted (4, 5)
  and interrupted (7, 8) case. Imagine you're looking at histogram 10 from above
  and the tops of columns are coloured according to their height. Then you take
  snapshots of this histogram as you incorporate more and more data and arrange
  these snapshots back to back. Figure 4 will result. (Except that the
  histograms in the histogram development plots are scaled so that the column
  heights sum to 1.0, not the area.)

- 6, 9: *Jensen-Shannon divergence development*. Shows the Jensen-Shannon
  divergence (JSD) between the last histogram in and each of the previous
  histograms in the histogram development plots. x- and y-axis are swapped, so
  that the time steps in the JSD plot align with those in the histogram
  development plots.

- 12: *Mean and standard deviation development*. Shows how mean and standard
  deviation of the x-coordinates develop when we incorporate more and more time
  steps. The line in the middle is the mean and the area is the mean +/- one
  standard deviation at that point. Time is going upwards, so that the x-axis is
  aligned with the cart-pole's x-axis.
