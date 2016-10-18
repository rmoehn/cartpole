## Method

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
