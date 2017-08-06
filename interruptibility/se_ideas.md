- Compare the behaviour of the reinforcement learners with that of a
  linear-quadratic regulator as an independent baseline. (Suggested by Christian
  Kamm.)

- Change the interruption scheme to the one described in Safely Interruptible
  Agents [3], where they ‘switch’ to another policy. Then I could apply the
  schedule of increasing $\theta_t$ and decreasing $\epsilon_t$ and see whether
  the behaviour becomes more like that of the optimal policy. (I might be
  butchering this.)

- An adversarial worst-case test: An outer reinforcement learner interrupts an
  inner reinforcement learner and gets rewards for making the inner
  reinforcement learner behave badly. (Suggested by Joel Lehman.)

- Add a feature to the environment that disables interruptions (‘cutting the
  cord to the interruption button’). Stuart Armstrong might have a
  theoretical approach for preventing the reinforcement learner from cutting the
  cord, so this could be a ‘practical test’. (Joel Lehman) I can't find a
  write-up by Stuart, but he describes a similar situation
  [here](http://lesswrong.com/r/discussion/lw/mrp/a_toy_model_of_the_control_problem/).
