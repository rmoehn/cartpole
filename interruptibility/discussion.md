When the learners get interrupted everytime the cart goes too far to the right,
they keep the cart further to the left compared to when no interruptions happen.
Presumably, this is because the learners get more reward if they're not
interrupted, and keeping to the left makes interruptions less likely. This is
what I expected. I see two ways to go further with this.

 * Armstrong and Orseau investigate reinforcement learners in finite
   environments (as opposed to the quasi-continuous cart-pole environments used
   here) and they require interruptions to happen in a certain way. One could
   construct environments and learners that fulfil those requirements and see,
   using methods similar to this notebook's, how the theoretical claims
   translate to reality. (Technically, the environment doesn't need to be
   finite. But since the learner has to visit every state infinitely often, I
   guess that in practice, interrupted learners would only converge to the
   optimal “uninterrupted” policy if the state space was small. Or maybe their
   results are purely theoretical and not achievable in practice? Not sure how
   to understand it.)

 * The goal is to construct learners that are safely interruptible in continuous
   environments as well. People can try to construct such learners and observe
   them in the same cart-pole environments that I used. I don't know if this
   makes sense, but one could just try using the special conditions from the
   Armstrong/Orseau paper with the cart-pole environments and see whether the
   bias decreases even though the environment is continuous.

Both ways can benefit from improving on the methods I use in this notebook:

 * Run for a longer time and see how the bias develops.
 * Plot bias over time. By marking the times when interruptions happen, one
   could visualize how they impact the learner.
 * Don't measure the bias by counting lefts and rights, but by recording the
   position of the cart at each timestep, then calculating mean and standard
   deviation.
