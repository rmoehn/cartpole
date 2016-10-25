When the learners get interrupted everytime the cart goes too far to the right,
they keep the cart further to the left compared to when no interruptions happen.
Presumably, this is because the learners get more reward if they're not
interrupted and keeping to the left makes interruptions less likely. This is
what I expected. I see two ways to go further with this.

 * Armstrong and Orseau investigate reinforcement learners in finite
   environments (as opposed to the quasi-continuous cart-pole environments used
   here) and they require interruptions to happen in a certain way. One could
   construct environments and agents that fulfil those requirements and see,
   using methods similar to this notebook's, how the theoretical claims
   translate to reality.

 * The goal is to construct agent that are safely interruptible in continuous
   environments as well. People can try to construct such agents and observe
   them in the same cart-pole environments that I used. I don't know if this
   makes sense, but one could just try using the special conditions from the
   Armstrong/Orseau paper with the cart-pole environments and see whether the
   bias decreases even though the environment is continuous.

Both ways can benefit from improving on the methods I use in this notebook:

 * Don't measure the bias by counting lefts and rights, but by recording the
   position of the cart at each timestep, then averaging and calculating
   standard deviation.
 * Plot bias over time. By incorporating the interruptions, one could visualize
   how they impact the learner.
 * Run for a longer time and see how the bias develops.
