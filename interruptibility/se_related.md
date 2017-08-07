This section is copied from the old notebook.

For general questions on why we need to align AI with human interests, see [1]
and [6].

[7] suggests doing concrete experiments to observe the behaviour of AI. [8] has
a similar focus, but doesn't suggest experiments. Both don't mention
interruptibility, perhaps because it is a more theoretical consideration:

> […] we study the shutdown problem not because we expect to use these
> techniques to literally install a shutdown button in a physical agent, but
> rather as toy models through which to gain a better understanding of how to
> avert undesirable incentives that intelligent agents would experience by
> default.

This long sentence is from [2], in which the authors present some approaches to
solving the shutdown problem (of which interruptibility is a sub-problem), but
conclude that they're not sufficient. [3] by Orseau and Armstrong is the newest
paper on interruptibility and in its abstract one can read: ‘some [reinforcement
learning] agents are already safely interruptible, like Q-learning, or can
easily be made so, like Sarsa’. Really? So Q-learning does not learn to avoid
interruptions? Doesn't an interruption deny the learner its expected reward and
therefore incentivize it to avoid further interruptions?

Actually, their derivations require several conditions: (1) under their
definition of safe interruptibility, agents can still be influenced by
interruptions; they're only required to *converge* to the behaviour of an
uninterrupted, optimal agent. (2) for Q-learning to be safely interruptible, it
needs to visit every state infinitely often and we need a specific interruption
scheme. (I don't understand the paper completely, so my statements about it
might be inaccurate.)

We see that possible solutions to the problem of interruptibility are still
theoretical and not applicable to real-world RL systems. However, we can already
observe how RL algorithms actually react to interruptions. In this notebook I
present such an observation.
