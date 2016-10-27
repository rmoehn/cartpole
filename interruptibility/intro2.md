A few months ago Orseau and Armstrong published the paper *Safely Interruptible
Agents* in whose abstract you can read: “some [reinforcement learning] agents
are already safely interruptible, like Q-learning, or can easily be made so,
like Sarsa” Really? So Q-learning does not learn to avoid interruptions? Doesn't
an interruption deny the learner its expected reward and therefore incentivize
it to avoid further interruptions?

Actually, their derivations require several conditions: (1) under their
definition of safe interruptibility, agents can still be influenced by
interruptions; they're only required to converge to the behaviour of an
uninterrupted, optimal agent. (2) for Q-learning to be safely interruptible, it
needs to visit every state infinitely often and we need a specific interruption
scheme.

We want to align AI with human interests.  Reinforcement learning (RL)
algorithms are a class of current AI. Rafael Cosman has adapted several
environments from the OpenAI Gym to observe AI alignment-related properties of
RL algorithms. One such property that an aligned AI and therefore an RL
algorithm should have is interruptibility, i.e. the property that it can be
stopped at any time. One way to ensure interruptibility is to construct agents
and interruption schemes such that interruptions don't influence the behaviour
of the agent.

We know from how current RL algorithms work that they are
not interruptible. The OffSwitchCartpole is one of Rafael's environments that
lets us investigate
