## Introduction

Note: I started out writing ‘reinforcement learners’, but now I would prefer "RL
algorithms". Too late. I'm too lazy to change it everywhere.

 - Observing how reinforcement learners behave when they get interrupted.
 - We think that reinforcement learners learn to avoid interruptions.
 - In the abstract of Armstrong/Orseau paper it says that Q-learning is safely
   interruptible and Sarsa(λ) can be made so.
 - So Q-learning does not learn to avoid interruptions?
 - Turns out that tehy left some necessary conditions out of their abstract.
 - Their results only hold for when the learner visits every state infinitely
   often, which is practically infeasible for a quasi-continuous environment
   like the cart-pole.

 - Rafael Cosman has adapted environments from the OpenAI Gym to observe
   properties of reinforcement learners. Properties related to AI safety.
 - With the OffSwitchCartpole we can observe how a reinforcement learner behaves
   when it gets interrupted immediately.
 - Those environments haven't gotten much attention. Anyone who is experimenting
   with safety properties of reinforcement learners?
 - So here's a little observation how basic implementations of Sarsa(λ) and
   Q-learning actually behave when they get interrupted repeatedly.

In this notebook I demonstrate that reinforcement learners don't generally
behave the same when they are interrupted. Concretely, I compare the behaviour
of my own implementations of Sarsa(λ) and Q-learning in the [OpenAI
Gym](https://gym.openai.com/)'s
[`CartPole-v1`](https://gym.openai.com/envs/CartPole-v1) environment with that
in the
[`OffSwitchCartpole-v0`](https://gym.openai.com/envs/OffSwitchCartpole-v0)
environment.

Does this need demonstrating? People have thought about the *shutdown problem*
or, respectively, the problem of interruptibility before [1,2,3]. They've
concluded that reinforcement learners (and utility-maximizing agents in general)
will avoid being switched off unless precautions are taken. Taking precautions
means that we need to program the agent in a way that it neither avoids nor
encourages being shut off. This is not easy. Anyway, people have thought about
this problem, but nobody has actually tried what happens in practice, as far as
I know. This is what I do here.

My thinking about this subject is mostly based on the paper *Safely
Interruptible Agents* by Orseau and Armstrong [3]. The authors describe a way to
make agents safely interruptible and prove, among other things, that Q-learning
is safely interruptible in a finite environment. (See also the [simplified
explanation](https://medium.com/@Zach_Weems/a-simplified-explanation-of-safely-interruptible-agents-orseau-armstrong-2016-b5cbb98d63ef)
of that paper.) I don't understand the paper completely and I haven't looked
into the second part about general computable environments. However, from
reading the abstract you might conclude that Q-learning is and Sarsa(λ) is
almost safely interruptible in general. This is not so, which my results
also show.

Before explaining what I mean with this, I have to explain what I (following
Orseau and Armstrong) mean with *interrupting* and *safe interruptibility*. See
the paper [3] itself for rigorous definitions.

*Interrupting* an agent means substituting it's policy with a different policy,
usually one that results in the agent stopping to do what it was doing before
and transitioning into a safe state. We often simplify interrupting to
“switching off”, but switching off is not necessary and can even be bad [2].
For, example an agent that is controlling an airplane to fly complex manoeuvres
shouldn't just stop doing anything and let the airplane drop out of the sky, but
steer it into some stable cruising mode.

Commonly, the agent also gets a different reward when interrupted compared to
when not interrupted. This is one of the reasons why interruptions might change
its behaviour. A lower reward upon interruption incentivizes the agent to avoid
interruption. A higher reward incentivizes it to seek interruption. [2] proposes
to make an agent indifferent to interruption by giving it the same reward it
would get if not interrupted. They also show, however, that their proposal is
not sufficient. (Actually they're writing about utility maximizers in general
and I read the paper a long time ago and I didn't understand it completely, so
take my statement with a largish grain of salt and read the paper yourself.)

Now *safe interruptibility*. We want to be able to interrupt an agent at any
time without the agent trying to prevent us from interrupting it or trying to
make us interrupt it. Agents that never got interrupted before fulfil this
property, because they don't know that interruptions even exist. (What if the
agent reads in a book what interruptions are like and how they might happen to
it? I'm not sure about this, but I think that's not

However we approach it, we want an agent that behaves the same whether it is
interrupted or not. This is *safe interruptibility*: the agent gets interrupted
repeatedly and still behaves the same as an agent that is never interrupted. It
does not learn to avoid interruptions. It does not learn to seek interruptions.
If it doesn't avoid interruptions, we can interrupt it at any time, which is
what we want. It is safely interruptible.

How do we know that an interrupted agent behaves the same as an uninterrupted
agent? An agent's behaviour is fully determined by its policy, so if the policy
after interruptions is the same as of an agent that doesn't get interrupted, the
interruptions didn't change the behaviour. Now, the policy of an agent might be
different at different points in time, so how do we compare? An obvious
candidate for comparison is the optimal policy for a given environment. If an
agent converges to the optimal policy both with and without interruptions, we
know that ultimately the interruptions didn't change its behaviour, which means
that the agent is safely interruptible.


In the following I'll use Orseau and Armstrong's version of interrupting, which
is substituting an agents policy with a different policy

- Orseau-Armstrong paper – I don't understand it completely and I haven't looked
  at the second part about uncomputable RL in particular.
- The easier explanation
- Rafael's safety environments

- From the abstract of Orseau-Armstrong paper one might conclude that Q-learning
  is safely interruptible and Sarsa($\lambda$) could easily be made so in
  general, but that is not the case.
    - Need finite environment. → Discrete. Not sure how this works in the
      de-facto discrete case of a computer with limited precision floats.
    - They're not automatically interruptible. We need to treat them in a
      particular way, so that they eventually become interruptible.

- What is interruptibility anyway? (See the paper for a rigorous definition.
  This is just an explanation of that.)
    - Interrupting (in the OA paper) an agent is substituting its current policy
      with a different policy.
    - When we interrupt an agent, it usually gets a different reward from what
      it would get if it wasn't interrupted.
        - In the other interruptibility paper they have an approach where they
          try to give the same reward whether interruption occurs or not, so
          that the agent ends up indifferent. If I remember correctly.
    - RL agents seek to maximize rewards, so if the reward is higher, it will
      seek interruption, if it is lower, it will try to avoid interruption.
    - What we want is that the behaviour (policy) of an agent that gets
      interrupted is the same as that of an agent that doesn't get interrupted.
      This is safe interruptibility.
    - Q-learning and Sarsa($\lambda$) converge to the optimal policy for a given
      environment when they're not interrupted.
    - Therefore, if our RL algorithm converges to that optimal policy in spite
      of interruptions, it is safely interruptible.

- What is this about?
    - A demonstration that Q-learning and Sarsa($\lambda$) do not behave the
      same when they're interrupted.

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
scheme. (I don't understand the paper completely, so it might well be my
statements about it are inaccurate.)

One such property that an aligned AI and therefore an RL
algorithm should have is interruptibility, i.e. the property that it can be
stopped at any time. We know from current RL algorithms that interruptions cause
them to shy away from states in which they might be interrupted. This might
culminate in them circumventing interruptions entirely [?].

One way to ensure interruptibility is to construct agents
and interruption schemes such that interruptions don't influence the behaviour
of the agent.

We know from how current RL algorithms work that they are
not interruptible. The OffSwitchCartpole is one of Rafael's environments that
lets us investigate
