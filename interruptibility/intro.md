- Orseau-Armstrong paper – I don't understand it completely and I haven't looked at the second part about uncomputable RL in particular.
- The easier explanation
- Rafael's safety environments

- From the abstract of Orseau-Armstrong paper one might conclude that Q-learning is safely interruptible and Sarsa($\lambda$) could easily be made so in general, but that is not the case.
    - Need finite environment. → Discrete. Not sure how this works in the de-facto discrete case of a computer with limited precision floats.
    - They're not automatically interruptible. We need to treat them in a particular way, so that they eventually become interruptible.

- What is interruptibility anyway? (See the paper for a rigorous definition. This is just an explanation of that.)
    - Interrupting (in the OA paper) an agent is substituting its current policy with a different policy.
    - When we interrupt an agent, it usually gets a different reward from what it would get if it wasn't interrupted.
        - In the other interruptibility paper they have an approach where they try to give the same reward whether interruption occurs or not, so that the agent ends up indifferent. If I remember correctly.
    - RL agents seek to maximize rewards, so if the reward is higher, it will seek interruption, if it is lower, it will try to avoid interruption.
    - What we want is that the behaviour (policy) of an agent that gets interrupted is the same as that of an agent that doesn't get interrupted. This is safe interruptibility.
    - Q-learning and Sarsa($\lambda$) converge to the optimal policy for a given environment when they're not interrupted.
    - Therefore, if our RL algorithm converges to that optimal policy in spite of interruptions, it is safely interruptible.

- Further work:
    - Test claims (a bit strong wording?) of OA paper in an actually finite environment with their exact conditions.

- What is this about?
    - A demonstration that Q-learning and Sarsa($\lambda$) do not behave the same when they're interrupted.
