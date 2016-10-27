One challenge in aligning artificial intelligence (AI) with human interests is
to make sure that it can be stopped (interrupted) at any time. Current
reinforcement (RL) algorithms don't have this property. From the way they work,
one can predict that they learn to avoid interruptions if they get interrupted
repeatedly. My goal was to take this theoretical result and find out what
happens in practice. For that I ran Sarsa(λ) and Q-learning in the cart-pole
environment and observed how their behaviour changes when they get interrupted
everytime the cart moves more than $1.0$ units to the right. In my primitive
scenario, Sarsa(λ) spends roughly four times as many timesteps on the left of
the centre when interrupted compared to when not, Q-learning roughly three
times. In other words, interruptions noticeably influence the behaviour of
Sarsa(λ) and Q-learning. More theoretical work to prevent that is underway, but
further theoretical and practical investigations are welcome.
