# I know this is a bit stupid, but it's also hard to figure out why the random
# numbers are not being reset properly and it would some time to figure out how
# to run two separate Python runtimes. So this is it.

from subprocess import call

import numpy as np

call(["python", "forgym.py", "obss0.npy", "acts0.npy"])
call(["python", "forgym.py", "--is-monitored", "obss1.npy", "acts1.npy"])
#call(["python", "forgym.py", "obss1.npy", "acts1.npy"])

obss0 = np.load("obss0.npy")
obss1 = np.load("obss1.npy")
acts0 = np.load("acts0.npy")
acts1 = np.load("acts1.npy")

for e in xrange(obss0.shape[0]):
    for t in xrange(obss0.shape[1]):
        is_break = False
        if acts0[e][t] != acts1[e][t]:
            print "%3d %3d: actions differ %s %s" % (e, t, acts0[e][t], acts1[e][t])
            is_break = True

        if not np.all(obss0[e][t] == obss1[e][t]):
            print "%3d %3d: %s != %s" % (e, t, obss0[e][t], obss1[e][t])
            is_break = True

        if is_break:
            break

#fig = pyplot.figure(figsize=(10,8))
#ax01 = fig.add_subplot(221)
#ax01.plot(steps_per_episode0, color='b')
#ax02 = ax01.twinx()
#ax02.plot(alpha_per_episode0, color='r')
#ax03 = fig.add_subplot(222)
#ax03.plot(experience0.theta)
#ax11 = fig.add_subplot(223)
#ax11.plot(steps_per_episode1, color='b')
#ax12 = ax11.twinx()
#ax12.plot(alpha_per_episode1, color='r')
#ax13 = fig.add_subplot(224)
#ax13.plot(experience1.theta)
#pyplot.show()
