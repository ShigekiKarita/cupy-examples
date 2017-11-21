# coding: utf-8
import pylab
f = open("pole.log", "r")
lines = f.readlines()
rewards = [float(l.split()[-1]) for l in lines]
def running_mean(x, N):
    cumsum = numpy.cumsum(numpy.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 
r10 = running_mean(rewards, 10)
pylab.plot(range(len(r10)), r10)
pylab.plot(range(len(r10)), [195 for _ in range(len(r10))])
pylab.savefig("pole.svg")
pylab.savefig("pole.pdf")
