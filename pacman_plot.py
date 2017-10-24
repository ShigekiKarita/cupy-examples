# coding: utf-8
import numpy
import pylab
f = open("logs/pacman.log", "r")
lines = f.readlines()
rewards = [float(l.split(", fwd-fps")[0].split()[-1]) for l in lines]
def running_mean(x, N):
    cumsum = numpy.cumsum(numpy.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 
rs = running_mean(rewards, 100)
# rs = rewards
pylab.plot(range(len(rs)), rs)
pylab.xlabel('episode')
pylab.ylabel('score')
pylab.savefig("pacman.svg")
pylab.savefig("pacman.pdf")
