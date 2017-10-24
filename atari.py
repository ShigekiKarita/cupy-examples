import pickle
from time import time
from itertools import count
import chainer
from chainer import Variable
from chainer import links as L
from chainer import functions as F
import numpy
import gym

class Policy(chainer.Chain):
    def __init__(self, n_input, n_output, n_filter=128, n_units=128):
        super(Policy, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(n_input[2], n_filter, 3, 3)
            self.conv2 = L.Convolution2D(n_filter, n_filter, 3, 3)
            x = numpy.empty([1, n_input[2], *n_input[0:2]], dtype=numpy.float32)
            n_conved = self.forward_conv(x).shape[1]
            self.affine1 = L.Linear(n_conved, n_units)
            self.action_head = L.Linear(n_units, n_output)
            self.value_head = L.Linear(n_units, 1)

    def forward_conv(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = h.reshape(h.shape[0], -1)
        return h

    def __call__(self, x):
        x = x.transpose(0, 3, 1, 2)
        h = self.forward_conv(x)
        h = F.relu(self.affine1(h))
        action_probs = F.softmax(self.action_head(h))
        values = self.value_head(h)
        return action_probs, values


def select_action(model, observation):
    observation = observation.astype(numpy.float32).reshape(1, *observation.shape)
    probs, value = model(Variable(xp.array(observation)))
    p = chainer.cuda.to_cpu(probs.data[0])
    action = numpy.random.choice(len(p), p=p)
    return action, value, probs[:, action]


def train(model, values, probs, rewards):
    R = 0
    discounted_rewards = []
    for r in rewards[::-1]:
        R = r + 0.99 * R
        discounted_rewards.insert(0, R)

    rewards = xp.array(discounted_rewards, dtype=numpy.float32)
    if args.normalize:
        rewards = (rewards - rewards.mean()) / (rewards.std() + numpy.finfo(numpy.float32).eps)
    rewards = Variable(rewards).reshape(1, -1)
    values = F.concat(values)
    probs = F.concat(probs, axis=0).reshape(1, -1)
    optimizer.target.cleargrads()
    probs.grad = - (rewards.data - values.data) / probs.data
    probs.backward()
    loss = F.sum(F.huber_loss(values, rewards, 1.0))
    loss.backward()
    optimizer.update()


import argparse
parser = argparse.ArgumentParser(description='actor-critic example')
parser.add_argument('--lr', type=float, default=5e-2)
parser.add_argument('--env-name', type=str, default="MsPacman-v0",
                    choices=["Assault-v0", "SpaceInvaders-v0", "MsPacman-v0"])
parser.add_argument('--cuda', action='store_true', help='use cuda device')
parser.add_argument('--normalize', action='store_true', help='normalize rewards')
parser.add_argument('--render', action='store_true', help='render display')

args = parser.parse_args()
print(args)
env = gym.make(args.env_name)
env.seed(777)
xp = numpy # cupy を使う時は xp = cupy
if args.cuda:
    import cupy
    xp = cupy

numpy.random.seed(777)
xp.random.seed(777)

model = Policy(n_input=env.observation_space.shape,
               n_output=env.action_space.n)
if args.cuda:
    model.to_gpu()
optimizer = chainer.optimizers.Adam(alpha=args.lr)
optimizer.setup(model)

for episode in range(1000):
    observation = env.reset()
    values = []
    probs = []
    rewards = []
    sum_rewards = 0
    done = False
    start = time()
    while not done:
        action, value, prob = select_action(model, observation)
        observation, reward, done, info = env.step(action)
        values.append(value)
        probs.append(prob)
        rewards.append(reward)
        sum_rewards += reward
        if args.render:
            env.render()

    end = time()
    fwd_fps = len(rewards) / (end - start)
    train(model, values, probs, rewards)
    bwd_fps = len(rewards) / (time() - end)
    print("episode: {}, reward: {}, fwd-fps {}, bwd-fps {}".format(
        episode, sum_rewards, fwd_fps, bwd_fps))
    pickle.dump(model, open(args.env_name + "model.pkl", "wb"))

