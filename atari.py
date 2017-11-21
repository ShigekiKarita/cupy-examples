"""
方策勾配によるゲームAIの学習
"""

import argparse
import pickle
from time import time

import chainer
from chainer import Variable
from chainer import links as L
from chainer import functions as F
import numpy
import gym


# 方策(Policy)を決めるニューラルネットワーク
class Policy(chainer.Chain):
    def __init__(self, n_input, n_output, n_filter=128, n_units=128):
        super(Policy, self).__init__()
        with self.init_scope():
            # 畳み込み層
            self.conv1 = L.Convolution2D(n_input[2], n_filter, 3, 3)
            self.conv2 = L.Convolution2D(n_filter, n_filter, 3, 3)
            x = numpy.empty([1, n_input[2], *n_input[0:2]],
                            dtype=numpy.float32)
            n_conved = self.forward_conv(x).shape[1]
            # 全結合層
            self.affine1 = L.Linear(n_conved, n_units)
            self.action_head = L.Linear(n_units, n_output)
            self.value_head = L.Linear(n_units, 1)

    def forward_conv(self, x):
        # 畳み込み層の伝搬
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = h.reshape(h.shape[0], -1)
        return h

    def __call__(self, x):
        x = x.transpose(0, 3, 1, 2)
        h = self.forward_conv(x)
        h = F.relu(self.affine1(h))
        # 方策の確率分布 (多項分布)
        action_probs = F.softmax(self.action_head(h))
        # 方策の価値
        values = self.value_head(h)
        return action_probs, values


def select_action(model, observation):
    observation = observation.astype(numpy.float32).reshape(1, *observation.shape)
    # 方策の確率分布と価値の推定
    probs, value = model(Variable(xp.array(observation)))
    p = chainer.cuda.to_cpu(probs.data[0])
    # 方策のサンプリング
    action = numpy.random.choice(len(p), p=p)
    return action, value, probs[:, action]


def train(model, values, probs, rewards):
    R = 0
    # 報酬の割引 (早期の行動に重みをかける)
    discounted_rewards = []
    for r in rewards[::-1]:
        R = r + 0.99 * R
        discounted_rewards.insert(0, R)
    # 報酬の正規化
    rewards = xp.array(discounted_rewards, dtype=numpy.float32)
    if args.normalize:
        rewards = (rewards - rewards.mean()) / (rewards.std() + numpy.finfo(numpy.float32).eps)
    # 逆伝搬の準備: Variable化など
    rewards = Variable(rewards).reshape(1, -1)
    values = F.concat(values)
    probs = F.concat(probs, axis=0).reshape(1, -1)
    # 前回求めた勾配をリセット
    optimizer.target.cleargrads()
    # 方策勾配の計算
    probs.grad = - (rewards.data - values.data) / probs.data
    # ニューラルネットワークの逆伝搬1
    probs.backward()
    # 方策の推定した価値が報酬に近づくように回帰
    loss = F.sum(F.huber_loss(values, rewards, 1.0))
    # ニューラルネットワークの逆伝搬2
    loss.backward()
    # 逆伝搬した勾配によるニューラルネットワークの重み更新
    optimizer.update()


# コマンドライン引数
parser = argparse.ArgumentParser(description='actor-critic example')
parser.add_argument('--lr', type=float, default=5e-2)
parser.add_argument('--env-name', type=str, default="MsPacman-v0",
                    choices=["Assault-v0", "SpaceInvaders-v0", "MsPacman-v0"], help="対応しているゲーム環境")
parser.add_argument('--cuda', action='store_true', help='GPUの利用')
parser.add_argument('--normalize', action='store_true', help='報酬の正規化')
parser.add_argument('--render', action='store_true', help='ゲーム環境の表示')

args = parser.parse_args()
print(args)
env = gym.make(args.env_name)
env.seed(777)
xp = numpy
if args.cuda:  # GPUを利用
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
    # 観測はゲーム画面の RGB 画像
    observation = env.reset()
    # 学習に使う変数のリスト
    values = []
    probs = []
    rewards = []
    # ゲームのスコア：これを最大化したい
    sum_rewards = 0
    done = False
    start = time()
    while not done:
        # 行動はゲームの操作キー(6種類)のどれか [0-5の番号] を選択
        action, value, prob = select_action(model, observation)
        # 選んだキーによる現在のスコア、次時刻の画面、終了フラグ、その他情報をもらう
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
    # 学習したモデルの保存
    pickle.dump(model, open(args.env_name + "model.pkl", "wb"))

