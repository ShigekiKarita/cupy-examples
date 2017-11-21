# cupy-interface

## cupy/chainerインストール方法

``` console
conda install anaconda
pip install cupy chainer "gym[atari]"
```

確認した version は以下の通りです。
+ cupy==2.0.0
+ chainer==3.0.0
+ gym==0.9.4

## cupyの例

中央値の計算をします

``` console
python median.py # 何もエラーがでなければOKです
```

## chainerの例(pole)

強化学習の入門問題であるポールと滑車の制御をします。方策勾配実装の簡単なテスト用

``` console
python pole.py | tee plot.log # スコアが200に到達すればOK
python plot_pole.py #　学習グラフ pole.pdf ができます
```

## chainerの例(atari)

実際にPacman用のゲームAIを学習してみます。
``` console
python atari.py |& tee pacman.log # スコアが700に到達すればOK
python plot_pacman.py # 学習グラフ pacman.pdf ができます
```


質問・不具合報告などは shigekikarita@gmail.com または twitter: @kari_tech まで

