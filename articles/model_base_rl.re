== Model-Based RL
前回までは、MDPにおける状態遷移について、何の仮定も置かない方法を扱ってきた。
例えば、TD誤差を用いる方法では、状態遷移確率はサンプリングによって近似していたし、
方策勾配法では状態遷移確率は扱わなくても良いものであった。
今回は、状態遷移確率をモデルとして陽に扱う手法を扱う。
これをモデルベース強化学習と呼ぶ。

さて、強化学習全般の大まかな流れは以下のようになっていると言える。

1. 適当な方策を実行し、サンプルを集める
2. モデルを更新する
3. 方策を更新する(方策の更新、最大のQ値を取る@<m>$a$の選択)

モデルベース強化学習では、上記2.については教師あり学習で行う。
そして、3.の部分で状態遷移確率を使って方策を更新する。

そのため、モデルベース強化学習の利点として

* サンプル数が少なくても学習できる
* 学習したモデルを他のタスクに転移して利用することができる。つまり汎用的な学習ができる。

ということが挙げられる。

=== 基本的なアルゴリズム
現在の状態@<m>$s_t$と行動@<m>$a_t$を取って、次の状態@<m>$s_{t+1}$を生成する関数を@<m>$f_{\phi}(s_t, a_t)$とする。
以下にモデルベース学習で用いられるアルゴリズムの概要を示す。
以下では、学習したいタスクの一連の系列を一イテレーションとして、そのindexを@<m>$k$としている。

1. サンプル@<m>$\mathcal{D}=\left\{(s, a, s')_i \right\{$を集めるために、何らかの方策@<m>$\pi_0(a_t|s_t)$(例えばランダム)を実行する
1. @<m>$for$ @<m>$k=1,2,\dots$ @<m>$do$
1. @<m>$f_{\phi}(s_t, a_t)$ を学習する (@<m>$\sum_i \| f_{\phi}(s_t, a_t)-s'_i\|^2$ を最小化)
1. @<m>$f_{\phi}(s_t, a_t)$ を使って@<m>$\pi_{\theta}(a_t | s_t)$を最適化する
1. @<m>$\pi_{\theta}(a_t | s_t)$実行して、タプル@<m>$(s, a, s')$を@<m>$\mathcal{D}$に記録する
1. @<m>$end$ @<m>$for$

=== Guided Policy Search
ただし、以上のように、全イテレーションにおいて、共通の@<m>$f_{\phi}(s_t, a_t)$を学習する方法では

* すべての状態において収束する性質の良いモデルを学習しなければならない
* モデルが未学習のため、状態を楽観的に評価してしまうなどして、誤った方向に方策を学習してしまう

などの問題がある。
そこで、時変の関数として@<m>$f_{\phi}(s_t, a_t)$を学習するようにする。
そのようなモデルをlocal modelと呼ぶ。

また、@<m>$\pi_{\theta}(a_t | s_t)$を直接算出することはせず、別の扱いやすい方策を計算し、それに@<m>$\pi_{\theta}(a_t | s_t)$をフィッティングすることで任意の方策を学習できるようにする方法をGuided policy searchと呼ぶ。
例えば、簡素な方策として線形ガウスモデル化を使い、より複雑な方策のモデルとしてニューラルネットワークを使うということがある。

=== 未知のダイナミクスを用いた guided policy search
Levineら[^1]はダイナミクス@<m>$f_{\phi}(s_t, a_t)$の学習と、guided policy search を組み合わせて、少ないサンプル数でもペグの穴への挿入や、歩行などの複雑なタスクをシミュレーション上で実現した。

[^1]: S.Levine et.al "Learning Neural Network Policies with Guided Policy Search under Unknown Dynamics" https://people.eecs.berkeley.edu/~svlevine/papers/mfcgps.pdf

さらに前イテレーションでの方策と現在の方策の間のKLダイバージェンスによる制約を適応的に変化させることで更にサンプル数を減少させ、実機でのタスクも実現した。[^2]

[^2]: S.Levine et.al "Learning Contact-Rich Manipulation Skills with Guided Policy Search" https://arxiv.org/pdf/1501.05611.pdf

guided policy searchと組み合わせたモデルベース学習の概略を以下に示す。

1. サンプル@<m>$\mathcal{D}=\left\{ (s, a, s' )_i \right\{$を集めるために、何らかの方策@<m>$\pi_0(a_t|s_t)$(例えばランダム)を実行する
1. @<m>$for$ @<m>$k=1,2,\dots$ @<m>$do$
1. local model @<m>$f_{\phi}(s_t, a_t)$ をサンプルにフィッティングする
1. サンプル@<m>$\mathcal{D}$を使って、任意の方策をguided policy searchする
1. local model @<m>$f_{\phi}(s_t, a_t)$ を使ってlocalな方策@<m>$p(a_t | s_t)$を最適化する (KLダイバージェンスで制約)
1. @<m>$\pi_{\theta}(a_t | s_t)$から生成される一連の行動を実行して、タプル@<m>$(s, a, s')$を@<m>$\mathcal{D}$に記録する
1. @<m>$end$ @<m>$for$

=== ダイナミクスのフィッティング
アルゴリズムのstep 3. では、仮定したダイナミクスのモデルをサンプルにフィッティングする。
モデルとしては、線形ガウスモデルや、ガウス混合モデル（Gaussian mixture model, GMM) や、ニューラルネットワークを用いることができる。

ロボットの関節が多くモデルが複雑で、高次元のシステムの場合、時変のモデルをフィッティングさせるために非常に長い訓練時間が必要になってしまう。
しかし、時間ステップ的に近いサンプルや、前イテレーションのサンプルも現在のイテレーションのある時間と強く関連していると想定できるならば、サンプル数をかさ増しし、訓練時間を短縮させることができる。
Levineら[^2]は方策更新ステップでのKLダイバージェンスの制約をコストに応じて適応的に変化させることで、現在時刻の周辺のサンプルと前イテレーションの同一時刻のサンプルを利用できるようにし、学習のイテレーション数を抑えている。

学習時に扱う方策@<m>$p(a_t | s_t)$とは別に実際に実行する方策として任意の関数@<m>$\pi_{\theta}(a_t | s_t)$(例えばニューラルネットワーク)を用意し、フィッティングするようにパラメタを調整する方法がguided policy searchであった。

方策を近似する仮定を入れることで、もとの価値最適化問題は

//texequation{
\underset{\theta, p(\tau)}{\text{minimize }}
E_{p(\tau)} [ l(\tau) ]
\text{ s.t. } D_{KL}(p(s_t) \pi_{\theta}(a_t | s_t) || p(a_t, s_t)) = 0
//}

を解く問題となる。
ここで、@<m>$\tau=(s_0, a_0, \dots, s_H, a_H)$であり、@<m>$l(\tau)$は軌道におけるコストを表す。(ここでは報酬の最大化ではなくコスト最小化問題として扱っているが、報酬の最大化と同一の問題である)
また、@<m>$p(\tau)$は@<m>$s, a$で表現される軌道における存在確率を表す。

これをラグランジュの未定乗数法で変形すると、以下の@<m>$\mathcal{L} (\theta, p, \lambda)_{GPS}$の最小化問題となる。
//texequation{
\mathcal{L} (\theta, p, \lambda)_{GPS}
= E_{p(\tau)} [ l(\tau) ]
+ \sum_{t=0}^T \lambda_t D_{KL}(p(s_t) \pi_{\theta}(a_t | s_t) || p(a_t, s_t))
//}

アルゴリズムのstep 4. では収集したサンプル@<m>$\mathcal{D}=\left\{ (s, a, s' )_i \right\{$を用いて、各時間ステップ@<m>$t$について以下を最小化する
//texequation{
\sum_{t=0}^T \lambda_t D_{KL}(p(s_t) \pi_{\theta}(a_t | s_t) || p(a_t, s_t))
//}

=== 方策の更新
状態遷移確率のモデルを
//texequation{
\begin{aligned}
p(s_{t+1} | x_t, a_t) =
\mathcal{N} (f(s_t, a_t), F_t) \\
f(s_t, a_t ) = \frac{df}{d s_t} s_t + \frac{df}{d a_t} a_t
\end{aligned}
//}

と仮定し、
アルゴリズムのstep 5. ではiLQR(iterative Linear Quadratic Regulator)を使って最適な方策を求める。
