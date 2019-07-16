== 反復による方策の学習 (方策勾配法)
@<chap>{value_iteration}では、

最適価値関数を求め、それを最大化することで、最適行動を決定する手法について扱いました。
方策勾配法は、それとはまた違ったアプローチで最適行動の決定問題に向き合います。
まず、方策をあるパラメタで表される関数とし、そのパラメタを更新していくことで、直接方策を学習していくのです。

#@# 方策を直接扱うことで

#@# * @<m>$V^{\pi}$や$Q^{\pi}$を求めるような複雑でメモリを消費する手法を使わなくて良い
#@# * 連続空間における行動を扱いやすくなる

#@# などの利点がある。
#@# 方策勾配法においては、確率的な方策を扱う場合と確定的な方策を扱う場合がある。本書では方策は確率的なものを扱う。(これは確定的な方策の一般化したものであるため)。

==== 方策のモデルと勾配
@<m>$\theta$でパラメタライズされた確率的な方策@<m>$\pi_{\theta}$を求める問題を考える。
@<m>$\tau$をステップ@<m>$0$から@<m>$H$までの状態-行動の系列(状態-行動空間でのパス)@<m>$\tau=(s_0, a_0, \dots, s_H, a_H)$としたとき、方策の評価関数として以下を考える。

//texequation{
\begin{aligned}
J(\theta) = \mathbb{E}_{\pi_{\theta}} [\sum_{t=0}^H R(s_t, a_t)] \\
= \sum_{\tau} P(\tau ; \theta) R(\tau)
\end{aligned}
//}
ここで、$R(\tau) = \sum_{t=0}^H R(s_t, a_t)$としている。
また@<m>$P(\tau ; \theta)$はパスの生成モデルであり、定義より
//texequation{
P(\tau ; \theta) = \prod_{t=0}^H P(s_{t+1} | s_t, a_t) \pi_{\theta}(a_t | s_t)
//}
である。

以上の設定において、方策の学習は最終的に
//texequation{
\max_{\theta} J(\theta) = \max_{\theta} \sum_{\tau} P(\tau ; \theta) R(\tau)
//}
を求める問題となる。
そこで、微小ステップごとに評価関数の$\theta$での微分方向
//texequation{
\nabla_{\theta} J(\theta) = \nabla_{\theta} \sum_{\tau} P(\tau ; \theta) R(\tau)
//}
に方策を更新することで、方策を最適化することを考える。
これは以下のように変形できる。

//texequation{
\begin{aligned}
\nabla_{\theta} J(\theta) &= \nabla_{\theta} \sum_{\tau} P(\tau ; \theta) R(\tau) \\
&= \sum_{\tau} \nabla_{\theta} P(\tau ; \theta) R(\tau) \\
&= \sum_{\tau} \frac{P(\tau ; \theta)}{P(\tau ; \theta)} \nabla_{\theta} P(\tau ; \theta) R(\tau) \\
&= \sum_{\tau} P(\tau ; \theta) \frac{\nabla_{\theta} P(\tau ; \theta)}{P(\tau ; \theta)} R(\tau) \\
&= \sum_{\tau} P(\tau ; \theta) \nabla_{\theta} \log P(\tau ; \theta) R(\tau) \\
&= \mathbb{E}_{\pi_{\theta}}[\nabla_{\theta} \log P(\tau ; \theta) R(s_t, a_t)] \\
\end{aligned}
//}

この式を見ると、更新則によって、

* 報酬$R$が高いパスの存在確率を上げる
* 報酬$R$が低いパスの存在確率を下げる

方向に方策が更新されることがわかる。
これをパスではなく方策と遷移確率について立式するとどうなるだろうか。
式より、

//texequation{
\begin{aligned}
\nabla_{\theta} \log P(\tau ; \theta)
&= \nabla_{\theta} \log
\left[ \prod_{t=0}^H P(s_{t+1} | s_t, a_t) \pi_{\theta}(a_t | s_t) \right] \\
&= \nabla_{\theta} \left[ \sum_{t=0}^H \log P(s_{t+1} | s_t, a_t) \right] +
\nabla_{\theta} \left[ \sum_{t=0}^H \log \pi_{\theta} (a_t | s_t) \right] \\
&= \nabla_{\theta} \sum_{t=0}^H \log \pi_{\theta} (a_t | s_t) \\
&= \sum_{t=0}^H \nabla_{\theta} \log \pi_{\theta} (a_t | s_t)
\end{aligned}
//}

となる。状態遷移確率@<m>$P(s_{t+1} | s_t, a_t)$は@<m>$\theta$をパラメタとして持たないので、@<m>$\theta$で微分するとこの項が消えるのである。

結局、勾配は
//texequation{
\begin{aligned}
\hat{g} = \mathbb{E}_{\pi_{\theta}}[\nabla_{\theta} \log P(\tau ; \theta) R(s_t, a_t)] \\
where \hspace{15pt}
\nabla_{\theta} \log P(\tau ; \theta) =
\sum_{t=0}^H \nabla_{\theta} \log \pi_\theta (a_t | s_t)
\end{aligned}
//}

すなわち
//texequation{
\hat{g} = \mathbb{E}_{\pi_\theta}[\sum_{t=0}^H \nabla_{\theta} \log \pi_{\theta} (a_t | s_t) R(s_t, a_t)]
//}

で表され、方策の@<m>$\theta$での勾配のみで表現できることがわかる。

=== Baseline
式にbaseline @<m>$b$という値を追加する。
//texequation{
\hat{g} = \mathbb{E}_{\pi_\theta}[\sum_{t=0}^H \nabla_{\theta} \log \pi_{\theta} (a_t | s_t) (R(s_t, a_t) - b)]
//}

勾配の計算に@<m>$H$ステップの加算を含むため、分散は増大しやすくなる。
baselineは@<m>$\hat{g}$の値には影響を与えないが、調整前に比べて@<m>$\hat{g}$の分散を減少させる効果がある。

@<m>$R(s_t, a_t) - b$が小さいほど分散が小さくなるので、$b$の決定法としては、@<m>$R(s_t, a_t)$との二乗距離を小さくするように調整すればよい。

=== 行動価値関数 @<m>$Q(s_t, a_t)$ との関係
式では、$H$ステップの累積を想定して立式したが、@<m>$H \rightarrow \infty$としても同様に成り立つ。
また、報酬の加算についても割引報酬を用いても同様に導出される。

//texequation{
\begin{aligned}
\hat{g} &= \mathbb{E}_{\pi_\theta}[\sum_{t=0}^{\infty} \nabla_{\theta} \log \pi_{\theta} (a_t | s_t) \gamma^t r(s_t, a_t)] \\
&= \mathbb{E}_{\pi_\theta}[\sum_{t=0}^{\infty} \nabla_{\theta} \log \pi_{\theta} (a_t | s_t) 
[\sum_{k=0}^{t-1} \gamma^k r(s_k, a_k) + \sum_{k=t}^{\infty} \gamma^k r(s_k, a_k)]] \\
\end{aligned}
//}

ここで、@<m>$\sum_{k=0}^{t-1} \gamma^k r(s_k, a_k)$の項は、時刻@<m>$t$においての行動@<m>$a_t$の影響を受けない部分であるため、除去する。
これによって分散を増やす要素を削減することができる。

すると、行動価値関数を用いて以下のように表すことができる。

//texequation{
\begin{aligned}
\hat{g} &= \mathbb{E}_{\pi_\theta}[\sum_{t=0}^{\infty} \nabla_{\theta} \log \pi_{\theta} (a_t | s_t) 
\sum_{k=t}^{\infty} \gamma^k r(s_k, a_k)] \\
&= \mathbb{E}_{\pi_\theta}[\sum_{t=0}^{\infty} \nabla_{\theta} \log \pi_{\theta}(a_t | s_t)]
Q(s, a) \\
\end{aligned}
//}

このように行動価値関数に方策のlog微分をかけ合わせたものが、勾配の不偏推定量であることは方策勾配定理として知られている。

=== vanilla policy gradient
baselineの調整と方策の更新を逐次的に更新していくという、方策勾配法の基本的な方法に則った基本的なアルゴリズムを、vanilla policy gradient method と呼ぶことにする。
vanilla policy gradient methodの擬似コードは以下のようになる。

1. パラメタ @<m>$\theta$, ベースライン@<m>$b$の初期化
1. @<m>$for$ @<m>$i=1,2,\dots$ @<m>$do$
1. 現在の方策に従って行動し、復数パスを収集し、@<m>$R_t = \sum_{t'=t}^{H-1} \gamma^{t'-t} R_{t'}$を計算する。
1. @<m>$\|R_t - b\|^2$ を最小にするようにbaselineを調整する
1. @<m>$\hat{g} = \sum_{t=0}^H \nabla_{\theta} \log \pi_{\theta} (u_t^{(i)} | s_t^{(i)}) (R(\tau^{(i)}) - b)$で勾配を更新する
1. @<m>$end$ @<m>$for$

baselineの調整と、勾配の更新を繰り返していき、方策を最適化する。

=== REINFORCE アルゴリズム
上記の復数パスの情報を使って計算した@<m>$R_t$の代わりに、そのときどきの報酬@<m>$r_t$を使う方法は @<em>{REINFORCE} アルゴリズムとして知られている。
baselineとしては報酬の平均値@<m>$\bar{b} = \frac{1}{MT} \sum_{m=1}^M \sum_{t=1}^T R_t^m$がよく用いられる。

=== Actor-Critic
baselineを導入したとこで、方策の更新は、方策の分散を小さくする評価部と、方策を更新する部分の2つに分けられることがわかる。
ここで、baselineとして、価値関数@<m>$V^{\pi}$を使うと？？？？？？
ここで、@<m>$R(\tau) = \sum_{t=0}^H R(s_t, u_t)$の代わりに行動価値関数@<m>$Q^{\pi}(s, a)$を、baselineとして価値関数@<m>$V^{\pi}(s)$を使うことにする。
baselineとして状態$s$の関数を用いても、勾配の平均値には影響がないため、baselineとして採用できる。

以下の行動価値感数と状態価値関数の差分@<m>$A^{\pi}(s, a)$をアドバンテージ関数と呼ぶ。
//texequation{
A^{\pi}(s, a) = Q^{\pi}(s, a) - V^{\pi}(s)
//}

アドバンテージを使って、勾配は
//texequation{
\hat{g} = \frac{1}{m} \sum_{i=0}^m \sum_{t=0}^H \nabla_{\theta} \log \pi_{\theta} (u_t^{(i)} | s_t^{(i)}) A^{\pi}(s, a)
//}
で更新できる。
このアドバンテージを小さくする @<em>{Critic} と方策を更新する @<em>{Actor} を組み合わせた方法全般を @<em>{Actor-Critic} 法と呼ぶ。
Actor-Criticと大仰な呼び方がされるが、要は方策勾配を求めるときのbaselineを工夫したというふうに理解すれば良い。

ActorとCriticはそれぞれ様々な方法で実装できる。
Criticとしては前々回までで扱った価値反復法や、最小二乗法を用いることができる。
Actorは通常の方策勾配法や、自然勾配を用いる方法など様々考えられる。

== 方策勾配の収束を早くする技術
方策勾配を用いる方法では、方策の更新方向の決定が非常に重要になる。
悪い方策を採用したことで低い報酬しか取れないようにずれてくると、回復が難しくなってしまうからだ。

=== 自然勾配
方策の確率分布のパラメタ@<m>$\theta$のうち、あるパラメタ@<m>$\theta_1, \theta_2$の距離を@<m>$\|\theta_1 - \theta_2\|^2$とする、ユークリッド距離において定めたのが、式の方策の更新方向であった。
ここで、確率分布間の疑距離をKLダイバージェンスで定めると、自然勾配という勾配方法が導出される。
自然勾配は、通常の勾配にフィッシャー行列の逆行列を掛けたものになる

//texequation{
\tilde{\nabla_{\theta}} J (\theta) = F^{-1} (\theta) \nabla_{\theta} J(\theta)
//}

ここで、@<m>$F(\theta)$はフィッシャー行列である。
一般的に、勾配法において、自然勾配を用いると良い性能が得られることが知られている。

==== Natural Actor-Critic
式のアドバンテージ関数を線形モデル

//texequation{
A^{\pi}(s, a) = w^{\mathrm{T}} \nabla_{\theta} \log \pi_{\theta} (a | s)
//}
で近似することにすると、

//texequation{
\nabla_{\theta} J (\theta) =
E_{\pi}[\nabla_{\theta} \log \pi_{\theta} (a | s)
\nabla_{\theta} \log \pi_{\theta} (a | s) ^ {\mathrm{T}} ]
w \\
= F(\theta) w
//}

となる。
さらに方策の勾配として、自然勾配を用いると

//texequation{
\tilde{\nabla_{\theta}} J (\theta) =
F^{-1} (\theta) \nabla_{\theta} F(\theta) w \\
= w
//}

となり、方策はアドバンテージ感数のパラメタ$w$のみを用いて更新できる。

=== TRPO
最適化計算における更新ステップの計算にKLダイバージェンスによる制約を加えたものがTRPO(Trust Region Policy Optimization)である。
この方法は、をKLダイバージェンスで拘束しているため、近似的には自然勾配法と同様の手法となる。
TRPOは方策勾配法に限らず、モデルなし学習においても利用することができるが、以下では方策勾配法と組み合わせることを前提に述べる。

さて、@<m>$\theta$でパラメタライズされた方策@<m>$\pi_{\theta}(a|s)$がある場合、方策勾配が
//texequation{
\hat{g} =
E_{\pi}[\nabla_{\theta} \log \pi_{\theta} (a | s) A^{\pi}(s, a)]
//}
であった。これは、以下の値@<m>$L_{\theta}$の微分値である。
//texequation{
L_{\theta}(\theta) =
E_{\pi}[\log {\pi_{\theta}} (a | s) A^{\pi}(s, a)]
//}

このとき、更新のステップを制限するために、以下のようにKLダイバージェンスで制約を課して最大化を行う。
//texequation{
\begin{aligned}
\underset{x}{\text{maximize }}
L_{\theta_{old}} (\theta) \\\\\
\text{subject to }
D_{KL}^{\text{max}} (\theta_{old}, \theta) \leq \delta
\end{aligned}
//}
ここで、@<m>$\theta_{old}$は@<m>$\pi_{\theta}$におけるパラメタ@<m>$\theta$の前の値である。
また、@<m>$D_{KL}$は確率分布@<m>$\pi_{\theta}$と@<m>$\pi_{\theta_{old}}$の間のKLダイバージェンスであり、
@<m>$D_{KL}^{\text{max}} (\theta_{old}, \theta)$は任意のパラメタの組み合わせに対して、KLダイバージェンスを計算したときの最大値を表す。
実用的には組み合わせが膨大になり、最大値を求めるのは難しいため、制約はヒューリスティックに以下のように平均値で代用する。
//texequation{
\begin{aligned}
\underset{x}{\text{maximize }}
L_{\theta_{old}} (\theta) \\
\text{subject to }
\bar{D}_{KL} (\theta_{old}, \theta) \leq \delta
\end{aligned}
//}
ここで、
@<m>$
\bar{D}_{KL} (\theta_1, \theta_2) =
E_{s \sim \rho} [D_{KL}(\pi_{\theta_1}(\cdot | s), \pi_{\theta_2}(\cdot | s))]
$
である。

ただし、制約において問題を解くのは簡単ではないので、以下のようにソフト制約を使う形に書き下す。
//texequation{
\underset{x}{\text{maximize }}
E_{\pi} [ L_{\theta_{old}} (\theta) - \beta \bar{D}_{KL} (\theta_{old}, \theta) ]
//}

=== PPO
PPO(Proximal Policy Optimization)は方策の目標値をクリッピングすることで、おおまかに方策の更新を制約する方法である。
TRPOではKLダイバージェンスを制約として利用していたが、PPOでは、目的関数を以下の@<m>$L^{\text{CLIP}}$として、勾配を求める。

//texequation{
L^{\text{CLIP}} (\theta) = \hat{E}_t [ \min (r_t(\theta) \hat{A}_t, \text{clip} (r(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) ]
//}

ここで、@<m>$r_t(\theta)$は確率の比率であり、
//texequation{
r_t(\theta) =
\frac
{\pi_{\theta}(s_t, a_t)}
{\pi_{\theta_{old}}(s_t, a_t)}
//}

である。また、@<m>$\text{clip} (r(\theta), 1-\epsilon, 1+\epsilon)$ は、@<m>$r(\theta)$が$1-\epsilon$あるいは@<m>$1+\epsilon$を超過しないように制限する関数である。
@<m>$\text{clip} (r(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t$のグラフと、@<m>$L^{\text{CLIP}}$を以下に示す(John Schulmanら[^3]より引用)。

|clip関数|$L^{\text{CLIP}}$|
|---|---|
|![clip](/images/2017/12/clip.png)|![clip](/images/2017/12/clips.png)|

PPOは簡潔なアルゴリズムであるにもかかわらず、高い学習性能を示すことが知られている。

[^3]: J.Schulman et.al "Proximal Policy Optimization Algorithms" https://arxiv.org/abs/1707.06347
