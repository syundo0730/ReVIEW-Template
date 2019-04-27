= 理論編
== マルコフ決定過程
マルコフ決定過程(Markov Decision Process; MDP)は状態の遷移が確率的に起こり、マルコフ過程を満たす過程のことをいう。
MDPは状態@<m>$s$、行動@<m>$a$、遷移先の状態を@<m>$s'$、状態遷移確率@<m>$P(s'|s, a)$の組で表現される。
また、状態@<m>$s$において行動@<m>$a$を選択したとき、即時報酬@<m>$r(s, a, s')$が得られるとする。

とくに時間的な過程の進展を表すため、特に時刻@<m>$t$から@<m>$t+1$の状態の遷移について

//texequation{
状態: s_t\\
行動: a_t\\
状態遷移確率: P(s_{t+1}|s_t, a_t)\\
報酬関数: r_t = r(s_t, a_t, s_{t+1})
//}

を考える。

=== 価値関数
時刻@<m>$t$において将来(@<m>$t \rightarrow \infty $)にわたって得られる報酬について、割引累積報酬@<m>$G_t$を定義する。
//texequation{
G_t = \sum_{k=0}^{\infty} \gamma ^k R_{t+k+1}
//}
ここで、@<m>$R_{t+1}$は@<m>$r(s_t, a_t, s_{t+1})$の値とする。
また、@<m>$\gamma$は@<m>$0 \le \gamma < 1$の値で、遠い将来に得られるであろう報酬を低く見積もるために使う。

状態$s$において、行動$a$が選択される確率を@<m>$\pi = \pi(a | s)$とする。
この@<m>$\pi$を@<em>{方策}と呼ぶ。
ロボットで言うと次の状態をどう選ぶかの判断を行う部分である。

さて、ある方策@<m>$\pi$を採用したときの報酬がどの程度のものか見積もりたい。
方策@<m>$\pi$のもとで、以下のように関数@<m>$V^{\pi}$を定義する。
//texequation{
V^{\pi}(s) = \mathbb{E}[G_t|s_t=s] = \mathbb{E}[R_{t+1} + \gamma R_{t+2} + \dots | s_t = s]
//}
これを@<em>{状態価値関数}(あるいは単に@<em>{価値関数}と呼ぶ。
期待値を取っているのは、方策@<m>$\pi$は確率的であるから、@<m>$s_t=s$となるのも確率的であるため、@<m>$s_t$について周辺化して評価したいためである。

同様に、状態だけでなく、行動についても条件として

//texequation{
Q^{\pi}(s,a) = \mathbb{E}[G_t|s_t=s, a_t=a] = \mathbb{E}[R_{t+1} + \gamma R_{t+2} + \dots | s_t = s, a_t = a]
//}
も考える。これを、@<em>{行動価値関数}と呼ぶ。

以上の枠組みにおいて、最も良い方策、@<em>{最適方策}を@<m>$\pi^{\*}$とし、
この方策を採用したときの価値関数を@<em>{最適行動価値関数}
//texequation{
Q^{*}(s,a) = Q^{\pi^{*}}(s,a) = \max_{\pi} Q^{\pi}(s,a)
//}
とする。

=== ベルマン方程式
式において@<m>$\mathbb{E}[*]$は線形の演算のため、

//texequation{
\begin{aligned}
V^{\pi}(s) &=  \mathbb{E}[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | s_t = s]\\
&= \mathbb{E}[R_{t+1} | s_t = s] + \mathbb{E}[\sum_{k=1}^{\infty} \gamma^k R_{t+k+1} | s_t = s]\\
&= \mathbb{E}[R_{t+1} | s_t = s] + \gamma \mathbb{E}[\sum_{k=1}^{\infty} \gamma^{k-1} R_{t+k+1} | s_t = s]
\end{aligned}
//}

とすることができる。

ここで、式右辺第1項は

//texequation{
\mathbb{E}[R_{t+1}\ | s_t = s]
=\sum_{a \in A(s)} \pi(a|s) \sum_{s' \in S} P(s'|s, a) r(s, a, s')
//}

となる。
@<m>$s_t=s$において行動@<m>$a$を取る確率が@<m>$\pi(a,|s)$、状態遷移して@<m>$s'$に移動する確率が@<m>$P(s'|s,a)$であるから、行動@<m>$a$を取って、状態@<m>$s'$に遷移する確率は@<m>$\pi(a,|s)P(s'|s,a)$である。
その上で、状態@<m>$s$において取れる行動の全集合@<m>$A(s)$と次に取れる全状態@<m>$S$について@<m>$r(s,a,s')$の期待値を取る、というのが上式で行われていることである。

次に、式右辺第2項は
//texequation{
\begin{aligned}
\mathbb{E}[\sum_{k=1}^{\infty} \gamma^{k-1} R_{t+k+1} | s_t = s]
&= \sum_{a \in A(s)} \pi(a|s) \sum_{s' \in S} P(s'|s,a) \mathbb{E}[\sum_{k=1}^{\infty} \gamma^{k-1} R_{t+k+1} | s_{t+1} = s']\\
&= \sum_{a \in A(s)} \pi(a|s) \sum_{s' \in S} P(s'|s,a) \mathbb{E}[\sum_{k=0}^{\infty} \gamma^k R_{(t+1)+k+1} | s_{t+1} = s']\\
&= \sum_{a \in A(s)} \pi(a|s) \sum_{s' \in S} P(s'|s,a) V^{\pi}(s')
\end{aligned}
//}

と変形できる。
以上, , 式により、

//texequation{
\begin{aligned}
V^{\pi}(s) = \sum_{a \in A(s)} \pi(a|s) \sum_{s' \in S} P(s'|s, a) \left(r(s, a, s') + \gamma V^{\pi}(s') \right)
\end{aligned}
//}

が導出される。これを@<em>{ベルマン方程式}と呼ぶ。

また、@<m>$Q^{\pi}(s,a)$の定義より
//texequation{
V^{\pi}(s) = \sum_{a \in A(s)} \pi(a|s) Q^{\pi}(s,a)
//}

であるから、
//texequation{
\begin{aligned}
Q^{\pi}(s, a) &= \sum_{s'} P(s'|s, a) \left(r(s, a, s') + \gamma V^{\pi}(s') \right)\\
&= \sum_{s'} P(s'|s, a) \left(r(s, a, s') + \gamma \sum_{a \in A(s')} \pi(a'|s') Q^{\pi}(s',a') \right)
\end{aligned}
//}

と行動価値関数についてのベルマン方程式を導出できる。

== 反復による価値の推定
ベルマン方程式を解くことができれば、@<m>$Q^{\pi}$を計算できるのだが、どう計算するのか、価値関数からどのように方策を決定するのかという問題がある。
そのためには状態遷移確率が既知である必要がある。
扱いたいのは環境が未知の問題であるため、状態遷移確率を用いずに、多数のデータを使って反復的に計算することでこれを求めたい。

価値関数についてのベルマン方程式において、常に最適な方策を取るという前提を置けば、以下の@<em>{最適ベルマン方程式}を定めることができる。
//texequation{
\begin{aligned}
V^{\pi}(s) = \max_{a \in A} \sum_{s' \in S} P(s'|s, a) \left(r(s, a, s') + \gamma V^{\pi}(s') \right)
\end{aligned}
//}

//texequation{
\begin{aligned}
Q^{\pi}(s, a) = \sum_{s'} P(s'|s, a) \left(r(s, a, s') + \gamma \max_{a' \in A} Q^{*}(s',a') \right)
\end{aligned}
//}

あるいは、取り得る方策が確率的でない、常に方策が決まっている定常方策を取るとすると、以下のようになる。
//texequation{
\begin{aligned}
V^{\pi}(s) =  \sum_{s' \in S} P(s'|s, a) \left(r(s, a, s') + \gamma V^{\pi}(s') \right)
\end{aligned}
//}

//texequation{
\begin{aligned}
Q^{\pi}(s, a) = \sum_{s'} P(s'|s, a) \left(r(s, a, s') + \gamma Q^{\pi}(s',a') \right)
\end{aligned}
//}

以上の性質は価値関数を逐次的に更新していくことで求めていくために重要になる。
また、その価値関数に従って最適な方策を決定し行動していくことで逐次的に方策を更新していくこともできる。
前者を価値反復、後者を方策反復と呼ぶ。
このとき、方策の決定方法としてはgreedy方策が考えられる。
//texequation{
a^* = \arg\max(Q^{*})
//}

また、実際に採用している方策と異なる方策を学習する方法は@<em>{方策オフ}学習と呼ばれる。
一方で、学習した方策をそのときどき採用する方法を@<em>{方策オン}学習と呼ぶ。

=== TD学習
TD学習(Temporal difference learning)は、現在の推定値を学習中の目標値として使用することで、問題を解いていく手法である。

ベルマン方程式は、状態遷移確率が未知の場合、そのまま解くことはできない。
そこで、状態遷移確率は実際に観測する(サンプリングする)ことによって、確率分布を近似し、ベルマン方程式を扱う。

==== TD(0)法
以下のように価値関数の推定値を更新していく手法がTD(0)法である。
//texequation{
V_{t+1}(s_t)
\leftarrow
(1 - \alpha_t) V_t(s_t) + \alpha_t (R_{t+1} + \gamma V_t(s_{t+1}))
//}

現在の値@<m>$V_t(s_t)$と目標値@<m>$R_{t+1} + \gamma V_t(s_{t+1})$との内分によって推定値を更新していくのがこの手法である。
TD誤差として@<m>$\delta_t$を以下のように定義することで、TD誤差を小さくする方向に更新するアルゴリズムとして捉えることもできる。

//texequation{
\begin{aligned}
\delta_t = (R_{t+1} + \gamma V_t(s_{t+1})) - V_t(s_t) \\
V_{t+1}(s_t)
\leftarrow
V_t(s_t) + \alpha_t \delta_t
\end{aligned}
//}

@<m>$\delta_t$は目標値と現在の値の差分であるから、この立式のほうがわかりやすいかもしれない。

==== Sarsa
TD(0)法を行動価値数に拡張したものが、Sarsaである。

時刻@<m>$t$において状態@<m>$s_t$であり@<m>$a_t$を行った結果、次の状態@<m>$s_{t+1}$と報酬@<m>$r_t$を観測したとき、@<m>$s_{t+1}$において行う予定の行動@<m>$a_{t+1}$をもとに、以下の更新則でQ値を更新する。

//texequation{
Q(s_t, a_t)
\leftarrow
(1 - \alpha) Q(s_t, a_t) + \alpha (r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}))
//}

=== Q-Learning
最適ベルマン方程式を解くためにで、以下の更新則でQ値を更新するのがQ-Learningである。

//texequation{
Q(s_t, a_t)
\leftarrow
(1 - \alpha) Q(s_t, a_t) + \alpha (r_{t+1} + \gamma \max_{a^{\prime} \in A} Q(s_{t+1}, a^{\prime}))
//}

Sarsaと違って、次に何の行動を取ったかはQ値の更新には関わってこない。
@<m>$\max_{a^{\prime} \in A} Q(s_{t+1}, a^{\prime})$つまり、次の状態において取れる行動のうち、最大の価値を得られる行動(つまり最適方策@<m>$\pi^{\*}$)を採った時のQ値を使って更新する。
Sarsaが実際に取った方策に更新が依存するのに対して、Q-LearningではQ値は環境に対して一定の値に収束する、方策オフ型学習である。
