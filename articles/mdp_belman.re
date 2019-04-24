
= 理論編
== マルコフ決定過程
マルコフ決定過程(Markov Decision Process; MDP)は状態の遷移が確率的に起こり、マルコフ過程を満たす過程のことをいう。
MDPは状態@<m>$s$、行動@<m>$a$、遷移先の状態を@<m>$s'$、状態遷移確率@<m>$P(s'|s, a)$の組で表現される。
また、状態@<m>$s$において行動@<m>$a$を選択したとき、即時報酬@<m>$r(s, a, s')$が得られるとする。

とくに時間的な過程の進展を表すため、特に時刻@<m>$t$から@<m>$t+1$の状態の遷移について

//texequation{
状態: s_t\\\\\
行動: a_t\\\\\
状態遷移確率: P(s\_{t+1}|s_t, a_t)\\\\\
報酬関数: r_t = r(s_t, a_t, s\_{t+1})
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

## ベルマン方程式の導出
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