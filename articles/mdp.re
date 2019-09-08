= マルコフ決定過程
マルコフ決定過程(Markov Decision Process; MDP)は状態の遷移が確率的に起こり、マルコフ性を持つ過程のことをいいます。
MDPは状態@<m>$s$、行動@<m>$a$、遷移先の状態を@<m>$s'$、状態遷移確率@<m>$P(s'|s, a)$の組で表現されます。
そして状態@<m>$s$において行動@<m>$a$を選択したとき、即時報酬@<m>$r(s, a, s')$が得られるとします。
MDPを概念的に表現したのが@<img>{mdp}です。
//image[mdp][MDPの概念図]

またMDPの遷移が時間的に起こることを考えると、時刻@<m>$t$から@<m>$t+1$まで状態が遷移するとして
//table[mdp_settings][MDPの設定]{
@<m>$s_t$	状態
@<m>$a_t$	行動
@<m>$P(s_{t+1}|s_t, a_t)$	状態遷移確率
@<m>$r_t = r(s_t, a_t, s_{t+1})$	報酬関数
//}
を扱います。


== 価値関数
時刻@<m>$t$において将来(@<m>$t \rightarrow \infty $)にわたって得られる報酬について、割引累積報酬@<m>$G_t$を考えます。
//texequation[gain_func]{
G_t = \sum_{k=0}^{\infty} \gamma ^k R_{t+k+1}
//}
ここで、@<m>$R_{t+1}$は@<m>$r(s_t, a_t, s_{t+1})$の値です。
@<m>$\gamma$は@<m>$0 \le \gamma < 1$の値で、遠い将来に得られるであろう報酬を低く見積もるために使います。
あまり遠い将来のことを考えても誤差が大きくて学習に悪い影響があるのでこのように減衰させます。

状態@<m>$s$において、行動@<m>$a$が選択される確率@<m>$\pi(a | s)$のことを@<em>{方策}と呼びます。
ロボットで言うと次の状態をどう選ぶかの判断を行う部分です。

さて、ロボットは方策を学習していくことで頭の良い行動を取るようになっていきます。
そのためにある方策@<m>$\pi$を採用したときの報酬はどの程度のものか、見積もりたくなります。
その報酬を最大化するような方策を求める、最適化問題として扱うことができるからです。
方策@<m>$\pi$のもとで、つぎのように関数@<m>$V^{\pi}$を定義します。
//texequation[v_func]{
V^{\pi}(s) = \mathbb{E}[G_t|s_t=s] = \mathbb{E}[R_{t+1} + \gamma R_{t+2} + \dots | s_t = s]
//}
これを@<em>{状態価値関数}(あるいは単に@<em>{価値関数}と呼びます。
期待値を取っているのは、方策@<m>$\pi$は確率的であって、@<m>$s_t=s$となるのも確率的であるため、@<m>$s_t$について周辺化して評価したいためです。

同様に、状態だけでなく、行動についても条件として

//texequation[q_func]{
Q^{\pi}(s,a) = \mathbb{E}[G_t|s_t=s, a_t=a] = \mathbb{E}[R_{t+1} + \gamma R_{t+2} + \dots | s_t = s, a_t = a]
//}
も考えられます。これを、@<em>{行動価値関数}と呼びます。

この枠組みにおいて、最も良い方策、@<em>{最適方策}を@<m>$\pi^{\*}$とし、
この方策を採用したときの価値関数を@<em>{最適行動価値関数}
//texequation[opt_q_func]{
Q^{*}(s,a) = Q^{\pi^{*}}(s,a) = \max_{\pi} Q^{\pi}(s,a)
//}
と呼びます。
エージェントの学習とは、この行動価値関数を最適にする方策を求める問題であるといえます。

== ベルマン方程式
@<eq>{v_func}において@<m>$\mathbb{E}[*]$は線形の演算のため、

//texequation[belman_eq_intro]{
\begin{aligned}
V^{\pi}(s) &=  \mathbb{E}[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | s_t = s]\\
&= \mathbb{E}[R_{t+1} | s_t = s] + \mathbb{E}[\sum_{k=1}^{\infty} \gamma^k R_{t+k+1} | s_t = s]\\
&= \mathbb{E}[R_{t+1} | s_t = s] + \gamma \mathbb{E}[\sum_{k=1}^{\infty} \gamma^{k-1} R_{t+k+1} | s_t = s]
\end{aligned}
//}

と計算を進めることができます。

ここで、@<eq>{belman_eq_intro}右辺第1項は

//texequation[belman_eq_right]{
\mathbb{E}[R_{t+1}\ | s_t = s]
=\sum_{a \in A(s)} \pi(a|s) \sum_{s' \in S} P(s'|s, a) r(s, a, s')
//}

となります。
@<m>$s_t=s$において行動@<m>$a$を取る確率が@<m>$\pi(a,|s)$、状態遷移して@<m>$s'$に移動する確率が@<m>$P(s'|s,a)$なので、行動@<m>$a$を取って、状態@<m>$s'$に遷移する確率は@<m>$\pi(a,|s)P(s'|s,a)$です。
その上で、状態@<m>$s$において取れる行動の全集合@<m>$A(s)$と次に取れる全状態@<m>$S$について@<m>$r(s,a,s')$の期待値を取る、というのが上式で行われていることです。

次に、@<eq>{belman_eq_intro}右辺第2項は
//texequation[belman_eq_right_second]{
\begin{aligned}
\mathbb{E}[\sum_{k=1}^{\infty} \gamma^{k-1} R_{t+k+1} | s_t = s]
&= \sum_{a \in A(s)} \pi(a|s) \sum_{s' \in S} P(s'|s,a) \mathbb{E}[\sum_{k=1}^{\infty} \gamma^{k-1} R_{t+k+1} | s_{t+1} = s']\\
&= \sum_{a \in A(s)} \pi(a|s) \sum_{s' \in S} P(s'|s,a) \mathbb{E}[\sum_{k=0}^{\infty} \gamma^k R_{(t+1)+k+1} | s_{t+1} = s']\\
&= \sum_{a \in A(s)} \pi(a|s) \sum_{s' \in S} P(s'|s,a) V^{\pi}(s')
\end{aligned}
//}

と変形できます。
以上@<eq>{belman_eq_intro}, @<eq>{belman_eq_right}, @<eq>{belman_eq_right_second}より、導出された、

//texequation[v_belman_eq]{
\begin{aligned}
V^{\pi}(s) = \sum_{a \in A(s)} \pi(a|s) \sum_{s' \in S} P(s'|s, a) \left(r(s, a, s') + \gamma V^{\pi}(s') \right)
\end{aligned}
//}

@<eq>{v_belman_eq}を@<em>{ベルマン方程式}といいます。

また、@<m>$Q^{\pi}(s,a)$の定義より
//texequation[def_v_func]{
V^{\pi}(s) = \sum_{a \in A(s)} \pi(a|s) Q^{\pi}(s,a)
//}

であるので、
//texequation[q_belman_eq]{
\begin{aligned}
Q^{\pi}(s, a) &= \sum_{s'} P(s'|s, a) \left(r(s, a, s') + \gamma V^{\pi}(s') \right)\\
&= \sum_{s'} P(s'|s, a) \left(r(s, a, s') + \gamma \sum_{a \in A(s')} \pi(a'|s') Q^{\pi}(s',a') \right)
\end{aligned}
//}

と行動価値関数についてのベルマン方程式を導出できます。
