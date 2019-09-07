= 反復による価値の推定
ベルマン方程式を解くことができれば、@<m>$Q^{\pi}$を計算できるのですが、実際にはどう計算するのかという問題があります。
計算するためには、状態遷移確率が分かっている必要があるためです。
環境が未知である場合に対応するために、状態遷移確率を用いずに近似的にでも計算したいです。
このように、環境が未知の状態で学習を行う問題を@<em>{モデルフリー学習}といいます。
ここでは、多数のデータを使って反復的に計算することで、最適な行動価値関数を求める方法を使います。

価値関数についてのベルマン方程式において、常に最適な方策を取るという前提を置けば、以下の@<em>{最適ベルマン方程式}を定めることができます。
//texequation[opt_v_belman_eq]{
\begin{aligned}
V^{\pi}(s) = \max_{a \in A} \sum_{s' \in S} P(s'|s, a) \left(r(s, a, s') + \gamma V^{\pi}(s') \right)
\end{aligned}
//}

//texequation[opt_q_belman_eq]{
\begin{aligned}
Q^{\pi}(s, a) = \sum_{s'} P(s'|s, a) \left(r(s, a, s') + \gamma \max_{a' \in A} Q^{*}(s',a') \right)
\end{aligned}
//}

更に、取り得る方策が確率的でない、常に方策が決まっている定常方策を取るとすると、以下のようになります。
//texequation[opt_const_v_belman_eq]{
\begin{aligned}
V^{\pi}(s) =  \sum_{s' \in S} P(s'|s, a) \left(r(s, a, s') + \gamma V^{\pi}(s') \right)
\end{aligned}
//}

//texequation[opt_const_q_belman_eq]{
\begin{aligned}
Q^{\pi}(s, a) = \sum_{s'} P(s'|s, a) \left(r(s, a, s') + \gamma Q^{\pi}(s',a') \right)
\end{aligned}
//}

以上の性質は価値関数を逐次的に更新していくことで求めていくために重要になります。

#@# また、その価値関数に従って最適な方策を決定し行動していくことで逐次的に方策を更新していくこともできる。
#@# このとき、方策の決定方法としてはgreedy方策が考えられる。
#@# //texequation{
#@# a^* = \arg\max(Q^{*})
#@# //}

#@# また、実際に採用している方策と異なる方策を学習する方法は@<em>{方策オフ}学習と呼ばれる。
#@# 一方で、学習した方策をそのときどき採用する方法を@<em>{方策オン}学習と呼ぶ。

価値関数を近似的に求めるために、状態、行動、報酬のセット @<m>$(s_t, a_t, r_t)$を多数記録し、それを用いて
//texequation[actual_gain]{
c_t = \sum_{k=t}^T \gamma ^{k-t} r_k
//}
@<eq>{actual_gain}のような実質的に得られた報酬を計算していくことで@<em>{モンテカルロ推定}していく方法が考えられます。
この状態、行動、報酬のセットを経験データ、経験と呼びます。
一方で、ベルマン方程式の持つ再帰性を用いれば、モンテカルロ推定する場合よりもより効率よく推定することができます。
直接モンテカルロ推定しようとすると、大きい時間ステップを推定した場合に偏りが出てしまうので、より多くの経験が必要になりますが、
再帰性を用いれば、前時間ステップまでの推定値を前提にして推定を進めることができるからです。

TD学習(Temporal difference learning)は、現在推定できる値を学習の目標値として使用することで、問題を解いていく手法です。

== TD法
@<eq>{opt_const_v_belman_eq}の状態遷移確率が不明なので以下のように価値関数の推定値を更新していく手法がTD法です。

//texequation{
V_{t+1}(s_t)
\leftarrow
(1 - \alpha_t) V_t(s_t) + \alpha_t (R_{t+1} + \gamma V_t(s_{t+1}))
//}

現在の値@<m>$V_t(s_t)$と目標値@<m>$R_{t+1} + \gamma V_t(s_{t+1})$との内分によって推定値を更新しています。
TD誤差として@<m>$\delta_t$を以下のように定義することで、TD誤差を小さくする方向に更新するアルゴリズムとして捉えることもできます。

//texequation{
\begin{aligned}
\delta_t = (R_{t+1} + \gamma V_t(s_{t+1})) - V_t(s_t) \\
V_{t+1}(s_t)
\leftarrow
V_t(s_t) + \alpha_t \delta_t
\end{aligned}
//}

@<m>$\delta_t$は目標値と現在の値の差分なので、この立式のほうがわかりやすいかもしれません。

このTD法の価値関数@<m>$V_t$の代わりに、@<m>$\lambda$ステップの累積報酬を用いる、TD(@<m>$\lambda$)法もあります。
TD(@<m>$\lambda$)法はモンテカルロ法とTD法を一般化したものと考えることができるので、TD法をTD(0)法と呼ぶこともあります。

== Sarsa
@<eq>{opt_const_q_belman_eq}を使って、TD法を行動価値関数に拡張したものが、Sarsaです。
時刻@<m>$t$において状態@<m>$s_t$であり@<m>$a_t$を行った結果、
次の状態@<m>$s_{t+1}$と報酬@<m>$r_t$を観測したとき、
@<m>$s_{t+1}$において行う予定の行動@<m>$a_{t+1}$をもとに、以下の更新則でQ値を更新します。

//texequation{
Q(s_t, a_t)
\leftarrow
(1 - \alpha) Q(s_t, a_t) + \alpha (r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}))
//}

Q関数が得られた上で、方策@<m>$\pi$の決定方法としてはgreedy方策が考えられます。
//texequation{
a^* = \arg\max(Q^{*})
//}
Q値を最大にするような行動@<m>$a^*$を選択するのです。
SARSAではQ関数の推定、推定に基づいた方策の実行、そして観測という繰り返しをしていきます。
このように行動をそのときどきの推定値に基づいて決めて学習を進めていく方法を@<em>{方策オン(On-Policy)}学習といいます。

== Q-Learning
@<eq>{opt_const_q_belman_eq}の最適ベルマン方程式を解くために、以下の更新則でQ値を更新するのがQ-Learningです。

//texequation{
Q(s_t, a_t)
\leftarrow
(1 - \alpha) Q(s_t, a_t) + \alpha (r_{t+1} + \gamma \max_{a^{\prime} \in A} Q(s_{t+1}, a^{\prime}))
//}

Sarsaと違うのは、次ステップに取る行動はQ値の更新には関わってこないというところです。
Q関数の値は、次の状態において取れる行動のうち、最大の価値を得られる行動@<m>$\max_{a^{\prime} \in A} Q(s_{t+1}, a^{\prime})$を採用したときのQ値を使って更新します。

SarsaとQ-Learningを比較すると、Q-Learningでは実際に採用している行動とは関係なくQ値の更新をしています。
このような学習方法は@<em>{方策オフ(Off-Policy)}学習と呼ばれます。

== Q関数の近似
Q関数を計算機的に実現する方法として、テーブル形式で@<m>$(s,a)$とQ関数の値のペアを保持しておく方法が考えられます。
この方法で、状態数が小さい場合は扱うことができますが、状態数が大きくなるとメモリで持つこともできなくなってしまいますし、テーブル上は経験したことのない@<m>$(s,a)$のペアばかりになってしまいます。

そのため、近似関数@<m>$Q_{\theta_t}(s_t, a_t)$を使ってQ関数を近似します。
@<m>$Q_{\theta_t}$はパラメタ@<m>$\theta$でパラメタライズされた関数です。
近似したQ関数の学習とは、この@<m>$\theta$をTD誤差を最小にするように更新することになります。
近似関数を用いたQ-Learningのアルゴリズムは以下になります。

//texequation[approx_q_learning_update]{
\begin{aligned}
\delta_t = R_{t+1} + \gamma \max_{a^{\prime} \in A} Q_{\theta_t}(s_{t+1}, a^{\prime}) - Q_{\theta_t}(s_t, a_t) \\
\theta_{t+1} = \theta_t + \alpha_t \delta_{t+1} \nabla_{\theta} Q_{\theta_t} (s_t, a_t)
\end{aligned}
//}

=== DQN
DQN(Deep Q-Network)は行動価値関数@<m>$Q(s,a)$をDeepニューラルネットワークを用いて近似し、Q-Learningを行う手法です。
DQNで用いられるニューラルネットワークを図に示します。
ニューラルネットワークの入力は状態で、出力は、ある行動をとったときのQ値となっています。
状態、行動を両方入力とするのでなく、このような形式にすることでニューラルネットワークは入力の処理に注力することができます。
また状態として画像を扱うときに、画像認識の分野で用いられているニューラルネットワークを利用しやすいという利点があります。
DQNはAtariのいくつかのゲームにおいて人間より良いスコアを記録するなど目覚しい成果を上げました。

DQNのようにQ関数を近似した場合、学習した結果が最適行動価値関数へ収束することは保証されません。
そこで、DQNでは学習が収束しない問題に対処するため、また学習されたネットワークを他のタスクに転用しやすくするためにいくつかの工夫をしています。

 * Experience Replayの利用
 ** 経験@<m>$(s_t, s_{t+1}, a_t, r_{t+1})$ をリプレイバッファに蓄積し、それをランダムに並べ替えてからバッチ的に学習する。
 ** これによって、サンプルの系列において時間的相関があるとニューラルネットワークの学習(確率的勾配法)がうまく働かなくなる問題を緩和している。
 * Target Networkの利用
 ** Q-Learningの更新則における目標値の計算において、学習中のネットワーク@<m>$Q_{\theta_t}$の代わりに、パラメータが古いもので固定されたネットワーク@<m>$Q_{\theta_t^{-}}$を利用する。
 ** これによって、Q-Learningの目標値@<m>$R_{t+1}+\gamma \max_{a^{\prime} \in A} Q_{\theta_t}(s_{t+1}, a^{\prime}) - Q_{\theta_t}(s_t, a_t)$と現在推定している行動価値関数@<m>$Q_{\theta_t}$の間の相関があるために、学習が発散、振動しやすくなる問題を緩和している。
 * 報酬のクリッピング
 ** 得られる報酬を@<m>$[-1, 1]$の範囲にしている。
 ** これによって、復数のゲーム間で同じネットワークを使って学習をすることができる。

結局、DQNのTD誤差は以下のようになります。
//texequation{
\delta_t = R_{t+1} + \gamma \max_{a^{\prime} \in A} Q_{\theta_t^{-}}(s_{t+1}, a^{\prime}) - Q_{\theta_t}(s_t, a_t)
//}

上記のTD誤差において、
@<m>$\max_{a^{\prime} \in A} Q_{\theta_t}(s_{t+1}, a^{\prime})$の部分で、最大のQ値を実現するという@<m>$a^{\prime}$は、推定中のQ関数がたまたまその@<m>$a^{\prime}$について最大値を取っているだけかもしれません。
Q関数が推定中で、特定の@<m>$a^{\prime}$について推定誤差がある場合にそういった問題が起こりますが、これは行動空間が大きくなるほど起こりやすくなります。

そこでDQNを拡張した、DDQN(Double DQN)では、TD誤差を以下の@<m>$y_t^{\text{DoubleDQN}}$と置き換えます。
//texequation{
y_t^{\text{DoubleDQN}}
= R_{t+1} + \gamma Q_{\theta_t^{-}}(s_{t+1}, \underset{a^{\prime} \in A}{\text{argmax }} Q_{\theta_t}(s_{t+1}, a^{\prime}))
//}
こうすると、仮に@<m>$Q_{\theta_t}(s_{t+1}, a^{\prime})$の推定誤差によって、@<m>$a^{\prime}$が選ばれても、@<m>$Q_{\theta_t^{-}}$にも推定誤差があるために、過度に目標値が大きく見積もられることを抑制する効果がでます。

DQNの後継の手法としては、近年の様々な知見をもとに、DQNを拡張していったRAINBOWというものもあります。
現時点で、価値関数の推定の反復を用いた手法で採用する候補としてはRAINBOWが考えられるでしょう。
DQNとその亜種、後継の実装についてはネット上に多くの情報があるので、調べてみると良いと思います。