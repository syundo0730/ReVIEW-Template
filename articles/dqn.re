== DQN
DQN(Deep Q-Network)は行動価値関数@<m>$Q(s,a)$を深層ニューラルネットワークを用いて推定し、Q-Learningを行う手法である。
DQNでは行動と状態の組@<m>$(s,a)$に対してスカラー値@<m>$Q^{*}(s,a)$を割り当てるのでは@<em>{なく}、
状態@<m>$s$において行動@<m>$a_1, \dots, a_N$を採用したときの値@<m>$Q^{\*}(s,a_1), \dots, Q^{\*}(s,a_N)$を予測している。
つまり、入力が状態、出力が各行動を採用した時のQ値となっているニューラルネットワークを使ってQ関数の代わりとする。

この工夫によって、例えば、状態としてニューラルに多次元の画像をそのまま入力することができるようになり、ゲーム画面を入力としてそのままゲームを学習することができるようになった。
Atariのいくつかのゲームにおいて人間より良いスコアを記録するなど目覚しい成果を上げた。

関数近似を用いたQ-Learningのアルゴリズムは以下になる。

//texequation{
\begin{aligned}
\delta_t = R_{t+1} + \gamma \max_{a^{\prime} \in A} Q_{\theta_t}(s_{t+1}, a^{\prime}) - Q_{\theta_t}(s_t, a_t) \\
\theta_{t+1} = \theta_t + \alpha_t \delta_{t+1} \nabla_{\theta} Q_{\theta_t} (s_t, a_t)
\end{aligned}
//}

関数近似を用いた場合、つまりDQNのような場合でもQ-Learningのアルゴリズムはほぼ同じだが、Q関数を近似しているために、最適行動価値関数への収束は保証されない。

そこで、DQNでは学習が収束しない問題に対処するためや、学習されたネットワークを他のタスクに転用しやすくするためにいくつかの工夫をしている。

* Experience Replayの利用
  * 観測したサンプル @<m>$(s_t, s_{t+1}, a_t, R_{t+1})$ をリプレイバッファに蓄積し、それをランダムに並べ替えてからバッチ的に学習する。
  * これによって、サンプルの系列において時間的相関があると確率的勾配法がうまく働かなくなる問題を緩和している。
* Target Networkの利用
  * Q-Learningの更新則における目標値の計算において、学習中のネットワーク@<m>$Q_{\theta_t}$の代わりに、パラメータが古いもので固定されたネットワーク@<m>$Q_{\theta_t^{-}}$を利用する。
  * これによって、Q-Learningの目標値@<m>$R_{t+1}+\gamma \max_{a^{\prime} \in A} Q_{\theta_t}(s_{t+1}, a^{\prime}) - Q_{\theta_t}(s_t, a_t)$と現在推定している行動価値関数@<m>$Q_{\theta_t}$の間の相関があるために、学習が発散、振動しやすくなる問題を緩和している。
* 報酬のクリッピング
  * 得られる報酬を@<m>$[-1, 1]$の範囲にしている。
  * これによって、復数のゲーム間で同じネットワークを使って学習をすることができる。

結局、DQNのTD誤差は以下のようになる。
//texequation{
\delta_t = R_{t+1} + \gamma \max_{a^{\prime} \in A} Q_{\theta_t^{-}}(s_{t+1}, a^{\prime}) - Q_{\theta_t}(s_t, a_t)
//}

試しに、Open AIのCartpoleを使ったサンプルを[実装してみた]。(https://github.com/syundo0730/RL-samples/blob/master/dqn/main.py)
Qネットワークの目標値の計算は[L59](https://github.com/syundo0730/RL-samples/blob/master/dqn/main.py#L59)で行っている。
//list[target][target()][python]{
target = self.mainQ.model.predict(state)
target[0][action] = reward if done else reward + self.gamma * np.max(self.subQ.model.predict(next_state)[0])
//}
ここで、`subQ`と名付けているのが、Target netrowkである。Target networkの同期は2エピソードに1回にしてみた。
//list[sync][sync()][python]{
if e % 2 == 0:
    self.mainQ = self.subQ
//}

=== DDQN
DQNでは目標値の計算のなかにmaxを取る操作があるが、
@<m>$\max_{a^{\prime} \in A} Q_{\theta_t}(s_{t+1}, a^{\prime})$で、最大のQ値を実現するという@<m>$a^{\prime}$は、推定中のQ関数がたまたまその@<m>$a^{\prime}$について最大値を取っているだけかもしれない。
Q関数が推定中で、特定の@<m>$a^{\prime}$について推定誤差がある場合にこういうことが起こるが、これは行動空間が大きくなるほど起こりやすくなる。

これを防ぐため、DDQN(Double DQN)では、目標値を以下の@<m>$y_t^{\text{DoubleDQN}}$と置き換える。
//texequation{
y_t^{\text{DoubleDQN}}
= R_{t+1} + \gamma Q_{\theta_t^{-}}(s_{t+1}, \underset{a^{\prime} \in A}{\text{argmax }} Q_{\theta_t}(s_{t+1}, a^{\prime}))
//}
こうすると、仮に@<m>$Q_{\theta_t}(s_{t+1}, a^{\prime})$の推定誤差によって、@<m>$a^{\prime}$が選ばれても、@<m>$Q_{\theta_t^{-}}$にも推定誤差があるために、過度に目標値が大きく見積もられることを抑制できるという。

DDQNの場合のQネットワークの目標値の計算は[L59](https://github.com/syundo0730/RL-samples/blob/master/dqn/main.py#L65)のようになる。
//list[ddqn][ddqn()][python]{
target = self.mainQ.model.predict(state)
next_action = np.argmax(target[0])
target[0][action] = reward if done else reward + self.gamma * self.subQ.model.predict(next_state)[0][next_action]
//}