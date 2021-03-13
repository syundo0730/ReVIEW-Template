= 非線形最適制御
強化学習で扱われる問題のなかで、部分的に動力学的なモデルが既知であるような場合について、強化学習ではなく非線形最適制御の手法を使って解かれるものがあります。
学習イテレーションを少なくするために、強化学習で解く必要がない部分は最適制御で扱われるからです。
また、教師データや目標軌道を生成するために使われることがあります。
更には、強化学習の結果の評価するために比較対象として使われることもあります。
この章では、非線形最適制御問題とは何か、大別してどういった種類の手法があるかなど説明します。
そして、実ロボットを用いた強化学習の論文で言及されることが多い最適制御の手法である、
Differential Dynamic Programming(DDP), iterative LQR(iLQR), Sequential LQR(SLQ) について扱います。

== 状態方程式
状態@<m>$x$をもつシステムについて入力@<m>$u$が与えられた場合の状態時間変化の最適化を扱います。
MDPにおける状態@<m>$s$、行動@<m>$a$と対応しているものであり、強化学習との関係は深いです。
強化学習の論文であっても状態、行動、状態遷移を表す記号として@<m>$x, u, f$を用いている場合がありますが、
それは著者のバックグラウンドが制御工学にある@<fn>{fn01}ことが現れているのです。
状態の時間変化について微分形式で表されたものを状態方程式といいます。
時間変化を離散時間形式にしたものが離散の状態方程式です。
つぎの離散状態方程式で表されるシステムについて考えます。

//texequation[control_system_eq]{
x_{t+1} = f_t(x_t, u_t), t = 0, \dots, H
//}
ここで、@<m>$f_t$は何らかの関数で、線形とは限りません。

== 最適制御
ある評価関数の値を最大化あるいは最小化する制御入力の系列を求めるのが最適制御問題の目的です。
制御入力を求めるためのアプローチとしては
フィードバック制御を行う方策@<m>$\pi_{\theta}(x)$を求める方法と
フィードフォワード的に開ループにおいて制御入力@<m>$u_0, ... u_{\mathrm{H}}$を求める方法があります。

フィードバック制御を求める最適化問題は@<eq>{feedback_policy_collocation}のようになります。
//texequation[feedback_policy_collocation]{
\begin{aligned}
\min_{x_0, u_0, \dots, x_{\mathrm{H}}, u_{\mathrm{H}}, \theta} \sum_{t=0}^{\mathrm{H}} L_t(x_t, u_t) \\
\text{s.t. } \hspace{5mm} x_{t+1} = f_t(x_t, u_t) \\
u_{t+1} = \pi_{\theta}(x_t)
\end{aligned}
//}
ここで、@<m>$L_t(x_t, u_t)$は最小化したいコスト関数、@<m>$\theta$は方策のパラメータです。
状態方程式、方策の式を等式制約として持つ最小化問題として表されます。
線形のシステムと2次形式の評価関数においては、@<eq>{linear_feedback}のような最適な線形状態フィードバックが解析的に求められますが、
システムが非線形の場合や評価関数の形が2次形式でなければ工夫が必要になってきますし、望ましい性能が出ないことがあります。
//texequation[linear_feedback]{
u = K x
//}
線形のフィードバック以外には例えばNeural Netoworkを用いることが考えられます。
数値的にコスト関数を最小化して重みを求めることになります。

フィードフォワードの方法というのは@<eq>{feedforward_collocation}初期状態、初期入力から、状態方程式に基づいて将来の状態の系列と制御入力の系列を予測する方法です。
//texequation[feedforward_collocation]{
\begin{aligned}
\min_{x_0, u_0, \dots, x_{\mathrm{H}}, u_{\mathrm{H}}} \sum_{t=0}^{\mathrm{H}} L_t(x_t, u_t) \\
\text{s.t. } \hspace{5mm} x_{t+1} = f_t(x_t, u_t) \\
\end{aligned}
//}
状態方程式を等式制約として持つ最小化問題として表されます。
しかし、もし初期状態から終端状態までを予測したとしても、それを実行している間にモデルの誤差などにより実際の動きは変わってくるおそれがあります。
そこで、予測を逐次的に行う@<em>{モデル予測制御}が用いられることがあります。
モデル予測制御では一定期間@<m>$T$までの将来を予測して、一番現在に近い一点の制御入力を実行するのを制御周期ごとに繰り返します。
そのため、状態フィードバックに比べて、モデル予測制御では計算時間の問題がシビアですが、CPUの動作速度が向上したことで、近年研究や応用が盛んになっています。

以降ではモデルを用いて将来を予測するフィードフォワードの考え方をメインに扱っていきます。
つぎの状態方程式を制約とした評価関数の最適化問題が最適制御問題で解きたい問題です。
//texequation{
\begin{aligned}
find x_0 \dots x_N, u_0 \dots u_N \\
where \\
J(x, u) = \sum_{i=0}^{N-1} L_i(x_i, u_i) + L_f(x_N) \\
subject to. \\
x_{i+1} = f_i(x_i, u_i), i = 0, \dots, N-1
\end{aligned}
//}
状態方程式の制約の他に、出力についての制約が必要になってきます。
例えば歩行ロボットであると関節角度や最大トルクであったり、車輪ロボットであると最大速度などが考えられます。
しかし以降では簡単のために状態方程式以外の制約条件を無視します。

@<m>$x, u$を目的変数として最適化問題を解く方法は、direct collocation、collocation methodなどと呼ばれます。
実際に問題を解くには非線形最適化ソルバを使って@<m>$x, u$を求めます。


== 離散システムの最適化問題
このシステムにおいて、つぎの評価関数を最小にしたいと考えます。
//texequation{
J(x, u) = \sum_{i=0}^{N-1} L_i(x_i, u_i) + L_f(x_N)
//}
@<m>$N$は有限の時間ステップ、@<m>$L_i(x_i, x_i)$はstep @<m>$i$におけるコスト関数であり、
@<m>$L_f(X_N)$は終端時刻におけるコストです。
このときに、あるステップ@<m>$i$から終端までのコストを

//texequation{
J_i(x, u) = \sum_{j=i}^{N-1} L_i(x_i, u_i) + L_f(x_N)
//}

とし、$i$から終端までの最適コストを
//texequation{
V_i(x) = \min_u J_i(x, u)
//}
とすると、最適コストは部分最適コストで表現できるため、つぎのように再帰的に表現できます。
//texequation{
\begin{aligned}
V_i(x) &= \min_u [ l(x_i, u_i) + V_{i+1}(x_{i+1}) ] \\
&= \min_u [ l(x_i, u_i) + V_{i+1}(f(x_i, u_i)) ]
\end{aligned}
//}

ここで、@<m>$x_{i+1} = f_i(x_i, u_i)$です。
また、
//texequation{
Q_i(x_i, u_i) = l(x_i, u_i) + V_{i+1}(f(x_i, u_i))
//}
とおくと、

//texequation{
V_i(x) = \min_u Q_i(x_i, u_i)
//}

であるので、最適制御入力は$x_i$の関数
//texequation[control_opt_u]{
u_i^{*}(x_i) = \text{argmin} Q_i(x_i, u_i)
//}
を求めることで実現できます。

以上の離散システムにおいて、最適化を行う流れはつぎのようになります。

 1. 初期軌道 @<m>$x_0, \cdots , x_N$を設定する。これはヒューリスティックに決定する。たとえば、ランダムに設定したり、スタートからゴールまでを結ぶ直線上に配置するなどが考えられる。
 2. 後退パスにおいて最適化する。@<m>$i=N$から@<m>$i=0$に向かって逆方向に@<eq>{control_opt_u}の最適化を行う
 3. 2.で計算した最適制御入力を用いるとして、順方向パスを@<eq>{control_system_eq}に従って計算する。この結果を1.の初期値の代わりに用いて2.以降を繰り返す。

== Shooting法
#@# @<m>$x, u$を設計変数として最適化を行うことはできるが、システム@<m>$f(x_i, u_i)$の最適化を想定しているので、@<m>$u$が分かれば@<m>$x$は得られる関係にある。
システム@<m>$f(x_i, u_i)$の最適化をするときには、@<m>$u$が分かれば@<m>$x$は得られる関係にあるため、@<m>$x, u$について別に扱わなくてもよいです。
そこで、@<m>$J$を@<m>$u$の関数として表し、@<m>$u$について最適化します。
このような方法は、初期値から入力を操作してゴールを目指すことが的当てのように見えることから、Shooting法と呼ばれます。
@<eq>{control_opt_u}において@<m>$u$のみを設計変数として最適化していることから、このあと説明するDDP, iLQR, SLPもShooting法(Single Shooting)の一種ということになります。
//image[shooting][Shooting法][scale=0.5]

== Neuton法を用いた最適化
関数@<m>$Q_i(x, u)$を@<m>$x_i + \delta x$, @<m>$u_i + \delta u$周りで2次のテイラー展開すると、近似した関数@<m>$\bar{Q}(x, u)$は
//texequation{
\begin{aligned}
\bar{Q}(x, u) =
Q +
Q_u \delta u + Q_x \delta x +
\frac{1}{2}
\left[
    \begin{array}{rr}
      \delta u & \delta x
    \end{array}
  \right]
\left[
    \begin{array}{rr}
      Q_{uu} & Q_{ux} \\
      Q_{xu} & Q_{xx}
    \end{array}
  \right]
\left[
    \begin{array}{r}
      \delta u \\
      \delta x
    \end{array}
  \right]
\end{aligned}
//}
です。
ただし、これ以降は@<m>$Q_i(x_i, u_i)$を単に$Q$と、
また、関数@<m>$f(x, y)$の偏微分@<m>$\frac{\partial f}{\partial x}, \frac{\partial f^2}{\partial x^2}, \frac{\partial f^2}{\partial xy}$は@<m>$f_x, f_{xx}, f_{xy}$とする表記を用います。

ここで、
@<m>$\xi = \left[
    \begin{array}{r}
      \delta u \\
      \delta x
    \end{array}
  \right]
$,
@<m>$p = \left[
    \begin{array}{r}
      Q_u \\
      Q_x
    \end{array}
  \right]
$,

@<m>$\mathrm{H} = \left[
    \begin{array}{rr}
      Q\_{uu} & Q\_{ux} \\
      Q\_{xu} & Q\_{xx}
    \end{array}
  \right]
$
とおくと、

//texequation{
\bar{Q}(x, u) = Q
+ p^{\mathrm{T}} \xi
+ \xi^{\mathrm{T}} \mathrm{H} \xi
//}
であり、さらに展開すると
//texequation{
\begin{aligned}
\bar{Q}(x, u)
&= Q
- \frac{1}{2} p^{\mathrm{T}} \mathrm{H}^{-1} p
+ \frac{1}{2} (
\xi^{\mathrm{T}} \mathrm{H} \xi
+ 2 p^{\mathrm{T}} \xi
+ p^{\mathrm{T}} \mathrm{H}^{-1} p) \\
&= Q
- \frac{1}{2} p^{\mathrm{T}} \mathrm{H}^{-1} p
+ \frac{1}{2} (
\xi^{\mathrm{T}} \mathrm{H}
+ p^{\mathrm{T}})
(
\xi + \mathrm{H}^{-1} p) \\
&= Q
- \frac{1}{2} p^{\mathrm{T}} \mathrm{H}^{-1} p
+ \frac{1}{2} (
\xi + \mathrm{H}^{-1} p)
\mathrm{H}
(
\xi + \mathrm{H}^{-1} p) \\
\end{aligned}
//}
なので、
@<m>$\bar{Q}$がもとの関数を十分に近似できていると仮定すると、
//texequation[qsi_eq]{
\xi = - \mathrm{H}^{-1} p
//}
のときに、@<m>$Q$が最小となります。
このように2次の近似を用いて最適解を求める方法は、Newton法にほかなりません。

@<eq>{qsi_eq}をさらに計算すると

//texequation{
\begin{aligned}
\left[
    \begin{array}{r}
      \delta u \\
      \delta x
    \end{array}
  \right]
&=
-\left[
    \begin{array}{cc}
      Q_{uu}^{-1} + Q_{uu}^{-1}Q_{ux}S^{-1}Q_{xu}Q_{uu}^{-1} & -Q_{uu}^{-1}Q_{ux}S^{-1} \\
      -S^{-1}Q_{xu}Q_{uu}^{-1} & S^{-1}
    \end{array}
  \right]
\left[
    \begin{array}{r}
      Q_u \\
      Q_x
    \end{array}
  \right]
\end{aligned}
//}

//texequation{
\begin{aligned}
=-
\left[
    \begin{array}{r}
      Q_{uu}^{-1}Q_u + Q_{uu}^{-1}Q_{ux}[S^{-1}Q_{xu}Q_{uu}^{-1} -S^{-1}Q_x] \\
      -S^{-1}Q_{xu}Q_{uu}^{-1}Q_u + S^{-1}Q_x
    \end{array}
  \right]
\end{aligned}
//}
となります。
これより、最適な制御入力はつぎのようになります。
//texequation[control_opt_u_ddp]{
\delta u^* (\delta x) = \mathrm{k} + \mathrm{K} \delta x
//}
//texequation[control_opt_u_k]{
\mathrm{k} = Q_{uu}^{-1}Q_u , \mathrm{K} = Q_{uu}^{-1}Q_{ux}
//}

ハミルトニアン$H$の要素は
//texequation[control_hamil]{
\begin{aligned}
Q_x &= l_x + f_x^{\mathrm{T}} V'_x \\
Q_u &= l_u + f_u^{\mathrm{T}} V'_x \\
Q_{xx} &= l_{xx} + f_x^{\mathrm{T}} V'_{xx} f_x + V'_x \dot f_{xx} \\
Q_{ux} &= l_{ux} + f_x^{\mathrm{T}} V'_{xx} f_x + V'_x \dot f_{ux} \\
Q_{uu} &= l_{uu} + f_u^{\mathrm{T}} V'_{xx} f_u + V'_x \dot f_{uu} \\
\end{aligned}
//}
であり、
//texequation[control_vs]{
\begin{aligned}
\Delta V &= - \frac{1}{2} \mathrm{k}^{\mathrm{T}} Q_{uu} \mathrm{k} \\
V_x &= Q_x - \mathrm{K}^{\mathrm{T}} Q_{uu} \mathrm{k} \\
V_{xx} &= Q_{xx} - \mathrm{K}^{\mathrm{T}} Q_{uu} \mathrm{K} \\
\end{aligned}
//}
です。

以上の@<eq>{control_opt_u_ddp}, @<eq>{control_opt_u_k}, @<eq>{control_hamil}, @<eq>{control_vs}
を用いて実際に最適制御入力を求めることができるのです。

== Differential Dynamic Programming(DDP)
以上をふまえて、離散システムの逐次最適化における手順2に、@<eq>{control_opt_u_ddp}を用いる方法がDDPです。

 1. 初期軌道 @<m>$x_0, \cdots , x_N$を設定する。
 2. @<m>$i=N$から@<m>$i=0$に向かって、@<eq>{control_opt_u_ddp}における係数を計算する。
 3. 2.で計算した最適制御入力を用いるとして、順方向パスを@<eq>{control_system_eq}に従って計算する。この結果を1.の初期値の代わりに用いて2.以降を繰り返す。

== 線形システムへの近似とQuadratic Cost functionの採用
非線形システム@<m>$f(x, u)$をstep @<m>$i$の状態@<m>$x_k$, 入力@<m>$u_k$において線形化することを考えます。

//texequation[control_linear_system_eq]{
(x - x_k) = \mathrm{A_k} (x - x_k) + \mathrm{B_k} (u - u_k)
//}

ここで、
//texequation{
\begin{aligned}
\mathrm{A_k} = \frac{\partial f_k}{\partial x} (x_k, u_k) \\
\mathrm{B_k} = \frac{\partial f_k}{\partial u} (x_k, u_k)
\end{aligned}
//}

です。
座標系を変換することで時変線形システムとしてモデリングできることがわかります。
線形システムの最適化問題としては、LQR(Linear Quadratic Regrator)がよく知られており、そのコスト関数は

//texequation{
l_k(x_k, u_k) = \frac{1}{2} x^{\mathrm{T}} \mathrm{Q} x +
\frac{1}{2} u^{\mathrm{T}} \mathrm{R} u
//}
です。
この離散システムを最小化する解はつぎのRicatti方程式を解くことで得られることが知られています。

//texequation[control_ricatti_eq]{
S(k) = Q + A_k^{\mathrm{T}} S(k+1) A_k
- A_k^{\mathrm{T}}S(k+1)B_k(R + B_k^{\mathrm{T}}S(k+1)B_k)^{-1}B_k^{\mathrm{T}}S(k+1)A_k
//}

== iterative LQR(iLQR)
@<eq>{control_ricatti_eq}を満たす最適制御入力は@<eq>{control_opt_u_ddp}から@<m>$f$の二階微分の項を除いたものになっています。
DDPがニュートン法を用いた最適化を用いているのに対してiLQRはガウスニュートン法を用いたものであるといえます。
iLQRはDDPのようにヘッシアンを計算する必要も逆行列を計算する必要もないため、DDPに比べて高速に処理できます。

//footnote[fn01][とはいえ制御工学、強化学習、生物学の歴史的経緯、強い関連性を踏まえてバックグラウンドがなにかと評するのは難しいように思います。]
