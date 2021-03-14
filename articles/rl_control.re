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
フィードバック制御を行う方策@<m>$\pi_{\theta}(*)$を求める方法と、
フィードフォワード的に将来にわたる制御入力の系列@<m>$u_0, ... u_H$を求める方法があります。

フィードバック制御を求める最適化問題は@<eq>{policy_direct_collocation}のようになります。
状態方程式、方策の式を等式制約として持つ最小化問題として表されます。
//texequation[policy_direct_collocation]{
\begin{aligned}
\min_{x_0, u_0, \dots, x_H, u_H, \theta} \sum_{t=0}^H l(x_t, u_t) \\
\text{s.t. } \hspace{5mm} x_{t+1} = f_t(x_t, u_t) \\
u_{t+1} = \pi_{\theta}(x_t)
\end{aligned}
//}
ここで、@<m>$l(x_t, u_t)$は最小化したいコスト関数、@<m>$\theta$は方策のパラメータです。
線形のシステムと2次形式の評価関数においては、@<eq>{linear_feedback}のような最適な線形状態フィードバックが解析的に求められますが、
システムが非線形の場合や評価関数の形が2次形式でなければ工夫が必要になってきますし、望ましい性能が出ないことがあります。
//texequation[linear_feedback]{
u = K x
//}
線形のフィードバック以外には例えばNeural Netoworkを用いることが考えられますが、そのパラメータは数値的にコスト関数を最小化して求めることになります。

フィードフォワードの方法というのは@<eq>{direct_collocation_basic}にあるように初期状態、初期入力から、状態方程式に基づいて将来の状態の系列と制御入力の系列を予測する方法です。
状態方程式を等式制約として持つ最小化問題として表されます。
//texequation[direct_collocation_basic]{
\begin{aligned}
\min_{x_0, u_0, \dots, x_H, u_H} \sum_{t=0}^H l(x_t, u_t) \\
\text{s.t. } \hspace{5mm} x_{t+1} = f_t(x_t, u_t) \\
\end{aligned}
//}
しかし、初期状態から終端状態までを予測したとしても、実行している間にモデルの誤差などにより実際の動きは変わってくるおそれがあります。
そこで、予測を逐次的に行う@<em>{モデル予測制御}が用いられることがあります。
モデル予測制御では一定期間@<m>$H$までの将来を予測して、一番現在に近い一点の制御入力を実行するのを制御周期ごとに繰り返します。
そのため、状態フィードバックに比べて、モデル予測制御では計算時間の問題がシビアですが、CPUの動作速度が向上したことで、近年研究や応用が盛んになっています。

@<eq>{policy_direct_collocation}、@<eq>{direct_collocation_basic}のように
@<m>{x, u}を離散化し、決定変数とした最適化問題を数値的に解くことで制御する方法を@<em>{Direct Collocation}法といいます。
一方で、状態方程式を拘束条件でなく、@<eq>{direct_shooting_basic}のようにコスト関数に含め、決定変数を@<m>$u$のみにしたものを@<em>{Direct Shooting}法といいます。
//texequation[direct_shooting_basic]{
\min_{u_0, \dots, u_H} L_0(x_0, u_0) + L_1(f(x_0, u_0), u_1) + \dots
//}
Direct Shooting法は、物理モデル@<m>$f(x_t, u_t)$を前提とした最適化では、一時刻前の@<m>$u$が分かれば@<m>$x$は得られるという関係を利用しています。
そのため、初期状態、初期入力が終端状態まで影響を与えます。
このように初期状態から入力を操作してゴールを目指すことが的当てのように見えることから、Shooting法と呼ばれます。
//image[shooting][Shooting法][scale=0.5]
DDP, iLQR, SLPもDirect Shooting法の一種です。
Direct Shootingには状態変数の次元が非常に大きくて入力の次元が小さい時に決定変数の次元を小さくできるという利点があります。
また物理モデルをコスト関数の中に埋め込んであるため、最適化が収束するまでのどの段階でも物理モデルと矛盾しない入力@<m>$u$が得られるのも利点です。
Direct Collocationでは収束するまでは物理モデルを満たしていない、実際には実行不可能な解が算出されることがあります。

しかし、Direct Shootingでは計算誤差が累積して@<m>$u_0, \dots, u_H$の値がうまく算出できないことがあるという問題があります。
また、入力@<m>$u_0, \dots, u_H$の初期値をどう定めるかというのも問題になります。
Direct Collocationでは初期状態から終端状態までの線形補間など、最適化の収束を助ける事前知識を初期値として与えることができますが、
Direct Shootingではランダムに@<m>$u$を初期化するなどするほかなく、収束させるのが難しいことがあります。

以上では最適化問題の制約は状態方程式だけであるという前提で考えてきましたが、実際にロボットに適用する最適化問題には出力についての制約が必要になってきます。
例えば歩行ロボットであると関節角度や最大トルクであったり、車輪ロボットであると最大速度などが考えられます。
Direct Shooting法のように状態方程式の制約を考慮しなくてもよい定式化をしても、入力についての制約条件は別途考慮する必要があり、さらに収束させるのが難しくなります。

== 動的計画法に基づくShooting法の定式化
@<eq>{direct_shooting_basic}のDirect Shootingの定式化について、つぎのように終端コストを分けた表現に書き換えます。
//texequation{
J(x, u) = \sum_{t=0}^{H-1} l(x_t, u_t) + l_f(x_H)
//}
@<m>$l(x_t, x_t)$は時間ステップ@<m>$t$におけるコスト関数であり、
@<m>$l_f(X_H)$は終端時刻におけるコストです。
このときに、あるステップ@<m>$t$から終端までのコスト(部分コスト)を
//texequation{
J_t(x, u) = \sum_{j=t}^{H-1} l(x_t, u_t) + l_f(x_H)
//}
とし、最適入力@<m>$u_t$が与えられたときの部分最適コストを
//texequation{
V(x, t) = \min_{u_t} J_t(x, u)
//}
とします。
初期状態から終端状態までの最適コストを実現しているならば、その部分コストも最適でなければならないという、@<em>$Bellmanの最適性の原理$によって、
部分最適コストはつぎのように再帰的に表現できます。
//texequation[partial_recursive_cost]{
\begin{aligned}
V(x, t) &= \min_u [ l(x_t, u_t) + V(x_{t+1}, t+1) ] \\
&= \min_{u_t} [ l(x_t, u_t) + V(f(x_t, u_t), t+1) ]
\end{aligned}
//}

また、今後の式展開の都合上、
//texequation[control_q_func]{
Q(x_t, u_t, t) = l(x_t, u_t) + V(f(x_t, u_t), t+1)
//}
とおきます。
つまり、以下を定義します。
//texequation[v_func_def]{
V(x_t, t) = \min_{u_t} Q(x_t, u_t, t)
//}
以上の定義により、時刻@<m>$t$での最適制御入力@<m>$t$は@<m>$x_t$の関数
//texequation[control_opt_u]{
u_t^{*}(x_t) = \text{argmin} Q(x_t, u_t, t)
//}
となることがいえます。

以上のように最適化問題を部分問題に分割して@<em>{動的計画法(Dynamic Programming)}によって解く流れはつぎのようになります。

 1. 初期入力 @<m>$u_0, \cdots , u_H$を設定する。これはたとえばランダムに設定するなどヒューリスティックに決定する。
 2. @<m>$i=H$から@<m>$i=0$に向かって逆方向に@<eq>{control_opt_u}の最適化を行う
 3. 2.で計算した最適制御入力を用いるとして、順方向パスを@<eq>{control_system_eq}に従って計算する。この結果を1.の初期値の代わりに用いて2.以降を繰り返す。

== コスト関数の2次近似とNewton法を用いた最適化
これまで、関数@<m>$Q$の形式は任意の非線形関数であると仮定してきましたが、計算の簡便さのために2次関数に近似することを考えます。
二次計画問題というよく知られた形式の最適化問題に帰着させることが目的です。
そのために関数@<m>$Q(x, u, t)$を@<m>$x_t + \delta x_t$, @<m>$u_t + \delta u_t$周りで2次のテイラー展開すると、近似した関数@<m>$\bar{Q}(x, u, t)$はつぎのようになります。
//texequation[approximated_q_func]{
\begin{aligned}
&\bar{Q}(x_t + \delta x_t, u_t + \delta u_t, t) = \\
&Q(x_t, u_t, t) +
\left[
  \begin{array}{rr}
    Q_u & Q_x
  \end{array}
\right]
\left[
  \begin{array}{r}
    \delta u_t \\
    \delta x_t
  \end{array}
\right] +
\frac{1}{2}
\left[
    \begin{array}{rr}
      \delta u_t^{\mathrm{T}} & \delta x_t^{\mathrm{T}}
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
      \delta u_t \\
      \delta x_t
    \end{array}
  \right]
\end{aligned}
//}
@<m>$Q(x_t, u_t, t)$を単に@<m>$Q$と、
関数@<m>$f(x, y)$の偏微分@<m>$\frac{\partial f}{\partial x}, \frac{\partial f^2}{\partial x^2}, \frac{\partial f^2}{\partial xy}$は@<m>$f_x, f_{xx}, f_{xy}$とする表記を用いています。
ここで、@<eq>{control_q_func}の定義より次が成り立っています。
//texequation[q_func_devs]{
\begin{aligned}
Q_x &= l_x + f^{\mathrm{T}}_x V_x(t + 1) \\
Q_u &= l_u + f^{\mathrm{T}}_u V_x(t + 1) \\
Q_{xx} &= l_{xx} + f^{\mathrm{T}}_x V_{xx}(t + 1) f_x + V_x(t+1) \cdot f_{xx} \\
Q_{uu} &= l_{uu} + f^{\mathrm{T}}_u V_{xx}(t + 1) f_u + V_x(t+1) \cdot f_{uu} \\
Q_{ux} &= l_{ux} + f^{\mathrm{T}}_u V_{xx}(t + 1) f_x + V_x(t+1) \cdot f_{ux}
\end{aligned}
//}
@<eq>{approximated_q_func}より明らかに、@<m>$\bar{Q}$を最小化する解は、つぎのとおりです。
//texequation[qsi_eq]{
\begin{aligned}
\left[
  \begin{array}{r}
    \delta u_t \\
    \delta x_t
  \end{array}
\right]
=
- \mathrm{H}^{-1}
\left[
  \begin{array}{rr}
    Q_u & Q_x
  \end{array}
\right]
\\
\mathrm{H}
=
\left[
  \begin{array}{rr}
    Q_{uu} & Q_{ux} \\
    Q_{xu} & Q_{xx}
  \end{array}
\right]
\end{aligned}
//}
このように2次までの微分項を用いて最適解を求める方法は、Newton法にほかなりません。
@<eq>{qsi_eq}をさらに計算するとつぎのようになります。
//texequation{
\begin{aligned}
\left[
    \begin{array}{r}
      \delta u_t \\
      \delta x_t
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
これより、最適な制御入力はつぎのようになります。
//texequation[control_opt_u_ddp]{
\begin{aligned}
\delta u_t^* (\delta x_t) = \mathrm{k} + \mathrm{K} \delta x \\
\mathrm{k} = -Q_{uu}^{-1}Q_u \\
\mathrm{K} = -Q_{uu}^{-1}Q_{ux}
\end{aligned}
//}

次に@<eq>{qsi_eq}の@<m>$\mathrm{H}$の要素を計算するために@<m>$V$に関する式を求めていきます。
@<eq>{v_func_def}に定義されているように、@<m>$V$は@<m>$Q$の最小値であるので、
最適入力の@<eq>{control_opt_u_ddp}を@<eq>{approximated_q_func}に代入するとつぎのようになります。

//texequation[v_func_result]{
\begin{aligned}
\bar{V}(x_t + \delta x_t, t) =
&Q(x_t, u_t) + Q_u k \\
+
&\frac{1}{2} k^{\mathrm{T}} Q_{uu} k +
(Q_x + Q_u K + k^{\mathrm{T}} Q_{ux} + k^{\mathrm{T}} Q_{uu} K) \delta x_t \\
+
&\frac{1}{2} \delta x^{\mathrm{T}}_t
(Q_{xx} + Q_{xu} K + K^{\mathrm{T}} Q_{ux} + K^{\mathrm{T}} Q_{uu} K) \delta x_t
\end{aligned}
//}

@<eq>{v_func_result}を@<m>$V$の2次近似であるつぎの@<eq>{v_approx_basic}と辺々を比較することで、
@<eq>{v_by_q}、@<eq>{vx_by_q}、@<eq>{vxx_by_q}が求まります。
@<eq>{control_opt_u}を用いて式を整理していく過程はつぎのようになります。

//texequation[v_approx_basic]{
\bar{V}(x_t + \delta x_t, t) =
V(x_t, t) +
V_x \delta x_t +
\frac{1}{2} \delta x^{\mathrm{T}}_t V_{xx} \delta x_t
//}

//texequation[v_by_q]{
\begin{aligned}
V(x_t, t)
&= Q(x_t, u_t, t) + Q_u k + \frac{1}{2} k^{\mathrm{T}} Q_{uu} k \\
&= Q(x_t, u_t, t) - k^{\mathrm{T}} Q_{uu} k + \frac{1}{2} k^{\mathrm{T}} Q_{uu} k \\
&= Q(x_t, u_t, t) -  \frac{1}{2} k^{\mathrm{T}} Q_{uu} k \\
&= Q(x_t, u_t, t) -  \frac{1}{2} Q^{\mathrm{T}}_u Q^{-1}_{uu} Q_u
\end{aligned}
//}

//texequation[vx_by_q]{
\begin{aligned}
V_x(x_t, t)
&= Q_x + Q_u K + k^{\mathrm{T}} Q_{ux} + k^{\mathrm{T}} Q_{uu} K \\
&= Q_x - k^{\mathrm{T}} Q_{uu} K - k^{\mathrm{T}} Q_{uu} K + k^{\mathrm{T}} Q_{uu} K \\
&= Q_x - K^{\mathrm{T}} Q_{uu} k \\
&= Q_x - Q_{xu} Q^{-1}_{uu} Q_u
\end{aligned}
//}

//texequation[vxx_by_q]{
\begin{aligned}
V_{xx}(x_t, t)
&= Q_{xx} + Q_{xu} K + K^{\mathrm{T}} Q_{ux} + K^{\mathrm{T}} Q_{uu} K \\
&= Q_{xx} - K^{\mathrm{T}} Q_{uu} K + K^{\mathrm{T}} Q_{ux} - K^{\mathrm{T}} Q_{ux} \\
&= Q_{xx} - K^{\mathrm{T}} Q_{uu} K \\
&= Q_{xx} - Q_{xu} Q^{-1}_{uu} Q_{ux}
\end{aligned}
//}

== Differential Dynamic Programming(DDP)
ここまでに導出してきた式を使って、以下の手順を行うのがDDPです。

 * 1. 初期化
 **  入力 @<m>$u_0, \cdots , u_H$をランダムに設定する。

 * 2. backward step (@<m>$t=H$から@<m>$t=0$に向かって計算を進める)
 ** @<eq>{control_opt_u_ddp}を使って制御入力を計算する。
 ** 次ステップ@<m>$t$の計算に用いるために@<eq>{vx_by_q}、@<eq>{vxx_by_q}を計算する。

 * 3. forward step (@<m>$t=0$から@<m>$t=H$に向かって計算を進める)
 ** backward stepで計算した制御入力を用いるとして、状態@<m>$x_t$を@<eq>{control_system_eq}に従って計算する。

2, 3を繰り返して@<m>$V(x_t, t)$の値が収束するまで計算します。

== iterative LQR(iLQR)
DDPの計算手順の@<eq>{q_func_devs}において、@<m>$f$の2次の微分項を削除した@<eq>{q_func_devs_ilqr}を用いるのがiterative LQR(iLQR)です。
//texequation[q_func_devs_ilqr]{
\begin{aligned}
Q_x &= l_x + f^{\mathrm{T}}_x V_x(t + 1) \\
Q_u &= l_u + f^{\mathrm{T}}_u V_x(t + 1) \\
Q_{xx} &= l_{xx} + f^{\mathrm{T}}_x V_{xx}(t + 1) f_x \\
Q_{uu} &= l_{uu} + f^{\mathrm{T}}_u V_{xx}(t + 1) f_u \\
Q_{ux} &= l_{ux} + f^{\mathrm{T}}_u V_{xx}(t + 1) f_x
\end{aligned}
//}

DDPにおいて関数@<m>$V(x_t, t)$を2次までに近似したうえで、@<m>$f$の2次の微分項を削除するということは、
状態方程式を1次の線形の方程式で表し、部分コスト関数を2次の式で表すということと等価です。
ですからiLQRは@<eq>{control_linear_system_eq}の線形状態方程式に逐次的にシステムを近似し、
2次の部分コスト関数@<eq>{linear_cost_func}、@<eq>{v_quadratic}において最適化する方法になります。

//texequation[control_linear_system_eq]{
\delta x_{t+1} = A_t \delta x_t + B_t \delta u_t
//}

//texequation[linear_cost_func]{
\begin{aligned}
&\bar{c}(x_t + \delta x_t, u_t + \delta u_t, t) = \\
&c(x_t, u_t, t) +
\left[
  \begin{array}{rr}
    w_t & r_t
  \end{array}
\right]
\left[
  \begin{array}{r}
    \delta u_t \\
    \delta x_t
  \end{array}
\right] +
\frac{1}{2}
\left[
    \begin{array}{rr}
      \delta u_t^{\mathrm{T}} & \delta x_t^{\mathrm{T}}
    \end{array}
  \right]
\left[
    \begin{array}{rr}
      R_t & P_t \\
      P^{\mathrm{T}}_t & W_t
    \end{array}
  \right]
\left[
    \begin{array}{r}
      \delta u_t \\
      \delta x_t
    \end{array}
  \right]
\end{aligned}
//}

//texequation[v_quadratic]{
\bar{V}(x_t + \delta x_t, t) =
s(t) +
\bm{s}(t) \delta x_t +
\frac{1}{2} \delta x^{\mathrm{T}}_t S(t) \delta x_t
//}

ここでは次の1次の線形近似をしています。
//texequation{
\begin{aligned}
A_t = \frac{\partial f}{\partial x} (x_t, u_t) \\
B_t = \frac{\partial f}{\partial u} (x_t, u_t)
\end{aligned}
//}

よって、最適制御入力は@<eq>{control_opt_u_ddp}の記号を置き換えることによって@<eq>{control_opt_u_ilqr}と表現できます。

//texequation[control_opt_u_ilqr]{
\begin{aligned}
&\delta u_t^* (\delta x_t) =
-(R_t + B^{\mathrm{T}}_t S(i+1) B_t)^{-1}
(r_t + B^{\mathrm{T}}_t \bm{s}(t+1) + P_t + B^{\mathrm{T}}_t S(t+1) A_t) \delta x_t \\
&S(t) =
W_t + A^{\mathrm{T}}_t S(i+1) A_t
-(P_t + B^{\mathrm{T}}_t S(i+1) A_t)^{\mathrm{T}}
(R_t + B^{\mathrm{T}}_t S(i+1) B_t)^{-1}
(P_t + B^{\mathrm{T}}_t S(i+1) A_t)
\end{aligned}
//}

@<eq>{control_opt_u_ilqr}をDDPのbackward stepの計算の代わりに用いるのが、iLQRということになります。

ちなみに、コスト関数が2次の項のみで、@<m>$u, x$の交差項が無いとしたとき、つまり
@<m>$w_t = r_t = P_t = 0$, @<m>$s(t) = \bm{s}(t) = 0$とみなすと、最適制御入力は@<eq>{control_opt_u_simple_ricatti}のようになります。

//texequation[control_opt_u_simple_ricatti]{
\begin{aligned}
&\delta u_t^* (\delta x_t) =
-(R_t + B^{\mathrm{T}}_t S(i+1) B_t)^{-1}
B^{\mathrm{T}}_t S(t+1) A_t \delta x_t \\
&S(t) =
W_t + A^{\mathrm{T}}_t S(i+1) A_t
-A^{\mathrm{T}}_t S(i+1) B_t
(R_t + B^{\mathrm{T}}_t S(i+1) B_t)^{-1}
B^{\mathrm{T}}_t S(i+1) A_t
\end{aligned}
//}

これはLQRに対する離散システムのRicatti方程式としてよく知られています。

== Sequential LQR(SLQ)
SLQ は iLQR のforwad stepの計算手順において、非線形の状態方程式の代わりに線形化した@<eq>{control_linear_system_eq}を用いることに相当します。

//footnote[fn01][とはいえ制御工学、強化学習、生物学の歴史的経緯、強い関連性を踏まえてバックグラウンドがなにかと評するのは難しいように思います。]
