== 現代制御とのつながり
=== 記号について
制御工学では@<m>$x, u$が使われるぽよー。
どっちの畑の人なんだなーと思えばよい。

=== 非線形最適制御
最適制御の手法である、Differential Dynamic Programming(DDP), iterative LQR(iLQR), Sequential LQR(SLQ) がたまに強化学習の論文に出てくるが、どういう文脈で生まれたものなのか、その関係性はどうなっているのかわからないので私は苦しめられてきた。
以下の資料を参考にして理解を深めていったので、分かる範囲のことをまとめたい。

=== 離散システムの最適化問題
以下の離散システムを考える。
//texequation{
x_{i+1} = f_i(x_i, u_i), i = 0, \dots, N-1
//}
ここで、@<m>$f_i$は何らかの関数で、線形とは限らない。

このシステムにおいて、以下の評価関数を最小にしたいと考える。
//texequation{
J(x, u) = \sum_{i=0}^{N-1} L_i(x_i, u_i) + L_f(x_N)
//}
@<m>$N$は有限の時間ステップ、@<m>$L_i(x_i, x_i)$はstep @<m>$i$におけるコスト関数であり、
@<m>$L_f(X_N)$は終端時刻におけるコストである。
このときに、あるステップ@<m>$i$から終端までのコストを

//texequation{
J_i(x, u) = \sum_{j=i}^{N-1} L_i(x_i, u_i) + L_f(x_N)
//}

とし、$i$から終端までの最適コストを
//texequation{
V_i(x) = \min_u J_i(x, u)
//}
とすると、最適コストは部分最適コストで表現できるため、以下のように再帰的に表現できる。
//texequation{
\begin{aligned}
V_i(x) &= \min_u [ l(x_i, u_i) + V_{i+1}(x_{i+1}) ] \\
&= \min_u [ l(x_i, u_i) + V_{i+1}(f(x_i, u_i)) ]
\end{aligned}
//}

ここで、システム表現@<m>$x_{i+1} = f_i(x_i, u_i)$を用いている。
また、
//texequation{
Q_i(x_i, u_i) = l(x_i, u_i) + V_{i+1}(f(x_i, u_i))
//}
とおくと、

//texequation{
V_i(x) = \min_u Q_i(x_i, u_i)
//}

であるから、最適制御入力は$x_i$の関数
//texequation{
u_i^{*}(x_i) = \text{argmin} Q_i(x_i, u_i)
//}
を求めることで実現できる。

以上の離散システムにおいて、最適化を行う流れは以下のようになる

1. 初期軌道 @<m>$x_0, \cdots , x_N$を設定する。これはヒューリスティックに決定する。例えば、ランダムに設定したり、スタートからゴールまでを結ぶ直線上に配置するなどが考えられる。
2. 後退パスにおいて最適化する。@<m>$i=N$から@<m>$i=0$に向かって逆方向に式の最適化を行う
3. 2.で計算した最適制御入力を用いるとして、順方向パスを式に従って計算する。この結果を1.の初期値の代わりに用いて2.以降を繰り返す。

=== Shooting法
@<m>$x, u$を設計変数として最適化を行うことはできるが、システム@<m>$f(x_i, u_i)$の最適化を想定しているので、@<m>$u$が分かれば@<m>$x$は得られる関係にある。
そこで、@<m>$J$を@<m>$u$の関数として表し、@<m>$u$について最適化している。
このような方法は、初期値から入力を操作してゴールを目指すことが的当てのように見えることから、Shooting法と呼ばれる。
式において@<m>$u$のみを設計変数として最適化していることから、このあと述べるDDP, iLQR, SLPもShooting法(Single Shooting)の一種ということになる。

=== Neuton法を用いた最適化
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
となる。
ただし、以下では@<m>$Q_i(x_i, u_i)$を単に$Q$と、
また、関数@<m>$f(x, y)$の偏微分@<m>$\frac{\partial f}{\partial x}, \frac{\partial f^2}{\partial x^2}, \frac{\partial f^2}{\partial xy}$は@<m>$f_x, f_{xx}, f_{xy}$とする表記を用いる。

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
であるから、
@<m>$\bar{Q}$がもとの関数を十分に近似できていると仮定したとき、
//texequation{
\xi = - \mathrm{H}^{-1} p
//}
のときに、@<m>$Q$が最小となる。
このように2次の近似を用いて最適解を求める方法は、Newton法にほかならない。

式をさらに計算すると

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
となる。
これより、最適制御入力は以下で与えられる。
//texequation{
\delta u^* (\delta x) = \mathrm{k} + \mathrm{K} \delta x
//}
//texequation{
\mathrm{k} = Q_{uu}^{-1}Q_u , \mathrm{K} = Q_{uu}^{-1}Q_{ux}
//}

ハミルトニアン$H$の要素は
//texequation{
\begin{aligned}
Q_x &= l_x + f_x^{\mathrm{T}} V'_x \\
Q_u &= l_u + f_u^{\mathrm{T}} V'_x \\
Q_{xx} &= l_{xx} + f_x^{\mathrm{T}} V'_{xx} f_x + V'_x \dot f_{xx} \\
Q_{ux} &= l_{ux} + f_x^{\mathrm{T}} V'_{xx} f_x + V'_x \dot f_{ux} \\
Q_{uu} &= l_{uu} + f_u^{\mathrm{T}} V'_{xx} f_u + V'_x \dot f_{uu} \\
\end{aligned}
//}
であり、
//texequation{
\begin{aligned}
\Delta V &= - \frac{1}{2} \mathrm{k}^{\mathrm{T}} Q_{uu} \mathrm{k} \\
V_x &= Q_x - \mathrm{K}^{\mathrm{T}} Q_{uu} \mathrm{k} \\
V_{xx} &= Q_{xx} - \mathrm{K}^{\mathrm{T}} Q_{uu} \mathrm{K} \\
\end{aligned}
//}

以上の式, , , を用いて実際に最適制御入力を求めることができる。

=== Differential Dynamic Programming(DDP)
以上をふまえて、離散システムの逐次最適化における手順2に、式を用いる、
すなわち、

1. 初期軌道 @<m>$x_0, \cdots , x_N$を設定する。
2. @<m>$i=N$から@<m>$i=0$に向かって、式における係数を計算する。
3. 2.で計算した最適制御入力を用いるとして、順方向パスを式に従って計算する。この結果を1.の初期値の代わりに用いて2.以降を繰り返す。

の手順を踏む手法が、DDPである。

# 線形システムへの近似とQuadratic Cost functionの採用
非線形システム@<m>$f(x, u)$をstep @<m>$i$の状態@<m>$x_k$, 入力@<m>$u_k$において線形化することを考える

//texequation{
(x - x_k) = \mathrm{A_k} (x - x_k) + \mathrm{B_k} (u - u_k)
//}

ここで、
//texequation{
\begin{aligned}
\mathrm{A_k} = \frac{\partial f_k}{\partial x} (x_k, u_k) \\
\mathrm{B_k} = \frac{\partial f_k}{\partial u} (x_k, u_k)
\end{aligned}
//}

である。
座標系を変換することで時変線形システムとしてモデリングできることがわかる。
線形システムの最適化問題としては、LQR(Linear Quadratic Regrator)がよく知られており、そのコスト関数は

//texequation{
l_k(x_k, u_k) = \frac{1}{2} x^{\mathrm{T}} \mathrm{Q} x +
\frac{1}{2} u^{\mathrm{T}} \mathrm{R} u
//}

この離散システムを最小化する解は以下のRicatti方程式を解くことで得られることが知られている。

//texequation{
S(k) = Q + A_k^{\mathrm{T}} S(k+1) A_k
- A_k^{\mathrm{T}}S(k+1)B_k(R + B_k^{\mathrm{T}}S(k+1)B_k)^{-1}B_k^{\mathrm{T}}S(k+1)A_k
//}

=== iterative LQR(iLQR)
式を満たす最適制御入力は式から@<m>$f$の二階微分の項を除いたものになっている。
このため、DDPがニュートン法であったのに対してiLQRはガウスニュートン法を用いたものであるとも言える。
iLQRはDDPのようにヘッシアンを計算する必要も逆行列を計算する必要もないため、DDPに比べて高速に処理できる。
一方で、DDPよりも収束は遅いため、iteration数は大きくなるようだ。

=== Sequential LQR(SLQ)
SLQはiLQRに対して、離散システムの逐次最適化における手順3で、式を用いたものである。