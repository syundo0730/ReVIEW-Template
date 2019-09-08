= 確定的なパスごとの勾配を用いた学習
何らかの関数@<m>$F$があるときに、その期待値の勾配@<m>$\nabla_{\theta} \mathbb{E}[F]$を求めたいとします。

後者の場合、後述するre-parametrization trickを使って

//texequation{
\nabla_{\theta} \mathbb{E}_{z \sim \mathcal{N}(0, 1)} [f(x(z, \theta))] 
= \mathbb{E}_z [\nabla_{\theta} f(x(z, \theta))]
//}
を扱います。

Q-Learningでは、値を求めてその最急方向を得たいがために離散行動の中からQ functionの@<m>$\arg\max$を計算していましたが、
Q functionの微分を取ることによって、連続行動の問題についても扱えるようにしたのがpathwise derivative methodといえます。
行動と方策の更新を必ずしも同時に行う必要がないため、方策オフの手法です。

== Stocastic Value Gradient(SVG)
ベルマン方程式において、終端状態を表現するためだけに時変の報酬@<m>$r^{(t)} = r(s^{(t)}, a^{(t)}, t)$を用いることを想定する
(一連の動作のうち最後の状態遷移で報酬@<m>$r^{(t)}$が得られるイメージ)。
このとき、連続空間におけるベルマン方程式はつぎのようになります。
時間を表すために上付き文字を用いています。

//texequation[svg_cont_belman_eq]{
V^{(t)}(s) = \int [ r^{(t)} + \gamma \int V^{(t+1)} (s')p(s'|s, a)ds' ] p(a|s;\theta) da
//}

=== re-parametrization trick
//texequation{
\begin{aligned}
a = \pi(s, \eta; \theta) = \mu(s; \theta) + \sigma(s; \theta) \eta \\
s' = f(s, a, \xi) = \mu(s, a) + \sigma(s, a) \xi \\
\eta = \xi = \mathcal{N}(0, 1)
\end{aligned}
//}
確率分布を使って、ある値が生成されるというモデルを利用します。
確率分布@<m>$\rho$から生成される確率変数@<m>$\eta \sim \rho(\eta)$、@<m>$\xi \sim \rho(\xi)$を使って、
遷移確率@<m>$f$、方策@<m>$\pi$のパラメータの変更をします。
これによって、確率的な振る舞いを入れ込みながら、変数自体は決定論的に扱うことができるのです。
これと、@<eq>{svg_cont_belman_eq}より

//texequation[svg_v_func]{
V(s) = \mathbb{E}_{\rho(\eta)} \left[r(s, \pi(s, \eta; \theta)) 
+ \gamma \mathbb{E}_{\rho(\xi)} 
[V'(f(s, \pi(s, \eta; \theta), \xi)) ]
\right]
//}

累積報酬の最大化をするために、累積報酬の期待値は状態価値関数@<m>$V$なので、その勾配を求めます。

詳細は[Hessら](https://arxiv.org/abs/1510.09142)のAppendix Aを参照していただくとして、
パラメタ@<m>$\theta$についての勾配@<m>$V_\theta$は@<eq>{svg_v_func}より、つぎのようになります。

//texequation{
\begin{aligned}
V_s &= \mathbb{E}_{\rho(\eta)} [
r_s +
r_a \pi_s +
\gamma \mathbb{E}_{\rho(\xi)}
V'_{s'}(f_s + f_a \pi_s)
] \\
V_{\theta} &= \mathbb{E}_{\rho(\eta)} [
r_a \pi_{\theta} +
\gamma \mathbb{E}_{\rho(\xi)}
[
V'_{s'} f_a \pi_{\theta} +
V'_{\theta}
]
]
\end{aligned}
//}

@<eq>{svg_v_func}の期待値の計算は、実際には確率変数@<m>$\eta, \xi$に基づくサンプリングをするモンテカルロ法を使って行います。
いちサンプルを@<m>$v_s, v_{\theta}$で表すと

//texequation[svg_v_func_monte]{
\begin{aligned}
v_s &= [
r_s +
r_a \pi_s +
\gamma v'_{s'}(f_s + f_a \pi_s)
] \|_{\eta, \xi'} \\
v_{\theta} &= [
r_a \pi_{\theta} +
\gamma v'_{s'} f_a \pi_{\theta} +
v'_{\theta}
] \|_{\eta, \xi'}
\end{aligned}
//}

です。

== SVG(@<m>$\infty$)
時間ステップ@<m>$T$について、@<eq>{svg_v_func_monte}を用いて逆向きに計算し、@<m>$v^{(0)}_{\theta}$を実際に用いるのがSVG(@<m>$\infty$)です。
SVG(@<m>$\infty$)のアルゴリズムは、つぎのようになります。
<img src="/images/2018/07/svg_inf.png" width="400px">

 1. @<m>$t=0 \dots T$の順方向パスにおいて、現在の方策@<m>$\pi$を実行し、@<m>$(s,a,r,s')$のペアを@<m>$\mathcal{D}$に記録する。
 2. @<m>$\mathcal{D}$から近似モデル@<m>$\hat{f}$を推定する。モデルとしては微分可能であれば、何でも使うことができる。論文ではNeural Networkを用いている。
 3. @<m>$t=T+1$での初期値として@<m>$v's_s=0, v'_{\theta}=0$とする
 4. @<m>$t=T \dots 0$について、@<m>$\xi, \eta$の推定、@<eq>{svg_v_func_monte}の計算を時間ステップの逆方向にしていく。@<m>$\xi, \eta$は@<m>$\mathcal{D}$における同じ時刻@<m>$t$の情報を用いて推定できる。
 5. @<m>$v^{(0)}_{\theta}$を用いてパラメタ@<m>$\theta$を更新する

以上の1.-5.を収束するまで繰り返し計算します。
ちなみに、SVG(@<m>$\infty$)の@<m>$\infty$はモデル@<m>$f$を使ってどれだけの時間ステップ軌道を計算しているかということを表しています。

== SVG(1)
SVG(1)では、SVG(@<m>$\infty$)とは違い、@<m>$\mathcal{D}$から価値関数@<m>$V$を学習します。
これによって、SVG(@<m>$\infty$)にあった@<m>$v_s$の時間方向の更新は必要なくなります。
また、@<m>$v_s$の更新はモデル@<m>$f$を用いて1時間ステップのみ行われます。
さらに、重点サンプリングを用いて@<m>$v_{\theta}$を更新します。
この重点サンプリングを用いる方法を論文中では特にSVG(1)-ER(Experience Replay)と呼んでおり、
論文中では$SVG(1)-ER$が最も性能が高いと報告しています。

SVG(1)のアルゴリズムをつぎに引用します。
<img src="/images/2018/07/svg_1_er.png" width="400px">

価値関数@<m>$\hat{V}$の推定にはFitted Policy Evaluationという手法が用いられています。
Fitted Policy Evaluationのアルゴリズムはつぎのようになります。
<img src="/images/2018/07/fitted_policy_eval.png" width="400px">

== SVG(0) と DPG/DDPG
SVG(0)はSVG(1)と似通った手法ですが、モデル$f$を用いない、モデルフリーな手法です。
そのかわりに、@<m>$V$でなく行動価値関数@<m>$Q$を学習します。
SVG(0)のアルゴリズムをつぎに引用します。
<img src="/images/2018/07/svg_0.png" width="400px">

SVG(0)はDeterministic Policy Gradient(DPG)あるいは特にActor, CriticにDeep Neural Networkを用いる
Deep Deterministic Policy Gradient(DDPG)とほとんど同じ手法となっています。
DPGとの違いは、SVG(0)では方策@<m>$\pi$を確率的なものとして扱っているので、

@<m>$\pi$の確率変数$\eta$を推定し、
@<m>$\mathbb{E}_{\rho(\eta)} \left[
  Q_a \pi_{\theta} | \eta
\right]$
を求めるということをやっています。
これは実際には上記のアルゴリズムにあるようにモンテカルロ法で計算されます。

== シミュレーション
プリメイドAI環境でSVGを使った学習を走らせてみます。

//cmd{
$ python run_svg.py --cuda 0
//}

CUDA環境がない状態で学習を行うと大変計算時間がかかってしまうため、pyTorchのセットアップを行ったうえで実行されることをお勧めします。