== pathwise derivativeを用いる方法
方策勾配を求めるために価値関数の微分を計算する手法(pathwise derivative metho)について扱う。

何らかの関数$F$があるときに、その期待値の勾配@<m>$\nabla_{\theta} \mathbb{E}[F]$を求めたいとする。
likelihood ratio methodとpathwise derivative methodの違いは、パラメタ@<m>$\theta$が期待値にかかってくる確率分布に設定されているか、モデル@<m>$F$に存在すると考えるか、ということだといえる。

つまり、前者の場合は、

//texequation{
\nabla_{\theta} \mathbb{E}_{x \sim p(\cdot | \theta)} [f(x)] 
= \mathbb{E}_x [\nabla_{\theta} \log p(x; \theta) f(x)]
//}
を計算する。

後者の場合、後述するre-parametrization trickを使って

//texequation{
\nabla_{\theta} \mathbb{E}_{z \sim \mathcal{N}(0, 1)} [f(x(z, \theta))] 
= \mathbb{E}_z [\nabla_{\theta} f(x(z, \theta))]
//}
を扱う。
ここで、@<m>$f$は連続で微分可能であるものとする。
そうでなければ、期待値計算の中に微分の操作を入れるような入れ替えを行うことはできない。

Q-Learningでは、値を求めてその最急方向を得たいがために離散行動の中からQ functionの@<m>$\arg\max$を計算していたが、
Q functionの微分を取ることによって、連続行動の問題についても扱えるようにしたのがpathwise derivative methodである。
よって、Q-Learningとよく関連している手法である。
また、行動と方策の更新が必ずしも同期する必要が無いため、方策オフの手法である。

=== Stocastic Value Gradient(SVG)
ベルマン方程式において、終端状態を表現するためだけに時変の報酬@<m>$r^{(t)} = r(s^{(t)}, a^{(t)}, t)$を用いることを想定する(一連の動作のうち最後の状態遷移で報酬$r^{(t)}$が得られるイメージ)。
このとき、連続空間におけるベルマン方程式は以下のようになる。

//texequation{
V^{(t)}(s) = \int [ r^{(t)} + \gamma \int V^{(t+1)} (s')p(s'|s, a)ds' ] p(a|s;\theta) da
//}

ここで、時間を表すために今までの記事のように下付き文字を使わずに、一貫性が無い上付き文字を用いているが、これは元論文の中で下付き文字はその変数での微分を表すために用いられているからである。
なかなか累乗とわかりにくいので、悪あがきかもしれないが、時間については括弧で囲んでみた(汗)。

=== re-parametrization trick とベルマン方程式
何らかの確率分布@<m>$\rho$から生成される確率変数@<m>$\eta \sim \rho(\eta)$、@<m>$\xi \sim \rho(\xi)$を使って、遷移確率@<m>$f$、方策@<m>$\pi$をre-parametrizationする。
具体的には
//texequation{
\begin{aligned}
a = \pi(s, \eta; \theta) = \mu(s; \theta) + \sigma(s; \theta) \eta \\
s' = f(s, a, \xi) = \mu(s, a) + \sigma(s, a) \xi \\
\eta = \xi = \mathcal{N}(0, 1)
\end{aligned}
//}
のように、確率分布を使って、ある値が生成されるというモデルを利用する。
これによって、確率的な振る舞いを入れ込みながら、変数自体は決定論的に扱うことができる。
これと、式より

//texequation{
V(s) = \mathbb{E}_{\rho(\eta)} \left[r(s, \pi(s, \eta; \theta)) 
+ \gamma \mathbb{E}_{\rho(\xi)} 
[V'(f(s, \pi(s, \eta; \theta), \xi)) ]
\right]
//}

累積報酬の最大化をするために、累積報酬のパラメタについての勾配を求めたい。
累積報酬の期待値は状態価値関数@<m>$V$なので、その勾配を求めればよい。

詳細は[Hessら](https://arxiv.org/abs/1510.09142)のAppendix Aを参照していただくとして、パラメタ@<m>$\theta$についての勾配@<m>$V_\theta$は式より、

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

式の期待値の計算は、確率変数@<m>$\eta, \xi$に基づくサンプリングをするモンテカルロ法を使って行う。
いちサンプルを@<m>$v_s, v_{\theta}$で表すと

//texequation{
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

となる。

=== SVG(@<m>$\infty$)
時間ステップ@<m>$T$について、式を用いて逆向きに計算し、@<m>$v^{(0)}_{\theta}$を実際に用いるのがSVG(@<m>$\infty$)である。
SVG(@<m>$\infty$)のアルゴリズムを見ると、以下のようになっている。
<img src="/images/2018/07/svg_inf.png" width="400px">
つまり、

1. @<m>$t=0 \dots T$の順方向パスにおいて、現在の方策@<m>$\pi$を実行し、@<m>$(s,a,r,s')$のペアを@<m>$\mathcal{D}$に記録する。
2. @<m>$\mathcal{D}$から近似モデル@<m>$\hat{f}$を推定する。モデルとしては微分可能であれば、何でも使うことができる。論文ではNeural Networkを用いている。
3. @<m>$t=T+1$での初期値として@<m>$v's_s=0, v'_{\theta}=0$とする
4. @<m>$t=T \dots 0$について、@<m>$\xi, \eta$の推定、式の計算を時間ステップの逆方向にしていく。@<m>$\xi, \eta$は@<m>$\mathcal{D}$における同じ時刻@<m>$t$の情報を用いて推定できる。
5. @<m>$v^{(0)}_{\theta}$を用いてパラメタ@<m>$\theta$を更新する

以上の1. - 5.を収束するまで繰り返し計算する。

ちなみに、SVG(@<m>$\infty$)の@<m>$\infty$はモデル@<m>$f$を使ってどれだけの時間ステップ軌道を計算しているかということを表している。
そうすると、@<m>$\infty$でなく@<m>$T$な気がするが、私はそのあたりのニュアンスはよく理解できていない。

=== SVG(1)
SVG(1)では、SVG(@<m>$\infty$)とは違い、@<m>$\mathcal{D}$から価値関数@<m>$V$を学習する。
これによって、SVG(@<m>$\infty$)にあった@<m>$v_s$の時間方向の更新は必要なくなる。
また、@<m>$v_s$の更新はモデル@<m>$f$を用いて1時間ステップのみ行われる。
さらに、重点サンプリングを用いて@<m>$v_{\theta}$を更新する。
この重点サンプリングを用いる方法を論文中では特にSVG(1)-ER(Experience Replay)と呼んでいる。
論文中では$SVG(1)-ER$が最も性能が高いと報告している。

SVG(1)のアルゴリズムを以下に引用する。
<img src="/images/2018/07/svg_1_er.png" width="400px">

価値関数@<m>$\hat{V}$の推定にはFitted Policy Evaluationが用いられている。
<img src="/images/2018/07/fitted_policy_eval.png" width="400px">

=== SVG(0) と DPG/DDPG
SVG(0)はSVG(1)と似通った手法だが、モデル$f$を用いない、モデルフリーな手法である。
そのかわりに、$V$でなく行動価値関数$Q$を学習する。
<img src="/images/2018/07/svg_0.png" width="400px">

SVG(0)はDeterministic Policy Gradient(DPG)あるいは特にActor, CriticにDeep Neural Networkを用いるDeep Deterministic Policy Gradient(DDPG)とほとんど同じ手法となっている。
DPGとの違いは、SVG(0)では方策$\pi$を確率的なものとして扱っているので、

@<m>$\pi$の確率変数$\eta$を推定し、
@<m>$\mathbb{E}_{\rho(\eta)} \left[
  Q_a \pi_{\theta} | \eta
\right]$
を求める。
これは実際には上記のアルゴリズムにあるようにモンテカルロ法で計算される。