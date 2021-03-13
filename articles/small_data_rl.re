= データ効率を高めるRL
プリメイドAIの歩行をモデルフリーの手法で学習させた結果、とても歩いているように見えない動きが学習されてしまいました。
そのうえ、学習イテレーション数も膨大になってしまいました。
学習した方策を実機で実現することを考えると、現実的でない方策が学習されてしまう、
シミュレーションの結果をベースにして実機で学習を行うとしても学習イテレーションが膨大である、という問題にぶつかってしまいます。
実機を複数台使って並列に学習をすすめる手法も提案されていますが、筆者が知る限りアームロボットでの報告に限られています。
ヒューマノイドロボットが転倒しない方策をモデルフリー教師なしの方法で学習するのは難しいといえるでしょう。
しかし、歩行には一定周期で繰り返す、左右の足を交互に出すなどのある程度のパターンがあるため、
そういった要素を学習に盛り込むことができればもっとうまく学習できるのではないかと考えられます。

この章では、人力で作ったサンプル、エージェント自身の試行錯誤でモデルを獲得する手法や、
人力で作った教師を模倣する方法や、人間の事前知識を入れ込む方法によって、高いサンプル効率、少ない試行回数で学習を行う手法を扱います。

== モデルベースRL
モデルフリーの学習手法では、MDPにおける状態遷移確率は未知のものとして扱っていました。
モデルベース強化学習は、状態遷移確率つまりモデルを陽に定義して学習したうえで、モデルを使って最適な入力を求めます。
現在の状態@<m>$s_t$と行動@<m>$a_t$を取って、次の状態@<m>$s_{t+1}$を生成する関数を@<m>$f_{\phi}(s_t, a_t)$とします。
モデルベース学習で用いられるアルゴリズムはつぎのような構成になります。
学習したいタスクの一連の系列をイテレーションとして、そのindexを@<m>$k$としています。

 1. サンプル@<m>$\mathcal{D}=\left\{(s, a, s')_i \right\{$を集めるために、何らかの方策@<m>$\pi_0(a_t|s_t)$(例えばランダム)を実行する
 1. @<m>$for$ @<m>$k=1,2,\dots$ @<m>$do$
 1. @<m>$f_{\phi}(s_t, a_t)$ を学習する (@<m>$\sum_i \| f_{\phi}(s_t, a_t)-s'_i\|^2$ を最小化)
 1. @<m>$f_{\phi}(s_t, a_t)$ を使って@<m>$\pi_{\theta}(a_t | s_t)$を最適化する
 1. @<m>$\pi_{\theta}(a_t | s_t)$実行して、タプル@<m>$(s, a, s')$を@<m>$\mathcal{D}$に記録する

モデルベース学習手法の一種である、@<em>{Guided Policy Search}では、ガウス過程を仮定して
@<m>$f_{\phi}(s_t, a_t)$モデルを学習し、得られたモデルを使ってiLQRにより最適入力を決定します。
その上で、得られた状態と最適入力のペアからNeural Networkを学習します。
モデルを使い、しかも最適な行動を決定する仕組みをNeural Networkに転写できているために、
画像inputから各関節への制御入力を生成するような、難易度の高い@<em>{End to End}方策を学習できます。

== 逆強化学習
経験データから報酬を推定し、推定した報酬を最大化する学習を行うのが逆強化学習の枠組みです。
画像を生成するネットワークであるGANと関連が高いことが知られており、Neural Networkを用いた逆強化学習の取り組みがされています。
machinaで実装されているアルゴリズムにはGAILやAIRLがあります。

== 方策に事前知識を持たせる
人間の事前知識を反映して、方策の構造を決定することも学習効率を上げることに効果的です。
たとえば、ヒューマノイドの右脚、左脚等の左右対称の部分に該当する行動価値関数を左右対称にするということが行われます。
また、深度画像を入力として、障害物を避けるような高次元の入力を扱うシステムでは、
目標位置に向う歩行制御を行うコントローラと、画像から目標位置を決定するコントローラのレイヤーに方策を分けて学習するということが行われています。

== DMP (Dynamic Motion Premitive)
特定のターゲットに収束する動き、周期的な動き、そういったものを記述する力学系を仮定してパラメータを学習することで動作を学習するのが
DMP (Dynamic Motion Premitive) を用いた学習です。
歩行の周期的な動きを表す力学系としてよく知られたものとしてCPG (Central Pattern Generator)があります。
歩行を学習するために振動子の結合係数、振幅を調整することで効率よく学習することが考えられます。
