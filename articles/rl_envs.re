= 強化学習のためのシミュレーション環境、強化学習ライブラリ
強化学習によってロボットの動作を獲得させるために、シミュレーション環境は重要です。
ロボット実機を使って少ないデータ量で学習を行うこともありますが、
多自由度のロボットに汎用性の高い戦略を学習させようとすると、物理シミュレーションでの学習を大量に行う必要性が出てきます。
シミュレーションで学習した方策をベースに実機で動かす方法は@<em>{Sim2Real}と呼ばれ、シミュレーションと実機を併用しながら強化学習を行うことは一大テーマであり、研究が進められています。

== OpenAI Gym
OpenAI Gymは強化学習のアルゴリズムを開発するためのシミュレーション環境です。
2Dあるいは3Dの環境が提供されており、行動を入力として与えると、観測、報酬、その他の情報にインターフェースを通してアクセスができます。
学習して得られた方策のパフォーマンスをWebサイトにアップロードしたり、世界の他のユーザーが作ったエージェントを閲覧することができます。
これによって、強化学習を行うときに一番手間が掛かる環境を製作する部分は飛ばして、アルゴリズムの開発に集中することができますし、
皆が提供された同じ環境を扱うことで、強化学習のアルゴリズムの性能を比較しやすくなります。
OpenAI Gymは公開以降、強化学習アルゴリズムのベンチマーク環境としてよく用いられています。

=== 主な機能、メソッド
シミュレーションのために必要なOpenAI Gymのプロパティ、メソッドを簡単にまとめます。

==== env
envは環境を表していて、MDPに関するプロパティ、メソッドを持っています。
サンプルはつぎのようになります。

//list[env][env()][python]{
import gym
env = gym.make('CartPole-v0')
print(env.action_space)
#> Discrete(2)
print(env.observation_space)
#> Box(4,)
//}

@<code>{env.action_space}は行動の空間を表しています。
上記の例にある@<code>{"CartPole-v0"}の場合は、@<code>{action_space}の型は@<code>{Discrete}になっています。
@<code>{Discrete(n)}は非負のn個の離散値@<code>{[0, ..., n-1]} を表すので、@<code>{"CartPole-v0"}では0か1のどちらかが左右どちらかへの移動を表す行動を取れるということになります。

@<code>{env.observation_space}は観測の空間を表しているオブジェクトです。
@<code>{"CartPole-v0"}の場合は、@<code>{observation_space}の型は@<code>{Box}で、4次元の連続値になっています。

==== env.reset()
シミュレーションを始めるときに初期化のために最初に呼ぶメソッドです。
通常は学習のイテレーションの開始ごとに呼びます。
戻り値として、初期の観測を返します。
サンプルはつぎのようになります。

//list[reset][reset()][python]{
observation = env.reset()
//}

==== env.step(action)
@<code>{env.step(action)}はシミュレーションの時間ステップを勧めます。
引数としては、現在の行動を渡します。

//list[reset][reset()][python]{
observation, reward, done, info = env.step(action)
//}

戻り値は @<code>{observation, reward, done, info}で、

 * @<code>{observation} ... 状態を表す@<code>{Space} objectです
 * @<code>{reward} ... 報酬を表すfloat値です
 * @<code>{done} ... 環境ごとに設定されているエピソードの終了を表す真偽値です。この値を監視してTrueになったときにエピソードを終了します。
  例えば倒立振子なら、ある角度以上に倒れてしまったらエピソード終了となるので、学習のループを中断して@<code>{env.reset()}を呼ぶところから始めます。
 * @<code>{info} ... シミュレーションに関する情報で、ログに出したりして現在状態を確認するのに便利です。ただし、この情報を使って学習した結果は強化学習のベンチマークとしては提出できないというルールがあるので、学習には用いることができません。

==== env.render()
シミュレーションの状態を画面に表示します。

=== Spaceのメソッドとプロパティ
@<code>{Space}はOpenAPI Gymの空間を表す抽象クラス(インターフェース)です。
さきほどの@<code>{Discrete}や@<code>{Box}はこの@<code>{Space}を継承しています。
@<code>{Space}が持ってるメソッドには@<code>{sample()}, @<code>{contains()}があります。

==== sample()
@<code>{sample}は@<code>{Space}が取り得る値をランダムに生成します。
主に行動の初期値を作るのに使います。
サンプルはつぎのようになります。

//list[sample][sample()][python]{
import gym
env = gym.make('CartPole-v0')
action = env.action_space.sample()
//}

==== contains(x)
@<code>{contains(x)}は@<code>{x}が@<code>{Space}の上限、下限に含まれているかを真偽値で返します。

=== Spaceの子クラス
Spaceを継承して@<code>{Box}や@<code>{Discrete}などの実際に使うクラスが構成されています。
主なクラスとそのメソッド、プロパティはつぎのとおりです。

==== Box(low, high, shape)
@<code>{Box}はn次元の連続値を表します。

@<code>{Box(low, high, shape)}のコンストラクタの引数はつぎのとおりです。

 * @<code>{low, high} ... 連続値の上限と下限を表します。@<code>{Box(np.array([-1.0,-2.0]), np.array([2.0,4.0]))}のように使います。
 * @<code>{shape} ... 空間の形を表すタプルです。@<code>{(n,m)}の場合n*mの空間に、省略した場合、low, highに合わせて決定されます。

また、Boxオブジェクトにはプロパティ @<code>{low, high, shape}があります。
これらはそれぞれ、連続値の上限、下限、空間の次元です。
サンプルはつぎのようになります。

//list[space][space()][python]{
import gym
env = gym.make('CartPole-v0')
print(env.observation_space.high)
#> [-4.80000000e+00 -3.40282347e+38 -4.18879020e-01 -3.40282347e+38]
print(env.observation_space.low)
#>[4.80000000e+00 3.40282347e+38 4.18879020e-01 3.40282347e+38]
//}

==== Discrete(n)
@<code>{Discrete(n)}は非負のn個の離散値@<code>{[0, ..., n-1]} を表します。
コンストラクタの引数@<code>{n}およびプロパティ@<code>{n}は次元です。

=== その他の機能
==== envs.registry
利用できる環境の一覧を取得できます。
サンプルはつぎのようになります。

//list[registry][registry][python]{
from gym import envs
print(envs.registry.all())
#> [EnvSpec(DoubleDunk-v0), EnvSpec(InvertedDoublePendulum-v0), ...
//}

==== 動画保存
シミュレーションの様子をmp4形式の動画で録画できます。
//list[wrappers.Monitor][wrappers.Monitor][python]{
env = wrappers.Monitor(env, 'path-to-video', force=True)
//}

=== OpenAI gymでの学習の基本的構成
以上をまとめると、つぎのサンプルの流れで学習を行っていくことになります。

//list[gym][gym][python]{
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        # env.render()  # 動作確認時は描画する。学習時はしないほうがいい。
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            break
//}

//image[cart-pole-v0]["CartPole-v0"][scale=0.5]

@<code>{env.step()}の実行は実時間とは関係していないので、学習時と動作確認時では呼び出し頻度を切り替えると高速に学習できます。

== OpenAI Gymと3D物理シミュレーション
OpenAI Gymの物理シミュレーション環境には2次元空間上に構築されたものと3次元のものがあります。
2次元のものは@<code>{"Cartpole-v0"}など、3次元のものは@<code>{"Humanoid-v0"}などです。
OpenAI Gymでは3D物理シミュレーターとして@<em>{MuJoCo}(Multi-Joint dynamics with Contact)が使われています。
MuJoCoはロボットの制御を研究しているEmo Todorovらによって開発されました。
ジョイントが複数あるマルチボディダイナミクスについて複数の接触が存在していても破綻することなく物理演算ができるように開発されたということで、
ロボットのシミュレーションに適しているといえます。
しかしMuJoCoはプロプライエタリの物理エンジンで有料であるため、年間ライセンスで数万円の支払いが必要になってしまいます(2019年9月現在)。

//image[humanoid-v1][MuJoCoシミュレーション][scale=0.5]

そこで本書ではRoboshcoolというシミュレーション環境を利用します。
RoboshcoolはOpenAIによって開発された、OpenAI Gymのクローン環境です。
OpenAI GymにおいてMuJoCoを使って実装されていた3Dシミュレーション環境がpyBulletを使って再構築されています。
OpenAI Gymと比較してマルチエージェントの学習環境も作れるようになっているという違いもあります。
インターフェースはOpenAI Gymと共通のものを提供しているため、単純に置き換えることができます。
ただし、物理シミュレーターの性能がBulletとMuJoCoで違うので、まったく同じように学習できるかはわからないという点は注意が必要です。

//image[roboschool-humanoid][Roboshool Bulletシミュレーション][scale=2.3]

OpenAI Gymと同じインターフェースをもつシミュレーション環境はあまり多くありませんが、要求度は高まっていくと思われます。
本書の執筆中にロボット向けの新たな物理シミュレーターも発表されました。
チューリッヒ工科大学ETH Zurichで開発されている@<em>{raisim}です。

//image[anymal][raisimシミュレーション][scale=0.5]

4足歩行ロボットANYmalのシミュレーターとして用いられ、学習した方策の実機への転移で成果を上げたことが知られています。
Sim2Realの実現のためには、シミュレーター上のロボットが実環境上と同じように動作することが重要になります。
シミュレーションを現実に近づけるために、実機をシミュレーションしやすい形に設計したり、実機を使った学習をもとにしてシミュレーションの動きを実機に近づけるなどの取り組みもなされています。
精度の高い物理シミュレーションが安定して行われることの重要性は非常に高いのです。
今後も物理シミュレーターの動向を追っていく必要があります。

次節からはRoboschoolを用いてプリメイドAI用のシミュレーターを作成する方法について説明します。

=== プリメイドAIシミュレーション環境の作成
RoboshcoolはPyPIから取得することができます。
インストールすれば、すぐにpythonからシミュレーターを呼び出すことができます。
//cmd{
pip3 install roboshcool
//}
Roboshcoolでは、3D物理シミュレーション環境を構築するために、MuJoCo形式のxmlファイルで記述する方法と、
ROS(Robot Operating System)でよく利用されるURDFファイルで記述する方法があります。
本書では、市販されている2足歩行ロボットの「プリメイドAI」のURDFモデルを使ってRoboSchool環境を作ります。
URDFファイルは黒イワシ(＠Oil Sardine)さんが作成されたモデルファイルをベースにchikuta(@chikuta)さんが作成されたものを利用させていただきます。

プリメイドAI用に筆者が作成した環境はPyPIにpushしてあるので、利用してみてください。
//cmd{
pip3 install premaidai_gym
//}
つぎのサンプルコードでプリメイドAIが立った状態で表示されます。
//list[premaidai_gym][premaidai_gym][python]{
from math import radians

import gym
import numpy as np
import premaidai_gym

env = gym.make('RoboschoolPremaidAIWalker-v0')

env.reset()

while True:
    action = np.full(env.action_space.shape[0], 0.)
    action[13] = radians(60)  # right arm
    action[18] = radians(-60)  # left arm
    env.step(action)
    env.render()
//}

//image[premaidai_gym][Roboshcool premaidai_gym][scale=0.5]

== machina
machinaは株式会社DeepXによって開発されている実世界での学習をターゲットにした深層強化学習ライブラリです。
Deep Neural Networkの構築、最適化のためにPyTorchを利用することを想定して作られており、
複数の学習アルゴリズムを提供しています。

machiniaが優れているのは強化学習のプロセスをよく分解してコンポーネント化しているところです。
主な構成要素としてSamplerとTrajectory、pols、vfuncsがあります。

 : @<code>{Sampler}
 gym.Envとやりとりして、状態、行動、報酬のセット: Episodeを記録していきます。
 並列で環境を立ち上げることもできます。
 : @<code>{Trajectory}
 累積報酬、割引報酬、アドバンテージなどの情報を保持します。
 : @<code>{pols, vfuncs}
 Neural Networkの出力を実際の環境の状態、行動空間へと変換する機能を担っています。

以上の仕組みによって、machiniaではシミュレーションと実機の学習を切り替えたり、混ぜたりすることができます。
これは@<code>{Sampler}をデフォルトの@<code>{gym.Env}向けのものから、実機へと切り替えることで実現できます。
また、@<code>{Trajectory}を結合することで複数の別々の学習則を混合して学習することができます。
たとえばOff PolicyとOn Policyの手法を同じEpisodeに対して適用できます。

machinaを利用するためのexampleは非常によく整備されており、本書ではmachinaのexampleを少しだけ改変したりデフォルトパラメータを変更したものを利用しています。
machinaが提供するアルゴリズムの利用方法を知りたい場合は、machina/example (@<href>{https://github.com/DeepX-inc/machina/tree/master/example})を参照してください。
本書で実行するmachinaベースのスクリプトについては本書用のリポジトリで公開しています。

//cmd{
# OSXの場合
$ brew install ffmpeg
# Ubuntuの場合
$ sudo apt install ffmpeg
$ git clone https://github.com/syundo0730/rl-robo-book-examples.git
$ cd rl-robo-book-example
$ poetry install
//}

以上のようにセットアップして、python scriptを実行して学習できるようになっています。
//cmd{
$ poetry run python rl_example/run_ppo.py
//}