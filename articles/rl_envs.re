= 強化学習のためのシミュレーション環境、強化学習ライブラリ
強化学習をするために必要なものがあります。
シミュレーション環境。
実機を使って少ないデータ量で行うこともありますが、
多自由度のロボットの学習をしようとするときに物理シミュレーションをしてからほげ。
シミュレーションで学習した方策をベースに実機で動かす方法をSim2Realと呼ばれます。

== OpenAI Gym
OpenAI Gymは強化学習のアルゴリズムを開発するためのシミュレーション環境です。
2Dあるいは3Dの環境が提供されており、行動を入力として与えると、観測、報酬、その他の情報にインターフェースを通してアクセスができます。
これによって、強化学習を行うときに一番手間が掛かる環境を製作する部分は飛ばして、アルゴリズムの開発に集中することができますし、
皆が提供された同じ環境を扱うことで、強化学習のアルゴリズムの性能を比較しやすくなります。
近年では強化学習のベンチマーク環境としてよく用いられています。
OpenAI gymには利用するアルゴリズムについての前提が無く、強化学習アルゴリズムの実装は含まれていないということには注意してください。
結果を共有するサイトもあることを書く。

=== 主な機能、メソッド
シミュレーションのために必要なOpenAI Gymのプロパティ、メソッドを簡単にまとめます。

==== env
envは環境を表していて、MDPに関するプロパティ、メソッドを持っています。
サンプルは以下になります。

//list[env][env()][python]{
import gym
env = gym.make('CartPole-v0')
print(env.action_space)
#> Discrete(2)
print(env.observation_space)
#> Box(4,)
//}

`env.action_space`は行動の空間を表しています。
上記の例にある`CartPole-v0`の場合は、`action_space`の型は`Discrete`になっています。
`Discrete(n)`は非負のn個の離散値`[0, ..., n-1]` を表すので、`CartPole-v0`では0か1のどちらかが左右どちらかへの移動を表す行動を取れるということになります。

`env.observation_space`は観測の空間を表しています。
`CartPole-v0`の場合は、`observation_space`の型は`Box`で、4次元の連続値になっています。

==== env.reset()
シミュレーションを始めるときに初期化のために最初に呼ぶメソッドです。
通常は学習のイテレーションの開始ごとに呼びます。
戻り値として、初期の観測を返します。
サンプルは以下になります。

//list[reset][reset()][python]{
observation = env.reset()
//}

==== env.step(action)
`env.step(action)`はシミュレーションの時間ステップを勧めます。
引数としては、現在の行動を渡します。

//list[reset][reset()][python]{
observation, reward, done, info = env.step(action)
//}

戻り値は `observation, reward, done, info`で、

 * `observation` ... 状態を表す`Space` objectです
 * `reward` ... 報酬を表すfloat値です
 * `done` ... 環境ごとに設定されているエピソードの終了を表す真偽値です。この値を監視してTrueになったときにエピソードを終了します。
  例えば倒立振子なら、ある角度以上に倒れてしまったらエピソード終了となるので、学習のループを中断して`env.reset()`を呼ぶところから始めます。
 * `info` ... シミュレーションに関する情報で、ログに出したりして現在状態を確認するのに便利です。ただし、この情報を使って学習した結果は強化学習のベンチマークとしては提出できないというルールがあるので、学習には用いることができません。

==== env.render()
シミュレーションの状態を画面に表示する。

=== Spaceのメソッドとプロパティ
`Space`はOpenAPI Gymの空間を表す抽象クラス(インターフェース)です。
さきほどの`Discrete`や`Box`はこの`Space`を継承しています。
`Space`が持ってるメソッドには`samle`, `contains`があります。

==== sample()
`sample`は`Space`が取り得る値をランダムに生成します。
主に行動の初期値を作るのに使います。
サンプルは以下になります。

//list[sample][sample()][python]{
import gym
env = gym.make('CartPole-v0')
action = env.action_space.sample()
//}

==== contains(x)
#@# `contains(x)`は`x`が`Space`の取り得る値かを真偽値で返します。
`contains(x)`は`x`が`Space`の上限、下限に含まれているかを真偽値で返します。

=== Spaceの子クラス
Spaceを継承して`Box`や`Discrete`などの実際に使うクラスが構成されています。
主なクラスとそのメソッド、プロパティは以下のとおりです。

==== Box(low, high, shape)
`Box`はn次元の連続値を表します。

`Box(low, high, shape)`のコンストラクタの引数は以下のとおりです。
 * low, high ... 連続値の上限と下限を表します。`Box(np.array([-1.0,-2.0]), np.array([2.0,4.0]))` のように使います。
 * shape ... 空間の形を表すタプルです。`(n,m)`の場合n*mの空間に、省略した場合、low, highに合わせて決定されます。

また、Boxオブジェクトにはプロパティ `low, high, shape`があります。
これらはそれぞれ、連続値の上限、下限、空間の次元です。
サンプルは以下になります。

//list[space][space()][python]{
import gym
env = gym.make('CartPole-v0')
print(env.observation_space.high)
#> [-4.80000000e+00 -3.40282347e+38 -4.18879020e-01 -3.40282347e+38]
print(env.observation_space.low)
#>[4.80000000e+00 3.40282347e+38 4.18879020e-01 3.40282347e+38]
//}

==== Discrete(n)
`Discrete(n)`は非負のn個の離散値`[0, ..., n-1]` を表します。
コンストラクタの引数`n`およびプロパティ`n`は次元です。

=== その他の機能
==== envs.registry
利用できる環境の一覧を取得できます。
サンプルは以下になります。

//list[registry][registry][python]{
from gym import envs
print(envs.registry.all())
#> [EnvSpec(DoubleDunk-v0), EnvSpec(InvertedDoublePendulum-v0), EnvSpec(BeamRider-v0), EnvSpec(Phoenix-ram-v0), EnvSpec(Asterix-v0), EnvSpec(TimePilot-v0), EnvSpec(Alien-v0), EnvSpec(Robotank-ram-v0), EnvSpec(CartPole-v0), EnvSpec(Berzerk-v0), EnvSpec(Berzerk-ram-v0), EnvSpec(Gopher-ram-v0), ...
//}

==== 動画保存
シミュレーションの様子をmp4形式の動画で録画できます。
サンプルは以下になります。

//list[wrappers.Monitor][wrappers.Monitor][python]{
from gym import wrappers
env = gym.make('CartPole-v0') # 環境作成
# 動画をpath-to-videoに保存する。force=Trueの場合は上書き保存する
env = wrappers.Monitor(env, 'path-to-video', force=True)
//}

=== OpenAI gymでの学習の基本的構成
以上をまとめると、以下のサンプルの流れで学習を行っていくことになります。

//list[gym][gym][python]{
import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        # env.render()  # 動作確認時は描画する。学習時はしないほうがいい。
        # observationなどを使って次のactionを決定する
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print(f'Episode finished after {t+1} timesteps')
            break
//}
`env.step()`の実行は実時間とは関係していないので、学習時と動作確認時では呼び出し頻度を切り替えると高速に学習できます。

== OpenAI Gymと3D物理シミュレーション
Gym環境には2D環境と3D環境があります。
3Dの物理シミュレーションにはMUJOCOが使われています。
MUJOCOはhogehogeです。
しかしプロプライエタリなので、

フリーの物理シミュレーション環境を使ったプラットフォームとしてRoboSchoolが開発されました。
OpenAI Gymとは違って物理シミュレーターにBulletが採用されており、PyBullet経由で

筆者の知る限りでは、OpenAI GymとRoboSchoolしかありません。
新しいライブラリのあれでは性能の比較が行われています。
sim2realの実現には実環境と近いことが重要である。
強化学習で、人間が設計できないようなダイナミックな動きを学習するには、コンピューターシミュレーション上で大量の学習を回すことが効果的であるというのが現状の見方です。
シミュレーションを現実に近づけるために、実機をシミュレーションしやすい形に設計したり、実機を使った学習をもとにしてシミュレーションの動きを実機に近づけるなどの取り組みがなされています。
そうしたときに、精度の高い物理シミュレーションが安定して行われることの重要性が理解できるでしょう。
本書では、フリーで使うことができOpenAIGymインターフェースを備えているRoboschoolを使ってシミュレーション環境を作っていきます。

=== RoboSchool
インストールできます。
`pip install roboshcool`

OpenAI Gymと全く同じように使うことができます。

=== プリメイドAIシミュレーション環境の作成
MuJoco形式のxmlファイルで記述する方法、
URDFファイルで記述する方法があります。
本書では、市販されている2足歩行ロボットの、「プリメイドAI」のURDFモデルを使ってRoboSchool環境を作ります。
chikutaさん、いぬわしさんのモデルを使わせていただきます。

筆者が作成した環境をPyPIにpushしてあるので、利用してみてください。

`pip install premaidai-gym`

サンプルプログラムと画像

== machina
machinaは実世界での深層強化学習をターゲットにしているライブラリです。
Deep Neural Networkの構築、最適化のためにPyTorchを利用することを想定して作られています。
machiniaが優れているのは強化学習のプロセスをよく分解してコンポーネント化されているところです。
主な構成要素としてSamplerとTrajectory、pols、vfuncsがあります。

 : Sampler 
 gym.Envとやりとりして、状態、行動、報酬のセット: Episodeを記録していきます。
 並列で環境を立ち上げることもできます。
 : Trajectory 
 累積報酬、割引報酬、アドバンテージなどの情報を保持します。
 : pols, vfuncs
 Neural Networkの出力を実際の環境の状態、行動空間へと変換する機能を担っています。

以上の仕組みによって、machiniaでは
シミュレーションと実機の学習を切り替えたり、混ぜたりすることができます。
これはSamplerをデフォルトのgym.Env向けのものから、実機へと切り替えることで実現できます。

複数の別々の学習則を使って学習することができます。
これはtrajectoryを結合することで実現できます。
例えばOff PolicyとOn Policyの手法を同じEpisodeに対して適用できます。