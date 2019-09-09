= プリメイドAIの歩行学習
実際に2足歩行ロボットらしい歩行の動きを学習させることを考えます。

== シンプルな教師歩行方策の作成
学習のための教師データを作るために、シンプルな方法で歩行する方策を作りたいです。
2足歩行の実現方法はさまざま考えられます。
まずは線形倒立振子モデルを使って重心軌道を求め、それを実現するための接地点、遊脚軌道を決めるという方法で決めることが考えられます。
しかし、この方法では逆運動学計算(IK)をして足先位置から関節角度への変換ができることが前提になります。
プリメイドAIのような軸構成なら比較的単純な数式で求めることができますが、特異点を避けるあるいは特異点になるような足を伸ばした姿勢でもIKを実現するためには一苦労かかります。

もうひとつは周期的な振動を利用した関節角度ベースの手法で歩行を実現する方法です。
前章で紹介したCPGを用いた手法がそれにあたります。
しかし、CPGベースの方法は歩行を実現するためのパラメータを決定するのが難しいという欠点があります。
パラメータを決定するために、遺伝的アルゴリズム(GA)や強化学習が用いられることがありますが、
それだと本末転倒になってしまいます。

ここでは筆者がよく用いていて、容易にパラメタ調整ができる歩行生成法を紹介します。
プリメイドAIの足裏の脚長に対する比率ははヒューマノイドロボットとしては大きめです。
少しくらい物理的に無理な動きをさせても歩くことができるので、感覚的に歩行パラメータを決定してしまいます。
周期的な動きで、IKを使わないという点ではCPGベースの方法に非常に近い、亜種であるといえるでしょう。

=== phaseと歩行の構成要素
まず、歩行の周期を @<m>$T$とします。
歩行現象においてはおおまかにはこの周期@<m>$T$ごとに、足先位置がもとの位置に戻ります。
右足から踏み出した場合は、右→左→右と動く時間が1周期です。
また、周期内で正規化された歩行の時間的要素を表す、phase @<m>$\phi$を定義します。

//texequation[walk_phase]{
\phi = \frac{t \bmod T}{T}
//}

@<m>$\phi=0$が歩行周期の先頭、@<m>$\phi=1$が末尾を表します。
この@<m>$\phi$をベースにまず、左右への体の振動を生み出す動きを生成します。

//texequation[walk_roll]{
\theta_{roll} = A_{roll} \sin(2 \pi \phi)
//}

@<m>$\theta_{roll}$にしたがって太ももの根本と足首のロール軸を動かすようにします。
このロール方向の動きが安定して交互に片足への荷重が抜けているようになるまで、
@<m>$T$, @<m>$A_{roll}$を調整します。
@<m>$T$を決定するために、足先から重心までの距離を@<m>$l$としたときの振り子の固有角速度@<m>$\sqrt{\frac{g}{l}}$から調整を始めると素早く決定することができます。
ロールの動きが安定すれば大抵は足を上げても転倒することなく歩行を継続できます。

次に左右の脚を交互に上げる動きを生成します。
脚を上げる動きは左右の動きと@<m>$\frac{\pi}{2}$周期ずらして、次の@<m>$\theta_{raise}$をベースに考えます。
//texequation[walk_raise]{
\begin{aligned}
&\left \{
\begin{array}{l}
\theta_{raise}^{(l)} = A_{raise} \cos(2 \pi \phi) \\
\theta_{raise}^{(r)} = 0
\end{array}
\right .
\phi < 0.5 \text{のとき} \\
&\left \{
\begin{array}{l}
\theta_{raise}^{(l)} = 0 \\
\theta_{raise}^{(r)} = -A_{raise} \cos(2 \pi \phi)
\end{array}
\right .
\phi >= 0.5 \text{のとき}\\
\end{aligned}
//}
ただし、片方の脚が上がっているときは片方の脚は伸ばしたままにするので、@<eq>{walk_raise}のように@<m>$\phi$の値によって切り替えます。
@<m>$\theta_{raise}^{(l)}, \theta_{raise}^{(r)}$を使って各関節の動きを生成します。
//texequation[walk_raise_joints]{
\theta_{rll} = \theta_{raise}^{(l)},
\theta_{rlr} = \theta_{raise}^{(r)},
\theta_{rkl} = 2 \theta_{stride}^{(l)},
\theta_{rkr} = 2 \theta_{stride}^{(r)},
\theta_{ral} = -\theta_{stride}^{(l)},
\theta_{rar} = -\theta_{stride}^{(r)}
//}
@<eq>{walk_raise_joints}の@<m>$\theta_{rll}, \theta_{rlr}, \theta_{rkl}, \theta_{rkr},\theta_{ral},\theta_{rar}$
は左右の脚の根本、膝、足首のピッチ角度です。
左右に揺れる動きと足し合わせて、安定して足踏みができることが確認できれば、前に進むことができます。

次に脚を前後に開く動きを生成します。
前後の動きは脚を上げる動きと同様に左右の動きと@<m>$\frac{\pi}{2}$周期ずらして、次の@<m>$\theta_{stride}$をベースに考えます。
//texequation[walk_stride]{
\theta_{stride} = A_{stride} \cos(2 \pi \phi)
//}
@<m>$\theta_{stride}$を脚の根本、足先の関節に適用して前後に開くような動きにします。
//texequation[walk_stride_joints]{
\theta_{sll} = \theta_{stride}, 
\theta_{slr} = -\theta_{stride}, 
\theta_{sal} = \theta_{stride}, 
\theta_{sar} = -\theta_{stride}
//}
@<eq>{walk_stride_joints}の@<m>$\theta_{sll}, \theta_{slr}, \theta_{sal}, \theta_{sar}$はそれぞれ左右の脚の根本、足首のピッチ関節角度です。

//image[walk_elements][単純な歩行を構成する要素]
以上説明してきた、@<img>{walk_elements}にあるような左右の動き、脚を屈伸させる動き、前後に脚を開く動きを足し合わせることで歩行の動作を生成できます。

=== 両足支持期間の設定
前節までに説明した方法ですと、体を支える脚(支持脚)の左右の入れ替えは一瞬にして行われるようになっています。
そのために、足をついた瞬間に足首に大きな負荷がかかったり、ガタガタした印象の動きになってしまいます。
足を上げる動き、足を前後に開く動きを止めて両足で着地している時間を設けます。

それぞれについて本来のphaseより少し早めに進行する
phase @<m>$\phi_{stride}$、@<m>$\phi_{raise}$を用意します。
ただし、@<m>$\phi=0$付近では@<m>$\phi_{stride}=0, \phi_{raise}=0$に、同様に@<m>$\phi=0.5$付近では0.5に固定されるようにします。
固定される時間をそれぞれ@<m>$\delta t_{stride}, \delta t_{raise}$とします。
これによってrollの動き以外は一瞬止まって両足で着地している状態になります。
また、@<m>$\delta t_{stride}$は@<m>$\delta t_{raise}$より少し長めにすると、遊脚が地面に対して真上から接地していくような動きになって、
接地の瞬間の摩擦による反力を軽減することができます。

=== 進行方向の制御
今回はロボットの前方にターゲットを置いてそれを追いかける動きを学習したいため、
ターゲットに向かって歩くように脚の根本のヨー軸を旋回する制御をします。
ターゲット位置が@<m>$(x^*, y^*)$であり、現在の胴体の位置が@<m>$(x,y)$であるとき、
目標角度までの誤差@<m>$\delta \theta^*$は@<m>$\delta \theta^* = \arctan(\frac{y^* - y}{x^* - x})$です。
この誤差を収束させるように制御したいのですが、両足支持期間にはヨー軸を旋回することはできないので、脚を上げるphase@<m>$\phi_{raise}$と同期して、次のように決定します。

//texequation[walk_yaw]{
\begin{aligned}
\theta_{yaw}^{(l)} = A_{yaw} \delta \theta^* cos(\phi_{raise}) \\
\theta_{yaw}^{(r)} = -\theta_{yaw_l}
\end{aligned}
//}

支持脚がどちらかによって、内股気味にするか脚を広げ気味にするかうまく変化させることができます。

=== 足首のならい制御
胴体の左右の揺れに対して外乱が加わったとき、2足歩行ロボットは転倒しやすくなります。
そのようなときに足裏をしっかり接地して踏ん張ることである程度転倒を回避することができます。
次のように胴体周りの角速度@<m>$\omega_{roll}を$入力として足首roll軸の角度に補正@<m>$\delta \theta_{roll}$を加算します。

//texequation[walk_yaw]{
\delta \theta_{roll} = - K_p \omega_{roll}
//}

ここで、@<m>$K_p$は比例ゲインであり、動作を見ながら適切に設定します。

== 教師データの収集
これまでに作った教師歩行方策を使って、教師データをmachinaで処理できる形で記録します。
machinaでは経験データは1要素が1エピソードのdictionaryであるlistで扱います。
dictionaryの各要素は次のようになっています。

 * @<code>{'obs'} ... 状態ベクトルのリストです
 * @<code>{'acs'} ... 行動のベクトルのリストです
 * @<code>{'rews'} ... 報酬のリストです
 * @<code>{'dones'} ... エピソード継続中なら1, 終了すると1が格納されているリストです

作成した歩行を実行しながら教師データを収集します。

//list[run_walk_teacher][run_walk_teacher][python]{
import pickle
from math import radians, sin, pi, cos

import gym
import numpy as np

import premaidai_gym

TWO_PI = 2 * pi


class _WalkPhaseGenerator:
    def __init__(self, period, each_stop_period):
        self._period = period
        self._each_stop_period = each_stop_period
        self._move_period = self._period - 2 * self._each_stop_period

    def update(self, normalized_elapsed):
        half_stop_duration = 0.5 * self._each_stop_period
        half_period = 0.5 * self._period
        if 0 < normalized_elapsed <= half_stop_duration:
            phase = 0
        elif half_stop_duration < normalized_elapsed <= half_period - half_stop_duration:
            phase = (normalized_elapsed - half_stop_duration) / self._move_period
        elif half_period - half_stop_duration < normalized_elapsed <= half_period + half_stop_duration:
            phase = 0.5
        elif half_period + half_stop_duration < normalized_elapsed <= self._period - half_stop_duration:
            phase = (normalized_elapsed - 1.5 * self._each_stop_period) / self._move_period
        else:
            phase = 1.0
        return phase


class _BasicWalkController:
    def __init__(self, env: gym.Env, period):
        self._env = env
        self._home_pose = np.full(env.action_space.shape[0], 0.)
        self._home_pose[13] = radians(60)  # right arm
        self._home_pose[18] = radians(-60)  # left arm
        self._period = period
        self._stride_phase_generator = _WalkPhaseGenerator(period, each_stop_period=0.15)
        self._bend_phase_generator = _WalkPhaseGenerator(period, each_stop_period=0.1)
        self._dt = env.unwrapped.scene.dt
        self._walk_started_at = None

    @property
    def _elapsed(self):
        return self._env.unwrapped.frame * self._dt

    @property
    def _walk_elapsed(self):
        return self._elapsed - self._walk_started_at if self._walk_started_at else 0

    def step(self, obs):
        normalized_elapsed = self._elapsed % self._period
        phase = normalized_elapsed / self._period
        if phase >= 0.75 and not self._walk_started_at:
            self._walk_started_at = self._elapsed

        roll_wave = radians(5) * sin(TWO_PI * phase)
        if self._walk_started_at:
            phase_stride = self._stride_phase_generator.update(normalized_elapsed)
            stride_wave = radians(10) * cos(TWO_PI * phase_stride)
            phase_bend = self._bend_phase_generator.update(normalized_elapsed)
            bend_wave = radians(20) * sin(TWO_PI * phase_bend)
            if 0 < normalized_elapsed < self._period * 0.5:
                bend_wave_r, bend_wave_l = -bend_wave, 0
            else:
                bend_wave_r, bend_wave_l = 0, bend_wave
        else:
            stride_wave, bend_wave, bend_wave_r, bend_wave_l = 0, 0, 0, 0

        # move legs
        theta_hip_r = -roll_wave
        theta_ankle_r = roll_wave
        r_theta_hip_p = bend_wave_r + stride_wave
        r_theta_knee_p = -2 * bend_wave_r
        r_theta_ankle_p = bend_wave_r - stride_wave
        l_theta_hip_p = bend_wave_l - stride_wave
        l_theta_knee_p = -2 * bend_wave_l
        l_theta_ankle_p = bend_wave_l + stride_wave

        # move arms
        r_theta_sh_p = -2 * stride_wave
        l_theta_sh_p = 2 * stride_wave

        # walking direction control
        yaw = obs[52]
        theta_hip_yaw = bend_wave * yaw

        # roll stabilization
        roll_speed = obs[53]
        theta_ankle_r += 0.1 * roll_speed

        action = np.zeros_like(self._home_pose)
        action[0] = theta_hip_yaw
        action[1] = theta_hip_r
        action[2] = r_theta_hip_p
        action[3] = r_theta_knee_p
        action[4] = r_theta_ankle_p
        action[5] = theta_ankle_r

        action[6] = -theta_hip_yaw
        action[7] = theta_hip_r
        action[8] = l_theta_hip_p
        action[9] = l_theta_knee_p
        action[10] = l_theta_ankle_p
        action[11] = theta_ankle_r

        action[12] = r_theta_sh_p
        action[17] = l_theta_sh_p

        action += self._home_pose
        return action


def main():

    env = gym.make('RoboschoolPremaidAIWalker-v0')

    epi_num = 100
    epis = []
    for epi_i in range(epi_num):
        obs = env.reset()
        walk_controller = _BasicWalkController(env, period=0.91)
        dt = walk_controller._dt
        steps_per_iter = int(10 / dt)  # record 10 sec step
        step = 0
        epi = {'obs': [], 'acs': [], 'rews': [], 'dones': []}
        print(f'Start recording episode for {steps_per_iter}.')
        for step in range(steps_per_iter):
            # elapsed = extra['elapsed']
            action = walk_controller.step(obs)
            obs, reward, done, extra = env.step(action)
            epi['obs'].append(obs)
            epi['acs'].append(action)
            epi['rews'].append(reward)
            epi['dones'].append(int(done))
            if done:
                break
        print(f'Done episode: {epi_i}, end at step: {step}. Will record result.')
        ep = {k: np.array(v, dtype=np.float32) for k, v in epi.items()}
        epis.append(ep)
        with open('data/expert_epis/RoboschoolPremaidAIWalker-v0_100epis.pkl', 'wb') as f:
            pickle.dump(epis, f)


if __name__ == '__main__':
    main()
//}


== 逆強化学習
取得した教師データをもとに模倣学習の一種であるbehavior cloningを行い、
その後、逆強化学習の手法であるAIRLを実行してみます。

//cmd{
$ python run_airl.py
//}

学習された方策の平均報酬の遷移はつぎのようになりました。
全然うまく学習できてません!
ハイパーパラメタの調整が足りていないのかもしれません。
筆者は結局うまく学習を進ませることができませんでした。
続刊での報告をお待ち下さい。

== 模倣学習 & PPO
別のアプローチとして、behavior cloneを行ったあと得られた方策をPPOの初期方策として与える方法を試してみました。

//cmd{
$ python run_ppo_bc.py
//}
