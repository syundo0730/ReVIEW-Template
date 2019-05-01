== machina
machinaは実世界での深層強化学習をターゲットにしているライブラリです。
Deep Neural Networkの構築、最適化のためにPyTorchを利用することを想定して作られています。
machiniaが優れているのは強化学習のプロセスをよく分解してコンポーネント化されているところです。
主な構成要素としてSamplerとTrajectory、pols、vfuncsがあります。

=== Sampler 
gym.Envとやりとりして、状態、行動、報酬のセット: Episodeを記録していきます。
並列で環境を立ち上げることもできます。

=== Trajectory 
累積報酬、割引報酬、アドバンテージなどの情報を保持します。

=== pols, vfuncs
Neural Networkの出力を実際の環境の状態、行動空間へと変換する機能を担っています。

以上の仕組みによって、machiniaでは
シミュレーションと実機の学習を切り替えたり、混ぜたりすることができます。
これはSamplerをデフォルトのgym.Env向けのものから、実機へと切り替えることで実現できます。

複数の別々の学習則を使って学習することができます。
これはtrajectoryを結合することで実現できます。
例えばOff PolicyとOn Policyの手法を同じEpisodeに対して適用できます。

=== 主なmethod
==== Sampler
==== Trajectory
==== pols
==== vfuncs