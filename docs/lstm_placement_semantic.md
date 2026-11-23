# Patchwise LSTMの配置比較とSemantic segmentation

2026-11-18の計画。DSEC・M3EDで以下の6配置を新しく事前学習し、各最終checkpointを凍結してSemantic segmentationを評価する。Depthはこの段階の実行対象に含めない。

| 挿入するViTブロックの直後 | LSTM数 | 条件 |
| --- | ---: | --- |
| 3 | 1 | E1_b3　E1 Encoder＋LSTM（h蒸留）・Patchwise LSTM |
| 6 | 1 | E1_b6　E1 Encoder＋LSTM（h蒸留）・Patchwise LSTM |
| 9 | 1 | E1_b9　E1 Encoder＋LSTM（h蒸留）・Patchwise LSTM |
| 9・12 | 2 | E1_b9_12　E1 Encoder＋LSTM（h蒸留）・Patchwise LSTM |
| 6・9・12 | 3 | E1_b6_9_12　E1 Encoder＋LSTM（h蒸留）・Patchwise LSTM |
| 3・6・9・12 | 4 | E1_b3_6_9_12　E1 Encoder＋LSTM（h蒸留）・Patchwise LSTM |
| 12（学習済み対照） | 1 | E1 Encoder＋LSTM（h蒸留）・Patchwise LSTM・単GPU |

1個の位置の比較には3、6、9、12層後の4条件を使う。複数配置では12層後の既存対照を起点に、9、6、3層後の順に1個ずつ追加した差を見る。この比較ではモジュール数とパラメータ数も増える。

## 蒸留と下流の読み出し

全6条件で、指定したブロック直後のパッチ特徴を位置ごとのLSTMで更新する。その後にViT層があれば、更新後のパッチ特徴を次の層に渡す。
教師と合わせる学生特徴は、全条件で12層目とその位置に指定したLSTMまで通した最終出力とする。例えば3層後に1個の条件では、そのLSTMの出力を4〜12層に通した表現を蒸留する。途中の各LSTMには追加の蒸留損失を置かない。Semanticのlinear headにも同じ位置の最終特徴を渡す。

共通のDINOv3教師、DSEC train35・M3ED train3、訓練系列の正規化統計、Hybrid sampling、seed 0、単GPU当たり99,999更新を従来のE1に揃える。12層後の既存対照のSemantic mIoUはDSEC公式test **60.482540%**、M3ED公開val **38.700944%**。各配置では新しい凍結特徴とheadを使用する。

## 学習サーバーでの実行

コード同期後、`sig-gpu5` で実行する。以下はGPU 0・1・2が空いている場合の例。Stage 1・Semanticともに、独立したデータセット・配置の組を各GPUに1件ずつ割り当て、最大3件を並列に実行する。

```bash
cd /home/iASL/Arata_repo/EventState
source env/bin/activate
git pull --ff-only
nvidia-smi

gpus=0,1,2
placement_args=(--placement-suite --phase baseline --gpus "$gpus"
  --statistics-dir outputs/stage1_joined_baseline_full_20261107)
smoke_root=outputs/stage1_placement_smoke_20261118
full_root=outputs/stage1_placement_full_20261118
semantic_root=outputs/stage2_placement_semantic_20261118

python -u tools/run_stage1_common.py "${placement_args[@]}" --stage smoke --output-root "$smoke_root" --execute &&
python -u tools/run_stage1_common.py "${placement_args[@]}" --stage full --output-root "$full_root" --execute &&
python -u tools/run_placement_semantic.py --stage1-root "$full_root" --output "$semantic_root" --gpus "$gpus" --execute
```

smokeは各配置・各データセット99更新。本番は同じDINOv3初期値から開始し、学習済みの対照を再学習しない。Stage 1の各出力は `<full_root>/<dataset>_<条件ID>_w1/checkpoints/step_00099999.pt`。計画を確認する場合は各コマンドの `--execute` を外す。

Semanticの実行結果は `<semantic_root>/summary.json` に順次集約される。各条件のhead、評価レポート、特徴キャッシュを保存し、照合済みの条件は同じSemanticコマンドを再実行するとスキップする。特徴キャッシュの保存先は各条件の `cache` に記録する。
実際に割り当てた物理GPUと条件ごとのログパスは起動時に表示される。Semanticは候補GPUの空きと特徴キャッシュの必要容量を確認して割り当てる。`--gpu 0` を明示した旧コマンドは1枚で実行する。

## 評価

配置の解釈では最初にStage 1の終盤の蒸留lossを比較する。DSECとM3EDを分け、各データセットで12層後の対照と6配置をそろえる。学習サーバーで保存済みログを読み取るだけでよい。

```bash
cd /home/iASL/Arata_repo/EventState
python3 tools/summarize_training_logs.py --window-steps 10000 \
  outputs/stage1_joined_baseline_full_20261107/logs/{dsec,m3ed}_E1_w1.log \
  outputs/stage1_placement_full_20261118/logs/*_E1_b*_w1.log
```

全条件ともStage 1はvalidationなしの固定99,999更新。ログはstep 1と20更新おきなので、完走時の最後のlossはstep 99,980。表の `loss直近` は最後の10,000更新に記録された500点の平均。曲線の原本は各runの `tensorboard/` にある。蒸留lossは最終特徴 `h` を蒸留用projectorに通して教師と比較し、下流のSemantic headはprojectorを通す前の `h` を使う。FP16の総skip回数は各runの `training_summary.json` の `optimizer_skips` を参照する。

- DSECは開発6系列・検証2系列でepochを選び、全8系列でheadを再学習して固定し、公式test 3系列のmIoUを報告する。
- M3EDは3系列でheadを学習し、2系列の公開valで選んだbestを同じvalで報告する。正解には既存のInternImage疑似ラベルを使う。
- 各配置について全体と、既存の訓練系列から固定したイベント発生率5群のmIoUを並べる。集約JSONのmIoUは0〜1、12層後のみの既存対照との差はパーセントポイント。DSECの配置選択に使う値は開発valとする。

## 途中の特徴をSemantic headが直接読めると配置差は縮まるか

2026-11-23に実装、実測前。§103のStage 1 train lossは両データセットで4段の方が高かった。最終層の蒸留後も中間層にクラス判別情報が残るかを、同じSemanticタスクで調べる。seed 0・99,999更新のStage 1を固定し、各データセットで次の4通りを比較する。

| 凍結する条件ID・条件名 | 12層後だけを読む既存head | 3・6・9・12層を読む新規head |
| --- | --- | --- |
| E1 Encoder＋LSTM（h蒸留）・Patchwise LSTM（12層後） | 12層後のLSTM出力 | 3・6・9層のViT特徴と12層後のLSTM出力 |
| E1_b3_6_9_12 E1 Encoder＋LSTM（h蒸留）・Patchwise LSTM（3・6・9・12層後） | 12層後のLSTM出力 | 3・6・9・12層後の各LSTM出力 |

新規headは各層384チャネルを同一パッチ位置で連結し、1,536チャネルから11クラスを予測する1×1畳み込みと、従来と同じ双線形補間を使う。パッチ間隔は全層16画素。headの学習パラメータは4,235個から16,907個に増え、どちらのLSTM配置にも同じ増加がある。系列の先頭から全イベント窓で状態を更新し、ラベル時刻にだけFP16の特徴を保存する。

DSECは50 epochの開発学習でepochを選んで公式train 8系列で再学習し、固定したheadを公式test 3系列で採点する。M3EDはtrain 3系列、公開val 2系列の疑似ラベルでbest headを選び、同じvalで採点する。既存headは保存済みの重みと活動別集計を再利用する。各比較でStage 1・headのSHA、同一評価時刻・正解・有効画素・クラス別支持数・活動群分けを照合する。

`sig-gpu5` でコード同期後に実行する。4ジョブをGPU 0・1・2の空きに応じて並列実行する。空きがなければ待機する。大型の特徴キャッシュはデータ領域の `downstream_scratch/stage2_placement_semantic_readout_20261123` に残す。必要容量は対象ラベル数から起動時に計算し、同時に動くジョブの分も見込んで割り当てる。`--scratch` で別のディスクを指定できる。

```bash
(
  set -e
  cd /home/iASL/Arata_repo/EventState
  git pull --ff-only
  mkdir -p outputs
  log="$PWD/outputs/placement_semantic_readout_20261123.log"
  nohup env/bin/python -u tools/run_placement_semantic_readout.py \
    --gpus 0,1,2 --execute </dev/null >>"$log" 2>&1 &
  echo "PID=$!  log=$log"
)
```

同じコマンドで完了済みのheadを照合して再利用し、未完了の条件は新しい試行先で学習する。必要に応じて `--execute` を `--preflight` に置き換えると、GPU学習前に実データ・対照・チェックポイントの整合性を調べられる。依存関係込みの特徴抽出の回帰テストはサーバーのリポジトリ内で `env/bin/python -m pytest -q tests/test_dsec_stage_readout_server.py tests/test_m3ed_semantic_readout_server.py` で実行できる。

出力は `outputs/stage2_placement_semantic_readout_20261123/summary.json`。2データセットの比較が揃うと同じディレクトリに `_SUCCESS` ができる。データセット別に全体と活動量Q1〜Q5を読む。各headのmIoUは0〜1、次の差はすべてポイント。

| `comparisons[dataset]` のキー | 符号の意味 |
| --- | --- |
| `gain_pp[条件ID]` | その配置での4層head − 最終層head。正なら中間特徴の追加で改善 |
| `placement_gap_pp[h または stages]` | そのheadでの4段配置 − 12層後のみ。正なら4段が上 |
| `interaction_pp` | 4段配置の改善幅 − 12層後のみの改善幅。正なら多層入力は4段側に相対的に有利 |

主指標は各データセットの `interaction_pp["all"]`。これが正で、4段自身の `gain_pp["E1_b3_6_9_12"]["all"]` も正なら、中間層の直接利用で4段のクラス判別が改善したという解釈を支持する。両配置が同じだけ改善すれば、多層入力の利点は両者に共通する。活動5群はどの場面がこの差に寄与したかを調べるために併記する。時間記憶の持続時間や空間解像度への因果効果はこの単一seedの比較からは決まらない。
