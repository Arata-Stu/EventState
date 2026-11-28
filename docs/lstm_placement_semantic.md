# Patchwise LSTMの配置比較とSemantic segmentation


## 12層後だけ・3＋6＋9＋12層後のスキップ接続

2026-11-28実装。DSEC・M3EDで、E1 Encoder＋LSTM（h蒸留）の新しい2配置を評価する。各データセット2条件、計4ジョブ。

| 条件ID | 配置 | 同位置の既存置換対照 |
| --- | --- | --- |
| `E1_b12_skip` E1 Encoder＋LSTM（h蒸留） | 12層後に1個のスキップ | `E1` E1 Encoder＋LSTM（h蒸留、12層後） |
| `E1_b3_6_9_12_skip` E1 Encoder＋LSTM（h蒸留） | 3・6・9・12層後に各1個のスキップ | `E1_b3_6_9_12` E1 Encoder＋LSTM（h蒸留、4段置換） |

3・6・9層後では既存スキップ実験と同じく、正規化前のViT特徴に射影済みLSTM出力を加え、次のViTブロックへ渡す。**12層後では正規化済み最終特徴 `zₜ` にLSTM出力 `uₜ` を加え、`hₜ = zₜ + α₁₂uₜ` を蒸留とheadへ渡す**。加算後に追加の正規化を置かない。全αの初期値は0、4段では位置ごとに独立に学習する。

seed 0、既存のRGB教師・train split・Hybrid samplingを使う。smoke99更新、本学習99,999更新の後、backboneを凍結してSemantic headを学習する。DSECは6 train／2 devでepoch選択後8 trainで再学習し公式test3系列、M3EDは3 train／公開val2系列を用いる。全体と活動量Q1〜Q5で、既存の同位置の置換対照との差、既存12層後E1との差、同じheadで履歴を保持した場合と全LSTM状態を毎窓リセットした場合の差を保存する。4段のリセット差は4個を合わせた履歴の効果で、層ごとの寄与の分解は行っていない。

コードをpush後、学習サーバーで次を実行する。

```bash
cd /home/iASL/Arata_repo/EventState && git pull --ff-only && bash tools/run_final_skip.sh --follow
```

GPU候補は0・1・2、空いたGPUへ独立したジョブを1件ずつ割り当てる。候補を限定する場合は `--follow 0,2`。進捗・残り時間と日本時間の終了見込みを端末に表示し、`outputs/final_skip_20261128.log` に保存する。Ctrl+Cで表示を閉じても処理は継続し、`bash tools/run_final_skip.sh --watch` で再確認、`--status` で1回表示できる。完了したStage 1の完了印がある場合は再利用する。Stage 1の途中再開は未対応。

過去の12層後と4段置換の実測速度から、Stage 1終了まで3 GPUで約34時間、2 GPUで約41時間が初期目安。GPU待ち、起動・保存・テスト、Semantic・履歴評価は別途かかる。実行時の進捗から予測を更新する。

- Stage 1の出力は `outputs/stage1_final_skip_{smoke,full}_20261128/`。
- Semanticの集計は `outputs/stage2_final_skip_semantic_20261128/summary.json`。4条件のheadと履歴評価、比較対照の照合、全ワーカー正常終了後に親の `_SUCCESS` を作る。
- 各行の `delta_from_replacement_pp`、`delta_from_final_block_only_pp`、`history.continuous_minus_reset_pp` と `history.learned_skip_scales` を読む。12層後だけの条件は、同位置の対照が既存12層後E1なので前二者が同じ値になる。
- 今回新規に作る一時特徴キャッシュは、そのジョブの必要な評価が終わった後に解放する。backbone、head、評価レポート、履歴比較は保存する。


## 6層後・9層後・3＋6＋9層後のスキップ接続

3層後スキップ接続で得たDSECの改善（実験台帳§111）が、別の深さや複数配置でも得られるかを調べる。seed 0、DSEC・M3EDそれぞれの既存Stage 1設定でsmokeを99更新、本学習を99,999更新とし、backboneを凍結して最終特徴を読むSemantic headを学習する。新しい全条件で12層後は通常のViT出力を使う。

| 新条件（すべてE1 Encoder＋LSTM（h蒸留）、Patchwise LSTM） | 同位置の置換対照 | 比較の意図 |
| --- | --- | --- |
| `E1_b6_skip` 6層後に1個 | `E1_b6` 6層後、学習済み | 中間層の元特徴を残す効果 |
| `E1_b9_skip` 9層後に1個 | `E1_b9` 9層後、学習済み | 後段の元特徴を残す効果 |
| `E1_b3_6_9_skip` 3・6・9層後に各1個 | `E1_b3_6_9` 3・6・9層後、新規学習 | 同じ3位置・3個のLSTMで接続方式を比較 |

各挿入位置では `x_l + α_l × u_l` を次のブロックに渡す。元の特徴 `x_l` をそのまま渡す初期状態から学習し、3段モデルのαは各位置で独立に更新する。6・9層後の置換対照は従来の6配置の検証済み結果を再利用する。DSEC公式test・M3ED公開valの全体と活動量Q1〜Q5で、同位置の置換対照との差と12層後E1との差を集計する。各スキップモデルについては同じheadで全LSTM状態を毎窓リセットした場合とも比較し、履歴を保持した効果と層別のαを保存する。3段のリセット比較が測るのは3個の状態を合わせた履歴の効果。

学習サーバーにコードを同期後、初回は次で開始する。

```bash
cd /home/iASL/Arata_repo/EventState && git pull --ff-only && bash tools/run_skip_placements.sh --follow
```

候補GPUは既定で0・1・2。Stage 1、Semanticともに同時に使えるGPUへ独立した条件を1件ずつ割り当てる。ログは起動端末と `outputs/skip_placement_20261128.log` に出る。Ctrl+Cで表示を閉じた後は `bash tools/run_skip_placements.sh --watch` で既存の出力を確認できる。起動時にGPUを限定する場合は `--follow 0,2`、同じ並列数で残り時間を見る場合は `--watch 0,2` とする。

3種のスキップ＋1種の新規対照を2データセットで実行し、各段階は8ジョブ。3 GPUを継続利用できる想定でStage 1終了まで約57時間、2 GPUなら約80時間。6・9層後は同じ深さの既存モデル、3・6・9層後は同じLSTM数の既存6・9・12層後モデルの速度からの初期予測。GPU待ちや起動・保存、後続のSemantic・履歴評価には別途時間がかかる。実行中は進捗と日本時間の終了見込みを更新する。

Stage 1の出力は `outputs/stage1_skip_placement_{smoke,full}_20261128/`。Semanticの集計は `outputs/stage2_skip_placement_semantic_20261128/summary.json`。新規スキップの各行には同位置の置換対照との差 `delta_from_replacement_pp`、12層後との差 `delta_from_final_block_only_pp`、履歴の効果 `history.continuous_minus_reset_pp`、層番号別の `history.learned_skip_scales` が入る。3・6・9層後スキップの比較行は、同じデータセットの置換対照も検証されてから集計される。今回のSemanticワーカーが新規に作る一時特徴キャッシュは、そのジョブの必要な評価が完了した後に解放する。学習済みbackbone、head、評価レポートは保存される。

## 3層後のスキップ接続

2026-11-28にDSEC公式testの結果を受領した（[実験結果台帳](experiment_results.md) §111）。M3ED公開valは結果待ち。DSEC・M3EDで、3層後に1個のPatchwise LSTMを置く2条件を比較する。どちらもE1 Encoder＋LSTM（h蒸留）で、Semantic headは12層まで通した最終特徴を読む。

| 条件ID・配置 | 第4ブロックに渡すパッチ特徴 |
| --- | --- |
| `E1_b3` 3層後・置換（既存の対照） | 正規化した第3ブロック出力をLSTMに通し、線形射影した更新量 `u₃` |
| `E1_b3_skip` 3層後・スキップ接続 | 正規化前の第3ブロック出力 `x₃ + α × u₃`。αは層全体で共有する学習可能なスカラーで初期値0 |

seed 0、同じRGB教師・訓練系列・イベント窓・99,999更新を使う。SemanticはDSEC公式testとM3ED公開valで、全体・活動量Q1〜Q5のmIoUを求める。置換対照の全体mIoUはDSEC 51.218%、M3ED 30.963%。測る差は次の2つ。

- **接続を変えた効果**は、連続推論での `E1_b3_skip − E1_b3`。12層後の既存E1との差も集計する。
- **スキップ接続モデルが履歴を使う効果**は、同じ学習済みbackboneとheadでの「連続推論 − 毎イベント窓の直前にLSTM状態をゼロにした推論」。後者でも現在窓と残差枝の計算を使う。学習後のαも記録する。αは枝の出力スケールにも依存するため、履歴の寄与はmIoU差で判断する。

サーバーに変更を同期後、**新しく開始する場合**は次を実行する。GPU 0・1・2の空きに合わせてDSEC・M3EDを最大2件並列にし、smoke、本学習、Semantic、履歴の評価まで順に進める。起動端末に進捗が流れ、同じ内容を `outputs/b3_skip_20261126.log` に追記する。Ctrl+Cでログ表示を閉じてもサーバーの実験は続く。

```bash
cd /home/iASL/Arata_repo/EventState
bash tools/run_b3_skip.sh --follow
```

**実行中のものを見る場合**は `bash tools/run_b3_skip.sh --watch` で既存の出力を1分ごとに読む。`--status` は1回だけ表示する。どちらも学習を起動しないため、以前のコマンドで開始した実験にも使える。起動時に候補GPUを1枚に限定した場合は `--watch 0` のように同じ候補を渡す。新しい親ログの追跡は `tail -n 30 -F outputs/b3_skip_20261126.log`。

全工程の見積もりは起動時と5分ごとに表示する。Stage 1は今回のstepと処理速度を使い、開始前は既存の3層後置換の実測速度を使う。2 GPUなら各段階でDSEC・M3EDの長い方、1 GPUなら両方を足す。空きGPUを待つ時間は別途かかる。終了見込みは日本時間。未計測の工程が残る間は、見積もれる部分の残り時間と未計測部分を併記する。

集計先は `outputs/stage2_b3_skip_semantic_20261126/summary.json`。スクリプトは完了印のあるStage 1を再利用する。中断後にSemanticを再実行すると、検証済みのheadを再利用できる。

## 配置6条件の比較

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

2026-11-23に実装し、11月25日に全4件の比較が完了した。4層連結で12層後のみと4段配置の全体mIoU差はDSECで0.191ポイント、M3EDで0.163ポイント縮まった。[実験台帳 §107](experiment_results.md#107-semanticの4層連結でlstm配置間の差は約02ポイント縮小2026-11-25)

§103のStage 1 train lossは両データセットで4段の方が高かった。最終層の蒸留後も中間層にクラス判別情報が残るかを、同じSemanticタスクで調べる。seed 0・99,999更新のStage 1を固定し、各データセットで次の4通りを比較する。

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
