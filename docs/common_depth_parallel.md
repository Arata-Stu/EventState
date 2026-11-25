# Common Depthを最大2並列で実行する

2026-11-18の運用方針。独立した条件を各GPUに1本ずつ割り当て、Depthの下流はデータセットを合わせて同時2本までとする。
Common Depth、ConvLSTMの混合ランナー、F³の並列ランナーは同じホスト・ユーザーで実行枠を共有する。以下は他にDepthが走っていないときの例。旧版や個別スクリプトで別途1本実行中なら、ここには空きGPUを1枚だけ指定する。
既存runの完了済みhead・評価を再利用する。Depth headのbatch、loss、seed、epoch数、split、best選択基準は変更しない。
条件名の旧型DDP・新型DDPはStage 1重みの学習方式を表し、今回のDepth headをDDPで学習する意味ではない。

## 4段配置のDepth評価

`--four-stage-placement` は `E1_b3_6_9_12` E1 Encoder＋LSTM（h蒸留、3・6・9・12層後）について、DSEC・M3ED各1本のDepth headを学習・評価する。両方ともseed 0、既存の99,999更新のStage 1を凍結し、VKITTI初期化のDPTを50 epoch、batch 4で学習する。対照のE1 Encoder＋LSTM（h蒸留、12層後）もDPTが同じ4位置の特徴を読む。4段配置では4位置すべてがLSTM通過後、対照では最後の位置がLSTM通過後である。

11月10日のCommon Depthと同じtarget indexを使う。DSECはtrain 37系列・公開val 4系列、M3EDはtrain 3系列・公開val 2系列。全体と共通活動Q1〜Q5の画素単位MAE・フレーム平均MAE、およびDSEC公開valの視差MAEを集計する。比較対象は完了済み12層後E1の記録で、公開valのpooled MAEはDSEC 2.445824 m、M3ED 3.678082 m。

変更をMacからpush後、`sig-gpu5` のbashで実行する。過去の同じtarget件数でscratch予約はDSEC約99.6 GiB、M3ED約119.2 GiB、2件合計約218.8 GiB。ここでは以前の `df` で `/` に余裕があったため `/var/tmp` を使う。現在の空きと必要量はpreflightに表示される。GPU 0・1・2の空きと共有Depth枠に合わせて最大2並列で実行し、同時に容量が足りなければ順に処理する。

```bash
cd /home/iASL/Arata_repo/EventState
git pull --ff-only
df -h /home /var/tmp
args=(--four-stage-placement --gpus 0,1,2 --scratch "/var/tmp/$USER/eventstate-placement-depth")
if env/bin/python -u tools/run_common_depth.py "${args[@]}" --preflight; then
  log="outputs/placement_depth_$(date +%Y%m%d_%H%M%S).log"
  nohup env/bin/python -u tools/run_common_depth.py "${args[@]}" >"$log" 2>&1 </dev/null &
  echo "PID=$!  log=$log"
  tail -f "$log"
fi
```

完了時の `outputs/stage2_placement_depth_20261126/summary.json` に両条件の結果が入る。各データセットの `comparison_to_E1.groups` は、時刻・正解・有効画素・活動群の一致を照合した上で、`all` と5群の4段配置と12層後E1のMAEを並べる。`delta_m` は4段配置から12層後E1を引いた差で、負の値が改善を表す。再実行でも同じ出力先を指定すると、完了したheadと評価を再利用する。

## 現在の直列実行から切り替える

**先に、現在の学習を実行した元のターミナルでCtrl+Cを1回押し、shellのpromptへ戻るまで待つ。**
その後、同じrunの特徴抽出・学習・評価の子プロセスも終了したことを確認する。
別jobまで止める`pkill python`やscratch全体の削除は行わない。

学習途中のheadは保全用の別名へ移し、その条件の学習を最初からやり直す。
既存の特徴cacheは一時パスを含むため、途中epochからの継続は保証しない。
`_SUCCESS`と`best.pt`が揃う完了済みheadは再学習せず、不足している評価だけ行う。
受領ログのDSEC E0 Encoder（z蒸留）は50epochと公開val採点まで到達しているが、切替時点の実ファイルはサーバー側で確認する。

sig-gpu5で、元のrunが停止してから実行する。

```bash
cd /home/iASL/Arata_repo/EventState
source env/bin/activate
nvidia-smi
git pull --ff-only

# 既存target index・重み・head・評価の整合性と実行計画を確認。推論・出力作成はしない。
python tools/run_common_depth.py --gpus 0,1 --preflight

# 同じrun rootを使い、完了済み成果物から再開する。
python -u tools/run_common_depth.py --gpus 0,1
```

`nvidia-smi`で他の学習がGPUを使っている場合は、そのGPUを割り当てない。
preflightはGPU上でのモデルロードやVRAM容量を検証するものではない。
今回のrun rootは既定の`outputs/stage2_common_depth_vkitti_20261110`。
元の起動時にdata/output/scratch等を変更した場合は、preflight・本実行の両方へ同じ引数を追加する。

## 並列実行と保存先

### 2GPU並列への切替（2026-11-12）

ユーザーがCPU負荷と実測時間を踏まえ、まず2並列を試すと決定。
現行コードのまま`--gpus 0,1`で実行できるため、コード更新は不要。
今回はCPU thread数等を変更せず、並列数だけを変える。

元の3GPU実行ターミナルでCtrl+Cを1回押し、promptへ戻り子処理が終了してから実行する。

```bash
cd /home/iASL/Arata_repo/EventState
source env/bin/activate
nvidia-smi

python tools/run_common_depth.py --gpus 0,1 --preflight &&
python -u tools/run_common_depth.py --gpus 0,1
```

同じ`outputs/stage2_common_depth_vkitti_20261110`を使うため、完了head・採点結果を再利用する。
学習途中のheadは保全して最初から学習する。中断cacheは残る場合があるが、scratch全体の削除はしない。
起動表示のGPUは0/1のみとなる。2026-11-13受領ログで2GPU実行とDSEC5条件/M3ED E0の完了を確認（台帳§78）。
2026-11-13、DSEC新型DDPのconfig.json→_SUCCESS時刻差を受領し、2並列時は平均15.35分/epochだった（台帳§79）。
以前の3並列DSEC3条件平均35.99分/epochより約57%短い。条件・同時実行相手が異なる比較として扱い、当面2並列を維持する。
2026-11-15の最新親ログで全10条件、DSEC test552枚ZIP、最終summaryの生成完了を確認（台帳§82）。
次は全体スコア回収と、Semantic/Depthの共通活動5分位による追加評価へ進む。
同じDSECのhead処理なら現在の3並列約36分/epochに対し、2並列で約24分/epoch未満なら合計処理量が上回る目安。
M3EDや特徴抽出・後続採点へこの数値をそのまま適用しない。

### 共通動作

- DSEC/M3EDそれぞれ5条件：E0 Encoder（z蒸留）、E1 Encoder＋LSTM（h蒸留）、E2 Encoder＋LSTM（z+h蒸留）、
  E1 Encoder＋LSTM（h蒸留）・旧型DDP、E1 Encoder＋LSTM（h蒸留）・新型DDP。
- GPUごとに1条件を割り当て、終了したGPUに次の条件を割り当てる。
  並列実行ではDSECとM3EDが重なる場合がある。`--gpus`なしの既存直列経路はDSEC→M3EDの順を維持する。
- 特徴cacheの必要容量を合算して予約し、10%余裕を含めて入る条件だけ起動する。
  容量不足時は`Scratch wait`を表示する。小さいM3ED条件や特徴cache不要の視差採点を先に実行できる。
  単独でも入らない場合は明示的に停止する。
- 各workerは個別のログ・状態を保存する。起動表示の条件ID・条件名・GPU番号・logパスで進捗を確認する。
  既存のhead・JSONと最終`summary.json`の配置は維持する。
- run全体のlockを共有し、同じrun rootへの二重起動を拒否する。
  最終集計とDSEC選択条件の公式test ZIP生成は親が一度だけ実行する。
  全体完了は`Completed: outputs/stage2_common_depth_vkitti_20261110/summary.json`で確認する。

停止・失敗時は今回のworkerと子プロセスをまとめて終了し、学習済み成果物とログを残す。
中断したworkerの一時cacheが残る場合がある。容量が不足した場合は、ログに記録された**今回の一時cacheだけ**を確認して整理する。
破損した通常評価JSONは別名に保全して不足分を再評価し、入力のidentity不一致は停止する。
活動別reportの破損・来歴不一致は停止するため、表示された対象を確認する。

## 学習完了後

Semantic・Depthの共通活動5分位評価は[既存の一括手順](semantic_depth_activity.md)で実行する。
Depthの追加評価には、活動manifestを用意して `--activity-only --gpus 0,1` を指定できる。

2026-11-11の実装時はMacで関連stdlib91件（並列処理18件を含む）、変更新規Python34ファイルの構文、CLI、差分確認が通過。
模擬workerの3同時実行、GPU環境変数の分離、scratch予約・待機、完了E0の再利用、停止時の子孫終了を確認した。
