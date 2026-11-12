# Common Depthを3GPUで実行する

2026-11-11追加。`tools/run_common_depth.py --gpus 0,1,2`で、独立した条件を各GPUに1本ずつ、最大3本同時に実行する。
既存runの完了済みhead・評価を再利用する。Depth headのbatch、loss、seed、epoch数、split、best選択基準は変更しない。
条件名の旧型DDP・新型DDPはStage 1重みの学習方式を表し、今回のDepth headをDDPで学習する意味ではない。

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
python tools/run_common_depth.py --gpus 0,1,2 --preflight

# 同じrun rootを使い、完了済み成果物から再開する。
python -u tools/run_common_depth.py --gpus 0,1,2
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
起動表示のGPUは0/1のみとなる。切替や速度改善の実確認はサーバー出力を受領後に行う。
同じDSECのhead処理なら現在の3並列約36分/epochに対し、2並列で約24分/epoch未満なら合計処理量が上回る目安。
M3EDや特徴抽出・後続採点へこの数値をそのまま適用しない。

### 共通動作

- DSEC/M3EDそれぞれ5条件：E0 Encoder（z蒸留）、E1 Encoder＋LSTM（h蒸留）、E2 Encoder＋LSTM（z+h蒸留）、
  E1 Encoder＋LSTM（h蒸留）・旧型DDP、E1 Encoder＋LSTM（h蒸留）・新型DDP。
- GPUごとに1条件を割り当て、終了したGPUに次の条件を割り当てる。
  並列実行ではDSECとM3EDが重なる場合がある。`--gpus`なしの既存直列経路はDSEC→M3EDの順を維持する。
- 特徴cacheの必要容量を合算して予約し、10%余裕を含めて入る条件だけ起動する。
  容量不足時は`Scratch wait`を表示する。小さいM3ED条件や特徴cache不要の視差採点を先に実行できる。
  単独でも入らない場合は明示的に停止する。3GPU同時使用や3倍の速度は保証しない。
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
通常のDepth学習を3GPU化するだけでは、まだ生成していない活動manifestや追加評価は自動実行しない。
必要ならDepthの既存`--activity-only`にも`--gpus 0,1,2`を併用できる。

Macで関連stdlib91件（並列処理18件を含む）、変更新規Python34ファイルの構文、CLI、差分確認が通過。
模擬workerの3同時実行、GPU環境変数の分離、scratch予約・待機、完了E0の再利用、停止時の子孫終了を確認した。
実GPU・実データの3並列実行はサーバーで確認する。
