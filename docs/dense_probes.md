# DSEC / M3ED のDepth・視差評価

既存のDetection/Semanticに加え、frozen EventState特徴 `z/h/concat` から
非再帰の小さなheadを学習する。今回追加するタスクは **単眼metric depthとDSEC視差**。
Optical Flow・Poseのheadは今回の実装範囲に含めない。
学習・Torch/NumPyテスト・データ処理はLinux学習サーバーで実行する。

## 分割の根拠とラベル提供状況

| protocol | head train | val | official test | ローカル精度評価 |
|---|---:|---:|---:|---|
| `dsec_f3` | 37系列 | 4系列 | 12系列・正解非公開 | F3の4val |
| `m3ed_f3` | 車載昼間3系列 | 同2系列 | 昼間2系列・正解/RGB非公開 | F3の2val |
| `dsec_official` | 公式41系列 | なし | 公式12系列・正解非公開 | trainは学習のみ、test提出用PNG生成 |
| `m3ed_official_car_day` | 昼間ラベル付き6系列 | なし | 公式昼間2系列・正解/RGB非公開 | testのローカル採点不可 |

独自のランダム分割や時刻でのtrain/val分割は行わない。
F3のvalをtestへ呼び替えない。`m3ed_official_car_day`は公式配布表の車載昼間部分集合であり、
全プラットフォームの公式Depthランキング用splitがあるという意味ではない。
M3EDに新しいラベル付きtestを独自に作ることもしない。
公開GTがないofficial testへの完全な学習/採点実験は実行できないため、
**M3EDのローカル比較にはF3の公開train/valを使用する**。
M3ED official testの時刻生成・イベントのみの前処理は未対応で、既存のRGB時刻に基づく
prepare_m3ed.pyをtestへ流用しない。`m3ed_official_car_day`は分割監査用でもある。

F3設定は次の公開revisionに固定した。
`833e2c65233f9787449ce0675ae41e3f6e9c193a`（2026-10-27確認）。
JSON manifestには参照URL・revision・元設定のSHA256を保存した。

- [F3 Depth README](https://github.com/grasp-lyrl/fast-feature-fields/blob/833e2c65233f9787449ce0675ae41e3f6e9c193a/src/f3/tasks/depth/README.md)
- [DSEC train](https://github.com/grasp-lyrl/fast-feature-fields/blob/833e2c65233f9787449ce0675ae41e3f6e9c193a/confs/monocular_depth/full_dsec_gt_train.yml)、[val](https://github.com/grasp-lyrl/fast-feature-fields/blob/833e2c65233f9787449ce0675ae41e3f6e9c193a/confs/monocular_depth/full_dsec_gt_val.yml)
- [M3ED train](https://github.com/grasp-lyrl/fast-feature-fields/blob/833e2c65233f9787449ce0675ae41e3f6e9c193a/confs/monocular_depth/full_car_urban_day_gt_train.yml)、[val](https://github.com/grasp-lyrl/fast-feature-fields/blob/833e2c65233f9787449ce0675ae41e3f6e9c193a/confs/monocular_depth/full_car_urban_day_gt_val.yml)
- [DSEC公式データ](https://dsec.ifi.uzh.ch/dsec-datasets/download/)、[データ形式](https://dsec.ifi.uzh.ch/data-format/)、[提出形式](https://dsec.ifi.uzh.ch/disparity-submission-format/)
- [M3ED公式系列表](https://m3ed.io/sequences/)、[データの説明](https://m3ed.io/data_overview/)

F3と揃えるのは系列分割。F3はDepthAnything V2 decoder等を使用しており、
本実装の2層Conv head、入力解像度、時間窓、事前学習、指標集計が同一ではない。
さらに本実装は左イベントのみの単眼probeで、DSECのステレオ手法と同じ入力条件ではない。
splitだけの一致をもってF3完全再現やステレオランキングの公平比較とは主張しない。

## リーク防止

### DSEC全タスク共通splitと共通backboneの区別（2026-10-31調査）

追記: ユーザー指定により独自splitを除外した公開分割監査は `dsec_published_split_audit.md` と
同名JSONを正本とする。以下のSemantic開発6/2や35系列案はその監査に含めない。
DSEC-DepthにはF3の37/4だけでなくDepthAnyEvent公開31/10もあり、同一splitとして扱わない。

確認した公式配布/各タスク実装には、Detection/Semantic/Depth/Flowのすべてに対して同一系列・
同一train/val/testを保証する統一標準splitは見つからなかった。
元DSECのtrain/test区分は共通の土台だが、Semanticは11系列のみ、Detectionは追加系列を含む60系列。
全系列に全タスクGTがあるわけではなく、Depth/Flowのtest GTは非公開。
公式根拠: https://dsec.ifi.uzh.ch/dsec-datasets/download/ 、
https://dsec.ifi.uzh.ch/dsec-semantic/ 、https://dsec.ifi.uzh.ch/dsec-detection/ 。
F3のDepth4valや本repoのSemantic開発2valは、元DSECのtrain内に確保されたholdoutである。

全タスクで同じ事前学習backboneを使うことは設計可能。下流の公式/公開splitを変更せず、
共通の事前学習候補から全対象タスクのval/test系列の和集合を除外し、そのtrain-only統計で学習する。
共有重みをfrozenで使いタスクごとに独立headを学習すれば、他タスクのhead学習でbackboneへ
評価系列が混入することも避けられる。再帰状態も系列境界でresetする。
例としてDetection公式+Semantic開発6/2+F3 Depth37/4だけなら、候補41系列から
Depth4valとSemantic2valを除外し35系列となる（stdlibでmanifestから確認）。
これは独自の「共通事前学習集合」の設計例であって、公式の35系列splitや全タスク標準ではない。
Flow等を含めるなら、採用する先行研究のholdoutを追加監査してから確定する。
既存dsec_joint_cleanはDetection/Semantic用であり、Depth/Flowまで監査済みと扱わない。
今回完了したDSEC37重みはSemantic開発val07_a/08_aを含むため、この35系列案とは異なる。
現時点で再学習や既存分割の変更は開始していない。

### GEPとの比較を検討する場合（2026-10-31調査）

Generative Event Pretraining with Foundation Model Alignmentの論文§4.4/Table3のDepthはMVSECであり、
DSEC/M3EDのDepthではない。train=outdoor_day2、validation=outdoor_day1/outdoor_night1/2/3。
独立testへの呼び替えや新しいhold-outはしない。
論文: https://rpg.ifi.uzh.ch/docs/CVPRF26_Cao.pdf
公式config: https://github.com/uzh-rpg/generative_event_pretraining/blob/master/src/config.py
ローカル参照checkoutはreference_repo/generative_event_pretraining、HEAD=b07c78c8124ab60682d2aa440a2aa52911de9c36。
DEPConfigのmvsec_train_sequences/valid_sequencesでも上記系列を確認。
ただし論文は評価最大距離制限なし、公開configのmvsec_eval_max_depth=30かつDepthClampあり。
公表値再現にはこの差、decoder/encoder fine-tuning/EventScape追加学習等を監査する必要がある。
現在のDepth probe既定0.1–80mとそのまま数値比較しない。

GEPのDSEC Semantic公開loaderは公式train/testフォルダを使用（config上valid_datasetがsplit=test）。
本repoの開発6train/2valと同じ母集団ではなく、公表値比較では公式train8/test3で最終評価を揃える。
GEP論文の主評価にM3EDやDetectionのsplitは示されていないため、今回のF3 splitをGEP由来と呼ばない。

比較案は2系統: (1)MVSECを追加しGEP論文protocolに合わせる、
(2)GEPを今回のDSEC37/4・M3ED3/2で再評価し「F3 splitでの追加実験」と明記する。
GEP論文はEventScape/N-ImageNet/DSECのEvent-1.8M事前学習を使用、MVSECは事前学習から除外。
DSECを含む既存GEP重みはF3 valの使用歴を監査するまでstrict未見評価に使えない。
strict DSEC比較ならGEP側もtrain37のみで学習/統計算出する等の統制が必要。
M3EDは論文記載の事前学習集合にないが、実際の使用重み履歴も確認する。
事前学習データ量/教師/時間窓/headとfreeze条件を別途報告する。
以上は比較方法の調査・提案であり、新規実装/学習/重みdownloadは未実施。

事前学習の勾配更新系列だけでなく、validationによるcheckpoint選択系列、明示された過去の
pretraining_history、正規化統計の算出系列を監査する。
下流val/testと重複する重みを新しいstrict評価に使うと停止し、bypassは設けていない。
train/val/testの特徴を混ぜる、別backbone/geometryのheadを評価に使う、
非公開testをローカル採点する、といった操作も拒否する。
正規化統計は下流trainだけから算出し、係数と系列をcheckpoint configへ埋め込む。
事前学習は固定100kでvalidation無効。headはF3ならval MAEで選択、officialなら固定最終epoch。
testへ合わせてepoch・閾値・距離範囲を変更しない。

**既存重みの扱い:** DSEC train41はF3の4valを事前学習に含むため、このF3評価へは使用不可。
M3ED既存4train/1valもucity_small_loopの検証使用がF3のvalと重なる。
これらの従来結果を遡及的に否定する変更ではなく、今回の新しいstrict protocolへの適合性の問題。
新規F3比較にはclean splitでの事前学習が必要。公式DSEC train/testならtrain41は系列上適合するが、
別途その重みの正規化統計の出典確認が必要。
RGB基盤モデルそのものの学習データまでは追跡できず、その範囲は監査済みとはしない。
resume/warm-startで過去重みを使った場合は、以前の学習系列をpretraining_historyへ明示する。

## 実行順序

### 統計・4条件smoke通過後の一括コマンド（2026-10-27）

ユーザーから統計生成とDSEC/M3ED × baseline/alpha1 smoke成功を受領（台帳§49）。
既存RGB間隔GEPとtrain-only統計をそのまま使う。次のコマンドはサーバー専用。
スクリプトを同期後、同じRUN_ROOTを全段階で指定する。

```bash
cd /home/iASL/Arata_repo/EventState
source env/bin/activate
export DATA_ROOT=/home/iASL/Arata_repo/dataset
export CUDA_VISIBLE_DEVICES=0
RUN_ROOT="$PWD/outputs/dense_f3_full_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_ROOT"
printf '%s\n' "$RUN_ROOT" > outputs/dense_f3_latest_run.txt

# 4条件を順次、smokeから継続せずDINO初期値から各100k step。
bash tools/run_dense_f3_pipeline.sh pretrain "$RUN_ROOT"

# 公開GTの所在が下記既定値と異なる場合、この2変数だけ修正する。
export DSEC_TARGET_ROOT="$DATA_ROOT/DSEC/train_disparity"
export DSEC_CALIBRATION_ROOT="$DATA_ROOT/DSEC/train_calibration"
export M3ED_TARGET_ROOT="$DATA_ROOT/m3ed_cache/m3ed_downstream"
bash tools/run_dense_f3_pipeline.sh targets "$RUN_ROOT"

# 新規特徴cacheを保存する場所。大きくなるため空き容量を確認する。
export CACHE_ROOT="$DATA_ROOT/downstream_scratch/$(basename "$RUN_ROOT")"
mkdir -p "$CACHE_ROOT"
df -h "$CACHE_ROOT"
bash tools/run_dense_f3_pipeline.sh depth "$RUN_ROOT"
```

端末再接続時は `RUN_ROOT="$(cat outputs/dense_f3_latest_run.txt)"` で同じ出力先を復元し、
export変数も再設定する。既存出力への再実行は各Python CLIが拒否する。
途中停止からの自動resume/既存cache削除は行わない。ログはRUN_ROOT/logsへ追記する。
一部条件だけ実行するなら `DATASETS=m3ed CONDITIONS=alpha1 bash ... depth "$RUN_ROOT"`。
これは未実行条件の選択用であり、途中まで生成された同条件cacheの再開ではない。

targetsはDSEC37train/4valとM3ED3train/2valをindex化するだけで、正解をダウンロード/生成しない。
M3ED既存正解にdepthまたは全5系列がない場合は、新しい出力先に準備する。
下記M3ED raw rootは実際の公式HDF5配置に合わせる（今回存在は未確認）。

```bash
export M3ED_TARGET_ROOT="$RUN_ROOT/m3ed_depth_targets"
python tools/prepare_m3ed_downstream.py --root "$DATA_ROOT/M3ED" \
  --prepared-root "$DATA_ROOT/m3ed_cache/half_dagr" --output-root "$M3ED_TARGET_ROOT" \
  --tasks depth --require-all \
  --sequences car_urban_day_city_hall car_urban_day_penno_big_loop car_urban_day_penno_small_loop \
              car_urban_day_rittenhouse car_urban_day_ucity_small_loop
# 未作成のM3ED indexのみを作る。既存DSEC indexを再生成しない。
DATASETS=m3ed bash tools/run_dense_f3_pipeline.sh targets "$RUN_ROOT"
```

Depth段階は4条件×3特徴で12 head、50epoch/seed0、val MAE最良best.ptを使い計24評価。
状態保持と毎frame resetは同一headで比較する。
結果: `heads/DATASET_CONDITION/{z,h,concat}/{continuous,frame,state_delta}.json`、
モデル差分: `comparison/DATASET_FEATURE_POLICY.json`（alpha1−baseline）。
state_deltaはcontinuous−frame。MAE/RMSE/AbsRel等の誤差は負が改善。
全体/低イベントframe/継続時間別を同時出力する。空群はnull。
正規化と低イベント率p10はtrain由来で固定、head選択は公開val、testは不使用。
単一seedの小型frozen probeであり、論文のdecoderや完全な評価条件の再現ではない。
新規cacheは自動削除せず蓄積する。容量不足時は停止して保存先を検討し、既存M3ED cacheを削除しない。

以下は個別CLIの説明。repoを同期済みの **サーバー** で実行する。

```bash
cd /home/iASL/Arata_repo/EventState
source env/bin/activate
python -m pytest -q tests/test_dense_protocol_stdlib.py tests/test_dense_probe.py tests/test_training_runtime.py
df -h /home/iASL/Arata_repo/dataset

# 最初はsplit表示だけ。checkpointを追加すると重複監査できる。
python tools/audit_dense_protocol.py --protocol dsec_f3
python tools/audit_dense_protocol.py --protocol m3ed_f3
```

### 1. trainだけの正規化とclean事前学習

既存の未正規化GEP/固定RGB教師cacheは再利用する。既存cacheを削除・上書きしない。

```bash
DATA=/home/iASL/Arata_repo/dataset
TEACHER=/home/iASL/Arata_repo/models/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
python tools/compute_dense_normalization.py \
  --protocol dsec_f3 --event-cache-dir "$DATA/DSEC_cache/events/gep_rgb" \
  --output "$DATA/DSEC_cache/dsec_f3_gep_stats.json"

# 既定はコマンド表示のみ。smokeを実行する場合だけ--executeを追加。
python tools/run_dense_pretrain.py --protocol dsec_f3 \
  --statistics "$DATA/DSEC_cache/dsec_f3_gep_stats.json" \
  --data-root "$DATA/DSEC" --event-cache-dir "$DATA/DSEC_cache/events/gep_rgb" \
  --teacher-cache-dir "$DATA/DSEC_cache/dinov3_vits16" --teacher-checkpoint "$TEACHER" \
  --condition baseline --stage smoke --output-dir outputs/dsec_depth_f3_smoke
```

成功後、新しいoutput-dirで`--stage full --execute`。smokeから継続しない。
α1比較は`--condition alpha1`と別のoutput-dir、同じseed/統計を使う。
DSEC固定20msなら統計生成も`--event-cache-dir`の代わりに
`--raw-event-root "$DATA/DSEC" --window-us 20000`で別JSONへ行う。
その統計でpretrain launcherへ`--window-us 20000`を追加し、下流でもraw固定20msを使う。
このlauncherは既存の事前学習設定・旧manifestを変更せず、新しいprotocolを明示して起動する。
M3EDは`--protocol m3ed_f3`、event-cache-dir/data-rootを`$DATA/m3ed_cache/half_dagr`、
teacher-cache-dirを`$DATA/m3ed_cache/dinov3_vits16_640x352`に置き換える。
新しい統計は同じcompute_dense_normalization.pyでM3EDの3trainだけから計算する。

### 2. 正解のindex作成

DSECは公式`train_disparity`とcalibrationを用意する。対象タスクは`disparity`または`depth`。
depthへ変換する場合は各系列の`disparity_to_depth.cams_03`の4x4 Qを使用し、共通のf*b値は仮定しない。

```bash
python tools/prepare_dense_targets.py --protocol dsec_f3 --role train --task depth \
  --target-root "$DATA/DSEC/train_disparity" --calibration-root "$DATA/DSEC/train_calibration" \
  --output outputs/dsec_depth_f3_train_targets.json
python tools/prepare_dense_targets.py --protocol dsec_f3 --role val --task depth \
  --target-root "$DATA/DSEC/train_disparity" --calibration-root "$DATA/DSEC/train_calibration" \
  --output outputs/dsec_depth_f3_val_targets.json
```

root直下は`SEQ/disparity/event/*.png`と`SEQ/disparity/timestamps.txt`を想定する。
disparityだけの評価ではcalibration-rootは不要。
indexは系列ごとのtimestamp/PNG数・順序を検査し、元ファイルのSHA256を保持する。

M3EDは既存prepare_m3ed_downstream.pyのdepth出力（rectified 352x640、meters）を再利用する。
F3用の3trainと2valの全5系列に対してdepthが必要。旧4train/1valで用意していない
rittenhouseも、公式depthが提供されている。

```bash
python tools/prepare_m3ed_downstream.py --root "$DATA/M3ED" \
  --prepared-root "$DATA/m3ed_cache/half_dagr" --output-root "$DATA/m3ed_cache/downstream" \
  --tasks depth --require-all \
  --sequences car_urban_day_city_hall car_urban_day_penno_big_loop car_urban_day_penno_small_loop \
              car_urban_day_rittenhouse car_urban_day_ucity_small_loop
python tools/prepare_dense_targets.py --protocol m3ed_f3 --role train --task depth \
  --target-root "$DATA/m3ed_cache/downstream" --output outputs/m3ed_depth_f3_train_targets.json
python tools/prepare_dense_targets.py --protocol m3ed_f3 --role val --task depth \
  --target-root "$DATA/m3ed_cache/downstream" --output outputs/m3ed_depth_f3_val_targets.json
```

既存downstream outputで別のtaskを生成済みなら、上の例は**新しいoutput-root**に変更する。
旧出力に`--overwrite`を付けて上書きしない。prepared eventsがない系列は先にprepare_m3ed.pyで準備する。
M3EDの正解は既存処理で許容50msの最近傍時刻へ対応付ける。これはGTの整列であり、
特徴へ未来イベントを入れることとは異なるが、境界/動く物体では時間差の影響をQAする。

### 3. 特徴抽出

```bash
CKPT=/absolute/path/to/clean_dsec_depth_run/checkpoints/step_00100000.pt
python tools/cache_dense_features.py --checkpoint "$CKPT" \
  --target-index outputs/dsec_depth_f3_train_targets.json \
  --statistics-provenance "$DATA/DSEC_cache/dsec_f3_gep_stats.json" \
  --event-cache-dir "$DATA/DSEC_cache/events/gep_rgb" --teacher-checkpoint "$TEACHER" \
  --state-policy continuous --output-dir "$DATA/downstream_scratch/depth_f3_train"
```

val indexでcontinuousとframeの2回を、別output-dirへ抽出する。
固定20ms重みでは`--event-cache-dir`の代わりに`--raw-event-root "$DATA/DSEC"`。
M3EDも同じCLIで、M3ED checkpoint/index/stats/prepared-rootを使う。

事前学習configと統計を監査してから抽出する。全イベントフレームを順番に処理して系列境界だけでresetし、
評価時刻のz/hだけを保存する。frame条件は毎フレームh/c双方をresetする。
ターゲット時刻と特徴は完全一致を要求し、未来特徴による最近傍置換をしない。
最初のRGBフレームに因果的イベント窓がない場合だけ、理由付きでmetadata.skippedへ記録する。
途中の欠損は停止。出力先が既存なら停止し、途中cacheを自動削除しない。

DSECはSemantic用440行cropを流用せず、480x640の全rectifiedイベント画角で抽出する。
ViTのpatch gridは30x40となる。M3EDは既存の640x360→中央352x640を維持する。
比較時にはこの解像度・crop・DAGR処理の差を報告する。

### 4. head学習と保持/reset評価

```bash
python tools/train_dense_probe.py \
  --train-cache "$DATA/downstream_scratch/depth_f3_train" \
  --val-cache "$DATA/downstream_scratch/depth_f3_val_continuous" \
  --feature h --epochs 50 --seed 0 --output-dir outputs/depth_f3_h
python tools/evaluate_dense_probe.py --checkpoint outputs/depth_f3_h/best.pt \
  --cache "$DATA/downstream_scratch/depth_f3_val_continuous" --output outputs/depth_f3_h/continuous.json
python tools/evaluate_dense_probe.py --checkpoint outputs/depth_f3_h/best.pt \
  --cache "$DATA/downstream_scratch/depth_f3_val_frame" --output outputs/depth_f3_h/frame.json
python tools/compare_dense_probes.py --baseline outputs/depth_f3_h/frame.json \
  --candidate outputs/depth_f3_h/continuous.json --mode state --output outputs/depth_f3_h/state_delta.json
```

z、concatも同じ条件で学習する。state比較は同一head・backbone・GT・評価画素・活動時系列を要求し、
z入力では特徴のSHA256不変性も要求する。モデル比較は`--mode model`で別head/backboneを許すが、
評価母集団と活動時系列は一致が必要。

metric depthはlog-L1で学習し、MAE/RMSE/AbsRel/SqRel/log-RMSE/δ1–3を出力。
DSEC視差はL1で学習し、native pixel単位MAE/RMSEと1PE/2PE（%）を出力。
評価時のGTを用いたmedian/scale/shift alignmentは行わない。
既定評価範囲はdepth (0.1,80]m、disparity (0,1000]px。範囲はhead configへ固定する。
集計はGT有効画素をプールした指標で、フレーム平均とは異なる。予測失敗画素は除外せず停止する。
全体、低活動frame、その他、低活動継続0–100ms/100–500ms/500ms以上を集計する。
空群はnull。有効画素数とframe数も保存する。

低活動は今回、既定で**trainだけから求めたイベント発生率の下位10%閾値以下**。
同じwindowでcompute_dense_normalization.pyが算出したevent_rate_p10_hzをval/testにも固定適用する。
各frameのevent_rate_hz（実際の入力表現に残るイベント数/窓長）を保存する。
trainで別途測定・事前固定した閾値は`--low-event-rate-threshold`でも指定できる。
従来の活動画素率10%と比較したい場合は、特徴抽出へ
`--low-activity-metric activity_fraction --low-activity-threshold 0.1`を明示する。
両定義を混ぜた結果比較は拒否する。継続時間はGTのない途中フレームも含めて累積する。
M3EDではDAGR後・352行crop前のイベント数なので、DSECのセンサーcountと直接横断比較しない。
継続時間は連続して低活動と判定された入力フレームの時刻間隔の累積であり、
正確なイベント停止時刻や物理的停止時間ではない。特に20ms窓/約50ms更新では、
入力窓の間の時間までイベントが少なかったと証明するものではない。
物体の追跡や車→道誤分類の評価は追加していない。Depthの改善だけで物体保持を証明しない。

### DSEC非公開test

`dsec_official`でtrainだけを固定epoch学習する（val-cacheなし、last.ptを使用）。
公式test_disparity_timestampsを取得し、次のようにindex化する。

```bash
python tools/prepare_dense_targets.py --protocol dsec_official --role test --task disparity \
  --target-root "$DATA/DSEC" --test-timestamps-root "$DATA/DSEC/test_disparity_timestamps" \
  --output outputs/dsec_disparity_test_targets.json
```

このindexでtest特徴を抽出した後、disparity学習headに
`evaluate_dense_probe.py --export --checkpoint .../last.pt --cache ... --output outputs/dsec_submission`
を使う。公式のtimestamp/indexに従うuint16 PNG（disparity×256）を生成する。
出力ディレクトリの外へmetadataを書き、公式checkerが要求するSEQ/*.pngだけをzipに入れられる。
自動アップロードやtestでのパラメータ選択は行わない。公式check_disparity_submission.pyで検査後に提出する。
これは単眼イベント手法であること、RGBは蒸留時のみ使用することを提出説明へ明記する。

## 検証境界

Mac: 標準ライブラリによるprotocol/リーク/対応比較テスト、AST、CLI help、コマンド計画のみ。
サーバー用数値テストにはPNG decoding/Qによる深度変換、無効画素、指標、head勾配、factory guardを含む。
ユーザーから修正後のサーバー全テスト、統計生成、4条件の事前学習smoke通過を受領（台帳§48–49）。
実Depth正解整列・Depth用全画角特徴抽出・本学習・下流精度・公式提出checkerは未確認。
run_dense_f3_pipeline.shはMacでbash構文と既存CLI引数の静的照合を実施。
新しいDepth精度結果はまだない。

2026-10-27の初回サーバーpreflightは6 failed/47 passed（13.22s）。
Hydra設定の過剰解決と保存一時ファイル名を修正し、子プロセスstderrの表示を追加した。
修正後のサーバー再実行は全通過報告あり。初回失敗の詳細は実験台帳§47を参照する。


## 自然な低イベント区間のz/h比較（2026-11-02）

主評価を、自然な入力を変更しないcontinuousのz対h（concatは補足）とする。
同じbackboneのzにもh蒸留の学習効果が入るため、これは読み出し比較であり、
独立に学習したLSTMなしモデルとのアーキテクチャ比較ではない。
各入力専用の同型・同じ学習設定の非再帰headを学習する。concatは入力次元が増える。
baseline/alpha1は両方LSTMありで、別軸の損失比較として保持する。

下流をまだ開始していないRUN_ROOTで、サーバー上で実行:

```bash
cd /home/iASL/Arata_repo/EventState
source env/bin/activate
RUN_ROOT=/home/iASL/Arata_repo/EventState/outputs/dense_f3_full_20261027_172342
CACHE_ROOT=/home/iASL/Arata_repo/dataset/downstream_scratch/dense_f3_full_20261027_172342 \
CUDA_VISIBLE_DEVICES=0 bash tools/run_dense_f3_pipeline.sh natural "$RUN_ROOT"
```

4モデル各train/val continuousだけを抽出し、12head/12評価、z対h/concat計8比較を生成。
frame-reset cache/評価は作らない。全入力を系列先頭から時系列順に処理し、系列境界でのみreset。
低イベントはtrain由来p10、継続時間は未ラベル入力も含めて計測する。
全体・低イベント・それ以外・低イベント継続0–100ms/100–500ms/500ms以上を集計。
空群はN/A。MAE/RMSE/AbsRel差は負が改善、delta1/2/3は正が改善。
継続時間群は同じ物体の追跡でも、一定時間後の因果効果の推定でもない。
信号待ちの場面ラベル、物体単位の保持/退出評価、細かい秒単位の長時間分析は未実装。

旧depth経路が完了済みなら、学習・cache抽出を繰り返さず既存continuous JSONから集計:

```bash
bash tools/run_dense_f3_pipeline.sh natural-summary "$RUN_ROOT"
```

比較JSONは `comparison/DATASET_CONDITION_z_vs_{h,concat}_natural.json`。
双方の絶対値・対象frame/画素数と差を保存し、端末には短いMAE表を表示。
backbone/cacheハッシュ、GT/有効画素、低イベント閾値、全frameの活動量/継続時間の一致を検査。
既存headやcache、比較出力は上書きしない。途中まで進んだrunにnaturalを再実行すると
既存出力で停止する（自動resumeではない）。natural-summaryは全必要continuous結果のある条件で使う。
DATASETS=m3ed CONDITIONS=baselineのように対象を限定できる。

Macでstdlib計20テストとbash構文・diff検査を実施。GPU・実データでの新経路は未実行。

## 中断したF3 Depthを3GPUで再開（2026-11-02）

元の実行端末でCtrl-Cして、親shellと子のcache/train/evalが止まったことを確認する。
新runnerは既存runを指す旧pipeline/workerをpsで検出すると拒否する。PIDの決め打ちはしない。
サーバーへコード同期後、次を実行する（Macでは学習しない）。

```bash
cd /home/iASL/Arata_repo/EventState
source env/bin/activate
python tools/run_dense_f3_parallel.py \
  outputs/dense_f3_full_20261027_172342 --gpus 0,1,2
```

既存DSEC alpha1 hのconfigに保存されたtrain cache metadata SHAで、run/featuresまたは
既定dataset/downstream_scratch直下の既存cacheを特定。曖昧なら停止し、元の--cache-rootを明示する。
GPU0=DSEC alpha1、GPU1=M3ED baseline、GPU2=M3ED alpha1を優先し、DSEC baselineも再利用確認する。
独立条件並列でbatch4/seed0/50epochは維持。全worker終了後にモデル比較を直列集計。
完了cacheはcheckpoint/target index hashとfeature存在確認、評価はhead/cache hashで再利用。
未完cacheは削除せず停止（別名に退避してから再生成）。各headはconfig一致を検査してlast.ptから再開。
途中epochは再実行。旧checkpointのloader乱数はseedからsamplerを再生するがbitwise一致は未保証。
新checkpointにはloader RNGを保存。最良val MAEのbest.ptを保持しlastの評価がより良ければ復旧する。
Macはstdlib3件・AST・shell構文通過。Torch resume/実サーバー3GPUは未検証。

## M3ED Depth動画（2026-11-04）

`tools/visualize_m3ed_dense_probe.py`は既存continuous cacheと評価済みbest headを使い、
RGB/GEP/GT/z/h/concatの6画面MP4を生成する。固定0–80m配色、GT無効画素は黒。
manifestに重み/cache SHAと表示条件、frames.csvにtimestamp/低活動継続時間/各head MAEを保存。
評価JSONのhead/cache hash、readout比較の母集団整合性、prepared metadata hashを検査する。
元系列から状態を保持して生成済みのcacheなので表示区間だけ切り出してもstateをresetしない。
1正解sample=1動画frame、既定20fpsは再生速度。欠損/不等間隔は再標本化せず実時間動画と主張しない。
新規学習/特徴cache不要。MacでAST/help確認のみ、依存テスト・GPU描画はサーバーで未確認。

## F³ M3EDネイティブ採点・DSEC公式test（2026-11-07）

既存F3 clean runのweightsを使用。旧continuous.jsonの上書き・再学習なし。
サーバー上で env を有効化し、`bash tools/run_f3_depth_benchmarks.sh <stage>` を使う。
`RUN_ROOT`, `DATA_ROOT`, `BENCH_ROOT`で既定パスを変更できる。出力先は新規のみ。

**M3ED**: 旧26,611 RGB時刻のtargets.h5ではなく、rawの `{car_urban_day_rittenhouse,car_urban_day_ucity_small_loop}_{data,depth_gt}.h5` が必要。rawは従来aten-22側の/mnt/ssd-4tb/dataset/m3edに存在報告。
4ファイルを読める場所をM3ED_RAW_ROOTに指定（もしくは重みをraw側へ持っていく）。
固定版のF³ cloneをF3_REPOに指定する。GPU実行前にソース・raw・依存を準備：

```bash
git clone https://github.com/grasp-lyrl/fast-feature-fields.git /home/iASL/Arata_repo/fast-feature-fields
git -C /home/iASL/Arata_repo/fast-feature-fields checkout --detach 833e2c65233f9787449ce0675ae41e3f6e9c193a
export F3_REPO=/home/iASL/Arata_repo/fast-feature-fields
# EventStateとF³双方の依存が必要。既存学習envを丸ごと上書きしないこと。
PYTHONPATH="$F3_REPO/src" python -c 'from f3.tasks.depth.utils.dataloader import TimeAlignedDepthAndEvents; import hdf5plugin'
export M3ED_RAW_ROOT=/path/to/raw/m3ed
CUDA_VISIBLE_DEVICES=1 bash tools/run_f3_depth_benchmarks.sh m3ed-audit
CUDA_VISIBLE_DEVICES=1 FEATURE=z bash tools/run_f3_depth_benchmarks.sh m3ed
CUDA_VISIBLE_DEVICES=2 FEATURE=concat bash tools/run_f3_depth_benchmarks.sh m3ed
```

実データの読み出しは公式TimeAlignedDepthAndEventsを直接使用。50kHzの左イベント索引だけ原コードと同じsearchsortedで一時生成。各系列で新しい一時dirを作り過去のmetadata_depths.jsonを拾わない。raw座標1280×720、LiDAR時刻中心20ms、公式コードの200k以上・最大800k・決定的subsample、0.1<GT<80かつ10点以上、frameごとの公式eval_depth関数の算術平均。SqRelも従来EventStateとは式が違うため公式関数に従う。5,387件が必要、さらにauditと本番の系列/時刻/GT/マスクのdigestを照合し、異なる場合metrics.jsonを出さない。samples.csv/support.jsonは診断用。元論文が個別IDを公表したわけではなく、論文件数+公開実装で確認する。

EventState側は同じ選択済みevent列からGEPを作り、全画角を352×640にresize、固定headの予測をnativeに補間。旧DAGR/rectified cropやRGB更新時刻と異なる**新しい推論条件**。イベント窓は未来10msも含むオフライン比較。LSTMは系列冒頭でresetしサンプル間で状態保持。F³自身は極性を除いた3列を使う。教師/学習過程の一致ではなく、評価対象と採点規約を合わせる。自然低活動の検証は別表に残す。

F³モデル自身を同じ採点器で再実行する場合は、公式の**車載昼間metric stage-2**のdepth_config.yml（参照先F³ backbone設定/重みを含む）とbest.pthが必要。通常の公式download_models.sh depthが取得するpseudo stage-1は代用不可。相対モデルパスはF³ repoをcwdとして解決すること。公式F³環境で実行例：

```bash
# ES / OUT / METRIC_CFG / METRIC_CKPT は絶対パスを指定
(cd "$F3_REPO" && python "$ES/tools/eval_m3ed_f3_native.py" --f3-repo "$F3_REPO" \
  --raw-root "$M3ED_RAW_ROOT" --reference-support "$OUT/m3ed_reference/support.json" \
  --f3-model-config "$METRIC_CFG" --f3-model-weights "$METRIC_CKPT" --output "$OUT/f3_reference_model")
```

**DSEC**: 事前に公開valで選んだdsec_alpha1/concat bestを全12系列で固定。左イベントと全画像時刻列のみを使い、テスト画像の画素は読まない。元RGB intervalの全時刻で更新し、公式指定時刻のみ予測。native 480×640の深度を各sequenceのtest_calibration/disparity_to_depth/cams_03の同次Qで逆変換（主点offsetも含む）。PNGの表現範囲のみ飽和し個数を記録。旧視差head exportとは独立。公式ZIPのSHA256と全CSVのhashを検査。552枚の名前・CRC・IHDRを検査してからsubmission.zipを確定する。大きなtest特徴cacheは作らない。

```bash
bash tools/run_f3_depth_benchmarks.sh dsec-preflight
CUDA_VISIBLE_DEVICES=0 bash tools/run_f3_depth_benchmarks.sh dsec
git clone https://github.com/uzh-rpg/DSEC.git /tmp/dsec-official-checker
DSEC_CHECKER=/tmp/dsec-official-checker/scripts/check_disparity_submission.py bash tools/run_f3_depth_benchmarks.sh dsec-check
```

[公式サイト](https://dsec.ifi.uzh.ch/uzh/disparity-benchmark/)のinstitutional accountで、検査済み dsec/submission.zip を一度だけ提出。RGB教師を訓練時に使用、推論は左イベントのみと申告。提出条件・report要件は公式ページ参照。スコアとsubmission IDを記録する。Macでは軽量数値テストのみ、実重み/実rawでの推論・ZIP・投稿は未検証。

検証範囲: 公式時刻ZIP本体を取得、12系列・552枚を実CSVで確認。仮のデータ配置/校正と公式CSVを組み合わせたpreflight通過、CSV改変を拒否。数値テストは主点offset付きQの往復、50kHz時刻の重複・ブロック境界、PNGのuint16形式を確認。実HDF5の内容/モデル/公式checkerでの検証はサーバーで行う。M3EDはF³のuint64索引算術のためNumPy 2.x（公式環境は2.1.2）を使う。support.jsonでは正解・画素・時刻に加え、公式ローダが返したイベント列のSHA256も比較する。

### DSEC公式規則と蓄積窓の選択（2026-11-07）

[公式規則](https://dsec.ifi.uzh.ch/uzh/disparity-benchmark/)と[提出形式](https://dsec.ifi.uzh.ch/disparity-submission-format/)で1Hzは指定出力時刻。20ms入力窓の指定なし。
DSEC_WINDOW_MSはrgb-interval（従来通り）または正の整数ms。固定窓は毎画像時刻tで(t-W,t]の同一系列のrawイベントを読む。画像間隔より長ければ隣接更新の入力は重なる。時刻列/LSTM更新は変えず、公式552枚だけ出力。既存重みと正規化はRGB間隔で学習したものに固定。これは再学習でなく推論窓の比較。
同じモデル・同じ窓を全test系列で使用し、test上で窓選定をしない。正解なし事前学習、モデル選択、統計計算、申告済みの先行学習履歴にもtest系列を混ぜない。既存の重み監査を通さず推論しない（外部RGB教師の全履歴は不明）。窓と監査結果を提出manifestに保存。

候補比較は **dsec-val** のみ（公開4系列、保存済みdsec_val.jsonから元視差GTを使用）。例：
```bash
for w in rgb-interval 20 50; do
  DSEC_WINDOW_MS="$w" bash tools/run_f3_depth_benchmarks.sh dsec-val || break
done
```
出力のmean_sequence_mae_pxを事前に主指標と決め、1条件を固定後に限り、同じDSEC_WINDOW_MSでdsec-preflight→dsec→dsec-check。提出先ディレクトリは窓別に増やさず既存ZIPを保護。M3EDのF³再現用中央20ms窓は変更しない。
