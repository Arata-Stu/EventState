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

以下はrepoを同期済みの **サーバー** で実行する。新しい実験はまだ起動していない。

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
Torchテスト・実データ整列・全画角での実重み推論・GPU学習・公式提出checkerは未実行。
新しい精度結果はまだなく、サーバーのpreflight/smokeから確認する。
