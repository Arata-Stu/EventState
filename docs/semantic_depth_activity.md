# Semantic・Depthの共通イベント活動5分位評価

2026-11-10実装。現在のtask優先順位はSemantic segmentation・Depth、その次に可能ならFlow。
Detectionは低優先度。既存の学習済みheadを使い、活動分位別の追加評価を行う。

## Depth完了後にSemantic・Depthをまとめて実行

今回の変更をサーバーへ反映し、現在のDepth学習が完了してからsig-gpu5で実行する。
既定はDSEC/M3ED × Semantic/Depth × 基本3条件・E1旧新DDPの計20評価。
DSEC Semanticは既存final headによる公式test3系列、M3ED Semanticと両datasetのDepthは公開valを使う。

```bash
cd /home/iASL/Arata_repo/EventState
source env/bin/activate

# head・学習時config・cache・targetsの存在とDepth完了を確認。推論・出力作成は行わない。
python tools/run_semantic_depth_activity.py \
  --depth-run outputs/stage2_common_depth_vkitti_20261110 \
  --preflight

# 上の確認が通ったら、manifest生成 → Semantic → Depth追加評価をGPU 0で直列実行。
CUDA_VISIBLE_DEVICES=0 python -u tools/run_semantic_depth_activity.py \
  --depth-run outputs/stage2_common_depth_vkitti_20261110 \
  --output outputs/semantic_depth_activity_20261110
```

`--depth-run`には実際に学習したrun rootを指定する。基本3条件だけを学習した場合は、
確認・実行の両方に`--baselines-only`を追加する。
Semanticのcache・ラベルrootは既存headの`config.json`から復元する。
cacheを手動削除した場合は事前確認で止まるため、対応するStage 1重みからの再抽出が必要。
新しいhead学習は起動しない。Depthはval特徴を一時的に再抽出するため、時間・scratch空き容量が必要。

出力は`outputs/semantic_depth_activity_20261110/`の`manifests/`、`semantic/`と`summary.json`。
Depthの詳細JSON/CSVと`activity_summary.json`は元のDepth run配下に保存する。
同じコマンドを再実行すると、入力が一致する完了済み評価を再利用する。
この一括入口のstdlibテスト7件を含め、Macの関連stdlibテスト63件が通過。
実データ/GPU実行はサーバー未検証。

## 共通の区分

- DSECは共通train35、M3EDは共通train3の全prepared RGB時刻から、それぞれ20/40/60/80%境界を固定する。
  同じdatasetのSemantic/Depth・全モデルで同じ活動manifestを使う。両datasetへ同じevent/s閾値を流用するものではない。
- 指標はイベント数/実RGB入力間隔秒数。DSECはrectified480×640全体、M3EDはDAGR downsampling・rectification後の
  360×640全体のcountを使う（モデル用中央352行cropの前）。活動画素率や生イベントの間引き率ではない。
- `p00_20`〜`p80_100`を評価し、同値境界は統合する。評価splitの各群が20%ずつになるとは限らない。
  先頭RGBには過去の入力区間がないため、従来通り評価対象に含めない。
- Semanticは群ごとに混同行列を合算してmIoU・クラス別IoUを算出する。
  Depthは有効GT画素の誤差和を集計し、MAE/RMSE/AbsRel/SqRel/RMSElog/δ1–δ3を算出する。
  この既存欄はframe別mIoUやRMSEの単純平均ではない。Depth再評価では下記のF³式frame平均も併記する。
  既存の全体・低活動/継続時間別指標も保持する。
- 各群のframe数・有効GT画素数・系列別件数・実活動率範囲をJSON/CSVへ保存する。
  空群・有効GTなし群の指標はnull。推論後の採点だけを区分し、系列の状態継承方針は変えない。

## Depthの計算式とframe平均（2026-11-10追加）

ユーザーの指摘を受け、Depthの再評価CLIは同一予測・同一GT maskから2通りの集計を出力する。
既存`metrics`と活動群内の`metrics`は全有効画素をpoolした値として保持する。
追加`metrics_frame_mean_f3`は、各frameでF³公式の式を計算し、有効GTのあるframeを等重みで平均した値。
全体・低活動/継続時間別はrootの`metrics_frame_mean_f3.<group>`、活動5分位は
`activity_evaluation.groups.<group>.metrics_frame_mean_f3`に保存する。
比較表でF³と計算・集計を揃える際はこちらの欄を使用し、旧集計は来歴を確認する参考値として残す。
SemanticのmIoUは従来通り群内の混同行列から算出する。

| 項目 | 既存の画素pool欄 | 追加のF³式frame平均欄 |
|---|---|---|
| frameの重み | 有効GT画素数に比例 | 有効GTのあるframeごとに同じ重み |
| RMSE | 全画素のMSEの平方根 | 各frameのRMSEを平均 |
| SqRel | 平均`(予測−GT)² / GT` | 各frameの平均`(予測−GT)² / GT²`を平均 |
| その他 | MAE・AbsRel・RMSElog・delta1–3 | MAE・AbsRel・RMSElog・d1–3・log10・silog |

数式の参照は[F³固定版eval_depth](https://github.com/grasp-lyrl/fast-feature-fields/blob/833e2c65233f9787449ce0675ae41e3f6e9c193a/src/f3/tasks/depth/utils/utils.py)、
集計は[同評価loop](https://github.com/grasp-lyrl/fast-feature-fields/blob/833e2c65233f9787449ce0675ae41e3f6e9c193a/src/f3/tasks/depth/evaluate.py)。
公式のSqRelは分母GT²、RMSElogは自然対数、silogは係数0.5・100倍なし、d1–3は割合0–1。
ここでの「公式」はF³公式実装を指す。分母GTの式はDAv2/Monodepth2で使われる通常のSqRelであり、
旧欄が誤りという意味ではない。分母GT²のF³互換値は別の量なので、同じSqRelとして数値を混ぜない。
通常SqRelとF³互換SqRelを区別して報告する。
MAEは公式関数の返却欄にない追加指標。固定revisionとsource SHAを`depth_frame_metric_protocol`に記録する。
内部計算はfloat64なので、公式float32とbit単位の一致を主張しない。

追加欄は計算・集計だけを揃え、RGB時刻・resize/crop・0.1<GT≤80m・低活動を含む評価集合を維持する。
有効GTゼロのframeは平均の分母から除外し、scored frame数と有効画素数を併記する。
F³の有効10点以上・中心20ms内20万イベント超の選別を追加していないため、F³論文と完全同条件のスコアではない。
学習中のlossやbest headの選択指標は既存のpooled val MAEを維持し、再学習は起動しない。

前述の一括コマンドで両方式を同時出力する。CSVの追加列は`f3_frame_mean_`で始まり、旧列と区別する。
旧`activity_continuous.json`に追加統計がない場合は上書きせず、同headの
`activity_continuous_frame_mean.json`/CSVへ再評価し、`activity_summary.json`が実reportを参照する。

## サーバーでのテスト

以下はsig-gpu5の既存env用。Macではstdlibと構文確認だけを行う。

```bash
cd /home/iASL/Arata_repo/EventState
source env/bin/activate
python -m pytest -q tests/test_activity_quantiles_stdlib.py \
  tests/test_m3ed_activity_quantiles_stdlib.py tests/test_activity_reporting_stdlib.py \
  tests/test_semantic_activity_stdlib.py tests/test_dense_activity_stdlib.py \
  tests/test_common_depth_runner.py \
  tests/test_semantic_depth_activity_runner_stdlib.py \
  tests/test_depth_frame_metrics_stdlib.py \
  tests/test_semantic_activity_server.py tests/test_dense_activity_server.py \
  tests/test_depth_frame_metrics_server.py
```

## Datasetごとにmanifestを一度生成

```bash
python tools/prepare_dsec_activity_quantiles.py \
  --event-cache-dir /home/iASL/Arata_repo/dataset/DSEC_cache/events/gep_rgb \
  --output outputs/dsec_activity_quantiles_20261110.json

python tools/prepare_m3ed_activity_quantiles.py \
  --prepared-root /home/iASL/Arata_repo/dataset/m3ed_cache/half_dagr \
  --sequences car_urban_day_rittenhouse car_urban_day_ucity_small_loop \
  --output outputs/m3ed_activity_quantiles_20261110.json
```

`--sequences`は追加する系列であり、各datasetの参照train系列は常に含む。
既存出力は上書きしない。以前作ったDSEC manifestのv1はそのまま使える。M3EDは専用contractのv2。
metadataと時刻・countを読み、重いevent tensorの展開や新しい教師推論は行わない。
初版は既存GEP RGBの完全なRGB間隔のみ。固定窓や短縮窓の混用は拒否する。

## DSEC Semanticの既存headを再評価

`--checkpoint`、`--feature-cache-dir`、`--labels-root`は対応する既存head・cache・Semanticラベルrootを指定する。
E0 Encoder（z蒸留）は`--feature z`、E1 Encoder＋LSTM（h蒸留）とE2 Encoder＋LSTM（z+h蒸留）は`--feature h`。

```bash
CUDA_VISIBLE_DEVICES=0 python tools/evaluate_dsec_semantic.py \
  --checkpoint outputs/stage2_dsec_semantic_linear_20261109/E1/final/last.pt \
  --feature-cache-dir /path/to/matching/dsec_semantic_features \
  --labels-root /path/to/dsec_semantic_labels \
  --feature h --role test \
  --activity-manifest outputs/dsec_activity_quantiles_20261110.json \
  --condition-label 'E1 Encoder＋LSTM（h蒸留）・単GPU' \
  --output outputs/semantic_activity/E1_dsec_test.json
```

headの上記パスは例であり、実際の配置は未確認。開発valなら対応する開発headと`--role val`を使う。
既存全体スコアだけから層別値は復元できないため、headを再推論する。

## M3ED Semanticの既存headを再評価

再学習を伴わないstandalone評価入口を追加した。既定は共通3train/2公開valのmanifest。
`--target-cache-dir`にはRittenhouse/UCITYのSemantics付きtargetsが揃うroot、または既存runのtargets viewを指定する。

```bash
CUDA_VISIBLE_DEVICES=0 python tools/evaluate_m3ed_semantic.py \
  --checkpoint outputs/stage2_m3ed_semantic_linear_20261109/E1/h/best.pt \
  --feature-cache-dir /path/to/matching/m3ed_semantic_features \
  --target-cache-dir /path/to/semantic_targets_view \
  --prepared-root /home/iASL/Arata_repo/dataset/m3ed_cache/half_dagr \
  --feature h \
  --activity-manifest outputs/m3ed_activity_quantiles_20261110.json \
  --condition-label 'E1 Encoder＋LSTM（h蒸留）・単GPU' \
  --output outputs/semantic_activity/E1_m3ed_val.json
```

既存common cacheの`identity.json`にある元prepared metadata hash、または新cacheに埋めたmetadataを照合する。
元入力の来歴がない古いgeneric cacheは受け付けない。必要なら既存Stage 1重みからcacheを再抽出する。
新規`run_common_m3ed_semantic.py`にも同じ`--activity-manifest`/`--condition-label`を渡せる。
従来の`continuous_groups.json`と、新しい`activity_groups.json`/CSVを出力する。

## Depth完了後、同じrunのheadだけを追加評価

現在の学習が完了してから、元のrun rootを`--output`へ指定する。

```bash
CUDA_VISIBLE_DEVICES=0 python tools/run_common_depth.py \
  --output outputs/stage2_common_depth_vkitti_20261110 \
  --dsec-activity-manifest outputs/dsec_activity_quantiles_20261110.json \
  --m3ed-activity-manifest outputs/m3ed_activity_quantiles_20261110.json \
  --activity-only
```

単GPU基本3条件だけなら`--baselines-only`を追加する。片方のdatasetだけを追加評価する場合は、そのmanifestだけ渡す。
この経路は完了済み`best.pt`と`_SUCCESS`を要求し、val特徴だけを一時的に再抽出する。
元の`continuous.json`/`summary.json`を保持し、各headの`activity_continuous.json`/CSVと
run rootの`activity_summary.json`へ保存する。元runのlockを共有するため、同じrunの学習中には並行実行しない。

既存val特徴cacheが残っている場合は直接評価も可能：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/evaluate_dense_probe.py \
  --checkpoint /path/to/completed/depth_head/best.pt \
  --cache /path/to/matching/val_feature_cache \
  --activity-manifest outputs/dsec_activity_quantiles_20261110.json \
  --condition-label 'E1 Encoder＋LSTM（h蒸留）・単GPU' \
  --output outputs/depth_activity/E1_dsec_val.json
```

同じ入口でDepth/Disparityの公開GTを採点できる。GT非公開testのPNG exportとの併用はしない。
公式testの全体スコアと公開valの活動別指標は評価先を区別する。

## 出力の読み方と検証範囲

群別は`activity_evaluation.groups.<群名>.metrics`。SemanticのmIoUは0–1、Depthは従来のm単位。
活動manifest identityは同一dataset内の全taskで共有するが、GT時刻や有効画素はtask別に維持する。
条件間比較では各task内の入力/GT・活動割当digestを確認する。

Macのstdlib・構文確認のみ。実データのmanifest生成・head再推論・GPU採点はサーバー未検証。
新しい精度結果や学習完了を示す記録ではない。
