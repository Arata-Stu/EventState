# DSEC Detectionのイベント活動分位別mAP

2026-11-10実装。ユーザー指定の「イベント量の下位20%ごとの5群」による追加評価。
対象は学習済みDetection headの評価であり、新しい事前学習・head学習は起動しない。
SemanticのmIoU・Depth指標にも同じdatasetのmanifestを使う経路を追加済み。
現在はこちらが主対象であり、手順は [semantic_depth_activity.md](semantic_depth_activity.md) を参照する。

## 区分と評価単位

- 活動量はprepared DSECの`event_count / RGB入力間隔の秒数`（events/s）。
  rectified 480×640全体での場面活動であり、活動画素率・ScaleEvent mask率・bbox内活動ではない。
- 既定の参照集合は`tools/manifests/dsec_stage1_common_v1.json`のtrain35系列。
  ラベルの有無によらず同集合の全prepared frameから20/40/60/80%点を算出する。
  各境界はソート済みtrain値の`floor(p/100 * (N-1))`番目とし、val/testで再推定しない。
  Detectionの41/6/13 splitを変更するものではない。
- `p00_20`、`p20_40`、`p40_60`、`p60_80`、`p80_100`に分類する。
  境界と等しい値は下側へ入れる。同値境界は統合し、同じ活動率を無理に別群へ分けない。
  trainでの実件数、評価時の実件数は必ずしも20%ずつにならない。
- 推論・状態継承は既存経路を維持し、採点時だけframeを分類する。
  各群の全予測・全GTをCOCO評価器へ渡してAPを再計算する。frame別APの平均ではない。
  GTなしframeの誤検出も、その群のAPに含める。
- 空群、GTなし群、GTが存在しないクラス/サイズのAPは`null`。GTがあるのに予測が0ならAPは0。
  全体mAP/AP50/AP75も同じ推論結果から保存する。AP値は従来と同じ0–1表記。

## サーバーでの検証

Macではstdlibテスト・構文確認のみ。PyTorch/COCOを使うテストと実データ評価はsig-gpu5のenvで行う。

```bash
cd /home/iASL/Arata_repo/EventState
source env/bin/activate
python -m pytest -q tests/test_activity_quantiles_stdlib.py \
  tests/test_detection_activity_stdlib.py tests/test_detection_activity_server.py
```

## 共通の活動manifestを一度だけ生成

```bash
python tools/prepare_dsec_activity_quantiles.py \
  --event-cache-dir /home/iASL/Arata_repo/dataset/DSEC_cache/events/gep_rgb \
  --output outputs/dsec_activity_quantiles_20261110.json
```

既定ではroot内の全系列を走査する。`--sequences SEQ ...`で追加対象を絞っても、参照train35は常に含める。
Detection val/testの追加系列も必要。例えばDetection val6は`zurich_city_16_a`〜`21_a`であり、
共通Stage 1 manifestのval6とは異なる。評価対象がmanifestに欠ける場合は評価前に停止する。

大きなevent tensorを展開せず、`torch.load(..., mmap=True)`でcount等のmetadataを読む。
`_SUCCESS`、frame数・時刻列・入力来歴・metadata digestを照合する。既存outputは上書きしない。
初版はGEP RGB・フルRGB間隔だけに対応し、固定20ms窓や短縮窓は混用を拒否する。

## Frozen headの評価

`DET_CKPT`と`FEATURE_CACHE`は、評価するモデルに対応する既存head・特徴cacheに置き換える。
cacheが容量整理で削除済みなら、既存Stage 1重みから特徴を再抽出する必要がある。
今回の実装だけでは、新しい共通Stage 1用の未学習Detection headは生成されない。

```bash
DET_CKPT=/path/to/detector/best.pt
FEATURE_CACHE=/path/to/matching/detection_features

CUDA_VISIBLE_DEVICES=0 python tools/evaluate_dsec_detection.py \
  --checkpoint "$DET_CKPT" \
  --feature-cache-dir "$FEATURE_CACHE" \
  --labels-root /home/iASL/Arata_repo/dataset/DSEC/dsec_det_labels \
  --dataset-root /home/iASL/Arata_repo/dataset/DSEC \
  --feature h --role val \
  --activity-manifest outputs/dsec_activity_quantiles_20261110.json \
  --condition-label 'E1 Encoder＋LSTM（h蒸留）・単GPU' \
  --output outputs/dsec_activity_map/E1_val.json
```

同じmanifestをE0 Encoder（z蒸留）/z、E1 Encoder＋LSTM（h蒸留）/h、
E2 Encoder＋LSTM（z+h蒸留）/hやDDP・ConvLSTM・配置比較で共有する。
各headの学習時protocolを既定で引き継ぐ。testの設定はvalで固定してから`--role test`に切り替える。

## End-to-end / Scratch / Fine-tuning済みheadの評価

```bash
CUDA_VISIBLE_DEVICES=0 python tools/evaluate_dsec_detection_end_to_end.py \
  --checkpoint /path/to/end_to_end/best.pt \
  --event-cache-dir /home/iASL/Arata_repo/dataset/DSEC_cache/events/gep_rgb \
  --labels-root /home/iASL/Arata_repo/dataset/DSEC/dsec_det_labels \
  --dataset-root /home/iASL/Arata_repo/dataset/DSEC \
  --role val \
  --activity-manifest outputs/dsec_activity_quantiles_20261110.json \
  --condition-label 'E1 Encoder＋LSTM（h蒸留）・Fine-tuning' \
  --output outputs/dsec_activity_map/E1_finetune_val.json
```

ラベルなしframeも学生モデルを更新し、系列先頭だけstateをresetする。
Frozen経路はcacheに記録されたstate_policyを出力に残すため、frame-reset cacheをcontinuousと表示しない。
`--activity-manifest`なしの従来の評価・学習validationは従来経路を維持する。

## 成果物と比較時の確認

指定JSONに`activity_evaluation.groups`を追加し、同stemのCSVも出力する。
JSONのoutput自体が`.csv`の場合は衝突を避け、CSVは`<output名>.groups.csv`に保存する。
各群にはmAP/AP50/AP75・クラス/サイズ別AP、frame数・系列別件数・GT物体数・クラス別GT数、
実イベント率の最小/最大、空群の理由を保存する。CSVにも条件IDと条件名を含む。

manifestにはtrain系列、区分境界、分位法、trainの各群件数、入力metadata hashを記録する。
評価JSONにはmanifestのidentity、評価時刻/GTと活動割当のdigest、detector checkpointのSHA256を保存する。
モデル間の差を取る前にrole・protocol・座標系、`samples_targets_sha256`、`activity_assignments_sha256`、
活動manifest identityが一致することを確認する。群別mAPの単純平均で全体mAPを再構成しない。

実データの活動manifest生成・head推論・COCO採点はサーバー未検証。新しいmAP値はまだない。
