# DSEC Detectionのイベント活動分位別mAP

## 共通Stage 1の9条件を比較する（2026-11-22）

DSECの学習済みbackboneを凍結し、条件ごとに同じYOLOX型検出headを初期化して学習する。対象は下表の9条件。12層後のみのPatchwise E1は、時間モデルの種類比較と配置比較で同じ結果を使う。

| 条件 | 検出に使う最終特徴 |
| --- | --- |
| E0 Encoder（z蒸留） | 12層通過後の `z` |
| E1 Encoder＋LSTM（h蒸留）・Patchwise LSTM（12層後） | `h` |
| E1_convlstm　E1 Encoder＋LSTM（h蒸留）・ConvLSTM（12層後） | `h` |
| E1_b3 / E1_b6 / E1_b9　E1 Encoder＋LSTM（h蒸留）・Patchwise LSTM（各1個、順に3・6・9層後） | 各配置で12層まで通した最終 `h` |
| E1_b9_12 / E1_b6_9_12 / E1_b3_6_9_12　E1 Encoder＋LSTM（h蒸留）・Patchwise LSTM（2・3・4個） | 各配置で12層後のLSTMまで通した最終 `h` |

共通Stage 1のtrain35系列はすべてDetectionのtrain41系列に含まれ、Detectionのval6・test13との重複は各0。物体ラベルがある41系列でheadを50 epoch学習し、val6全体のCOCO mAPが最大のepochを固定する。test13は固定済みheadで採点できる。過去のtrain41で事前学習したFrozen Detectionのスコアは実験結果台帳§2にある。[公式split](https://github.com/uzh-rpg/dsec-det/blob/master/config/train_val_test_split.yaml)

各系列の最初で状態をリセットし、未採点フレームも含め時系列順に状態を更新する。評価は既存の `dsec-det` 設定で、native scaleのイベントカメラ座標（640×430）上のcar・pedestrianの2クラスを扱う。元ラベルのbus・truckもこの設定のcarに統合する。DAGRのサイズ・クラス条件を満たすラベルが連続した画像ペアの2枚目が採点対象。全体と活動量5群別にCOCO mAP@[.50:.95]・AP50・AP75を確認する。活動量の境界は共通Stage 1のtrain35系列のevents/sから決め、評価系列にも同じ値を適用する。[DSEC-Detectionデータ](https://dsec.ifi.uzh.ch/dsec-detection/)

### sig-gpu5での起動

Semanticの完了後はGPU 0・1・2を候補とし、各フェーズ開始時に空いているGPUで別々の条件を並列実行する。GPU 0を別タスク用に確保する場合は `--gpus 1,2` を指定する。以下は全9条件のvalが成功するとtestへ進む。valを見て設定を変更する場合は、フェーズを分けて起動する。

```bash
(
  set -eo pipefail
  cd /home/iASL/Arata_repo/EventState
  mkdir -p outputs
  log="$PWD/outputs/dsec_detection_20261122.log"
  date -Is | tee -a "$log"
  git pull --ff-only 2>&1 | tee -a "$log"
  nohup bash -ec '
    git rev-parse --short HEAD
    for phase in val test; do
      echo "DSEC Detection $phase"
      env/bin/python -u tools/run_placement_detection.py --gpus 0,1,2 --phase "$phase" --execute
    done
  ' </dev/null >>"$log" 2>&1 &
  echo "Detection PID=$!  log=$log"
)
```

経過は `tail -f outputs/dsec_detection_20261122.log` で確認する。既定の出力先は `outputs/stage2_placement_detection_20261122`。集計は `summary_val.json` と `summary_test.json`、個別のheadは `<条件ID>/head/best.pt` に保存される。事前に実データを点検するには該当フェーズの `--execute` を `--preflight` に替える。testはvalで確定済みのheadを必要とする。

中断後は同じコマンドで再開できる。head学習を再開した場合、確定済みのepochまでの重みとoptimizerを復元し、再開したepochを結果に記録する。既存のtrainerはデータ順序とaugmentationの乱数状態を保存していないため、連続実行と再開後で学習軌跡に差が生じうる。

特徴キャッシュが容量の大半を占める。旧h特徴1条件でrectified版約64 GiB・Detection座標版約61 GiBの記録がある（実験結果台帳§25の `du` 記録）。今回の必要容量は対象系列と実ファイル数で変わる。各条件で検証したDetection座標の特徴を保持し、その変換に使った専用の一時特徴を解放する。実行中のSemanticでも同じファイルシステムを使う場合は、その増加分も見込む。

### Detectionの追加入力

この実験は `/home/iASL/Arata_repo/dataset/DSEC/dsec_det_labels` の `tracks.npy`、系列別の画像時刻・較正、および `DSEC_cache/events/gep_rgb` のRGB間隔GEPを使う。Detectionのval6（zurich_city_16_a〜21_a）は物理的には追加配布のtrain、testのthun_02_aは追加配布のtestにある。Semantic・Depth用の評価系列とは異なる。追加配布rawがあり、これらのGEPだけが不足している場合の準備例は次のとおり。

```bash
data=/home/iASL/Arata_repo/dataset
python tools/prepare_dsec.py --root "$data/DSEC" --layout dsec-det-extra \
  --split train --sequences zurich_city_{16,17,18,19,20,21}_a \
  --event-cache-dir "$data/DSEC_cache/events/gep_rgb"
python tools/prepare_dsec.py --root "$data/DSEC" --layout dsec-det-extra \
  --split test --sequences thun_02_a --event-cache-dir "$data/DSEC_cache/events/gep_rgb"
```

## 既存のDetection headに活動量別評価を追加する

2026-11-10実装。ユーザー指定の「イベント量の下位20%ごとの5群」による追加評価。
対象は学習済みDetection headの評価であり、新しい事前学習・head学習は起動しない。
SemanticのmIoU・Depth指標にも同じdatasetのmanifestを使う経路を追加済み。
Semantic・Depthの活動量別評価の手順は [semantic_depth_activity.md](semantic_depth_activity.md) を参照する。

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
CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/test_activity_quantiles_stdlib.py \
  tests/test_detection_activity_stdlib.py tests/test_detection_activity_server.py \
  tests/test_detection.py tests/test_placement_detection{,_protocol}_stdlib.py
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

2026-11-24に共通Stage 1の凍結9条件について、サーバーでのval・test各9/9の完了印と全体・活動5群のCOCO mAPを受領した。[実験結果台帳 §105](experiment_results.md) に両splitの数値を収録している。
