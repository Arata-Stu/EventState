# EventState セッション引き継ぎメモ

最終更新: 2026-09-24

新しい会話セッションは、最初にこの文書と `docs/experiment_results.md` を読む。
数値の正本は `experiment_results.md` であり、この文書は研究状況を素早く復元するための要約である。

## 1. 研究の中心

イベントカメラの現在特徴 `z` と、LSTMの時系列状態 `h` にRGB基盤モデルの知識を蒸留し、
物体検出・Semantic Segmentationなどへ転用可能な汎用表現を学習する。

現在の中心仮説は次のとおり。

- `z` は現在イベントが活発な領域を表現する。
- `h` は過去の観測を保持し、現在イベントが少ない領域を補完する。
- Random clipだけでなくStream/TBPTTを混ぜるHybrid学習で状態継承を学ばせる。
- ただし現結果ではHybridの改善は小さく、長期記憶が獲得された証拠はまだ弱い。
- 次の主実験はactivity-awareな `z/h` 複合蒸留である。

ScaleEvent（CVPR 2026）は高activity領域だけをRGB教師へ蒸留する。EventStateではこれを拡張し、
`z`をactive領域、`h`をinactive領域へ蒸留する。ただし一度も観測されていない領域への
無理な蒸留を避けるため、将来的には「現在inactiveかつ過去Kフレームでactive」の
history-aware maskを検討する。

詳細仕様は `docs/scale_event_distillation.md` を参照する。

## 2. 学習サーバーの固定パス

```text
repository:
/home/iASL/Arata_repo/EventState

DSEC root:
/home/iASL/Arata_repo/dataset/DSEC

DSEC GEP-RGB event cache:
/home/iASL/Arata_repo/dataset/DSEC_cache/events/gep_rgb

DSEC DINOv3 teacher cache:
/home/iASL/Arata_repo/dataset/DSEC_cache/dinov3_vits16

DINOv3 ViT-S/16 checkpoint:
/home/iASL/Arata_repo/models/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth

M3ED prepared event cache:
/home/iASL/Arata_repo/dataset/m3ed_cache/half_dagr

M3ED DINOv3 teacher cache:
/home/iASL/Arata_repo/dataset/m3ed_cache/dinov3_vits16_640x352

M3ED downstream targets:
/home/iASL/Arata_repo/dataset/m3ed_cache/m3ed_downstream
```

GPUはTesla V100 32 GBが3台で、通常 `CUDA_VISIBLE_DEVICES=0,1,2` を使う。

## 3. 事前学習条件の呼称

| 条件 | 蒸留 | 時系列 | dropout | sampling |
|---|---|---|---|---|
| E0 | `z` | なし | なし | Random |
| E1 | `h` | LSTM | なし | Random |
| E2 | `z+h` | LSTM | なし | Random |
| E3 | `h` | LSTM | あり | Random |
| E4 | `z+h` | LSTM | あり | Random |
| Hybrid | `h` | LSTM | 条件による | Random + Stream |

Randomはclipごとに状態を初期化してclip内BPTT、Streamは状態をclip間で継承し境界でdetachするTBPTT。
Hybridは両方のmicro-batchを同じoptimizer stepで学習する。

## 4. DSEC分割と漏洩方針

事前学習は必ず `dataset=dsec_det_train41` を用いる。DSEC-Det公式train 41系列のみで
100,000 stepのfinal fitを行い、事前学習validationは無効。公式Detection val/testと
Semantic testを事前学習へ入れない。

- Detection: train 41 / val 6 / test 13。
- Semantic開発: 公式train 8系列をtrain 6 / val 2に分けてepoch選択。
- Semantic最終: 8系列でheadを再学習し、公式test 3系列で評価。
- Semantic開発valの `zurich_city_07_a`, `08_a` はラベルなし事前学習には含まれるが、
  公式trainに属する。公式testは未使用。

正確な系列名は次を参照する。

- `tools/manifests/dsec_det_official_split.yaml`
- `tools/manifests/dsec_semantic_split.yaml`

## 5. 主要結果の要約

詳細値と出力パスは `docs/experiment_results.md` を正本とする。

### DSEC Detection（Frozen backbone）

- E1 `h` Random: mAP `0.38693`
- Hybrid `h` Random+Stream: mAP `0.39209`
- Hybridは対応するE1より `+0.00516`。
- Scratch no-memory: `0.25065`、Scratch LSTM: `0.26290`。
- 全層Fine-tuningはFrozenを上回らなかった。

### DSEC Semantic（Frozen）

- E1 Linear+CE: mIoU `0.60383`
- Hybrid Linear+CE: mIoU `0.60843`
- Hybrid GEP patch+CE+Dice: mIoU `0.61462`
- 改善の中心はDiceであり、GEP patch単体では改善しなかった。

### M3ED Semantic validation（Frozen Hybrid h）

- Random no-dropout, Linear+CE: `0.36889`
- Hybrid no-dropout, Linear+CE: `0.37388`
- Hybrid dropout, Linear+CE: `0.37611`
- Hybrid dropout, Linear+CE+Dice: **`0.38392`**
- Hybrid dropout, GEP patch+CE: `0.37552`
- Hybrid dropout, GEP patch+CE+Dice: `0.38280`

M3EDの最高値はLinear+CE+Diceの38.392%。異なる18クラス／RGB-event融合などの
公開手法とはプロトコルが異なるので直接SOTA比較しない。E2VIDの39.40%は参考値に留める。

## 6. 現在の実装・直近の作業

activity-aware蒸留と下流activity weightingを実装済み。作業ツリーにはユーザー／別セッションの
未コミット変更があるため、上書き・reset・checkoutをしない。

主比較はGPU 0/1/2で次の3条件を走らせる。共通の学習設定を使い、3番目は損失形式も変える追加比較である。

1. 従来E2相当の `z+h` 全領域蒸留
2. `activity_dual`: activeを`z`、inactiveを`h`へ従来cosine+MSEで蒸留
3. `scale_event_dual`: 同じ領域分担にScaleEvent型構造損失を追加

100 step smokeはユーザーから完了報告あり。保存先は
`/home/iASL/Arata_repo/EventState/outputs/dsec_activity_smoke_20260922_231607`。
続いてユーザーが `--stage full --skip-tests` で100,000 stepの3条件を起動した。
2026-09-24にユーザー提供ログで3条件とも100,000 step到達と
`checkpoints/step_00100000.pt`（各277M）の存在を確認。重みのロード検証は未実施。
最終stepのloss・勾配は有限、optimizer skipは0（全期間の集計ではない）。
次は従来の分割・下流損失でFrozen評価を行う。新しい重みで下流特徴を再抽出する。
ただし先にサーバーストレージを整理する。2026-09-24のユーザー提供容量一覧では
`/home`は98%使用・空き77G。DSEC下流特徴はDetection 671G、Semantic 53G、
M3ED Semantic特徴は65G。旧特徴キャッシュの内訳・再生成元を確認中で削除は未実施。
DSEC GEP RGB入力270Gとデータ本体は次の特徴抽出でも必要。
M3ED関連は別PCでの再生成が必要なため、全キャッシュを保持し削除対象から除外する。
旧DSEC Detection/Semantic特徴の削除をユーザーが実施予定（削除完了・空き容量は未確認）。
下流はまず3条件のh特徴で従来E2と同じFrozen Detectionを比較する。
キャッシュは新しい専用ディレクトリへ作成し、Semanticとz/concat評価は別段階で進める。
その後ユーザーから3条件のFrozen Detection hの完了とtest結果を受領。
seed 0のmAPはbaseline_e2=0.37778238、activity_only=0.37342058、
scale_event_full=0.37427887。下流activity重みなし。詳細は実験台帳§14。
出力は `outputs/dsec_detection_activity_20260923_h/<条件名>/seed_0/test_metrics.json`。
次は予定済みのSemantic評価とz/concat比較。単一seedのh検出では改善は確認できていない。
本番の出力先は `outputs/dsec_activity_full_20260923_000937`、
各条件のコンソールログはその配下の `logs/<条件名>.log`。
サーバーでは `source env/bin/activate` を使い、依存関係はuvで管理する。
起動ツールは
`tools/run_dsec_activity_comparison.sh`。正確な仕様と検証条件は
`docs/scale_event_distillation.md`を読む。

## 7. activity-aware実験の解釈上の注意

- 「event activityでlossを変える」一般概念自体は新規ではない。
- EventDAM（ICCV 2025）とScaleEvent（CVPR 2026）は高activity領域を重視する。
- EventStateの差分は、時系列状態`h`を低activity領域へ割り当てる点。
- 現在inactiveでも過去にactiveだった領域へ限定するhistory-aware maskはまだ今後の候補。
- active/inactive率、各枝のloss、低activity区間の下流性能を必ず記録する。
- 通常mAP/mIoUだけでは長期記憶を証明できない。continuous/reset、gap長、低activity subset、
  静止物体の評価が必要。

## 8. 新しい結果を受け取ったとき

次の情報を `docs/experiment_results.md` に追記する。

1. 条件名と変更点
2. dataset/split
3. seed
4. checkpoint stepまたはbest epoch
5. mAP/mIoUなどの正式な評価値
6. `test_metrics.json`または`validation_metrics.json`の出力パス
7. smoke、途中値、単一seedである場合の注記

会話だけに数値を残さない。

## 9. DSEC論文用の単一フレームPNG出力

`tools/export_dsec_detection_frame.py`で、DSEC-Detの1フレームからRGB、GEP Event、
PCA特徴、Detection結果、manifestを個別PNG/JSONとして出力できる。

ポスター素材として使用した既知のフレーム:

```text
sequence: zurich_city_14_b
timestamp: 55314607586
visualizer frame: 414 / 576
model: Hybrid no-augmentation cached-teacher
feature: h
```

サーバー上での再生成例:

```bash
cd ~/Arata_repo/EventState
source env/bin/activate

DET_DIR="outputs/dsec_detection_frozen_hybrid_noaug_cache/HYBRID/seed_0"
FEATURE_CACHE="/home/iASL/Arata_repo/dataset/DSEC_cache/detection_features/hybrid_noaug_cache_step100000/dsec_det/HYBRID"
EXPORT_DIR="$DET_DIR/publication/zurich_city_14_b_timestamp_55314607586"

CUDA_VISIBLE_DEVICES=0 python tools/export_dsec_detection_frame.py \
  --source "HYBRID:$DET_DIR/best.pt:$FEATURE_CACHE:h" \
  --event-cache-dir /home/iASL/Arata_repo/dataset/DSEC_cache/events/gep_rgb \
  --labels-root /home/iASL/Arata_repo/dataset/DSEC/dsec_det_labels \
  --dataset-root /home/iASL/Arata_repo/dataset/DSEC \
  --role test \
  --sequence zurich_city_14_b \
  --timestamp 55314607586 \
  --output-dir "$EXPORT_DIR" \
  --device cuda \
  --precision fp16 \
  --score-threshold 0.25 \
  --detection-background rgb
```

現在のスクリプトは既定でrectification paddingを全画像共通の有効領域へcropする。
paddingを意図的に残す場合だけ `--keep-padding` を付ける。PCAは選択フレームだけでfitせず、
対象シーケンス全体からfitする。Macへコピーした素材の既知の保存先は次である。

```text
/Users/at/Library/Mobile Documents/com~apple~CloudDocs/プレゼン/プレゼン/中間発表/自分/素材/
```
