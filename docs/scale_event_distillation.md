# ScaleEvent に合わせた z/h 蒸留

## 再現の基準

参照論文: [Scaling Dense Event-Stream Pretraining from Visual Foundation Models](https://arxiv.org/html/2603.03969v1)。
参照実装: `reference_repo/ScaleEvent`、commit `92f005b0f2cbb19dfc8391e3019ca042b1a9f587`。
判定は `scale_event/dataset/event_utils.py::prepare_mask`、損失は
`scale_event/pretrain/criterion.py::DistillLoss_With_CrossGram` に合わせた。
参照リポジトリへの実行時依存はない。

論文とコードは一致していないため、今回の再現基準は **公開コード** とする。

| 項目 | 論文 | 公開コード／今回の基準 |
|---|---|---|
| 活動領域 | パッチ内密度、閾値 64 | 白背景 uint8 イベント画像を OpenCV で W/4,H/4 → W/16,H/16 に２回縮小し、RGB 合計 <765 |
| 関係損失 | 式では L1 | MSE |
| 関係の選択 | 構造の整合 | 負の類似度を 0 にし、teacher 類似度 >0.1 のペアだけ残す |

この実装の「活動あり」は生イベント数の上位群ではない。弱い信号は uint8 化や
縮小で消え、パッチ内にイベントが存在しても非活動になる場合がある。
平均プーリング、単なる非ゼロ判定、分位点による分類には置き換えない。

## データと損失

既存の GEP RGB 表現（白背景、正負のイベントを着色）を使う。RGB と同じ crop を
イベント画像に適用し、`round(255*x)` で uint8 化してから参照通り２回縮小する。
水平反転も RGB と一致させる。入力正規化後の値から活動を推定しない。
参照データの PNG 生成処理は公開コードだけでは同一性を保証できないため、
**同じ uint8 画像に対するマスク演算の再現**と、イベント画像生成の同一性は区別する。
GEP の画像生成、既存の因果的時間窓、入力正規化は従来通りであり、ScaleEvent 全体の完全再現ではない。

活動マスクを M として、z は M、h は 1−M で教師特徴を蒸留する。
これは ScaleEvent を EventState に拡張した部分であり、h の補集合蒸留は原論文の手法ではない。
人工的な event dropout を使う場合は、そのフレームの M を 0 にして h 側に渡す。
デフォルトでは dropout は無効。

主比較の `experiment=activity_dual` は、従来の cosine＋正規化 squared L2 を保つ。
選択 token の重み付き平均という従来の分母も維持する。変更するのは M / 1−M による
枝ごとの選択だけ。空集合の枝は微分可能なゼロにする。teacher は detach する。
h から encoder・過去状態への勾配は通常通り流れる。

別比較の `experiment=scale_event_dual` は公開コードの
`L1 + 10 * intra_MSE + 4 * cross_MSE` を使う。
こちらはマスク後も全 token／全 token ペアを分母とする。関係行列だけ L2 正規化し、
cross は `teacher @ student.T`。フレーム間の関係は計算しない。
これは損失の種類も変わるため、activity だけの効果の主比較とは区別する。

## 従来と同じ分割・学習条件

主実験は **`dataset=dsec_det_train41`、validation 無効、固定最終 step**。
E0〜E4／Hybrid と同じ41系列を使い、下流の分割も変更しない。

| 段階 | train | validation | test |
|---|---|---|---|
| 事前学習 | DSEC-Det 公式41系列 | なし | 不使用 |
| 検出 | 41系列 | 6系列 | 13系列 |
| Semantic 開発 | 6系列 | 2系列 | 不使用 |
| Semantic 最終 head 学習 | 8系列 | 選択済み epoch 数 | 3系列 |

起動時に、事前学習が公式41系列と完全一致し、validation が無効で、検出 val/test・
semantic test と系列が重複しないことを検査する。
semantic 開発 val の `07_a/08_a` は公式 train に属し、ラベルなし事前学習には含める。
これは従来のプロトコルを維持するためであり、公式 test の使用ではない。
`interlaken_00_c…g` も公式検出 train として維持する。

追加した `dsec_joint_clean` は内部 val の入力まで未見にする任意の別条件であり、
主比較には使わない。その条件を使う場合だけ `training.validation_enabled=true` とする。
従来結果を test 漏洩として扱ったり、従来モデルに新分割での再学習を要求したりしない。

`activity_dual` は E2 の `h_distill_lstm_zloss` と同じモデル、cosine/MSE の係数、
optimizer、学習率、batch size、勾配累積、step 数を継承する。
Hybrid や dropout との比較では、その比較相手の sampling/dropout 設定をそのまま使い、
`dataset.activity_mask=true` を追加する。異なる蒸留枝を持つ E0/E1 との差には、
枝数の違いもあることを明記する。

マスクは既存イベント窓 `(window_start,t]` の入力だけから作り、RGB、下流ラベル、
評価集合統計は判定に使わない。教師 cache の幾何整合検査と収録境界の state reset は維持する。

## 前処理と事前学習

既存の **rectified GEP RGB イベントキャッシュ** と、同じ crop・解像度・教師の
DINO cache は再利用できる。活動マスクは読み込み時に生成するため、マスク用の
キャッシュ再生成は不要。voxel cache は使用不可。旧学習済み重みから抽出した
下流特徴キャッシュは、新しい事前学習 checkpoint で作り直す。
イベント窓・画像位置合わせ・解像度を変える場合は、対応する入力／教師キャッシュを再生成する。

以下は学習サーバーで実行する例。パスは実データに合わせて指定する。
OpenCV はこの実験では学習時にも必要（`pip install -e '.[scale-event]'`）。

```bash
python train.py dataset=dsec_det_train41 experiment=activity_dual \
  dataset.root=/data/DSEC \
  dataset.event_cache_dir=/data/cache/dsec_gep_rgb \
  teacher.cache_dir=/data/cache/dsec_dinov3 \
  teacher.checkpoint=/models/dinov3_vits16.pth
```

batch size 8、勾配累積1など、既存設定を変更しない。`active_patch_fraction` と各枝の
損失を記録する。活動率がほぼ1なら h の直接教師信号が弱くなるため、その率を確認する。
閾値を評価集合に合わせて自動調整することはしない。
`scale_event_dual` は token 数の二乗に比例する追加計算が必要で、GPU 実測は未実施。

最終 checkpoint は従来通り `checkpoints/step_00100000.pt` を使う（100,000 step の場合）。
事前学習 validation がないため、`best.pt` は作らない。

## 検出・セグメンテーションへの接続

新しい事前学習 checkpoint は既存の z/h 出力契約を維持している。
同じ checkpoint を凍結し、まず z・h・concat を両タスクで比較する。
タスクごとの head 学習と選択は、それぞれの既存 split を用いる。
検出側で fine-tune した backbone を semantic の未見検証用に流用しない。

検出では `cache_dsec_detection_features.py` で train/val/test それぞれの z/h を抽出し、
`prepare_dsec_detection_benchmark_features.py` で公式検出座標へ変換する。
その後 `train_dsec_detection.py --protocol dsec-det --feature h` を用いる
（z/concat も同様）。変換前の rectified 特徴を公式検出座標として扱わない。
test の評価は val で設定を固定した後に `evaluate_dsec_detection.py --role test` で行う。
各ツールの `--help` と既存 README の DSEC-Detection 手順に他の必須パスを記載している。

semantic は既存 runner がキャッシュ・6/2 分割の開発・公式 train での最終 head 学習・
test 評価を実行できる。

```bash
bash tools/run_dsec_semantic_frozen.sh \
  --checkpoint /outputs/activity/checkpoints/step_00100000.pt \
  --event-cache-dir /data/cache/dsec_gep_rgb \
  --labels-root /data/DSEC_semantic \
  --feature-cache-dir /data/cache/scale_event_semantic_h \
  --output-dir /outputs/scale_event_semantic_h \
  --feature h
```

## 下流の活動量による損失変更

Frozen backbone の z/h は固定されているため、そこに蒸留損失を追加しても学習されない。
今回の下流機能は **教師あり損失の空間重み付け**であり、RGB 教師との追加蒸留ではない。
学習中だけ `w = active_weight * M + inactive_weight * (1−M)` を用いる。

- 検出: objectness BCE、分類 BCE、box IoU loss を anchor 中心の活動重みで重み付け。
  assignment と正例数による従来の分母は変えない。
- Semantic: CE を有効画素の重み付き平均にし、CE+Dice では Dice の交差・和にも同じ重みを使う。
  ignore label は従来通り除外する。
- 評価 mAP/mIoU、検証での checkpoint 選択は重み付けしない。
- `--activity-active-weight 1 --activity-inactive-weight 1` がデフォルトで、従来の処理経路を通る。
  head 構造、backbone の凍結方針、augmentation、optimizer は変更しない。
- 重みは checkpoint に保存し、resume と Semantic の開発→最終 fit での変更を拒否する。

対応エントリーポイント:
`train_dsec_detection.py`（Frozen）、`train_dsec_detection_end_to_end.py`（既存 Fine-tuning/Scratch）、
`train_dsec_semantic.py`（Frozen）と `run_dsec_semantic_frozen.sh`。
Semantic の backbone fine-tuning や下流 RGB 追加蒸留は、この変更には含まない。

例えば z の活動領域を重視する試行は `--activity-active-weight 1 --activity-inactive-weight 0.25`、
h の低活動領域を重視する試行は逆の `0.25 / 1` と指定できる。
0 を使えば選択領域だけの損失になる。これらの値は実装例で、検証済み推奨値ではない。
concat では一つの head の損失に重みを掛けるため、z/h を別々に監督する効果はない。

Frozen 特徴の抽出時は次の両ツールに **`--include-activity`** を付ける。

```text
cache_dsec_detection_features.py
cache_dsec_semantic_features.py
```

検出の `prepare_dsec_detection_benchmark_features.py` はマスクも公式座標へ nearest で変換する。
loader の水平反転も特徴・ラベル・マスクに同時適用する。
Semantic のマスクは実イベントの上440行を使い、人工 padding は白（イベントなし）として扱う。
patch mask を元の stride で展開し、440行へ切り詰める。448行を440行へ縮めない。
マスクには `activity_format` を記録し、欠けた古い cache を重み付き学習へ渡すとエラーにする。
従来 cache で重みなし評価は引き続き可能。特徴・入力は変えず、マスク情報だけを追加する。
Semantic runner は非デフォルトの重みを指定すると、必要なマスクも生成する。

公平な比較は次の4条件を同じ seed・head・学習条件で行う。

| 条件 | 事前学習の活動マスク | 下流の活動重み |
|---|---|---|
| baseline | なし | 1 / 1 |
| pretrain のみ | あり | 1 / 1 |
| downstream のみ | なし | 指定値 |
| 両方 | あり | 同じ指定値 |

検出 test／Semantic test の結果を重み選択に使わない。

## 検証状況

### V100 32GB × 3台で開始する

`tools/run_dsec_activity_comparison.sh` は GPU 0/1/2 に、従来の E2、activity のみ、
ScaleEvent の損失全体をそれぞれ割り当てる。事前学習は全条件41系列、同じ seed・LSTM・
batch size・optimizer とし、下流損失の変更はこの最初の比較では使わない。

```bash
bash tools/run_dsec_activity_comparison.sh \
  --root /path/to/DSEC \
  --event-cache-dir /path/to/DSEC_cache/events/gep_rgb \
  --teacher-cache-dir /path/to/DSEC_cache/dinov3_vits16 \
  --checkpoint /path/to/dinov3_vits16.pth \
  --gpus 0,1,2 --stage smoke
```

既定は100 step、warmup 10の試運転。Python の数値テストを CPU で実行し、通過した場合だけ
3条件を起動する。学習ホストに既存依存関係に加えて pytest と OpenCV が必要。
参照比較のため ScaleEvent checkout も必要。Mac ではこのランチャーを実行しない。

各条件のログは出力先の `logs/`、checkpoint は条件別の `checkpoints/` に保存する。
活動あり／なしの両方が存在すること、損失・勾配が有限であること、optimizer skip、GPUメモリを確認する。
`--stage full` で100,000 step、warmup 1000の本学習を新規に開始する。
試運転 checkpoint から本学習へは継続せず、全条件を同じ DINO 初期値から開始する。
既存出力先への上書きは拒否する。GPU番号は `--gpus`、全条件共通のbatchは `--batch-size` で変更可能。
ScaleEvent の条件だけメモリ不足になる場合も、比較条件の変更を隠さず、全条件に同じ設定を適用する。

`--dry-run` は Python や学習を実行せず起動コマンドを表示する。
`--skip-tests` は同じ revision を学習ホストで検証済みの場合に限って用いる。

この Mac では、標準ライブラリの分割テスト、Python 構文・シェル構文検査のみ実行する。
依存ライブラリを使う次のテストと実学習は、ユーザーの指示があるまで実行しない。

```bash
python -m pytest tests/test_scale_event.py tests/test_activity_losses.py \
  tests/test_split_guard_stdlib.py tests/test_transforms.py tests/test_dsec.py \
  tests/test_training_runtime.py tests/test_detection.py tests/test_segmentation.py
```

テストには公開コードのマスク・CrossGram 値／勾配との比較、既存損失との全1マスク時の一致、
空領域、Semantic ignore 画素、検出 assignment の不変性、設定の同一性を含む。
参照 checkout がない環境では参照比較は skip される。
