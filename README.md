# EventState

> **Built with DINOv3.** This project uses Meta AI's DINOv3 ViT-S/16 as a
> frozen RGB teacher and as event-encoder initialization. See
> [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for license and attribution details.

イベントカメラの短時間特徴 `z_t` と、時間方向に統合したpersistent visual state
`h_t` を分けて学習するためのpretraining実装です。Phase 0/1ではFrozen DINOv3
ViT-S/16のdense patch tokenを教師とし、次を比較します。

```text
E0: Event → Encoder → z ─────────────→ DINOv3
E1: Event → Encoder → z → LSTM → h ─→ DINOv3
E2: Event → Encoder → z ─────────────→ DINOv3
                         └→ LSTM → h ─→ DINOv3
```

実装の詳細な根拠は [docs/blueprint.md](docs/blueprint.md) と
[docs/design_decisions.md](docs/design_decisions.md) にあります。`blueprint.md`は当初の20ch案を
残した原案です。その後の方針変更により、現行実装ではGEP互換の実効3chを最初のbaselineとし、
20chは追加実験として扱います。相違点は`design_decisions.md`を正とします。

## 現在の実装範囲

- DSECのsequence loader（sequence boundaryを跨がない）
- RGB時刻に対する厳密なevent window `(T_(t-1), T_t]`
- `t_offset`を含むHDF5 event slicing
- DSEC公式rectification mapによるevent座標の整列
- RGBからevent rectified座標へのrotation-only homography warp
- GEP互換の実効3ch event frame（既定）
- 10 bins × 2 polaritiesの20ch voxel grid（追加実験）
- DINOv3 ViT-S/16 Event Encoder / Frozen RGB teacher
- patch-wise LSTMとstreaming state interface
- cosine + normalized MSE distillation
- teacher feature cache、TensorBoard logging、resume可能なcheckpoint
- `z` / `h` alignmentとlow/medium/high event-count別評価

Mamba、future prediction、flow probe、CMax、下流segmentation/flow/detection headは
次のphaseです。Mambaは設定名と明示的なplaceholderだけを用意しています。

## データセット選択

最初は **DSEC** を使用します。event/RGB/calibrationが揃い、同じdataset familyで
flow、semantic segmentation、detectionへ評価を広げられるためです。M3EDはPhase 1の
比較が安定した後に同じdataset interfaceへ追加します。

既定のevent入力は20chではなく、GEPとの比較を優先した3chです。negative=赤、
positive=青の白背景event frameを90 percentile clipで作り、GEPで使われるDSEC event
統計で正規化します。20chは独立した設定で残してあります。

## セットアップ

Python 3.11以上を想定します。学習機のTesla V100（Volta、sm_70）に合わせ、PyTorchは
Volta対応wheelが提供される最後の系列である2.14.0 + CUDA 12.6に固定しています。CUDAでは
既定のFP16 autocastを使います。macOS/MPSでは未対応のautocast設定を明示的な警告付きで
FP32へfallbackします。

```bash
uv venv --python 3.12 env
source env/bin/activate
uv pip install --torch-backend=cu126 -e '.[prepare]'
```

V100環境では、ドライバの`CUDA Version: 13.0`からcu130を自動選択させないことが重要です。
`--torch-backend=cu126`を必ず明示し、PyTorch公式のCUDA 12.6 indexから`torch`と
`torchvision`を取得します。`uv sync`でcu130が選択された環境は流用せず、作り直してください。
確認時は次の表示になり、architecture一覧に`sm_70`が含まれることを確認します。

既存フローの都合で`requirements.txt`を直接使う場合も、
`uv pip install --torch-backend=cu126 -r requirements.txt`と明示します。

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_arch_list()); print(torch.ones(1, device='cuda'))"
```

DINOv3はvendorしていません。既定では公式repositoryの固定commitをTorch Hub経由で
取得します。クラスタがofflineの場合は、DINOv3を`reference_repo/`以外へcloneし、weightも
事前にdownloadしたうえで、学習時に次を指定してください。Event Encoderは同じrepositoryと
checkpointを継承します。DINOv3の利用前に公式licenseを確認してください。

```bash
python train.py \
  teacher.source=local \
  teacher.repository=/absolute/path/to/dinov3 \
  teacher.checkpoint=/absolute/path/to/dinov3_vits16_weights.pt \
  dataset.root=/path/to/DSEC
```

teacher cache生成toolはHydra設定を読まないため、offline環境では同じcloneとweightを
明示します。

```bash
python tools/cache_dinov3_features.py \
  --root /path/to/DSEC \
  --split train \
  --output-dir /path/to/cache/dinov3_vits16 \
  --source local \
  --repository /absolute/path/to/dinov3 \
  --checkpoint /absolute/path/to/dinov3_vits16_weights.pt
```

このcacheを使うtrain/evaluateにも、同じ内容の`teacher.checkpoint`を指定します。local weightは
内容SHA-256がmetadataとcheckpoint条件へ記録されるため、別machineでは同一ファイルを別pathへ
置けます。

## 想定するDSEC配置

```text
/path/to/DSEC/
├── train_events/<sequence>/events/left/
│   ├── events.h5
│   └── rectify_map.h5
├── train_images/<sequence>/images/
│   ├── timestamps.txt
│   └── left/rectified/*.png
├── train_calibration/<sequence>/calibration/cam_to_cam.yaml
├── test_events/...
├── test_images/...
└── test_calibration/...
```

pathはすべてCLI/configから渡し、コード内には固定しません。sequenceを明示したい場合は
`dataset.train_sequences`と`dataset.val_sequences`へlistを指定します。frame単位の
random splitは実装していません。

## 0. DSECの取得

EventStateに必要なevents、左rectified images、calibrationだけを公式配布元から取得する
scriptを用意しています。macOS標準のBash 3.2でも動作し、転送途中の`.part`は削除せず、
同じcommandを再実行すると続きから取得します。展開途中の場合も同じZIPを再利用し、全sequenceの
必須fileを確認できた後だけ完了markerを作ります。resume前には保存済みのContent-Length、ETag、
Last-Modifiedを現在のserver応答と照合し、配布物が変わっていれば継ぎ足さず停止します。一時的な
接続切断は`.part`の現在位置から自動で最大20回再試行します。

```bash
# まずURLと保存先だけを確認
bash tools/download_dsec.sh \
  --root /path/to/DSEC \
  --split train \
  --dry-run

# original DSEC train 41 sequenceを取得・展開
bash tools/download_dsec.sh \
  --root /path/to/DSEC \
  --split train

# original DSEC test 12 sequenceも必要な場合
bash tools/download_dsec.sh \
  --root /path/to/DSEC \
  --split test
```

eventsとimagesだけでtrainは圧縮時でも約341 GB、testは約70 GBあります。展開先にはさらに
大きな空き容量が必要です。`--download-only`で取得だけ、`--keep-archives`で展開後もZIPを
保持できます。公式配布ページには暗号学的checksumがないため、HTTPS転送とZIP内CRC、展開後の
必須file・sequence数で完全性を確認します。公式の配布内容と利用条件は
[DSEC download page](https://dsec.ifi.uzh.ch/dsec-datasets/download/)を確認してください。
`--download-only`ではContent-LengthとZIP central directoryまでを検査し、CRC全件検査は
後で展開するときに行います。

### DSEC-Detectionの追加sequence

DSEC-Detectionはoriginal DSEC 53件に7件を追加した全60件です。追加分の物理的な配布場所と
論理的なbenchmark roleは異なります。

- `train/`に配布される公式validation 6件:
  `zurich_city_16_a`〜`zurich_city_21_a`
- `test/`に配布される公式test 1件: `thun_02_a`

公式split全体は
[tools/manifests/dsec_det_official_split.yaml](tools/manifests/dsec_det_official_split.yaml)
に記録しています。追加raw dataを取得する場合も、誤って自動探索されないようoriginal DSECへ
mergeせず、`dsec_det_extra/`へ隔離します。

```bash
bash tools/download_dsec.sh \
  --root /path/to/DSEC \
  --dataset dsec-det-extra \
  --split all
```

このmodeはpretrainingに必要な追加raw events/images/calibrationだけを公式形式のまま
`/path/to/DSEC/dsec_det_extra/{train,test}/`へ展開します。detection labelとDSEC-Det提供の
distorted-event-view imageは取得しません。本projectは独自にevent-rectified座標へRGBを整列するため、
後者をteacher画像として混用しないでください。

標準的なDSEC-Detectionのinductive評価を行う本実験では、pretrainingに使うのは公式train
41件だけとし、その中からsequence単位の内部validationを切ります。公式validation 6件と
公式test 13件はteacher/event cache、正規化統計、early stopping、checkpoint選択にも使いません。
全60件でのself-supervised pretrainingは有効な追加実験ですが、標準結果とは分けて
`transductive / test-exposed pretraining`と明記します。

現在の既定`val_split=test`はGEPとの比較・alignment開発用であり、DSEC-Detectionに対する
benchmark-cleanな設定ではありません。本実験用の`dataset=dsec_benchmark_clean`は、公式train
41件だけを33 train / 8 internal validationへ固定分割します。validationはrecording group単位で
`interlaken_00_{c..g}`と`zurich_city_11_{a..c}`を全てhold outし、公式validation/testは使いません。

## 1. DSECの準備

RGBをevent camera座標へwarpします。event representationも同時にcacheする場合は
`--event-cache-dir`を指定します。既存ファイルは`--overwrite`なしでは変更しません。

```bash
python tools/prepare_dsec.py \
  --root /path/to/DSEC \
  --split all \
  --event-cache-dir /path/to/cache/events/gep_rgb
```

`--split all`はtrainを完了してからtestを処理します。片方だけ処理する場合は`train`または
`test`を指定します。`--sequences`による部分実行はsplitを一つに限定した場合だけ使用できます。

event cacheを省略すると、学習時にraw HDF5から同じ表現を生成します。
既存outputがあるのに対応する`metadata.json`がない場合、その生成条件を確認できないため
自動では再利用しません。内容を置き換えてよいことを確認して`--overwrite`を指定してください。

## 2. Frozen teacher featureのcache

```bash
python tools/cache_dinov3_features.py \
  --root /path/to/DSEC \
  --split train \
  --output-dir /path/to/cache/dinov3_vits16

python tools/cache_dinov3_features.py \
  --root /path/to/DSEC \
  --split test \
  --output-dir /path/to/cache/dinov3_vits16
```

複数GPUでは、同じsplitのsorted sequence一覧を重複のないshardへ分割できます。例えば3 GPUなら
3 processすべてに同じ`--num-shards 3`を与え、`--shard-index 0,1,2`を一つずつ割り当てます。
各processは別sequence directoryだけを書き換えるため、同じoutput directoryを共有できます。
中断後も同じ指定で再実行すると既存fileを検証して再利用します。

各sequenceには`metadata.json`、timestamp名の`.pt`、全frame完了後だけ作られる
`_SUCCESS`を保存します。各frame payloadの主要fieldは次のとおりです。

```python
{
    "format_version": 2,
    "dataset": "DSEC",
    "split": str,
    "sequence_name": str,
    "frame_index": int,
    "timestamp": int,
    "source_image": str,
    "input_fingerprint_digest": str,
    "manifest_digest": str,
    "patch_tokens": Tensor[1120, 384],
    "grid_size": [28, 40],
    "input_size": [448, 640],
    "model_identifier": str,
    "checkpoint_identifier": dict,
    "cache_dtype": str,
}
```

sequence metadataには入力画像・timestamp・前処理、DINOv3 source tree/revision、weightの
識別情報を記録します。datasetやcacheを別machineへ移しても、host固有の絶対pathではなく
相対pathと内容hashで同一性を判定します。

cacheは固定geometryに対応するため、`teacher.cache_features=true`とrandom crop/flipは
併用できません。augmentationを使う場合は`teacher.cache_features=false`にしてonline
teacherを使います。

cacheの完全性検証はstrict correctnessを優先し、起動時にsourceと期待outputの全byteを
SHA-256で再確認します。大規模なDSEC cache、外付けdisk、network filesystemでは起動に
相応のI/O時間がかかります。複数実験では検証済みcacheを同じ高速なlocal storageから使う
ことを推奨します。検証中は`[cache-validation]`としてsplit、sequence、進捗数を表示し、GPUへの
model配置と`Starting ...`表示は検証完了後に行います。

## 3. 学習

E1（既定）:

```bash
python train.py \
  dataset.root=/path/to/DSEC \
  dataset.event_cache_dir=/path/to/cache/events/gep_rgb \
  teacher.cache_dir=/path/to/cache/dinov3_vits16
```

同一条件の3実験:

```bash
# E0: z → DINOv3
python train.py model=no_memory experiment=gep_baseline dataset.root=/path/to/DSEC

# E1: z → LSTM → h → DINOv3
python train.py model=lstm experiment=h_distill_lstm dataset.root=/path/to/DSEC

# E2: z と h の両方をDINOv3へalign
python train.py model=lstm experiment=h_distill_lstm_zloss dataset.root=/path/to/DSEC
```

V100 32GB × 3台では、benchmark-clean splitのE0/E1/E2を1 GPUに1実験ずつ割り当てるlauncherを
使えます。既定batch sizeは、V100 smoke runの実測をもとに8としています。

```bash
bash tools/run_v100_baselines.sh \
  --root /path/to/DSEC \
  --event-cache-dir /path/to/cache/events/gep_rgb \
  --teacher-cache-dir /path/to/cache/dinov3_vits16 \
  --checkpoint /path/to/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```

launcherはGPU 0/1/2をE0/E1/E2へ固定し、同一timestampのrun root以下へ個別checkpoint、
TensorBoard、console logを保存します。最初は`--max-steps 2000`などで長めのpilotを行い、
GPU memoryとvalidation推移を確認してから100000 stepの本実験へ進みます。

上の短い例ではcache pathを環境変数`EVENT_STATE_CACHE`または追加overrideで与えてください。
Hydraの最終設定は各run directoryへ保存され、checkpointにはstudent model、optimizer、
scheduler、step、data iteration位置、乱数状態、解決済みconfigが含まれます。Frozen teacher
weightは含めません。

20ch追加実験:

```bash
python train.py \
  experiment=h_distill_lstm_voxel20 \
  dataset.root=/path/to/DSEC \
  dataset.event_cache_dir=/path/to/cache/events/voxel20
```

20ch cacheは3ch cacheと混ぜず、別directoryへ生成します。

```bash
python tools/prepare_dsec.py \
  --root /path/to/DSEC \
  --split train \
  --event-cache-dir /path/to/cache/events/voxel20 \
  --representation voxel_grid \
  --event-bins 10

python tools/prepare_dsec.py \
  --root /path/to/DSEC \
  --split test \
  --event-cache-dir /path/to/cache/events/voxel20 \
  --representation voxel_grid \
  --event-bins 10
```

sequence長は、たとえば`dataset.sequence_length=1,2,4,8,16`で変更できます。
LSTMのtruncated BPTTは`model.temporal.detach_state_every=4`のように指定します。

中断したrunを再開する場合は、元と同じexperiment/model指定にcheckpointを加えます。
model、event表現、loss、optimizerなどの研究条件が異なるcheckpointは、stateを読み込む前に
明示的なエラーにします。dataset/cacheのmount先やworker数は変更できます。

```bash
python train.py \
  model=lstm \
  experiment=h_distill_lstm \
  dataset.root=/path/to/DSEC \
  training.resume=/path/to/checkpoints/step_00010000.pt
```

## 4. 評価

```bash
python evaluate.py \
  evaluation.checkpoint=/path/to/checkpoints/best.pt \
  dataset.root=/path/to/DSEC
```

評価時はcheckpointに保存したmodel・event representation・teacher・loss条件を復元します。
別machineや別validation集合では、`dataset.root`、`dataset.event_cache_dir`、
`teacher.cache_dir`、`dataset.val_split`、`dataset.val_sequences`をCLIで明示した項目だけ
差し替えます。学習splitは構築しないため、評価対象のsplitだけを配置して実行できます。
local DINOv3 source/checkpointのmount先が変わった場合は`teacher.repository`と
`teacher.checkpoint`（Event Encoderで個別指定した場合は対応する`model.event_encoder.*`）
も明示してください。保存済みsource tree/weight hashと一致しない内容は起動前に拒否します。

validationでは少なくとも次を記録します。

- raw `z` / DINO cosine similarity
- raw `h` / DINO cosine similarity
- projected `z` / `h` similarity
- total / cosine / normalized-MSE loss
- low / medium / high event-count別の`z` / `h` similarity

TensorBoard log、JSON metrics、checkpointはHydra run directory以下へ保存されます。

## 5. E0/E1/E2 feature可視化

可視化用依存を追加した後、学習完了後の3実験の`best.pt`から同一validation clipを
順番にexportして比較します。学習中のGPUと競合させないため、原則として3本の学習が
終了してから実行してください。

```bash
uv sync --active --extra prepare --extra visualize

CUDA_VISIBLE_DEVICES=0 bash tools/visualize_v100_baselines.sh \
  --run-dir outputs/v100_baselines_YYYYMMDD_HHMMSS \
  --root /path/to/DSEC \
  --event-cache-dir /path/to/DSEC_cache/events/gep_rgb \
  --teacher-cache-dir /path/to/DSEC_cache/dinov3_vits16 \
  --teacher-checkpoint /path/to/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --sequence interlaken_00_c \
  --clip-index 0
```

出力先は既定で
`RUN_DIR/feature_visualization/SEQUENCE_clipN/`です。

- `projected_alignment.png`: L2正規化した`Pz`/`Ph`とDINOv3を同一PCA基底・色範囲で比較
- `projected_alignment_raw.png`: projector前の`z`/`h`を個別PCAで確認
- `projected_alignment_inputs.png`: 8 frameのevent/RGB入力とevent count
- `projected_alignment_stability.png`: temporal lag別のfeature安定性
- `projected_alignment.json`: frame別alignment cosineと安定性の数値

main画像の下段は中央patch（`--query-index`で変更可能）に対するtoken cosine mapです。
`Pz`/`Ph`はlossが有効なbranchだけを表示し、未学習projectorを結果として誤読しないように
しています。複数clipを見る場合は`--clip-index`を変更します。

### シーケンス全体の動画

LSTM stateをclip間で維持したまま1シーケンスを最後まで流し、E0/E1/E2と教師を
同期した動画にできます。3モデルはGPUメモリを共有しないよう順番に推論します。

```bash
CUDA_VISIBLE_DEVICES=0 bash tools/visualize_v100_sequence.sh \
  --run-dir outputs/v100_baselines_YYYYMMDD_HHMMSS \
  --root /path/to/DSEC \
  --event-cache-dir /path/to/DSEC_cache/events/gep_rgb \
  --teacher-cache-dir /path/to/DSEC_cache/dinov3_vits16 \
  --teacher-checkpoint /path/to/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --sequence interlaken_00_c
```

`RUN_DIR/feature_visualization/SEQUENCE_sequence/`に次を保存します。

- `alignment.mp4`: event/RGB、共通PCA、query cosine、全区間のteacher cosine推移
- `alignment.csv`: frameごとのevent数と各branchのteacher cosine
- `alignment.json`: シーケンス平均
- `context/`と`E0/`〜`E2/`: 再描画用のclip artifact

exportが中断しても同じコマンドで再開できます。時系列stateを正しく復元するため先頭から
forwardは再実行しますが、検証済みartifactの書き込みは省略します。別シーンは
`--sequence`だけを変えて実行してください。

### State reset ablation

同じE1/E2 checkpointを、stateをシーケンス全体で保持する条件、8-frame validation clipごとに
resetする条件、毎frame resetする条件で比較します。

```bash
CUDA_VISIBLE_DEVICES=0 bash tools/visualize_state_reset_ablation.sh \
  --run-dir outputs/v100_baselines_YYYYMMDD_HHMMSS \
  --root /path/to/DSEC \
  --event-cache-dir /path/to/DSEC_cache/events/gep_rgb \
  --teacher-cache-dir /path/to/DSEC_cache/dinov3_vits16 \
  --teacher-checkpoint /path/to/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --sequence interlaken_00_c
```

`SEQUENCE_state_reset/`以下のE1/E2別MP4で3条件を同期表示します。CSVとJSONには
teacher cosineの平均・標準偏差・frame間変動量・event数三分位別の値も保存します。
`continuous > clip reset > frame reset`なら8 frameを越える履歴、`clip reset > frame reset`なら
clip内の短期履歴が寄与しています。ほぼ同値ならLSTMは主に平滑化器として働いています。

## テスト

```bash
uv pip install --torch-backend=cu126 -e '.[prepare,dev]'
pytest
```

event表現、timestamp境界、sequence boundary、paired transform、DINO patch shape、
3ch/20ch Event Encoder、LSTM/state、teacher freeze、loss backward、checkpoint resumeを
対象にしています。

## 参考コードとの関係

`reference_repo/` は仕様調査のみに使います。本プロジェクトのPython/configはそこをimportせず、
checkpointやlocal DINOv3 repositoryとして指定することも拒否します。そのdirectoryを削除しても
学習コードは影響を受けません。
