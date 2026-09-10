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
41件だけとします。設計開発中はその中からsequence単位の内部validationを切りますが、条件を
固定した本学習では41件すべてに勾配更新を行い、固定stepの最終重みを採用します。公式validation
6件と公式test 13件はteacher/event cache、正規化統計、early stopping、checkpoint選択にも使いません。
全60件でのself-supervised pretrainingは有効な追加実験ですが、標準結果とは分けて
`transductive / test-exposed pretraining`と明記します。

現在の既定`val_split=test`はGEPとの比較・alignment開発用であり、DSEC-Detectionに対する
benchmark-cleanな設定ではありません。開発用の`dataset=dsec_benchmark_clean`は、公式train
41件だけを33 train / 8 internal validationへ固定分割します。本学習用の
`dataset=dsec_det_train41`は41件すべてを学習に使い、validationを構築しません。どちらも
公式validation/testは使いません。

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

### 短時間event蓄積cache

再学習なしの疎event評価では、各RGB intervalの末尾50%、25%、12.5%だけを使ったGEP cacheを
raw eventから生成します。windowは常に現在のRGB timestampで終わるため因果的です。既存cacheとは
別directoryへ保存され、中断後に同じcommandを実行すると既存frameを検証して続きから再開します。

```bash
bash tools/prepare_dsec_event_windows.sh \
  --root /path/to/DSEC \
  --output-root /path/to/DSEC_cache/events/sparse_windows \
  --split train \
  --sequences \
    interlaken_00_c interlaken_00_d interlaken_00_e interlaken_00_f \
    interlaken_00_g zurich_city_11_a zurich_city_11_b zurich_city_11_c
```

生成先は`gep_rgb_tail_0p5`、`gep_rgb_tail_0p25`、`gep_rgb_tail_0p125`です。
各frameには短縮前のraw event数、短縮後かつrectify前のevent数、実際に表現へ入ったrectify後の
event数を記録します。単一条件だけなら`prepare_dsec.py --event-window-fraction 0.25`も使えます。
既存GEP画像の値を薄める処理ではなく、短縮したtimestamp区間からpercentile正規化をやり直します。

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

現在の本番比較としてE1を後回しにし、E0/E2/E4を公式train 41件すべてで同じ100000 stepだけ
学習する場合は次を使います。

```bash
bash tools/run_v100_e0_e2_e4.sh \
  --root /path/to/DSEC \
  --event-cache-dir /path/to/DSEC_cache/events/gep_rgb \
  --teacher-cache-dir /path/to/DSEC_cache/dinov3_vits16 \
  --checkpoint /path/to/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```

GPU 0/1/2をそれぞれE0/E2/E4へ割り当てます。E2/E4のtemporal modelは1層LSTMです。
3実験とも`dsec_det_train41`を使い、公式train 41件すべてに勾配更新を行います。表現validationと
`best.pt`生成は無効で、既定runでは`checkpoints/step_00100000.pt`を採用します。公式validation/test
は表現学習やcheckpoint選択へ使用しません。E1の本学習は[実験TODO](docs/todo.md)に記録しています。

長時間runのconsole logは、全行を表示せず次の集約コマンドで比較できます。既定では直近10,000
stepについて、初期区間からのloss変化、直近区間内の変化、h/z loss、projected cosine、勾配の
中央値・95 percentile、速度、optimizer skipや非有限値の有無をE0/E2/E4横並びで表示します。

```bash
RUN_DIR="$(ls -dt outputs/v100_dsec_det_e0_e2_e4_* | head -n 1)"
python tools/summarize_training_logs.py "$RUN_DIR"/logs/*.log --window-steps 10000
```

この41-sequence final fitではvalidationを意図的に無効化しているため、表示される
`VALIDATION: なし`は異常ではありません。

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

## Prophesee 1MpxのDAGR downsample推論

1Mpxの1280×720 event streamは、画像化してから縮小せず、DAGRと同じ極性付きの
状態保持filterでイベントのまま640×360へ縮小できます。filter stateはHDF5 chunk間でも
維持されます。モデル入力では内容を上詰めし、下88 pxをpaddingして640×448にします。

```bash
uv sync --active --extra prepare

python tools/prepare_1mpx_dagr.py \
  /path/to/sequence_td.h5 \
  /path/to/sequence_dagr_640x360.h5

CUDA_VISIBLE_DEVICES=0 python tools/export_1mpx_feature_sequence.py \
  --input /path/to/sequence_dagr_640x360.h5 \
  --checkpoint /path/to/EventState/checkpoints/step_00100000.pt \
  --window-ms 50 \
  --start 0 \
  --end 30 \
  --device cuda \
  --output-dir outputs/1mpx/sequence

python tools/render_1mpx_feature_sequence.py \
  --input-dir outputs/1mpx/sequence \
  --output outputs/1mpx/sequence/alignment.mp4
```

前処理HDF5には640×448のpixel mask、28×40のpatch mask、境界patchの実画素率も保存します。
推論時は完全なpadding patchをencoder出力と時系列モデル出力の両方でmaskし、保存featureと
`metrics.csv`の集計から除外します。360行目をまたぐpatch rowは実画素率0.5として記録し、
有効tokenとして扱います。`--start`を0より後にしても、DAGR filterとEventStateのLSTMは
sequence先頭からwarm-upし、指定区間より前の状態を引き継ぎます。artifactの保存だけを
指定時刻から開始します。
最終動画にはevent入力、各featureのPCA空間map、前frameとのcosine推移を表示します。

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

### E0/E2/E4最終重みの複数scene動画

`dsec_det_train41`の100,000 step最終重みではcheckpoint選択用validationを持たないため、
`--split test`を明示して未使用sceneを可視化します。次のrunnerは、既定で事前学習に使っていない
original DSEC test 12本を対象に、E0/E2/E4を1 GPUずつに割り当てます。各checkpointは一度だけ読み込み、
12本を順に処理します。各動画にはE0 `Pz`、E2 `Ph/Pz`、E4 `Ph/Pz`、DINOv3 teacher、
event/RGB入力と全区間のcosine推移が入ります。

```bash
bash tools/visualize_v100_e0_e2_e4_scenes.sh \
  --run-dir outputs/v100_dsec_det_e0_e2_e4_YYYYMMDD_HHMMSS \
  --root /path/to/DSEC \
  --event-cache-dir /path/to/DSEC_cache/events/gep_rgb \
  --teacher-cache-dir /path/to/DSEC_cache/dinov3_vits16 \
  --teacher-checkpoint /path/to/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --gpus 0,1,2 \
  --step 100000
```

最初は`--sequences zurich_city_13_b`を追加して1本だけ完走確認するのが安全です。その後同じcommandを
既定scene集合で再実行します。出力は
`RUN_DIR/feature_visualization/test_scenes_step100000/SEQUENCE/alignment.mp4`、scene別JSON/CSV、
全sceneをまとめた`summary.csv`です。`summary_comparisons.csv`にはE2/E4それぞれの`Ph - Pz`を含む
差分をscene別と全scene集約（`__overall__`）で保存します。各sequenceではstateを先頭だけでresetし、
最後まで連続させます。

feature export中の進捗は、別terminalで次のようにモデル別ログを確認できます。

```bash
VIS_DIR=outputs/v100_dsec_det_e0_e2_e4_YYYYMMDD_HHMMSS/feature_visualization/test_scenes_step100000
tail -n 20 -F \
  "$VIS_DIR/logs/export_E0.log" \
  "$VIS_DIR/logs/export_E2.log" \
  "$VIS_DIR/logs/export_E4.log"
```

export完了後に動画生成だけ失敗した場合は、可視化依存関係を追加して`--render-only`で再開できます。
このモードではE0/E2/E4のGPU推論を繰り返しません。

```bash
uv sync --active --extra prepare --extra detection --extra visualize

bash tools/visualize_v100_e0_e2_e4_scenes.sh \
  --run-dir "$RUN_DIR" \
  --root /path/to/DSEC \
  --event-cache-dir /path/to/DSEC_cache/events/gep_rgb \
  --teacher-cache-dir /path/to/DSEC_cache/dinov3_vits16 \
  --teacher-checkpoint /path/to/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --gpus 0,1,2 \
  --step 100000 \
  --sequences zurich_city_13_b \
  --render-only
```

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

### Event-drop ablation

再学習せず、一定間隔でevent入力をwhite-backgroundの空GEP frameへ置換し、履歴が欠落を
補完できるかを調べます。既定ではE2について、欠落なしと1/2/4/8 frame連続欠落を64 frame
間隔で挿入し、continuous `Ph`、毎frame resetした`Ph`、現在入力だけの`Pz`を比較します。

```bash
CUDA_VISIBLE_DEVICES=0 bash tools/visualize_event_drop_ablation.sh \
  --run-dir outputs/v100_baselines_YYYYMMDD_HHMMSS \
  --root /path/to/DSEC \
  --event-cache-dir /path/to/DSEC_cache/events/gep_rgb \
  --teacher-cache-dir /path/to/DSEC_cache/dinov3_vits16 \
  --teacher-checkpoint /path/to/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --sequence interlaken_00_c \
  --model E2
```

E1も含める場合は`--model both`を使います。`MODEL_gapN.mp4`では欠落frameを
`Event input [DROPPED]`と表示します。各JSONは欠落中／観測中のteacher cosineと条件間差を、
`event_drop_summary.csv`は全gap長・全条件を一覧で保存します。欠落中に
`continuous Ph - frame reset Ph`または`continuous Ph - current Pz`が正なら、履歴による補完です。

### Event-drop学習

E3/E4では学習clipの50%に1/2/4 frameの連続event欠落を1区間挿入します。先頭2 frameを
context、末尾1 frameをrecoveryとして必ず観測し、欠落中も`h`は同時刻DINOv3 tokenで
教師します。E4の`z` lossは欠落frameだけmaskし、空入力から平均的な教師特徴を学ぶことを
防ぎます。validationにはdropoutを適用しません。

```bash
bash tools/run_v100_event_dropout.sh \
  --root /path/to/DSEC \
  --event-cache-dir /path/to/DSEC_cache/events/gep_rgb \
  --teacher-cache-dir /path/to/DSEC_cache/dinov3_vits16 \
  --checkpoint /path/to/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --batch-size 8 \
  --max-steps 2000
```

- GPU 0: E3（E1 + event dropout）
- GPU 1: E4（E2 + event dropout、欠落中の`z` lossをmask）

通常条件のE1/E2と公平に比較するため、既存checkpointからfine-tuneせず同じDINOv3初期値から
学習します。学習ログには`event_dropout_fraction`、欠落／観測frame別の`h` cosineも出力します。
学習後は、生成された`v100_event_dropout_*`を`--run-dir`に指定し、`--model E4`または
`--model dropout`で上記のevent-drop ablationを実行できます。

## DSEC-Detection frozen probe

検出評価はDAGRと同じ公式41 train / 6 validation / 13 test splitをそのまま使います。
独自33/8 pretraining splitはcheckpoint開発だけのもので、検出headのsplitには流用しません。
`probe` protocolはrectified 448×640座標で`z` / `h`の診断を行う内部比較です。公開benchmarkの
主結果には、後述するnative-resolution `dsec-det` protocolの`car` / `pedestrian` 2 classとCOCO
mAP@[.50:.95]を使います。

DSEC-Det labelはraw dataへmergeせず、resume可能なdownload scriptで別directoryへ取得します。

```bash
bash tools/download_dsec.sh \
  --root /path/to/DSEC \
  --dataset dsec-det-labels \
  --split all \
  --yes
```

公式validationの6 sequenceは物理的にはDSEC-Detection追加train archiveにあります。隔離した
`dsec_det_extra`から通常のGEP cacheへ追加します。

```bash
python tools/prepare_dsec.py \
  --root /path/to/DSEC \
  --layout dsec-det-extra \
  --split train \
  --sequences \
    zurich_city_16_a zurich_city_17_a zurich_city_18_a \
    zurich_city_19_a zurich_city_20_a zurich_city_21_a \
  --event-cache-dir /path/to/DSEC_cache/events/gep_rgb
```

EventState backboneをsequence先頭からstreaming実行し、projection前の`z`/`h`を一度だけcacheします。
これ以降のdetector学習ではbackboneを読み込まないため、frozen probeを高速かつ再現可能に比較できます。

```bash
python tools/cache_dsec_detection_features.py \
  --checkpoint /path/to/event_state_best.pt \
  --event-cache-dir /path/to/DSEC_cache/events/gep_rgb \
  --output-dir /path/to/DSEC_cache/detection_features/E4 \
  --role train \
  --features z h \
  --state-policy continuous \
  --device cuda \
  --teacher-checkpoint /path/to/dinov3_vits16_pretrain_lvd1689m-08c60483.pth

python tools/cache_dsec_detection_features.py \
  --checkpoint /path/to/event_state_best.pt \
  --event-cache-dir /path/to/DSEC_cache/events/gep_rgb \
  --output-dir /path/to/DSEC_cache/detection_features/E4 \
  --role val \
  --features z h \
  --state-policy continuous \
  --device cuda \
  --teacher-checkpoint /path/to/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```

3 GPUでfeature cacheを作る場合は、同じroleについて`--num-shards 3`を全processへ与え、
`--shard-index 0`、`1`、`2`を一つずつ割り当てます。sequence単位で分割するためrecurrent stateは
各sequence内で保たれ、同じoutput directoryへ安全に書き込めます。

YOLOX型headはstride 16のtoken mapからstride 8/16/32 pyramidを作り、decoupled headと
dynamic-k assignmentで学習します。DSEC-Det bboxはdistorted event座標なので、loaderが公式
`rectify_map.h5`でEventStateと同じrectified event座標へ変換します。

```bash
uv sync --active --extra detection

CUDA_VISIBLE_DEVICES=0 python tools/train_dsec_detection.py \
  --feature-cache-dir /path/to/DSEC_cache/detection_features/E4 \
  --labels-root /path/to/DSEC/dsec_det_labels \
  --dataset-root /path/to/DSEC \
  --feature h \
  --output-dir outputs/dsec_detection/E4_h \
  --batch-size 16 \
  --epochs 50 \
  --device cuda
```

`z`と`concat`も同じcommandの`--feature`と出力先だけ変えて別headを学習します。state効果は、
同じE4 checkpointから`--state-policy frame`で別feature cacheを作り、continuous h用に学習した
`best.pt`を`tools/evaluate_dsec_detection.py`でそのcacheへ適用して測ります。headを再学習しないため、
差はstate利用によるものです。

3 GPUで`z`、`h`、`concat`を同時に学習する場合は次を使います。

```bash
bash tools/run_dsec_detection_probes.sh \
  --feature-cache-dir /path/to/DSEC_cache/detection_features/E4 \
  --labels-root /path/to/DSEC/dsec_det_labels \
  --dataset-root /path/to/DSEC \
  --output-dir outputs/dsec_detection/E4 \
  --batch-size 16 \
  --epochs 50
```

最良checkpointを公式validationまたはtestへ適用し、COCO mAPをJSONで保存できます。test評価の前には、
上と同じfeature-cache commandを`--role test`でも実行してください。

```bash
CUDA_VISIBLE_DEVICES=0 python tools/evaluate_dsec_detection.py \
  --checkpoint outputs/dsec_detection/E4/h/best.pt \
  --feature-cache-dir /path/to/DSEC_cache/detection_features/E4 \
  --labels-root /path/to/DSEC/dsec_det_labels \
  --dataset-root /path/to/DSEC \
  --role val \
  --feature h \
  --output outputs/dsec_detection/E4/h/val_metrics.json \
  --device cuda
```

YOLOXの設計はApache-2.0の公開方式を参照していますが、本実装はEventState内で独立実装しており、
GPL-3.0のDAGR/RVT sourceをimportまたはcopyしません。

### DSEC-Det event-only benchmark protocol

公表値にはDSEC-Detのdistorted event viewをnativeに近い640×430座標で使います。DAGRがgraph数を
抑えるために採用した1/2 event downsamplingはEventStateの必須条件ではありません。物理的なbox選択は
DAGRと揃え、native座標で最小辺`>20`・対角`>30`、連続する有効label frame対、公式41/6/13 splitを
固定します。EventState cacheはrectified座標なので、sequence固有の`rectify_map.h5`を使って特徴mapを
一度だけdistorted座標へwarpします。

`tools/prepare_dsec_detection_benchmark_features.py`の既定は`--scale 1`です。計算条件をDAGRへ近づける
補助実験だけ`--scale 2`を指定し、320×215座標と縮小後の最小辺`>10`・対角`>15`を使います。ただし、
これはfeature座標を縮小する条件であり、DAGR固有のsigned-event間引きを完全には再現しません。

trainとvalidationのbenchmark feature cacheは、roleごとに3 GPUで準備できます。

```bash
INPUT=/path/to/DSEC_cache/detection_features/E4
OUTPUT=/path/to/DSEC_cache/detection_features/E4_dsec_det
LOGS=/path/to/DSEC_cache/logs/dsec_det_warp
mkdir -p "$OUTPUT" "$LOGS"

for ROLE in train val; do
  PIDS=""
  for GPU in 0 1 2; do
    CUDA_VISIBLE_DEVICES="$GPU" python \
      tools/prepare_dsec_detection_benchmark_features.py \
      --input-dir "$INPUT" \
      --output-dir "$OUTPUT" \
      --dataset-root /path/to/DSEC \
      --role "$ROLE" \
      --features z h \
      --batch-size 64 \
      --device cuda \
      --num-shards 3 \
      --shard-index "$GPU" \
      > "$LOGS/${ROLE}_gpu${GPU}.log" 2>&1 &
    PIDS="$PIDS $!"
  done
  STATUS=0
  for PID in $PIDS; do wait "$PID" || STATUS=1; done
  [ "$STATUS" -eq 0 ] || exit 1
done
```

`probe`のheadとは座標とframe集合が異なるため、benchmark headは別に学習します。

```bash
bash tools/run_dsec_detection_probes.sh \
  --feature-cache-dir /path/to/DSEC_cache/detection_features/E4_dsec_det \
  --labels-root /path/to/DSEC/dsec_det_labels \
  --dataset-root /path/to/DSEC \
  --output-dir outputs/dsec_detection_benchmark/E4 \
  --protocol dsec-det \
  --batch-size 16 \
  --epochs 50
```

E0/E2/E4の100,000 step最終重みが揃った後は、cache作成と主比較を次のrunnerで固定できます。
E0は`z`、E2/E4は`h`を使い、各seedで3モデルをGPU 0/1/2へ一つずつ割り当てます。

```bash
bash tools/prepare_frozen_dsec_detection_e0_e2_e4.sh \
  --run-dir outputs/v100_e0_e2_e4_YYYYMMDD_HHMMSS \
  --event-cache-dir /path/to/DSEC_cache/events/gep_rgb \
  --dataset-root /path/to/DSEC \
  --output-root /path/to/DSEC_cache/detection_features/final \
  --teacher-checkpoint /path/to/dinov3_vits16_pretrain_lvd1689m-08c60483.pth

bash tools/run_frozen_dsec_detection_e0_e2_e4.sh \
  --feature-root /path/to/DSEC_cache/detection_features/final/dsec_det \
  --labels-root /path/to/DSEC/dsec_det_labels \
  --dataset-root /path/to/DSEC \
  --output-dir outputs/dsec_detection_frozen \
  --seeds 0,1,2 \
  --batch-size 16 \
  --epochs 50
```

cache作成を中断して同じcommandを再実行した場合、metadataと全frameが揃った完了済みsequenceは
model forwardも座標変換も丸ごとskipします。不完全なsequenceだけは、continuous LSTM stateを
再構築するため先頭からforwardします。`--overwrite`を指定した場合はこのresume判定を無効化します。
座標変換cache format v3では各frameをbatch tensorから独立したcompact storageへcloneして保存します。
v2以前のwarped cacheはstorageがbatch全体を保持する可能性があるため再利用せず、作り直します。

Frozenの次は、同じ公式splitとYOLOX recipeでend-to-end Fine-tuneとScratchを実行します。
Fine-tuneはRGB画像・DINO teacher weight/cacheを読まず、event encoder・1層LSTM・headを更新します。
Scratchは最終checkpointを
architecture定義としてだけ読み、event encoder、LSTM、headをランダム初期化します。学習loaderは
label frameで終わる固定長clip、validation loaderは未ラベルframeを含む連続streamです。

```bash
bash tools/run_dsec_detection_finetune_scratch.sh \
  --run-dir outputs/v100_e0_e2_e4_YYYYMMDD_HHMMSS \
  --event-cache-dir /path/to/DSEC_cache/events/gep_rgb \
  --labels-root /path/to/DSEC/dsec_det_labels \
  --dataset-root /path/to/DSEC \
  --output-dir outputs/dsec_detection_end_to_end \
  --seeds 0,1,2 \
  --batch-size 4 \
  --epochs 50 \
  --validate-every 5
```

同じ`--output-dir`で再実行すると、各条件に`last.pt`があればepoch単位で自動再開します。
checkpointは各epoch末にatomic保存されるため、途中停止時に失うのは実行中のepochだけです。
runnerは開始前に`pycocotools`も検査し、長時間学習後のvalidationで初めて依存不足が判明することを
防ぎます。

最初は`--seeds 0 --epochs 1 --validate-every 1`で実データsmoke testを行い、V100のVRAM使用量と
validation完走を確認してから本実行へ進みます。研究上の評価順とJEPA接続前に固定する契約は
[downstream roadmap](docs/downstream_roadmap.md)にまとめています。

各epochのログを直接読む代わりに、全seedのbest validation値を集約できます。このツールは
Python標準ライブラリだけで動作します。

```bash
python tools/summarize_detection_runs.py \
  outputs/dsec_detection_frozen \
  outputs/dsec_detection_end_to_end \
  --output outputs/dsec_detection_summary.json
```

最終testでは、まず元のfeature cacheを`--role test`で生成し、benchmark feature変換も
`--role test`で実行します。その後`tools/evaluate_dsec_detection.py`へbenchmarkの`best.pt`を
渡します。checkpointに記録された`dsec-det` protocolは評価時にも強制されます。

### Frozen detectionの動画可視化

公式validation/test sequenceの全frameについて、左列にDSEC-Det座標へwarpしたaligned RGBと
GEP 3-channel event入力、上段にE0/E2/E4の予測、下段に各detectorが使った
`E0-z` / `E2-h` / `E4-h`特徴を表示できます。検出背景は既定でRGB、`--detection-background event`
でeventへ切り替えられます。特徴の色はsequence内の
全model・全表示frameから求めた共通のL2-normalized PCAなので、model間で直接比較できます。
GTは白い破線、car予測は橙、pedestrian予測は水色です。動画と同時にframe別の予測数・最大scoreを
CSVへ、使用checkpointなどをJSONへ保存します。GTは公式mAP評価対象frameだけに表示し、それ以外は
`GT unavailable`と明記します。既定の`--frame-mode all`はfeature cache内の全frameを飛ばさず描画し、
従来の評価frameだけを確認するときのみ`--frame-mode evaluated`を指定します。

```bash
uv sync --active --extra detection --extra visualize

DET_DIR=outputs/dsec_detection_frozen_step100000
FEATURE_ROOT=/path/to/DSEC_cache/detection_features/e0_e2_e4_step100000/dsec_det

bash tools/visualize_frozen_dsec_detection_e0_e2_e4.sh \
  --det-dir "$DET_DIR" \
  --feature-root "$FEATURE_ROOT" \
  --event-cache-dir /path/to/DSEC_cache/events/gep_rgb \
  --labels-root /path/to/DSEC/dsec_det_labels \
  --dataset-root /path/to/DSEC \
  --output-dir outputs/dsec_detection_visualization/seed_0 \
  --role test \
  --seed 0 \
  --gpu 0 \
  --score-threshold 0.25 \
  --sequences zurich_city_13_b
```

`--sequences`を省略すると選択したroleの全sequenceを順番に処理します。最初の動作確認には
`--max-frames 100`を追加します。比較図のseedをtest結果から選ぶと恣意性が入るため、seed 0または
validation mAPの中央値に最も近いseedをtest評価前に固定します。

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
