# 20ch / 320×240 event encoder の事前学習

2026-10-12準備。事前学習済みRGB DINOv3 ViT-S/16を固定教師にし、イベントencoderを蒸留する。
RGB教師自身の追加学習は行わない。初回はDSEC公式train41、seed0、100k step。

## 入出力と学習条件

- Event入力: `[B,20,240,320]`。10時間bin × 正負2、polarity_major。
- RGB教師: 同じ整列画像・同じ320×240の幾何変換。300 patch tokens、各384次元。
- 既存DINOv3重みでTransformerを初期化し、20ch patch projectionだけrandom初期化。
- zをprojector経由で教師特徴に蒸留（cosine + MSE）。h蒸留なし、temporal identity。
- Hybrid Random4+Stream4、batch8、clip8、augmentation/dropoutなし、FP16。
  identityなので持ち越す再帰状態はない。今回はencoder単体の重み作成であり、記憶効果の実験ではない。
- rawイベントを窓ごとにvoxel化し、教師もオンライン計算。既存cacheは変更せず、新しい全量cacheは作らない。
  保存容量を抑える代わりに、raw読み出し・voxel化・教師推論の時間が増える。
- 10kごとに学習checkpoint保存。正規化統計はtrain41のraw voxelを逐次集計しJSONのみ保存。
  統計は既存規約に合わせresize前の640×480 countsで求め、resize後に固定係数を適用する。

## サーバーでの実行

変更ファイルをサーバーに同期した後、以下を実行する。Macでは学習しない。

```bash
cd /home/iASL/Arata_repo/EventState
source env/bin/activate
python -m pytest -q tests/test_voxel20_encoder.py tests/test_fixed_normalization.py tests/test_models.py

python tools/compute_dsec_raw_voxel_normalization.py \
  --root /home/iASL/Arata_repo/dataset/DSEC \
  --output /home/iASL/Arata_repo/dataset/DSEC_cache/voxel20_linear_train41_stats.json

bash tools/run_voxel20_encoder.sh smoke \
  /home/iASL/Arata_repo/dataset/DSEC_cache/voxel20_linear_train41_stats.json
```

100 step smokeのloss、メモリ、処理速度、export成功を確認してからfullを実行する。
統計生成は全train41を読むため時間がかかるが、一度作成したJSONは再利用する。
出力済み統計への上書きは拒否する。

```bash
GPU=0 bash tools/run_voxel20_encoder.sh full \
  /home/iASL/Arata_repo/dataset/DSEC_cache/voxel20_linear_train41_stats.json
```

出力先は `outputs/voxel20_encoder_320_<stage>_<timestamp>/`。
`DATA_ROOT` / `TEACHER`環境変数でサーバーの入力パスを変更できる。

## 再利用する重み

正常終了後の `event_encoder.pt` にbackboneのstate_dictとmetadata、
`event_encoder.json` に入力設定・埋め込み済み固定正規化係数・教師設定・SHA256を保存する。
optimizer、LSTM、蒸留projectorはexportに含めない。
将来のロード先は、同じDINOv3実装のpatch projectionを20ch化し、
`bundle['state_dict']` をbackboneへ `strict=True` でロードする。
forward_featuresの `x_norm_patchtokens` がencoder出力。今回のexportは保存/再読込一致まで確認する。
実際のkartモデルへのロードと推論は別途検証が必要。

## kart移植で合わせる項目

このpresetは既存EventStateの時間線形補間voxelを使う。
rectification後の640×480でvoxel化し、全画角をbilinearで320×240へ縮小する。
320×240座標へ直接イベントを集計する方法とは値が異なる。
窓はDSEC RGB間隔 `(previous_timestamp,timestamp]`。
kartの補間なし・40ms窓・4ms stride tensorとは一致しないため、解像度だけ合わせても互換ではない。
移植時にはbin定義、極性順、窓、空間変換、rectification、固定正規化を合わせる。
既存GEP実験とは入力表現と解像度が変わるため、精度差を単一要因のablationとして扱わない。

## 検証状況

MacではPython構文・shell構文・diff整合のみ確認。
Torch/Hydraテストを追加しランチャー先頭で実行するが、サーバー実行とGPU smokeは未実施。

## 他プロジェクトへの移植

export時に `event_encoder_model.py` も重みの隣にコピーする。
次の3ファイルをkart等へ移せばよい（EventStateパッケージ、Hydra、学習checkpoint、教師重みは不要）。

- `event_encoder.pt`: backbone重みと入力metadata
- `event_encoder.json`: 人が確認する入力契約・SHA256
- `event_encoder_model.py`: PyTorch推論loaderと固定正規化

モデル構造自体は公式DINOv3に依存する。移植先にもtorchとDINOv3の依存ライブラリ、
固定revision `adc254450203739c8149213a7a69d8d905b4fcfa` の公式ソースを用意する。
ネットワークを使わない運用ではそのcheckoutを `dinov3_repo` に指定する。
省略すると固定revisionをtorch.hubで取得する。RGB教師重みのダウンロードはしない。
DINOv3コード・元重みの利用条件も移植先で維持する。

```python
import torch
from event_encoder_model import load_encoder

encoder = load_encoder('event_encoder.pt', device='cuda',
                       dinov3_repo='/path/to/pinned/dinov3')
# counts: [B,20,480,640], rectified, linear temporal voxel, unnormalized
inputs = encoder.prepare_counts(counts.cuda())
with torch.inference_mode():
    tokens = encoder(inputs)              # [B,300,384]
    feature_map = encoder.feature_map(inputs)  # [B,384,15,20]
```

`forward` は正規化済み入力を受けるため、
既に前処理済みの場合は `prepare_counts` を重ねて適用しない。
`prepare_counts` はraw packetのvoxel化は行わず、学習と同じVGA voxelのresize/正規化だけを行う。
320×240へ直接binningしたkart tensorをそのまま互換入力とは扱わない。
loaderはstate_dictをstrictロードし、入力形状と出力形状を検査する。
移植先での実モデル推論・Jetson速度・ONNX/TensorRT化は未検証。

## 学習後の特徴量動画

encoderのexport重みのみを読み、DSECシーケンスを時刻順に処理する。
RGB / 正負イベント分布 / zのPCAカラーの3列MP4を出力する。全量特徴cacheは作らない。

```bash
python tools/visualize_voxel20_encoder.py \
  --weights outputs/voxel20_encoder_320_full_<日時>/event_encoder.pt \
  --root /home/iASL/Arata_repo/dataset/DSEC \
  --sequence zurich_city_16_a \
  --start 0 --frames 400 \
  --output outputs/voxel20_features_zurich_city_16_a.mp4
```

実在し整列RGBが準備済みのシーケンスを指定する。DSECのtrainフォルダ内から選び、
学習外の公式validationシーケンスも利用可能。必要依存は既存visualize extraのimageio-ffmpeg。
PCAは動画全体から最大32frameを等間隔にサンプルして一度だけfitし、色範囲も固定する。
`.pca.pt`に基底と色範囲、`.json`に全frame時刻と使用重みを記録する。
再生fpsは時刻差の中央値から決めた固定fpsなので、不規則な時刻差を厳密には再現しない。
PCA色は意味クラスではなく、別動画間の色は自動的には対応しない。
教師との比較ではなく、後から使うencoder生特徴zの可視化。動画生成はサーバー未実施。

## 160×120での学習

ランチャー第3引数を `160` にすると `voxel20_encoder_160` を選ぶ。省略時は従来の320×240。
VGA全画角を160×120に縮小し、RGB/eventsの上下に4pxずつゼロpaddingする。
ViT入力は160×128、パッチ数は10×8=80。画像を160×128へ引き伸ばす処理ではない。
イベントpaddingは正規化前のcount=0なので、正規化後は `-mean/std` になる。
上下の境界patchも蒸留lossに含まれる。activity mask・教師cacheとの併用は未対応で拒否する。

```bash
bash tools/run_voxel20_encoder.sh smoke \
  /home/iASL/Arata_repo/dataset/DSEC_cache/voxel20_linear_train41_stats.json 160

# smoke成功後
bash tools/run_voxel20_encoder.sh full \
  /home/iASL/Arata_repo/dataset/DSEC_cache/voxel20_linear_train41_stats.json 160
```

出力は `outputs/voxel20_encoder_160_<stage>_<日時>/`。
正規化統計はresize前のVGAで求める契約なので320用と同じJSONを再利用できる。
portable loaderはmetadataからcontent=160×120とmodel input=160×128を復元する。
`prepare_counts`はresize・padding・正規化まで行う。
`encoder`の出力は `[B,80,384]`、`feature_map`は `[B,384,8,10]`。
正規化済み入力を直接渡す場合は `[B,20,128,160]` を用意する。
動画ツールは重みから解像度を判定し、paddingを含む入力・特徴を表示する。
raw読み出しとVGA voxel化の処理は共通であり、全実行時間がパッチ数の比率で減るとは限らない。
Mac構文確認済み、GPU smokeとTorchのpadding/loader一致テストはサーバーで実行する。

## kart向け時間補間なし（2026-10-15追加）

既存linear版と別条件としてランチャー第4引数 `none` を追加。
40ms窓、10bin、`[start,end)`、positive10ch→negative10ch、binへの整数カウント。
整数microsecondの読み出し境界も合わせる。teacherは窓終了時刻に対応するRGB。
640×480のcountを整数比sum poolingしてcontent解像度へ縮小し、padding後に固定正規化。
これは同じ座標イベント集合をOpenEBのfloor座標縮小で直接集計した結果と一致する規約。
RGBは従来のbilinear resize。DSECのrectificationは維持されるため、kartのraw座標カメラと
同じ校正にはならない。窓位置もDSEC RGB終了時刻とdriverの4ms周期で異なる。
ROS/実カメラまで完全一致したという意味ではない。

**linear版の統計/重みは流用しない。** histogram統計は解像度ごとに別ファイル。

```bash
cd /home/iASL/Arata_repo/EventState
source env/bin/activate
python -m pytest -q tests/test_histogram_encoder.py tests/test_voxel20_encoder.py \
  tests/test_fixed_normalization.py tests/test_models.py tests/test_transforms.py

python tools/compute_dsec_raw_voxel_normalization.py \
  --root /home/iASL/Arata_repo/dataset/DSEC \
  --temporal-interpolation none --resolution 160 \
  --output /home/iASL/Arata_repo/dataset/DSEC_cache/voxel20_histogram_160_train41_stats_v2.json

GPU=0 bash tools/run_voxel20_encoder.sh smoke \
  /home/iASL/Arata_repo/dataset/DSEC_cache/voxel20_histogram_160_train41_stats_v2.json 160 none
# smoke成功後
GPU=0 bash tools/run_voxel20_encoder.sh full \
  /home/iASL/Arata_repo/dataset/DSEC_cache/voxel20_histogram_160_train41_stats_v2.json 160 none
```

320版は統計生成/ファイル名/ランチャーの160を320へ置換する。
出力prefixは `voxel20_encoder_160_histogram_...` / `voxel20_encoder_320_histogram_...`。
既存linear版は第4引数省略で維持される。実行中の学習は停止しない。

kartへ最新コードとexportを配置した後、サーバーで同じ重みの出力比較を実行できる。

```bash
python tools/verify_kart_histogram_encoder.py \
  --weights /path/to/event_encoder.pt \
  --kart-root /path/to/kart \
  --dinov3-repo /path/to/pinned/dinov3 \
  --device cuda --output outputs/kart_histogram_parity.json
```

この比較は人工VGA countと学習済み重みを使い、EventStateの前処理/encoderとkart adapterの
入力・patch tokenを比較する。校正、ROS転送、OpenEB CUDA集計の実行検証は含まない。
Macでは構文確認とkart stdlib契約2テスト通過、Torchテストは未実行。
kart既存EVSテストはMacのPyYAML不足で起動できず、サーバー/ROS環境で再実行が必要。

2026-10-17修正: histogram_version=2は時間境界[start,end)を正しく適用する。
旧histogram統計JSONは開始境界のイベント欠落の可能性があるため流用せず再生成する。
GEP/linearの境界は従来通り(start,end]。
