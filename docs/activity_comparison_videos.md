# 活動量で選ぶ連続シーンのSemantic・Depth比較動画

2026-11-16。ユーザー指定は「低活動frameだけを集めるのではなく、連続した数秒のscene」。
活動基準は、まず既存5分位による画面全体のevent rateとする。
新規学習は行わず、既評価のE0 Encoder（z蒸留）・E1 Encoder＋LSTM（h蒸留）・E2 Encoder＋LSTM（z+h蒸留）の重みを使う。

## 区間選択

- trainで固定したp20以下を低活動、p80超を高活動とする。境界を評価sceneへfitし直さない。
- 既定は連続10秒、低活動/高活動の時間占有率80%以上を満たす窓を各1本。
- 占有率はframe数ではなくRGB event intervalとの時間overlapで重み付けする。
- 選択窓の元の全入力frameを残す。残り20%に別活動群のframeが混ざっていても削らない。
- 窓は元timestampを先頭とする`[start_us,end_us)`。別系列、欠落interval、250ms超のgapを跨がない。
- 予測やGTスコアを見ず、活動量だけで選ぶ。選択scene同士は同じ系列内で非重複。
- 必要な候補がなければ不足を明記する。時間・閾値は自動で緩めず、rendererも不足時は推論前に停止する。

10秒という長さと80%は可視化用の既定値であり、先行研究由来の確立した閾値という意味ではない。
5秒等に変える場合は別の出力JSONへ明示的に指定し、選択条件を記録する。
局所的な車両のイベント量や物体追跡は今回の選択基準に含まれない。

## 時系列と表示の契約

動画開始でLSTMをresetせず、元sequenceの先頭から全入力を処理する。
表示前とscene間はhead計算だけ省略し、backbone/stateは更新する。
各モデルを1本ずつGPUへ載せるので、全モデルのGPU同時保持や全量特徴cache再生成は不要。
prefix replayには時間が掛かるが、任意の短いwarmupで既評価continuous状態と同一だとは扱わない。

全native入力frameを動画へ残し、[FFmpeg concatのduration](https://ffmpeg.org/ffmpeg-formats.html#concat-1)で元timestampの間隔を保存する。
GT付きframeだけを固定20fpsへ詰める処理は使わない。MP4はVFR、time base1µs。
最後のframeの表示時間を保つため、endpointで同じ画像を1枚だけ再掲する（追加の推論/GTではない）。
コンテナ長は指定秒数＋1µs。実FFmpeg/ffprobeの小さな合成動画で元PTS全保持を確認済み。

既定の6画面はRGB、GEP events、GT、E0予測、E1予測、E2予測。
元timestamp、scene内経過秒、実RGB区間のイベント数と区間長ms、活動率（百万events/s）、活動群、活動曲線、群の時間占有率を表示する。
条件IDと正式名を併記し、Semantic paletteとDepthの0–80m色範囲をモデル/時刻間で固定する。
RGBは参照表示のみで、推論入力はイベント。Depthの色clampは表示だけで、CSVのMAEは未clip予測から計算する。

GTはexact timestampに存在する場合だけ表示する。欠損時はGT unavailable、frame指標はN/A。
前のGTを現在のGTとして保持・補間しない。GTがなくても予測とstate更新は継続する。
動画中/CSVのframe mIoU・frame MAEは単一frameの診断値で、既報全体指標の集計とは区別する。
`frames.csv`にも`event_count`と`window_us`を保存する。イベント画像の色や非ゼロ画素から数を推定せず、既存活動manifestの実カウントを使う。

## 最初はM3EDの同一シーンで両taskを比較

M3EDは既存公開val2系列でSemantic・Depthを同じsceneに揃えられる。
ただしGT有効時刻はtaskによって異なるので、片方GTがないframeも間引かない。
DSECはSemantic公式test3とDepth公開val4が異なるため、既定はtaskごとにscene JSONを作る。
rendererはDSECで別taskのscene JSONを渡すと拒否する。

サーバーで、ユーザーが今回のコードをpushした後に実行する。
GPU0/1で事前学習中なら、`nvidia-smi`でGPU2の空きを確認してから使う。

```bash
cd /home/iASL/Arata_repo/EventState
source env/bin/activate
git pull --ff-only

# GPU不要：低活動/高活動の連続10秒を各1本選択する。
python tools/select_activity_scenes.py \
  --activity-manifest outputs/semantic_depth_activity_20261110/manifests/m3ed.json \
  --dataset m3ed --task semantic \
  --duration-seconds 10 --min-fraction 0.8 --per-kind 1 \
  --output outputs/activity_scenes_20261116/m3ed.json

# runtime/描画のCPU確認。実モデル・データはロードせず、小さい合成fixtureを使う。
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/test_activity_scenes_stdlib.py tests/test_comparison_timing_stdlib.py \
  tests/test_comparison_semantic_server.py tests/test_comparison_depth_server.py \
  tests/test_activity_comparison_server.py

nvidia-smi

# GPU2が空いている場合。Semantic/Depthに同じscene JSONを渡す。
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python -u tools/render_activity_comparison.py \
  --scenes outputs/activity_scenes_20261116/m3ed.json \
  --task semantic --conditions E0,E1,E2 \
  --output outputs/activity_videos_20261116/m3ed_semantic

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python -u tools/render_activity_comparison.py \
  --scenes outputs/activity_scenes_20261116/m3ed.json \
  --task depth --conditions E0,E1,E2 \
  --depth-run outputs/stage2_common_depth_vkitti_20261110 \
  --output outputs/activity_videos_20261116/m3ed_depth
```

scene選択JSONの`status=complete`と`counts.low/high`を確認してから動画を生成する。
出力は各task rootの`{scene_id}/comparison.mp4`と`frames.csv`、nativeframe PNG、
親`manifest.json`、`predictions/`、`assets/`。新規rootのみ許可し、途中成果物も上書き/削除しない。
sceneにはlow/high、sequence、元開始/終了timestamp、全frame時刻、占有率、元manifest SHAと選択条件が保存される。
動画manifestにはhead/backbone SHA・状態履歴・色範囲・入力と表示時刻の関係を保存する。

日本語条件名のためPillowが読める日本語fontを使う。Noto Sans CJK等を自動探索し、
見つからない場合は高価な推論を始める前に停止する。既存fontの実在パスを`--font /path/to/font.ttc`で指定できる。
ffmpegはPATHまたは既存`imageio_ffmpeg`の実行ファイルを使う。
動画依存は既存envのPyTorch/NumPy/Pillow/HDF5関連。Macへ依存を追加して実推論はしない。

旧新DDPも並べる場合は`--conditions E0,E1,E2,E1_old_ddp,E1_new_ddp`を指定する。
現在のConvLSTMは事前学習段階であり、対応するSemantic/Depth headが完成するまではこの動画へ追加しない。

## DSEC

同じselectorでdataset/taskを変え、それぞれ別JSONへ保存する。

```bash
python tools/select_activity_scenes.py \
  --activity-manifest outputs/semantic_depth_activity_20261110/manifests/dsec.json \
  --dataset dsec --task semantic --duration-seconds 10 --min-fraction 0.8 \
  --output outputs/activity_scenes_20261116/dsec_semantic.json

python tools/select_activity_scenes.py \
  --activity-manifest outputs/semantic_depth_activity_20261110/manifests/dsec.json \
  --dataset dsec --task depth --duration-seconds 10 --min-fraction 0.8 \
  --output outputs/activity_scenes_20261116/dsec_depth.json
```

両JSONで低/高の各1sceneが選ばれたことを確認し、GPU2が空いている場合に以下を逐次実行する。
M3ED動画をGPU2で生成中なら、その終了後に実行する。

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python -u tools/render_activity_comparison.py \
  --scenes outputs/activity_scenes_20261116/dsec_semantic.json \
  --task semantic --conditions E0,E1,E2 \
  --output outputs/activity_videos_20261116/dsec_semantic &&
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python -u tools/render_activity_comparison.py \
  --scenes outputs/activity_scenes_20261116/dsec_depth.json \
  --task depth --conditions E0,E1,E2 \
  --depth-run outputs/stage2_common_depth_vkitti_20261110 \
  --output outputs/activity_videos_20261116/dsec_depth
```

条件を満たせばSemantic/Depth各低活動・高活動の計4本を生成する。
出力は各task rootの`{scene_id}/comparison.mp4`。候補不足と既存出力の扱いはM3EDと同じ。
DSEC Semanticは表示440×640/入力448×640、Depthはfull480×640、M3EDは両taskとも中央352×640を維持する。

## 検証範囲

Mac：stdlib23件が通過し、実FFmpegの合成動画でnative PTS・末尾表示時間を確認した。
Torch/Hydra/Pillow/NumPyを使うbackend/合成描画テストと、実データ/GPU動画生成はサーバー未実行。
現在の数値結果を再学習する処理は含まれない。実際のscene時刻・動画本体はサーバー実行後に確定する。
