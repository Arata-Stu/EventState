# V-JEPA 2.1による因果的なh蒸留

2026-11-02実装。実重みロード、GPU smoke、本学習、下流スコアは未確認。
同日追加調査: 本学習の設定確定前に、末尾の「論文・公式コードとの照合」を参照。
単画像とtimestampベースの動画履歴を実装済み。動画既定16枚/4fpsは論文primaryの時間設定を参照するが、
最終配布モデルの全recipe再現や本タスクでの最適値という意味ではない。
既存DINO学習の分割・正規化・GEP入力を引き継ぎ、ViT-S/16 + LSTMのhを
固定V-JEPA 2.1 ViT-B/16 encoderの768次元特徴へ蒸留する。z直接損失なし。
推論時には教師・projector不要。DINO対照と比較する際はh-only、同split/予算/seedを揃える。
従来のz+hモデルとの比較だけで「動画教師の効果」と結論しない。

## 時系列の契約

CLIでは`--input-mode image`または`--input-mode video`を明示する。
教師の履歴長、Stream側LSTMの状態保持、BPTT区切り、EVS蓄積窓は別のparameter。

- 単画像: 現在時刻のRGB1枚を`[B,3,1,H,W]`で画像用tokenizerへ入力。
  `context_frames=1`、`sample_fps=null`。画像反復で動画tokenizerへ入れる処理ではない。
- 動画: 既定K=16、4fps。目標時刻は`t−(K−1)/fps, ..., t`。
  各目標時刻以下の最新RGBを選ぶ（floor、未来nearest禁止）。理想的な先頭末尾差は3.75秒。
  timestampは整数microsecondsとして扱い、サンプリング計算は有理数で行う。
  DSEC/M3EDの不規則間隔や欠損では実際の間隔が変わり、同じRGBを選ぶ場合もある。
- 現在フレームを含むが、tより後のRGBは入力しない。2フレームのtubeletのうち
  **末尾tubeletだけ**をtの教師にする。全clipの中間tokenを過去時刻の教師に配らない。
  V-JEPA内部は双方向attentionのまま。末尾特徴は履歴を使った表現であり瞬間値ではない。
- 教師履歴はデータセットの元系列から取得。Random/Streamのclip開始位置に依存しない。
  系列先頭で履歴が不足する場合だけ最初のRGBを左側に反復し、他系列で補わない。
  timestampの単調性と対象時刻への一致を検査する。RGB補間はせず、過去画像の選択だけを行う。
  prefix反復はlossから除外しない。同じ生徒学習frameを保ち、prefix割合を別途記録する。
- 生徒のHybridの状態保持/detachは既存実装を維持。Random側はクリップ先頭で状態を
  初期化するため、教師の方が長い文脈を持つ場合がある。未来参照とは別の条件差として記録する。
- RGB履歴とEVSへ同一のdeterministic crop/resizeを適用。教師の矩形入力も生徒と同じ
  H×W、patch16。384正方形へ追加cropしたり、tokenを空間補間したりしない。
  公式の矩形入力対応経路を使うが、実重みの矩形GPU推論はsmokeで確認する。
- 初版はaugmentation、event dropout、activity weightingなし。DINO特徴cacheは使用不可。
  GEP cacheは再利用可能。教師の全特徴cacheは作らず、重複RGBはclip内で一度だけdecode/変換し、CPUのunique画像bufferとindexで保持する。
  可変長bufferはcollate時にpadするが、pad画像を教師が読まないことを検査する。
  `teacher-micro-batch`ごとの動画窓だけGPUへ送る。教師FP32、生徒は元設定のAMP。

## サーバーの準備

Macでは実行せず、sig-gpu5のEventState envを使う。既存学習は継続してよい。
公式sourceは任意のmainを毎回自動取得せず、手元checkoutのcommitと重みSHA256を
run configに固定して照合する。sourceに変更/untrackedファイルがあると起動を拒否する。
重みはrepo外に配置する。既存ディレクトリ/ファイルがある場合は再作成しない。

```bash
cd /home/iASL/Arata_repo/EventState
source env/bin/activate

# 初回のみ。採用するcheckoutは記録されたcommitで以後固定する。
git clone https://github.com/facebookresearch/vjepa2.git /home/iASL/Arata_repo/vjepa2
git -C /home/iASL/Arata_repo/vjepa2 checkout --detach 204698b45b3712590f06245fbfba32d3be539812
mkdir -p /home/iASL/Arata_repo/models/vjepa2
curl --fail --location \
  https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt \
  --output /home/iASL/Arata_repo/models/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt

# 既存PyTorch環境は維持。公式encoderの追加依存。
python -m pip install einops timm
python -m pytest -q tests/test_vjepa21_stdlib.py tests/test_vjepa21.py \
  tests/test_training_runtime.py tests/test_models.py tests/test_m3ed.py
```

公式のhub encoder構築と同じパラメータでViT-Bを構築し、`ema_encoder`をstrict load。
predictorは構築しない。公式mainのhubにはlocalhost配布URLが残っていたため、
自動hub downloadを使わず、上記公式配布重みを明示してロードする。

## DSECの単画像・動画smoke（各10 updates、逐次）

baseは現在のF3 clean baselineの**resolved_config.yaml**。
既存のEventState重みをresumeするのではなく、同じDINO初期重みから新規学習する。
元のsplit・正規化統計・空間サイズ・sequence_length・seedを引き継ぐ。

```bash
VJEPA_REPO=/home/iASL/Arata_repo/vjepa2
VJEPA_REV=$(git -C "$VJEPA_REPO" rev-parse HEAD)
BASE=outputs/dense_f3_full_20261027_172342/pretrain/dsec_baseline/resolved_config.yaml

for MODE in image video; do
  CUDA_VISIBLE_DEVICES=0 python tools/run_vjepa21_pretrain.py \
    --base-config "$BASE" \
    --teacher-repo "$VJEPA_REPO" --teacher-revision "$VJEPA_REV" \
    --teacher-checkpoint /home/iASL/Arata_repo/models/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt \
    --input-mode "$MODE" --teacher-micro-batch 1 --batch-size 2 \
    --stage smoke --output-dir "outputs/dsec_vjepa21_${MODE}_smoke_20261102" \
    --execute || break
done
```

`--execute`を外すと設定表示だけ（cache生成・学習なし）。既存output-dirへの上書きなし。
動画の既定は`--context-frames 16 --sample-fps 4`。変更は動画側にだけ指定する。
単画像側にfpsや複数frameを指定すると拒否する。両条件で生徒・seed・loss・batch・splitは同じ。
出力は`train.log`、`resolved_config.yaml`、TensorBoard、checkpoint。
`teacher_input_example.json`は最初のbatchの系列名、目標/実timestamp、元frame index、prefix maskを保存。
以後は`teacher_span_ms`、`teacher_sampling_lag_ms`、`teacher_prefix_frame_fraction`、
動画の`teacher_repeat_fraction`を学習ログ/TensorBoardに集計する。全batchの画像/時刻列は保存しない。
時刻選択はconfigと元系列から決定的に再生成可能。
確認: 最終`step_00000010.pt`、finite loss、h_projected_cosine、勾配、OOM、step所要時間。
生徒384次元と教師768次元の生特徴のcosineは定義しないためz_cosine/h_cosineはNaN。
z_projected_cosineも未学習projectorの診断値なので収束判定には使わない。

本学習はsmoke確認後に`--stage full`と新しいoutput-dirへ変更し、
`--batch-size 2`を外すとbaseのbatch（DSECは8）を維持する。max_steps/scheduler/optimizerも
baseを引き継ぐ。checkpointは10kごと。VRAM/速度次第でbatchを変える場合は別条件として記録。
M3EDはbaseを`pretrain/m3ed_baseline/resolved_config.yaml`へ、output-dirも別名へ変更する。
教師のhistory Kは別ablationなのでOOM時に無記録で減らさない。

本学習の前に同条件のDINO h-only対照と計算予算を決める。
今回の実装だけではV-JEPAの効果、自然低活動での記憶、教師の元学習データとの非重複は証明しない。

## 検証状況と一次ソース

Mac: 依存なし17テスト（未来index排除、clip分割不変、prefix padding、timestamp異常、
設定継承等）、構文/help確認。PyTorchテストはサーバー用として追加し、Macでは実行していない。
PyTorchテストには未来画像の摂動、教師micro-batch不変、系列混入防止、DSEC/M3ED geometry、
teacher freeze、h-only逆伝播を含む。公開重みのstrict loadは別途上記smokeが必要。

- [公式repo](https://github.com/facebookresearch/vjepa2)
- [公式hub構築](https://github.com/facebookresearch/vjepa2/blob/main/src/hub/backbones.py)
- [encoderと時間・矩形格子](https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa_2_1/models/vision_transformer.py)
- [tubeletのtime-major順序](https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa_2_1/models/utils/patch_embed.py)
- [RGB正規化](https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa_2_1/transforms.py)

## 論文・公式コードとの照合（2026-11-02）

調査対象は[原論文v1](https://arxiv.org/html/2603.14482v1)と
公式repoのcommit `204698b45b3712590f06245fbfba32d3be539812`。
調査ファイルのURL/SHA256を `docs/vjepa21_source_audit.json` に保存。
一時cloneのMacではvitg/vitGの大文字小文字衝突があるため、証拠は全てGit objectを
`git show COMMIT:path`で読んだ。公式コードや重みは実行していない。

### 時間設定の根拠

| 出典・用途 | フレーム数 | 指定fps | 解像度 | EventStateでの位置づけ |
| --- | ---: | ---: | --- | --- |
| 論文Table 11、primary | 16 | 4 | 256×256 | 時間設計の参照候補 |
| 論文Table 11、cooldown | 64 | 4 | 384×384 | 配布モデルを理解する基準 |
| 固定commitのViT-B primary YAML | 16 | 4 | 256×256 | 論文と時間設定が一致 |
| 固定commitのViT-B cooldown YAML | 64 | 4 | **256×256** | 論文の384と不一致 |
| 固定commitのViT-B EK100評価YAML | 32 | 8 | 384×384 | 下流タスクで別の時間設定を使用 |
| EventState旧暫定実装 | 8 | 元RGBに従う | 既存GEPと同一 | 互換経路として残す |
| EventState動画の現既定 | 16 | 4 | 既存GEPと同一 | primaryの時間設定を参照、過去方向に採取 |
| EventState単画像 | 1 | 該当なし | 既存GEPと同一 | 現在RGBを画像用tokenizerへ入力 |

出典: [論文Appendix A](https://arxiv.org/html/2603.14482v1#A1)、
[ViT-B primary](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/configs/train_2_1/vitb16/pretrain-256px-16f.yaml)、
[ViT-B cooldown](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/configs/train_2_1/vitb16/cooldown-256px-64f.yaml)、
[EK100](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/configs/eval_2_1/vitb-384/ek100.yaml)。

名目クリップ長N/fpsは16/4で約4秒、64/4で約16秒。
理想的な等間隔サンプルの先頭〜末尾差は(N−1)/fpsだが、公式loaderは
動画のceil(avg_fps)、整数除算、N×step長の区間からlinspaceとclipでindexを生成する。
従って実際の間隔を常に厳密な250msとみなすのは誤り。再現には実index/timestampが必要。
[公式loader](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/src/datasets/video_dataset.py)。

提案: まず16枚・4fps相当を時間設計の参照候補とする。ただし最終384モデルの
64枚条件を完全に再現するわけではなく、計算予算を考慮した短縮条件と明記する。
64枚は長時間感度の候補。優位性を見ずに評価設計を固定し、必要ならtrain内で選定する。
現在のrunnerは動画既定を16枚/4fpsに設定する。枚数とfpsは別fieldとして保存する。

### 教師出力・因果性・空間対応

固定commitの[encoder](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/app/vjepa_2_1/models/vision_transformer.py)で確認:

- ViT-B encoderは768次元。通常forwardは最終blockのLayerNorm済みpatch特徴。
  1664次元はhubで指定する元の大教師側の次元であり、ViT-B encoderの出力次元ではない。
- T=1は画像用tokenizer、T>1はtubelet2の動画用tokenizer。単画像を2枚反復する処理は
  単画像tokenizerの対照とは異なる。現adapterはimageモードK=1とvideoモード偶数K>=2を明示的に分岐する。
- default is_causal=false。窓内の過去tokenも後方の入力を見る。末尾tubeletだけをtへ
  使う設計はEventState独自の因果性確保であり、公式のオンライン蒸留手順ではない。
- 矩形H/Wを受け取り、RoPEの空間座標を補間する経路がある。timestampは入力されず、
  時間位置はtoken index。fpsを20から4へ変える意味はモデル内では自動補正されない。
  [attention実装](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/app/vjepa_2_1/models/utils/modules.py)。
- 同じ矩形crop/resizeとpatch格子を共有することは幾何対応を保つ設計上の根拠。
  ただしtubelet内で動く物体の特徴を瞬間tのEVSへ対応させる近似誤差は残る。
  原論文の未来動作予測ではlast-frame-guided temporal poolingを使っており、
  現在の末尾tubelet抽出と同一手法ではない（Appendix C.4）。

### 「公式蒸留」と現手法の違い

論文Appendix Bの蒸留は固定大教師、masked prediction、最後の層の損失、
12-block predictor、学生EMA、低/高解像度の段階学習を使う。
公開B/L重みではhubが`ema_encoder`をstrict loadする。
[hub](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/src/hub/backbones.py)。

現EventStateのh→MLP→cosine+MSEは、既存EventState対照との条件固定を目的とした
**RGB動画からEVS状態への別の蒸留設計**。V-JEPAの蒸留recipeを再現したとは書かない。
公式trainのloss_exp=1設定も、現在のcosine+MSEの根拠にはならない。
[train](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/app/vjepa_2_1/train.py)。

### 公開物の不一致と未確認事項

1. 論文/READMEの最終384条件に対し、公開B cooldown YAMLは256。
2. B/L用YAMLが指す`app/vjepa_2_1/train.py`では`target_encoder=deepcopy(encoder)`と
   momentum updateを確認。固定外部ViT-G教師を読み込む蒸留経路は、このentrypointでは
   確認できない。公開YAML一式を配布蒸留重みの完全再現レシピと断定しない。
3. hubの配布base URLはlocalhost。公式README配布URLからローカル取得する方針を維持。
4. 元の教師の大規模事前学習データとDSEC/M3EDの非重複は、この設定公開だけでは保証できない。
   EventState側のF3 train/val/test・正規化統計の監査とは別の未確認事項として記載する。

### 論文で主張できる範囲と次の比較

原論文のDepth/Semanticは固定encoderの**単画像**linear probe評価（Appendix C.1/C.2）。
動画履歴やLSTMによる低イベント区間の改善を直接示す結果ではない。
V-JEPA 2.1を教師候補にする動機にはなるが、その効果はEventStateで測る必要がある。

提案する最小比較は、同じ生徒・F3 split・h-only損失・学習予算・下流headで:

1. DINOv3単画像教師: 既存方式の対照。
2. V-JEPA 2.1単画像教師: 教師モデル変更の寄与を測る。
3. V-JEPA 2.1過去動画教師: 同じ教師系列で履歴を加える寄与を測る。

2対3でも画像/動画tokenizerの差は残る。必要に応じて同一画像反復の動画tokenizer対照を
追加し、実際の履歴内容の効果を切り分ける。単画像/動画adapterは実装済みだが、これらの学習結果は未取得。反復動画対照は未実装。

timestampベースの4fps相当履歴、単画像モード、prefixの反復と計測を実装済み。
各tを右端に固定し、目標サンプル時刻以下の最新RGBを選び、未来nearestを使わない。
欠損・重複・実際の時間幅を保存する。これは因果的な適応案であり、公式loaderの厳密再現ではない。
教師を間引いても、生徒のEVS入力/状態更新は既存のRGB間隔で継続する。
初期履歴の大量反復による影響はfull-context区間と分けて確認する。
teacher K/fps、学生BPTT長、stream状態保持時間、EVS蓄積窓はそれぞれ独立に記録する。
