# 最終段Patchwise LSTM対ConvLSTM：共通Stage 1とSemantic・Depth

2026-11-15。活動5分位評価（実験台帳§85）の次の構造比較。
主比較はE1 Encoder＋LSTM（h蒸留）の時間モデルだけを変更する。
最終段1層の種類比較を先に行い、配置比較はその後に分ける。

## 比較条件

| 条件 | モデル指定 | 配置・幅 | 時間モデル＋出力projectionのparameter数 |
|---|---|---|---:|
| E1 Encoder＋LSTM（h蒸留）・Patchwise LSTM | `model=lstm` | ViT最終段のみ、1層、384→384→384 | 1,330,560 |
| E1 Encoder＋LSTM（h蒸留）・ConvLSTM | `model=convlstm` | 同じ最終段、1層、3×3/padding1、384→384→384 | 10,766,208 |

Patchwise LSTMはpatch位置ごとに同じLSTMを適用し、時間モデル内で隣接patchを混ぜない。
ConvLSTMは入力→状態・状態→状態に畳み込みを使う。
この考え方の一次資料は[Shi et al., ConvLSTM](https://arxiv.org/abs/1506.04214)。
本実装はpeepholeなし・単一層3×3の既存baselineで、原論文全体の構成・設定の完全再現ではない。
3×3、384幅、h蒸留、学習予算等はEventState内の比較設定であり、論文の推奨設定だとは説明しない。

同じhidden幅でもparameter数は約8.09倍、時間モデルの線形/畳み込みMACは約8.11倍。
encoder/teacherや非線形演算を除いた解析値であり、実測latencyの倍率ではない。
したがって主比較は同じ表現幅・学習recipeでの比較で、parameter/FLOPs一致の比較ではない。
精度とともにparameter数・MAC・実測推論時間/学習時間・VRAMを報告する。

## Stage 1：固定する学習条件

| 項目 | DSEC | M3ED |
|---|---|---|
| 共通train | 35系列 | 3系列（city_hall/penno_big_loop/penno_small_loop） |
| split正本 | `tools/manifests/dsec_stage1_common_v1.json` | `tools/manifests/m3ed_stage1_common_v1.json` |
| 入力 | RGB間隔GEP、448×640 | RGB間隔GEP、352×640 |
| encoder・教師 | DINOv3 ViT-S/16、同じDINO checkpointからencoder初期化、教師固定 | 同左 |
| 時間モデル | 最終段1層、input/hidden/output各384、dropout0 | 同左 |
| loss | h全tokenのcosine＋正規化MSE、係数各1 | 同左 |
| z loss / activity mask / event dropout | すべて無効 | 同左 |
| Hybrid batch | Random4＋Stream4をjoined forward、蓄積1 | Random2＋Stream2、蓄積1 |
| clip長 | 8frame | 16frame |
| Stream区間 | 原系列ごと1区間 | 原系列ごと2区間、合計6区間 |
| state | Streamだけclipを跨いで保持・clip境界でdetach、区間先頭reset、Randomはzero | 同左 |
| 成功更新数 | 99,999 | 99,999 |
| 予定入力frame | 6,399,936/run | 6,399,936/run |
| optimizer | AdamW、LR1e-4、encoder倍率0.1、WD0.05、clip norm1 | 同左 |
| schedule | warmup999更新、cosine、min LR ratio0.01 | 同左 |
| precision / seed | FP16 / 0 | 同左 |
| augmentation / Stage 1 val | 無効 / 無効 | 同左 |
| checkpoint選択 | 固定最終`step_00099999.pt` | 同左 |

単GPUE1の既存recipeを変更しない。DDP・E2 loss・活動mask・入力窓変更を同時に追加しない。
Stage 1最終checkpointはdataset・時間モデルごとに一つとし、SemanticとDepthで共用する。
taskごとに別のStage 1 stepを選択しない。headは各taskの従来条件で別々に学習する。

FP16 overflowでは成功更新数を進めずデータとstateは進むので、実frame数は予定より増え得る。
`training_summary.json`の`input_frames_seen`、`optimizer_attempts`、`optimizer_skips`も比較する。
同seedでも時間モデル構築時の乱数消費量が異なるため、後に作るprojectorの初期テンソルまで完全一致するとは限らない。
既存E1再利用は同recipe・同seed・同encoder初期値の比較であり、全非時間モデル重みの完全同一初期化は主張しない。

## 既存Patchwise側の再利用

次の単GPUE1は学習/両task評価済み（台帳§85）。旧新DDP版は今回の主比較へ混ぜない。

```text
outputs/stage1_joined_baseline_full_20261107/dsec_E1_w1/checkpoints/step_00099999.pt
outputs/stage1_joined_baseline_full_20261107/m3ed_E1_w1/checkpoints/step_00099999.pt
```

同rootの`dsec_common.json`と`m3ed_common.json`を正規化統計として再利用する。
plannerは統計のprotocol SHA、train系列、RGB interval、係数の妥当性を確認する。
比較表には実checkpoint SHA、`resolved_config.yaml`、`training_summary.json`、source revisionを紐付ける。
既存Patchwiseの出力・head・cacheを上書きしない。

## サーバーでのStage 1実行

実行場所はsig-gpu5の`/home/iASL/Arata_repo/EventState`。Macで学習しない。
今回追加したコードをGit同期後に実行する。
`tools/run_temporal_comparison.py`は既存common plannerからE1/w1だけを取り出す。
既定はConvLSTMのDSEC/M3ED計2run、1run1GPU、GPU0/1に各1本で実行する。

```bash
cd /home/iASL/Arata_repo/EventState
source env/bin/activate
git pull --ff-only
nvidia-smi

# 読み取り専用の計画表示。推論・学習・出力dir作成はしない。
python tools/run_temporal_comparison.py \
  --stage smoke \
  --statistics-dir outputs/stage1_joined_baseline_full_20261107 \
  --output-root outputs/stage1_temporal_smoke_20261115 \
  --gpus 0,1

# サーバー合成テスト→両dataset各99更新のsmoke。
# smoke全成功後、同じDINO初期値から両dataset各99,999更新の本番へ進む。
python -u tools/run_temporal_comparison.py \
  --stage smoke \
  --statistics-dir outputs/stage1_joined_baseline_full_20261107 \
  --output-root outputs/stage1_temporal_smoke_20261115 \
  --gpus 0,1 --execute &&
python -u tools/run_temporal_comparison.py \
  --stage full \
  --statistics-dir outputs/stage1_joined_baseline_full_20261107 \
  --output-root outputs/stage1_temporal_full_20261115 \
  --gpus 0,1 --execute
```

GPUは空きを`nvidia-smi`で確認してから使う。plannerはGPUの空きやV100のVRAM適合を保証しない。
smokeで実データ・AMP・学習・VRAMを確認する。smokeの重みからfullへ継続しない。
既存の出力rootは拒否する。途中で失敗した出力を削除せず、原因を確認する。
OOM時にConvLSTMだけbatch/clip/解像度を縮める自動処理はない。recipe変更が必要なら両方式の比較条件を再設計する。
`--skip-preflight`は同コードのサーバー確認済みの場合だけ使う。上の通常手順では指定しない。

本番の最終重み：

```text
outputs/stage1_temporal_full_20261115/dsec_E1_convlstm_w1/checkpoints/step_00099999.pt
outputs/stage1_temporal_full_20261115/m3ed_E1_convlstm_w1/checkpoints/step_00099999.pt
```

workerログは`outputs/stage1_temporal_full_20261115/logs/{dataset}_E1_convlstm_w1.log`、
preflightログは同rootの`logs/preflight.log`。
各runの`resolved_config.yaml`・`training_summary.json`と、親rootの`plan.json`・`revision.txt`・`source/`を保管する。
親の`_SUCCESS`は全runの終了codeと最終checkpoint存在を確認してから書く。下流評価完了を意味しない。

両方式を同じ新コードで学習し直す場合は、新規rootでsmoke/full双方に
`--architectures patchwise,convlstm`を追加する（計4run、2GPUで順次実行）。
既定の2runには不要。既存Patchwise重みをConvLSTMへ部分ロードして初期化しない。

## Stage 2：task内で方式間の条件を固定

SemanticとDepthは異なるtaskなので、loss・解像度・headを無理に共通化しない。
各task内でPatchwise/ConvLSTMの条件を一致させる。

| 項目 | DSEC Semantic | M3ED Semantic | DSEC Depth | M3ED Depth |
|---|---|---|---|---|
| backbone | 各方式のStage 1最終重みを凍結 | 同左 | 同左 | 同左 |
| readout | 最終段h、384ch | 同左 | block3/6/9のz＋最終段h、各384ch | 同左 |
| 入力 | 上440行crop＋下8行pad→448×640 | 中央352×640 | full480×640 | 中央352×640 |
| head | 1×1 Conv、384→11 | 同左 | DAv2 Metric DPT Small、VKITTI初期化 | 同左 |
| loss | 通常CE、活動重み1/1 | 同左 | log-L1、既存GT mask 0.1<GT≤80m | 同左 |
| train/val/test | 開発6/2→全8trainでrefit→公式test3 | train3/public val2、疑似ラベル | train37/public val4 | train3/public val2 |
| head batch/precision/seed | 8/FP16/0 | 8/FP16/0 | 4/FP32/0 | 4/FP32/0 |
| head optimizer | AdamW LR2e-4、WD0.05 | 同左 | AdamW LR2e-4、既存既定WD | 同左 |
| head予算/選択 | 開発50epoch、毎epoch val mIoU最大のepoch数でrefit | 50epoch、5epoch毎＋最終val mIoU最大 | 50epoch、毎epoch val pooled MAE最小 | 同左 |
| 全体指標 | mIoU・class IoU | 同左 | MAE/RMSE/AbsRel/通常SqRel等＋F³式frame平均 | 同左 |

Depthで最良checkpointをframe MAE基準へ後付けで選び直さず、既存のpooled MAE選択を固定する。
両集計は同じhead重みの評価として並記する。F³式frame平均は原論文の入力/サンプル条件の完全再現ではない。

活動5分位は既存manifestを変更せず、各datasetの両方式・両taskで共有する。

```text
outputs/semantic_depth_activity_20261110/manifests/dsec.json
outputs/semantic_depth_activity_20261110/manifests/m3ed.json
```

推論は全入力でcontinuous stateを更新してから全体/Q1〜Q5へ集計する。群が変わるたびにstateをresetしない。
frame/有効画素/系列件数、sample/活動割当/GT/mask hashを保存する。
DSEC Semantic test3とDepth val4は別系列、M3EDもtask別のGT有効時刻が異なり得るため、task間の時刻一致は強制しない。

## 実装・検証範囲

ConvLSTMのDepth抽出はPatchwiseと同じ4段readoutを許可し、実入力H/Wから矩形gridを渡すよう対応。
これによりDSECのStage 1/Semantic 28×40とDepth30×40を同じ畳み込み重みで処理する。
学習時grid属性は書き換えず、別gridの持越しstateは暗黙resetせずエラーにする。

既存`run_common_depth.py`と`run_semantic_depth_activity.py`は旧5条件・出力先が固定であり、
`--output`だけ変更して今回のConvLSTMを評価できるものではない。
低水準の特徴抽出・head学習・evaluatorは任意checkpointを受け取れるため、
Stage 1完了後は今回のcheckpointと上の条件を明示して接続する。新比較の下流一括runnerは今回追加していない。

## M3EDだけ先に完了した場合の並行作業（2026-11-17）

ユーザーからM3ED worker complete・現在1GPU使用との報告があった。
DSECのStage 1終了を待たず、完了したM3EDのSemantic Stage 2を先行できる。
complete行だけではsmoke/fullを判別できないので、以下は本番99,999更新の最終checkpointがある場合に限る。
出力rootを変更して学習していた場合は、その実rootへcheckpointを置き換える。

既存runnerで任意ConvLSTM checkpointをロードできるため、追加コードや学習中のGit同期は不要。
`--target-root`には既存Patchwise E1が使った5系列のラベルviewを指定し、追加Rittenhouse等の参照も揃える。
`--features h`で主比較に必要なlinear headだけを学習し、共通活動5分位も同時に採点する。

```bash
cd /home/iASL/Arata_repo/EventState
source env/bin/activate
nvidia-smi

# GPU1が空いている場合。空きGPUが別番号ならCUDA_VISIBLE_DEVICESを変更する。
m3ed_semantic_args=(
  --checkpoint outputs/stage1_temporal_full_20261115/m3ed_E1_convlstm_w1/checkpoints/step_00099999.pt
  --features h
  --statistics outputs/stage1_common_20261105/statistics/m3ed_common.json
  --prepared-root /home/iASL/Arata_repo/dataset/m3ed_cache/half_dagr
  --target-root outputs/stage2_m3ed_semantic_linear_20261109/E1/targets
  --cache /home/iASL/Arata_repo/dataset/downstream_scratch/stage2_m3ed_semantic_convlstm_20261117
  --output outputs/stage2_m3ed_semantic_convlstm_20261117
  --activity-manifest outputs/semantic_depth_activity_20261110/manifests/m3ed.json
  --condition-label 'E1 Encoder＋LSTM（h蒸留）・ConvLSTM'
)

test -s outputs/stage1_temporal_full_20261115/m3ed_E1_convlstm_w1/checkpoints/step_00099999.pt &&
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 \
  python -u tools/run_common_m3ed_semantic.py "${m3ed_semantic_args[@]}" --preflight &&
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 \
  python -u tools/run_common_m3ed_semantic.py "${m3ed_semantic_args[@]}"
```

preflightはcheckpoint metadata・split・ラベルの時刻/shape・統計・必要容量を確認し、推論やcache作成を行わない。
完了後は`outputs/stage2_m3ed_semantic_convlstm_20261117/h/`の`best.pt`、`continuous_groups.json`、
`activity_groups.json`等を既存Patchwise E1と比較する。既存出力は上書きせず、新規rootだけを使う。
学習条件は上表のM3ED Semanticと同じで、GT補完や追加lossを同時に導入しない。

次は同じ最終Stage 1 checkpointからM3ED Depthの4段特徴を抽出し、VKITTI初期化DPT headを学習する。
Depthは既存common runnerが旧5条件固定なので、そのまま`--output`だけ変えて使わない。
まずDSEC Stage1＋M3ED Semanticの2ジョブでCPU/I/Oの競合を確認する。GPU2まで重いcache抽出を同時起動することは既定にしない。
今回の提案時点で新しいSemantic/Depthジョブをサーバーへ起動した事実はない。

## M3ED Semanticの最初の比較結果（2026-11-17受領）

E1 Encoder＋LSTM（h蒸留）・ConvLSTMは50epoch中のepoch20を採用し、best mIoU38.601202%。
既存E1 Encoder＋LSTM（h蒸留）・Patchwise LSTMの38.700944%に対し−0.099742pt。
Q1は27.977294%対29.109403%（ConvLSTM−Patchwise=−1.132109pt）、
Q5は47.709706%対47.394163%（＋0.315543pt）。全体/全5群のframe・valid pixels件数は一致。
全体は近いが、低活動Q1での追加容量の利益は今回得られていない。単一seed/単一taskの結論に限定する。
全量と監査は台帳§94、`docs/evaluation_records/20261117_m3ed_semantic_convlstm_activity_best.json`。
DSEC Semanticと両datasetのDepthは引き続き同じ比較条件で進める。

残り3ジョブを空きGPUで並列に進める実行手順は`docs/temporal_remaining_jobs.md`。
`tools/run_temporal_downstream.py --gpus 0,1,2 --preflight`で読み取り確認し、`--execute`で新規runを起動する。

Macはstdlib planner回帰・AST/CLI確認のみ。PyTorch/Hydra/GPU/実データ/VRAMはサーバーで確認する。
専用runnerの実行前preflightには共通学習テスト、ConvLSTM矩形gridとDepth4段、
両dataset/両方式のHydra合成/runtime設定検証を含む。サーバー実行成功・新しいスコアはまだ未受領。
