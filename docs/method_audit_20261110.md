# EventState 学習方法と先行研究の対応

2026-11-10時点の共通Stage 1と各下流taskの確認用一覧。
現在の主対象はSemantic segmentationとDepth、その後に可能ならFlow。Detectionは過去実験として区別する。

SqRelは現在のEventStateでは評価指標。Depth学習はlog-L1、best headの選択は公開valのpooled MAEで行う。
損失、checkpoint選択指標、最終評価指標は別の役割を持つ。
通常SqRel（平均誤差²/GT）の現行定義は、DAv2/Monodepth2の利用例を踏まえて2026-11-10にユーザーが採用を了承した。
F³互換の平均誤差²/GT²は区別して報告する。

「直接参考」は構造・式・splitなど特定の部品が一次情報と対応することを指す。
「着想」は設計方針の共通性、「独自」はEventStateの変更・実験予算・工学的適応を指す。
独自設定を、先行研究と同一条件・最適性確認済みとは扱わない。

寸法は原則として高さ×幅。以下の細かいhyperparameterはローカルrunner/configの既定値。
現在のサーバーに保存された学習時config・plan・source snapshotをこの監査で直接取得した値ではない。
受領記録では共通Stage 1単GPU6件・旧新DDP4件、Semantic両dataset各5条件が完了。
Depthは学習中との報告だけで、実設定・進捗・最終値は未受領。Flowの実学習・提出も未受領。

## Stage番号の意味

| 呼称 | 前半 | 後半 | 下流task |
|---|---|---|---|
| GEP | Stage 1はRGB VFMとのalignment | Stage 2はcausal Transformerによる生成事前学習 | その後にtask学習 |
| 今回のEventState | Stage 1はencoderと必要ならLSTMの蒸留 | GEPの生成事前学習は実施していない | Stage 2と呼んでheadを学習 |

GEP全体の再現ではなく、RGB教師によるイベント特徴整合を参考にした再帰蒸留と、その凍結特徴の評価。
GEP論文と公開コードの既定値にも違いがあるため、両者を分けて比較する。[GEP論文](https://arxiv.org/html/2603.23032v1)、[GEP公式README](https://github.com/uzh-rpg/generative_event_pretraining/tree/b07c78c8124ab60682d2aa440a2aa52911de9c36)

## Stage 1の構造と入力

| 項目 | 現行EventState | 先行研究との関係 |
|---|---|---|
| 学習単位 | dataset×条件×seedにつき1回。その最終重みを各taskへ共有 | 共通表現を比較するEventStateの実験設計 |
| train | DSEC35系列、M3ED3系列。Stage 1 validationなし | 下流holdoutを保護する独自共通pool |
| 入力 | 3ch GEP RGB、白背景、極性別countの非ゼロ90 percentile正規化、優勢極性を着色 | GEP公開preprocessingを参考。論文のpercentile記述とは区別 |
| 時間窓 | RGB間隔 (前RGB時刻, 現RGB時刻]、fraction=1 | 現在の因果的入力設計。GEP公開前処理の中点までの蓄積とは同一でない |
| DSEC | rectified480×640から中央448×640、上下各16行crop | 既存cache/presetに合わせた独自適応。480も16で割れるのでpatch制約だけでは448を正当化できない |
| M3ED | native720×1280→DAGR符号付き2×downsamplingとrectify→360×640→中央352×640 | 360をpatch16で扱うため上下各4行crop。F³のnative評価とは別 |
| event正規化 | 共通trainだけから求めたdataset別3ch mean/std | holdoutを統計に含めない独自共通protocol |
| RGB教師 | frozen DINOv3 ViT-S/16、384次元、最終x_norm_patchtokens。CLS/storage tokenを除く | frozen RGB VFM教師という原理はGEP由来。GEP論文の標準はDINOv2 |
| 教師weight | runner既定dinov3_vits16_pretrain_lvd1689m-08c60483.pth | DINOv3の公開RGB事前学習重み |
| 教師のRGB正規化 | ImageNet mean/std、eventと対応した幾何 | 使用するRGB事前学習モデルの入力仕様 |
| student | 同じDINOv3 ViT-S/16 RGB重みで初期化。3ch patch projectionも保持し、encoder全parameterを学習 | 同じRGB重みからの初期化はGEPと共通する原理 |
| z | その時刻の最終encoder patch特徴、384次元 | frameごとのevent表現 |
| temporal | 最終encoder後に1層Patchwise LSTM。input/hidden/output384、dropout0、LSTM後Linear384→384 | EventState独自の再帰構造。GEPのGPTとは別 |
| LSTMの空間処理 | 固定patch位置ごとに独立して時系列処理。全patchでLSTMの重みを共有。空間warp・物体追跡なし | 独自設計。Patchwise memoryと世界座標上の同一物体追跡は区別する |
| h | 上記LSTM経路の出力特徴 | 同時刻のRGB教師へ蒸留する時系列表現 |
| 蒸留projector | z/h別のLinear384→768→GELU→Linear768→384 | 独自設計。GEP公開実装は通常cos/MSE時Identity、NCE時MLP |
| augmentation | 現common runでは全て無効 | 独自の統制条件。GEPのRRC/flip/極性色swapとは異なる |
| event dropout | 現common runでは無効 | 過去E3/E4とは別条件 |
| activity mask | 現common runでは無効。全時刻・全patchが蒸留対象 | 過去activity α1/ScaleEvent型の領域選択とは別条件 |

実装: [Stage 1 runner](/Users/at/project/research/EventState/tools/run_stage1_common.py:35)、[teacher設定](/Users/at/project/research/EventState/configs/teacher/dinov3_vits16.yaml:1)、[LSTM](/Users/at/project/research/EventState/event_state/models/temporal/lstm.py:21)、[projector](/Users/at/project/research/EventState/event_state/models/projectors.py:8)。

### 蒸留損失

Yを勾配停止した同時刻RGB教師特徴、Pz/Phを各projector、p̂とŶをtokenごとL2正規化した特徴とする。

D(p,Y) = 全batch・時刻・patchの平均 [(1 − p̂・Ŷ) + ‖p̂ − Ŷ‖²]

| 条件 | 主な学習経路 | 損失 |
|---|---|---|
| E0 Encoder（z蒸留） | encoderとz projector。temporalはIdentity | D(Pz(z),Y) |
| E1 Encoder＋LSTM（h蒸留） | encoder、LSTM、h projector | D(Ph(h),Y) |
| E2 Encoder＋LSTM（z+h蒸留） | encoder、LSTM、両projector | D(Ph(h),Y) + D(Pz(z),Y) |

- cosineと正規化squared-L2の係数は各1、eps=1e-6。
- squared-L2は384次元で割らず、各tokenの二乗距離をtoken平均する。
- E2は二枝の単純和であり、2で平均しない。E1と総loss尺度が同じという意味ではない。
- 教師には勾配を流さない。h lossはLSTM、encoder、clip内の過去時刻へ逆伝播する。
- 正常な単位ベクトルでは二乗L2=2(1−cos)なので、現在の各枝は3(1−cos)に等価。二種類の独立した制約ではない。
- InfoNCE、RGBをstudentに通すcapability preservation、未来token生成lossは入っていない。

根拠: [蒸留式](/Users/at/project/research/EventState/event_state/losses/distillation.py:53)、[二枝の適用](/Users/at/project/research/EventState/event_state/training/engine.py:511)。
GEP論文§3.3はcosine、InfoNCE、RGB preservationを含む。公開gra.pyのMSEはraw F.mse_lossであり、現在の正規化L2とは別。
したがってE0のconfig名がgep_baselineでも「GEP完全再現のbaseline」とは記載できない。
[GEP公式loss](/Users/at/project/research/EventState/reference_repo/generative_event_pretraining/src/gra.py:492)

### Samplingとoptimizer

| 項目 | DSEC単GPU | M3ED単GPU | 新型3GPU DDP |
|---|---|---|---|
| sampling | joined Hybrid | joined Hybrid | 各rankで同じRandom/Stream結合方針 |
| Random＋Stream | 4＋4 clips | 2＋2 clips | 各rank DSEC4＋4、M3ED2＋2 |
| clip長 | 8 frame | 16 frame | 同じ |
| accumulation | 1 | 1 | 1 |
| global clips/update | 8 | 4 | DSEC24、M3ED12 |
| optimizer updates | 99,999 | 99,999 | 33,333 |
| warmup | 999 updates | 999 updates | 333 updates |
| 計画frame提示数 | 6,399,936 | 6,399,936 | 各dataset同数 |

- 上記は計画提示数。FP16 overflowによるoptimizer更新skip時は実提示数が増え得るため、実数は保存training summaryで確認する。
- Randomはclipごとstate reset。Streamは時系列でstate保持、clip境界でdetachするTBPTT。
- joined batchはStream→Randomの順。Randomの初期stateはゼロで、Stream側だけstateを保存する。
- M3EDは3系列を各2区間に分けた6 stream lane。区間境界resetと短いlaneの周回がある。
- AdamW、base LR1e-4、encoder倍率0.1でpeak1e-5、LSTM/projector peak1e-4、WD0.05、betas0.9/0.999、eps1e-8。
- bias/norm等の1D parameterとCLS/mask/pos/storage tokenにはWDをかけない。
- warmup後cosine decay、最小LR比0.01、FP16、gradient clipping1.0、seed0。
- Stage 1 validationとbest選択はなし。単GPUstep99,999、DDPstep33,333の最終重みを固定する。
- 旧型DDPはseparate Hybrid、各rank Random1＋Stream1、蓄積DSEC4/M3ED2。同じglobal clip予算でもforward構成・提示順・state運用が異なる。
- RVTの混合samplingに着想を得た独自実装。単GPU/DDP差は純粋なbatch sizeの因果効果とは扱わない。

特定のLR、WD、clip長、LSTM幅、総update数をそのまま採った先行論文の根拠は未確認。
GEP論文のalignment既定は224×224、batch32、LR3e-5、WD0、25万step、augmentationあり。
公開GraConfigの既定はさらに異なるため、そのまま今回の設定の根拠にはできない。
[RVTの混合処理](https://github.com/uzh-rpg/RVT/blob/master/modules/utils/detection.py)、[現行optimizer](/Users/at/project/research/EventState/event_state/training/optim.py:46)、[DDP仕様](/Users/at/project/research/EventState/docs/stage1_common_ddp.md:26)

## Stage 2の共通条件

- 現行SemanticとDepthはStage 1 encoder/LSTMを固定する。taskごとに独立したheadを学習する。
- SemanticはE0 Encoder（z蒸留）のz、E1 Encoder＋LSTM（h蒸留）・E2 Encoder＋LSTM（z+h蒸留）・E1旧新DDPのhを使う。
- ラベルなし中間frameも系列順にLSTMを更新し、系列境界でresetする。保存済み特徴でheadを学習する際はshuffleする。
- Stage 1のtrain poolとStage 2 headのtrain splitを混同しない。例えばDSECはStage 1が35、Depth headが37系列。
- 凍結特徴を用いるprobeという比較方針はDINOv3やScaleEventに先例がある。Semanticはlinear head、Depthは非線形DPT headであり、以下の全数値が同一recipeという意味ではない。

## Semantic segmentation

| 項目 | DSEC Semantic | M3ED Semantic |
|---|---|---|
| event encoder入力 | rectified480×640の上440行をcropし、下8行zero padして448×640 | DAGR/rectify360×640から中央352×640 |
| 特徴map | 384×28×40 | 384×22×40 |
| logitsとlabel | 440×640 | 352×640 |
| label | 公式11クラスGT | 配布InternImage疑似labelをCityscapes19→DSEC11へmap |
| head | 1×1 Conv384→11＋bilinear upsampling、BNなし | 同じ |
| loss | 通常CEのみ。活動重みはactive/inactiveとも1 | 通常CEのみ |
| optimizer | AdamW、LR2e-4、WD0.05、batch8、seed0、FP16 | 同じ |
| schedule | cosine、最小LRは初期×0.01 | 同じ |
| gradient clipping | norm上限10 | 同じ |
| augmentation | cache特徴とlabelの水平反転0.5 | 同じ |
| 開発予算 | 最大50 epoch、毎epoch val | 50 epoch、5epochごと＋最終epoch val |
| head選択 | val mIoU最大epoch | val mIoU最大best.pt |
| 最終fit | headを再初期化し、全8trainで選択epoch数だけ学習 | val-bestをそのまま採点 |
| 最終採点 | 未使用の公式test3 | 同じ公開val2。独立hidden testではない |
| 指標 | mIoU、クラスIoU、pixel accuracy、mean class accuracy | 同じ |
| 指標集計 | 全体/各群で混同行列を合算。union>0のクラスのIoU平均。ignore255 | 同じ |

linear headではhead-width引数192は使わない。空間Conv refinementはなく、現在はDiceも使わない。
M3ED labelは同じRGB timestampを厳密に照合し、nearest rectificationと同じ中央cropを行う。
先頭RGBは過去入力intervalがないため除外する。

fitは両datasetで実施する。違いはモデル選択後の追加refit。
DSECは開発valで選んだepoch数だけを使い、新headを全8trainで再学習し、別のtest3を採点する。
M3EDはtrain3で学習したval-bestを、その選択に使った同じval2で採点する。
例えばDSECで12epochが最良なら、新headを全8系列で12epoch学習するという意味で、開発headの継続学習ではない。
M3EDを全5系列でrefitすると旧val2も学習集合に入るため、その2系列を汎化性能の評価には使えない。
現在のM3ED protocolにはこのrefit後に採点する独立testを設けていない。

### Semanticのsplit

- DSEC開発train6: zurich_city_00_a、01_a、02_a、04_a、05_a、06_a。
- DSEC開発val2: zurich_city_07_a、08_a。
- DSEC final train8: 上記6＋2。
- DSEC公式test3: zurich_city_13_a、14_c、15_a。
- M3ED train3: car_urban_day_penno_big_loop、car_urban_day_penno_small_loop、car_urban_day_city_hall。
- M3ED val2: car_urban_day_rittenhouse、car_urban_day_ucity_small_loop。

根拠: [DSEC head/手順](/Users/at/project/research/EventState/docs/dsec_semantic.md:1)、[DSEC trainer](/Users/at/project/research/EventState/tools/train_dsec_semantic.py:59)、[M3ED trainer](/Users/at/project/research/EventState/tools/train_m3ed_semantic.py:41)、[DSEC split](/Users/at/project/research/EventState/tools/manifests/dsec_semantic_split.yaml:5)、[M3ED split](/Users/at/project/research/EventState/tools/manifests/m3ed_semantic_common_v1.yaml:1)。

### Semanticの先行研究との対応

- DINOv3のdense linear probingを参考に、凍結特徴の質を軽量readoutで比較する。DINOv3にはtrained BNやLR/WD sweepがあり、今回のheadと全条件一致ではない。[DINOv3 Appendix D.1](https://arxiv.org/html/2508.10104v1)
- GEP論文の軽量headはpatch展開linear＋浅いConv refinement、lossはCE＋Dice。公開configの既定headはUPerNet系、encoder_frozen=Falseであり、論文とも区別する。現在の1×1凍結CE probeをGEP同条件とは説明できない。[GEP公式Semantic config](https://github.com/uzh-rpg/generative_event_pretraining/blob/b07c78c8124ab60682d2aa440a2aa52911de9c36/src/config.py)、[GEP公式Semantic loss](https://github.com/uzh-rpg/generative_event_pretraining/blob/b07c78c8124ab60682d2aa440a2aa52911de9c36/src/seg.py)
- DSEC公式8train/3testと11クラスはbenchmarkに対応。開発6/2→全8再fitはtestによるepoch選択を避けるEventStateの手順であり、GEPと同一の根拠はない。
- M3ED train3/val2は共通Stage 1・Depthに揃えた設計。F³ Semanticはtrain4、20ms中央窓の高活動例のみを採点するので同条件ではない。[F³補足S.3.2](https://arxiv.org/html/2509.25146v2#S3.SS2)
- CE単独、50epoch、LR2e-4、WD0.05、batch8は独自の固定予算。先行研究の同一recipeや最適性の根拠はない。

## Depth

### 構造と学習

| 項目 | 現行設定 |
|---|---|
| encoder readout | DINOv3 block3/6/9/12の各384ch、計1536ch |
| E0 Encoder（z蒸留） | 4段ともz |
| E1 Encoder＋LSTM（h蒸留）とE2 Encoder＋LSTM（z+h蒸留） | 前3段はz、最終段だけh。4段LSTMを学習済みという意味ではない |
| decoder | DAv2 Metric ViT-S互換DPT。投影48/96/192/384ch、fusion64ch、Sigmoid×80m |
| 初期値 | depth_anything_v2_metric_vkitti_vits.pthのdepth_head部分だけをロード |
| 学習対象 | decoderのみ。Stage 1 encoder/LSTMは凍結 |
| loss | 有効GT上のmean abs(log(pred)−log(GT))であるlog-L1 |
| optimizer | AdamW、LR2e-4。WD等は明示せずlibrary default |
| 予算 | 50epoch、batch4、seed0、固定LR、schedulerなし、augmentationなし |
| 学習範囲 | 0.1<GT≤80m |
| head選択 | 公開valの全有効画素pool MAE最小 |
| SqRel | 評価のみ。lossにもbest選択にも使わない |

### Datasetと評価

| 項目 | DSEC Depth | M3ED Depth |
|---|---|---|
| event入力 | 左rectified GEP RGB、480×640全画角 | DAGR/rectify後、中央352×640 |
| 特徴grid | 30×40 | 22×40 |
| 窓 | RGB間隔の因果的入力 | 同じ |
| GT | 公式disparityを系列別Qでmetric depthへ変換 | native LiDAR depthをhalf-scale rectified座標へforward splat＋z-buffer、中央crop |
| GT時刻 | 公式GT時刻と入力時刻のexact対応 | RGB時刻へ最近傍LiDARを対応、既定許容差50ms |
| head用split | F³公開37train/4val | F³ metric-depth公開3train/2val |
| ローカル採点 | 4公開val | 2公開val |
| 最終test | 公式12系列、depthからdisparityへ変換してZIP提出 | 現行ローカル経路でhidden test採点なし |

DSEC Depth val4: interlaken_00_g、zurich_city_02_e、zurich_city_09_e、zurich_city_11_c。
M3ED train3/val2は上記Semanticと同じ。全train/test系列名は以下manifestが正本。
[DSEC Depth split](/Users/at/project/research/EventState/tools/manifests/dense/dsec_f3.json:1)、[M3ED Depth split](/Users/at/project/research/EventState/tools/manifests/dense/m3ed_f3.json:1)

- 旧metrics欄はMAE/RMSE/AbsRel/SqRel/RMSElog/delta1–3を有効画素poolで計算する。
- 新metrics_frame_mean_f3欄はF³公式9指標＋追加MAEを各frameで計算し、frameを等重み平均する。
- F³式SqRelの分母はGT²、従来欄はGT。RMSEはframe内で平方根を取ってからframe平均する。
- 分母GTの式はDAv2/Monodepth2でも使う通常のSqRelで、誤った旧式という意味ではない。
  分母GT²はF³公式コードがsq_relと呼ぶ別の量で、深度による重み付けと単位も異なる。
  通常SqRelとF³互換SqRelを明確に区別し、異なる定義同士の順位/数値比較は行わない。
  [DAv2公式指標](https://raw.githubusercontent.com/DepthAnything/Depth-Anything-V2/main/metric_depth/util/metric.py)、
  [Monodepth2公式指標](https://github.com/nianticlabs/monodepth2/blob/master/evaluate_depth.py)
- 全体、低活動/その他、低活動継続時間別、活動5分位を報告する。
- 追加欄は同じmask・評価対象を使い、有効GTゼロframeだけ平均の分母から除外する。
- median/scale/shift alignmentは行わない。
- DSECの公式最終主指標はdisparity MAE(px)。ローカルDepth MAE(m)と同じ指標ではない。
- runnerは各headをdepth val MAEで選び、その後DSECの条件間ではvalのsequence平均disparity MAE(px)で提出候補を選ぶ。全trainへの再fitはしない。

根拠: [readout](/Users/at/project/research/EventState/event_state/models/event_state.py:126)、[DPT](/Users/at/project/research/EventState/event_state/dense/dav2.py:31)、[lossとselection](/Users/at/project/research/EventState/tools/train_dense_probe.py:127)、[追加指標](/Users/at/project/research/EventState/event_state/dense/depth_frame_metrics.py:18)、[公式disparity benchmark](https://dsec.ifi.uzh.ch/uzh/disparity-benchmark/)。

### Depthの先行研究との対応

- block3/6/9/12、DINOv3、DAv2 DPT、VKITTI decoder初期化はScaleEvent公式コードに直接対応する部品。[ScaleEvent固定版decoder](https://github.com/zhiwen-xdu/ScaleEvent/blob/92f005b0f2cbb19dfc8391e3019ca042b1a9f587/scale_event/models/build_depther_dav2.py)
- VKITTI outdoor metric重みはDAv2公式配布に根拠がある。[DAv2 metric depth](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth)
- LSTMのhを最終readoutに使う部分はEventState独自拡張。
- log-L1はGEP、DAv2、F³と同一recipeではない。GEPはSiLog＋0.25 multi-scale gradient、DAv2はSiLog。今回のlog-L1、50epoch、LR2e-4、fixedLR、best pooled MAEは独自の簡易probe設定。[GEP Depth loss](https://github.com/uzh-rpg/generative_event_pretraining/blob/b07c78c8124ab60682d2aa440a2aa52911de9c36/src/dep.py)、[DAv2 loss](https://raw.githubusercontent.com/DepthAnything/Depth-Anything-V2/main/metric_depth/util/loss.py)
- F³と直接対応するのは公開metric-depth split、および追加欄の式・frame集計。F³はpseudo-depth→LiDAR metricの二段学習であり、今回の学習法とは別。
- F³ M3ED採点はnative GT時刻、中心20ms、20万イベント超、10点以上、80m未満、車載昼間5,387例。今回のDAGR/crop・RGB時刻nearest50ms・全低活動例と異なる。[F³補足S.3.4](https://arxiv.org/html/2509.25146v2#S3.SS4)
- M3EDの半解像度、crop、GT許容差50msは工学的設定。F³同条件と説明する根拠はない。

## Optical flow

現行の共通runnerはDSEC向け。実GPU学習・精度・提出は未受領。M3EDの共通Flow runnerとRAFT/SEA-RAFT headは未実装。

| 項目 | 現行DSEC Flow probe |
|---|---|
| 入力 | 左rectified GEP RGB480×640、RGB間隔、全時刻でstate更新 |
| backbone | Stage 1凍結。z/h/concat対応、既定h |
| 特徴 | from/toの両端、stride16、30×40。toより未来のeventは使わない |
| head | 1×1で64chへ投影、5×5局所correlation、a/b/b−aと連結→3×3CNN128→64→2ch |
| 出力 | 480×640へbilinear補間しstride16を反映、forward displacement(px) |
| loss | GT valid画素上のmean EPE |
| optimizer/予算 | AdamW LR2e-4、50epoch、batch4、seed0。fixedLR、augmentationなし、WDはlibrary default |
| split | project train14/val4、official hidden test7 |
| val4 | zurich_city_02_e、07_a、08_a、11_c |
| selection | val各系列のpixel平均EPEを求め、その4系列平均EPE最小 |
| 指標 | EPE(px)、1/2/3PE(%)、AE(deg) |
| test | val-bestで公式416時刻対をexport。全18trainでrefitしない |
| 活動5分位 | 現Flow runnerには未接続 |

公式由来はleft rectified座標、forward flow、GT valid画素、指標・test時刻・PNG encoding。
14/4 holdout、小型CNN、correlation幅、学習予算は独自probe設計で、GEP/F³/E-RAFT/ScaleEventの再現ではない。
ScaleEventのFlowはSEA-RAFTであり、この直接CNNとは別。
[Flow head](/Users/at/project/research/EventState/event_state/flow/probe.py:7)、[Flow runner](/Users/at/project/research/EventState/tools/run_common_dsec_flow.py:123)、[DSEC公式Flow](https://dsec.ifi.uzh.ch/uzh/dsec-flow-optical-flow-benchmark/)

## Detectionの過去実験

現在は低優先度。以下を現在のcommon Stage 1を使った新しい結果として扱わない。
過去にはFrozen、scratch、fine-tuning、temporal-only fine-tuning、event gap/dropout評価を実施している。
個別runでのoverrideや使用checkpointは各configと台帳で区別する。

| 項目 | 実装と既定 |
|---|---|
| task/dataset | DSEC-Detection、car/pedestrianの2クラス。M3ED Detection共通runnerなし |
| split | 公式41train/6val/13test |
| encoder入力 | 従来rectified448×640特徴を使用する経路、native dsec-det特徴の経路がある |
| benchmark座標 | dsec-det protocolではdistorted上430×640。内部rectified probeとは区別 |
| head | 独自YOLOX-style。decoupled head、pyramid、dynamic-k matching |
| loss | 5×(1−IoU²)＋objectness BCE＋classification BCE、foreground数で正規化 |
| frozen | encoder/LSTMのcacheを固定しheadのみ。既定50epoch、batch16、AdamW LR2e-4/WD0.05、cosine |
| end-to-end | finetune/scratch。既定50epoch、batch4、LR2e-4、WD0.05、水平反転0.5 |
| selection | val mAP最大 |
| 評価 | test mAP@[.50:.95]、AP50/AP75、class AP、COCO式で全予測/GTを集計 |
| 活動5分位 | DSEC用追加採点実装済み、実データ再採点未確認 |

YOLOX-style構造・損失はYOLOX設計、splitとframe/box filteringはDSEC-Det/DAGRを参考。
独自head、学習予算、画像からのfeature変換を含むため、YOLOX/DAGRそのものの完全再現ではない。
[Detection head](/Users/at/project/research/EventState/event_state/detection/yolox.py:293)、[Frozen trainer](/Users/at/project/research/EventState/tools/train_dsec_detection.py:35)、[公式split](https://github.com/uzh-rpg/dsec-det/blob/master/config/train_val_test_split.yaml)、[YOLOX公式](https://github.com/Megvii-BaseDetection/YOLOX)

## 全taskの追加分析と実験状態

- 活動5分位はユーザー指定。DSEC train35/M3ED train3のevent/sの20/40/60/80%境界を固定し、同datasetの条件間・Semantic/Depthで共有する。
- val/testの結果を見て境界を選び直さない。同値境界は統合し、評価各群のframe数が20%ずつになるとは限らない。
- この5分位自体をGEPやF³の標準protocolとは説明しない。Semantic/DepthとDSEC Detectionへ実装、Flowは未接続。
- 自車速度別は未実装。M3ED pose由来の速度生成経路はあるが、現在のtargets配備と評価接続は未確認。DSECに直接の公開自車速度GTは確認できていない。
- 過去のE3/E4 event dropout、activity α1/ScaleEvent型蒸留、固定20ms GEP20、旧M3ED4trainの結果は、現在の共通baselineとは別。
- ConvLSTM、4段LSTM、LSTM配置変更は次の比較候補。実装の存在を学習完了と混同しない。
- 今回追加したF³式frame平均もサーバーでの数値/GPU実行前。ローカル73件は軽量テストであり新しい精度結果ではない。

## ユーザー確認項目

- [ ] Stage 1をGEP完全再現ではなく、画像教師整合を参考にしたEventState独自再帰蒸留として記述する。
- [ ] cosine＋正規化L2²の重複と、E2二枝のloss尺度差を把握した上で採用するか確認する。
- [ ] 全token蒸留の現common比較と、活動mask付き過去実験を分ける。
- [ ] Semanticのfrozen 1×1＋CEを表現比較用probeとして採用するか、GEPのCE＋Dice等との比較が必要か確認する。
- [ ] Depthの構造/初期化はScaleEvent/DAv2由来、log-L1と固定予算は独自設定として扱う。
- [ ] Depth lossを先行研究へ揃える追加実験が必要か確認する。現在の学習結果を遡って別lossの結果とは記載しない。
- [ ] M3EDの公開valでhead選択と報告を行っていること、F³と入力/GT時刻/母集団が異なることを明記する。
- [ ] LR/epoch/batch等の妥当性は先行研究の引用だけで保証せず、収束曲線・感度・seed比較で確認する。
- [ ] サーバー保存config/plan/source snapshotとこの一覧を照合して実行条件を確定する。

実験結果・出力先の正本は[experiment_results.md](/Users/at/project/research/EventState/docs/experiment_results.md:1)。
現在の優先順位とサーバー状況は[session_handoff.md](/Users/at/project/research/EventState/docs/session_handoff.md:1)。
