# 下流5タスクの分割・評価先（2026-11-07）
Stage 1の共通pool（DSEC 35 / M3ED 3）はhead用splitではない。

2026-11-10ユーザー確定のtask優先順位：Semantic segmentation・Depthを最優先とし、
次に可能ならFlow。Detectionは過去の評価経路を保持するが現在は低優先度。

| タスク | 開発・実測 | 論文での最終比較先／進捗 |
|---|---|---|
| DSEC Semantic | 公式train8内を独自6/2にして選択（val=07_a/08_a） | 全8でhead再学習し公式test3のmIoU。GEPのvalidはtest。過去のtest値あり（台帳§15）。 |
| DSEC Depth | F³公開37/4、既報は4valの距離(m) | DSEC公式disparity test12・1Hz・MAE(px)。変換/ZIP実装、公式スコア未確認。 |
| DSEC Flow | 公式GT train18。共通Stage 1から除外済みの02_e/07_a/08_a/11_cを開発val候補にするなら14/4（独自案） | 公式Flow test7、左rectified forward、指定時刻対（約100ms）・開始時刻2Hz、主指標EPE。head/cache/train/export実装済み、実データ未検証・未提出。 |
| M3ED Semantic | 旧実験は4train（city_hall/horse/penno_big_loop/penno_small_loop）、val=ucity_small_loop、別置きtest=rittenhouse | F³準拠にするにはhorseをucity_big_loop（grayscale InternImage疑似ラベル）に変え、2系列を一緒に評価。学習は1組おき、20ms中央窓>200Kの評価13,501件。既報は非準拠。共通3train・2valの新経路は実装済み（下記）。 |
| M3ED Depth | F³車載昼間3train/2val。既報26,611例は独自採点 | F³公式ローダ・metric採点（論文5,387件）への別経路を用意。実データ一致・スコアは未確認。 |

参照: [GEP設定](https://github.com/uzh-rpg/generative_event_pretraining/blob/b07c78c8124ab60682d2aa440a2aa52911de9c36/src/config.py)、[DSEC disparity](https://dsec.ifi.uzh.ch/uzh/disparity-benchmark/)、[DSEC Flow規則](https://dsec.ifi.uzh.ch/uzh/dsec-flow-optical-flow-benchmark/)、[Flow提出形式](https://dsec.ifi.uzh.ch/optical-flow-submission-format/)、[F³補足・Semantic](https://arxiv.org/pdf/2509.25146v2#page=35)、[F³補足・Depth](https://arxiv.org/pdf/2509.25146v2#page=39)。

2026-11-10：Depth再評価にはF³公式式・frame等重み平均の`metrics_frame_mean_f3`を追加。
既存pooled欄と区別して計算/集計を揃えた比較に使用する。特にF³ SqRelは分母GT²で既存欄と異なる。
元のGT mask/評価集合・入力/cropを維持するため、これだけでF³論文と同条件とはならない。
学習中のbest選択は既存pooled val MAEを維持。詳細[semantic_depth_activity.md](semantic_depth_activity.md)。
両DSEC benchmarkで全test系列に同じ設定を使い、testを事前学習・調整に使わない。Flowのみ論文の最終ablationとして複数提出可。

## 全評価に共通する活動・速度別分析（2026-11-10方針）

ユーザー指定：イベント活動・自車速度による性能差を全ての評価で調べる。
Depthに限定せず、Semantic・Depth・実施するDetection/Flowと全モデル比較に適用する。
E0 Encoder（z蒸留）、E1 Encoder＋LSTM（h蒸留）、E2 Encoder＋LSTM（z+h蒸留）、旧新DDP、
後続のPatchwise LSTM/ConvLSTM・配置比較で、同じ区分規則と集計方法を使う。
この方針は任意taskの新規学習を要求せず、Flowの保留も維持する。以下は評価要件であり、全経路の実装済みを意味しない。

- 全体指標を残した上で、イベント活動別・速度別を共通の分析軸とし、十分な件数があれば交差集計も加える。
  DSEC/M3EDのSemantic・DepthとDSEC Detectionはtrain由来5分位を実装。速度の数値境界は未確定。
  区分はtrain由来統計または事前に定めた物理閾値で固定し、test性能を見て選ばない。
  同じdataset・入力仕様ではtaskをまたいで同じ規則を使い、各taskのtrain/val/testの分割は保つ。
- フレーム全体のイベント率（event_count/実入力窓の秒数）と局所的なactive/inactiveは別に定義する。
  生イベント率、非白画素率、ScaleEvent活動マスク率を混同しない。
  既存の低活動継続時間別評価も保持する。物体保持の分析では局所活動・物体可視性の確認を補助分析として検討する。
- 速度は自車/カメラの並進速度として定義し、画面内の他車速度と区別する。
  GT/poseの出典・時刻対応・有効性・停止時ノイズを確認する。速度欠損をゼロやイベント量で埋めない。
  M3EDにはposeから速度を生成する前処理があるが、現在のサーバーへの配備状況と評価接続は未確認。
- 各task内では全モデルに同一の評価時刻・GT有効領域・区分を使い、系列全体を連続推論してから採点だけを分ける。
  Semanticは区分ごとの混同行列からmIoU、Detectionは区分内の予測/GTからAPを再計算する。
  Depth/Disparity/Flowは各protocolで定めた画素・系列の集約方法を保持し、frame平均への置き換えをしない。
  全体の集約スコアのみから層別スコアを復元できるとは扱わない。
- 各区分に絶対スコアと条件間差分、frame数・系列数・有効画素/物体数、速度情報のcoverageを併記する。
  task間でGT時刻やsplitが異なる場合はその違いを保つ。少数の系列・区分への偏りにも注意する。
- GT非公開の公式testは、評価サービスが区分採点に対応するか未確認。
  公開GT上の層別分析と公式test全体スコアを併記し、区分採点が得られない場合は理由を明記する。
  未採点や速度欠損を性能値として報告しない。

### Semantic・Depthへの共通活動5分位の実装（2026-11-10）

主対象のSemantic・Depthに、同一datasetで共有する活動manifestを接続した。
DSECは共通train35、M3EDは共通train3から境界を固定する。M3EDはDAGR/rectify後360×640全体のcountを使用する。
Semanticは混同行列、Depthは有効画素誤差和を群ごとに合算する。全体・既存の低活動/継続時間別指標も保持する。
M3EDの既存head再評価入口と、完了済みDepth runの`--activity-only`を追加。
詳細は [semantic_depth_activity.md](semantic_depth_activity.md)。Macはstdlib/構文のみ、実GPU採点は未検証。

### DSEC Detectionへの活動分位別mAPの実装（2026-11-10、現在は低優先度）

ユーザーは活動%を「イベント量の分位（下位20%・20〜40%…の5群）」と明示。
`tools/prepare_dsec_activity_quantiles.py`で共通train35のevents/s分布から境界を固定し、
同manifestをFrozen/End-to-endのDetection評価に渡す経路を実装した。
frame全体を分類して各群の全予測/GTからCOCO APを再計算し、全体指標と群別JSON/CSVを保存する。
全体活動はnative rectified480×640のcount/実RGB間隔。固定20ms/短縮窓の混用は拒否。
境界の同値は統合し、空群/GTなし群はnull。継続推論は維持する。
手順・制約は [dsec_detection_activity.md](dsec_detection_activity.md)。実データ・COCO採点はサーバー未検証。
同じmanifestをSemantic/Depthへ接続済み。Flowへの活動別評価はまだ未接続。

### Datasetごとの自車速度情報（2026-11-10公式配布確認）

- **M3ED**：公式の公開GT付き系列には `*_pose_gt.h5` と時刻があり、FasterLIO由来の左イベントカメラposeから
  相対並進量/時間差として速度を算出できる。CAN等の直接車速測定とは区別する。
  EventStateは `tools/prepare_m3ed_downstream.py` に `linear_velocity_mps` を生成する実装があるが、
  現在のサーバーtargetsに保存済みかは未確認。公式challenge test用ファイルはpose GTを配布しない。
  出典：[公式ファイル仕様](https://m3ed.io/data_overview/folderst/)。
- **DSEC**：RTK GPSを収録したことと公開データに含まれることは異なる。公式著者は
  [Issue #36](https://github.com/uzh-rpg/DSEC/issues/36)でGPS非公開、
  [Issue #7](https://github.com/uzh-rpg/DSEC/issues/7)でcamera軌跡GTを提供しないと説明。
  [現行配布一覧](https://dsec.ifi.uzh.ch/dsec-datasets/download/)にも速度/pose/GPSの配布経路は確認できない。
  LiDAR/IMUのraw rosbag（lidar_imu.zip、26GB）は配布されており、LIO等で速度を追加推定する道はあるが、
  これは公開車速GTではない。raw bagはtask系列へ切り出されていないため時刻対応も必要。
  bag全体のtopic実検査はしておらず、速度topicの絶対的な不在まで確認したわけではない。

速度別分析はM3EDの公開poseを使う経路と、DSECでの別途推定の要否を分けて設計する。
両datasetで同じ品質の自車速度GTが直ちに得られる前提にしない。

### 区分方法の先行例と設計案（2026-11-10調査）

確認範囲では全task・全datasetに共通するevent/sやkm/hの標準境界は得られなかった。
下記の数値は各研究固有であり、そのままEventStateの区分閾値として採用しない。

| 一次資料 | 実際の区別と用途 |
|---|---|
| [TAF / Motion Robust High-Speed Light-Weighted Object Detection with Event Camera, TIM 2023](https://arxiv.org/html/2208.11602#S5.SS1.SSS3) §V-A3、Tables II/III | 50ms間隔の推定flowの大きさをGT bbox内平均したBBOFDを、20%分位ごとの5 motion levelsに分けてmAP評価。自車m/sではなく画像上の相対運動。論文は各datasetの全注釈から分位点を定義し、train限定とは記載していない。 |
| [TEDNet / Tracking-Assisted Object Detection with Event Cameras, NEVI Workshop 2024](https://arxiv.org/html/2403.18330) §3.1/§4.1、補足Tables 7/8 | 50ms窓のbbox内イベント占有画素率と正規化中心変位でstill/movingを分類し、各群のGT/TP数を比較。閾値は0.1/0.03。GT清掃・学習用visibilityラベル追加も伴うため、評価専用binの標準ではない。 |
| [How Many Events Make an Object?, CVPRW 2023](https://tub-rip.github.io/eventvision2023/papers/2023CVPRW_How_Many_Events_Make_an_Object.pdf) §3.1、Fig.2 | bbox内イベント数・面積当たりイベント数と、100イベント未満が継続する時間を解析。閾値を変えたラベルfilterとbbox memoryも評価するが、全taskの共通低/中/高区分ではない。 |
| [HyperE2VID, TIP 2024](https://arxiv.org/html/2305.06382v2) 補足§I、Table II | HQFのdesk_slow/slow_hand、2,333 GT frameを自然低速のSLOW subsetとして固定し再構成指標を集計。数値速度閾値による区分ではない。 |
| [Haensel & Bertram, Sensors 2026](https://doi.org/10.3390/s26092595) §3.2/§4、Tables 1/2 | PWMヘッドライトの夜間制御実験で自車30/50km/h条件のAP@0.5を比較。自車の物理速度を用いる直接例だが、30/50は一般的な速度binの境界ではない。 |
| [F³ v2](https://arxiv.org/html/2509.25146v2) 補足§S.3.2/§S.3.4 | M3ED Semantic/Depthはラベル時刻中心20ms窓の200K超イベント例のみ評価。これは低活動の除外であり、低/高別の採点ではない。Fig.12(e)のランダム間引きも自然低活動とは別。 |

調査時点の設計案（後にDSEC/M3EDの活動5分位を実装、速度境界は未採用）：

1. 全task共通の広域分析では、フレーム全体のevent_count/実入力窓秒数と、自車並進速度の大きさを別々に測る。
   空間範囲・イベントcountの定義・窓・単位を記録する。異なる窓のevent/sが同じなら同じ入力情報量になるとは仮定しない。
2. 最初の区分案は各datasetの共通Stage 1 train poolから求めた20/40/60/80%点による5群。
   TAFを参考にするのは分位点で層別する考え方であり、活動率/自車速度への適用とtrainだけでの境界固定は本研究の設計。
   同一dataset・入力仕様の全task/モデルに同じ境界を使う。val/testの各群が20%になるとは限らない。
   同値が多く境界が重なる場合は群を統合し、人工的に同じ速度値を別群へ割り当てない。
3. 最低速度群を直ちに「停止」と呼ばず、各群の実測km/h範囲と件数を併記する。
   停止を別群にするならpose/GPSのノイズと時刻精度を確認して停止閾値を別途固定する。
4. 記憶保持の分析は全体活動とは別に、局所的なactive/inactiveと低活動へ移行してからの継続時間を調べる。
   TEDNet/Kugeleが近い先例。ただし物体trackが必要な解析と、Semantic/Depthの画素・patch別解析は区別する。
   既存p10低活動と100/500ms継続時間別評価は保持し、これらの境界を先行研究標準とは呼ばない。
5. 自然な低活動例を保持した全例分析を行い、F³互換の高活動subsetや人工間引きは別分析として扱う。
   群別性能を見て境界を選び直さず、低活動/低速の件数・系列・class構成と速度有効率を先に確認する。

調査によるプロトコル候補の整理であり、新しい学習・速度抽出・層別評価は実行していない。

## 共通Stage 2の新経路（2026-11-07）

表の旧実験は保存。新経路 `tools/run_common_stage2.sh` は既定でjoined E1 w3
の最終step_00033333.ptを読む。DSEC35/M3ED3 common manifestと正規化の来歴を検査し、
以前の41系列/4系列やholdoutを含む重みは拒否する。Stage 1は再学習しない。

**M3ED Semantic（主実験）**: [追加manifest](../tools/manifests/m3ed_semantic_common_v1.yaml)の3 train/2公開val、
旧manifest不変。11クラスのpseudo GT。全因果GEP入力でLSTMを更新、系列境界のみreset、
ラベルのある時刻だけ特徴を保存する。各系列について入力時刻とGT時刻の完全一致を確認。
線形head z/h/concat別、seed0/50epoch、公開valでbest mIoU選択。continuous_groups.jsonに全体/系列別、
train-only event-rate p10以下/それ以外/500ms以上の継続低活動のmIoUと画素・frame数を保存。
F³ Semanticと同一train数・20ms高活動フィルタではなく、因果的な全ラベル評価。

```bash
cd /home/iASL/Arata_repo/EventState
source env/bin/activate
# 必要なら、別に用意したsemantics付きtargetsの親を指定
# export EXTRA_TARGET_ROOT=/path/to/additional/semantic_targets
bash tools/run_common_stage2.sh semantic-check
CUDA_VISIBLE_DEVICES=0 bash tools/run_common_stage2.sh semantic
```

Rittenhouseがdepth専用targetsならそのまま使えない。rawがあるホストで、既存targetを上書きせず
`python tools/prepare_m3ed_downstream.py --root RAW_M3ED --prepared-root PREPARED --output-root NEW_TARGETS --sequences car_urban_day_rittenhouse --tasks semantics`。
その結果を学習サーバーのEXTRA_TARGET_ROOTに置く。UCITY等も同様に必要に応じて準備する。

**DSEC Flow（最初のfrozen probe）**: 14train/4公開val、公式7test。入力はRGB時刻ごとの左rectified GEP。
過去方向の蓄積窓は全系列共通。モデルの特徴はfrom/toの両時刻で取り、toを超える入力を使わない。
目標はfrom画素でのtoへの順方向変位。64ch投影、5×5局所相関と両端特徴・差分、convで2成分を予測。
これは本repoのprobeであり既存Flow専用モデルの再現ではない。学習は有効GT上のEPE、
公開valのsequence平均EPEでbest選択。history.jsonlに各系列EPE・1/2/3PE・AE（度）。
全18系列で再fitせず、14trainで学習したval-bestを提出する。チェックポイントに窓/特徴種/split/backboneを固定。
学習キャッシュはユニークな端点のみFP16で保存。testは全7系列をstreamし、巨大なtest cacheを作らない。

```bash
# FLOW_GT_ROOT/<SEQ>/.../forward_timestamps.txt とその隣の forward/*.png が必要
# export FLOW_GT_ROOT=/path/to/unpacked/train_optical_flow
python -m pytest -q tests/test_common_flow_contract.py tests/test_common_flow_server.py
bash tools/run_common_stage2.sh flow-check
CUDA_VISIBLE_DEVICES=1 bash tools/run_common_stage2.sh flow-cache &&
CUDA_VISIBLE_DEVICES=1 bash tools/run_common_stage2.sh flow-train
# 公開valで設定を固定した後に一度だけ:
CUDA_VISIBLE_DEVICES=1 bash tools/run_common_stage2.sh flow-export
```

既定FEATURE=h、DSEC_WINDOW_MS=rgb-interval、seed0、batch4、50epoch。別の公開val候補はFEATURE=z/concat、
DSEC_WINDOW_MS=20等。cache/outは窓と特徴ごとに別。Stage 1条件を変えるときは
DSEC_CHECKPOINT/M3ED_CHECKPOINTだけでなくOUTPUT_ROOT/CACHE_ROOTも別名にする。
flow-exportは[公式CSV ZIP](https://download.ifi.uzh.ch/rpg/DSEC/test_forward_optical_flow_timestamps.zip)のhashを照合し、
7系列416時刻対・RGB16（R=u, G=v, B=1）を検査対象のZIPに保存する。
提出前に[DSEC公式checker](https://dsec.ifi.uzh.ch/optical-flow-submission-format/)で検査すること。
両タスクとも実GPU学習・新しい精度・提出は未確認。空き容量の見積りは各cache単独なので、
同じファイルシステムで並行生成する場合は見積りの合計で判断する。
