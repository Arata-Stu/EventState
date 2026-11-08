# 下流5タスクの分割・評価先（2026-11-07）
Stage 1の共通pool（DSEC 35 / M3ED 3）はhead用splitではない。

| タスク | 開発・実測 | 論文での最終比較先／進捗 |
|---|---|---|
| DSEC Semantic | 公式train8内を独自6/2にして選択（val=07_a/08_a） | 全8でhead再学習し公式test3のmIoU。GEPのvalidはtest。過去のtest値あり（台帳§15）。 |
| DSEC Depth | F³公開37/4、既報は4valの距離(m) | DSEC公式disparity test12・1Hz・MAE(px)。変換/ZIP実装、公式スコア未確認。 |
| DSEC Flow | 公式GT train18。共通Stage 1から除外済みの02_e/07_a/08_a/11_cを開発val候補にするなら14/4（独自案） | 公式Flow test7、左rectified forward、指定時刻対（約100ms）・開始時刻2Hz、主指標EPE。head/cache/train/export実装済み、実データ未検証・未提出。 |
| M3ED Semantic | 旧実験は4train（city_hall/horse/penno_big_loop/penno_small_loop）、val=ucity_small_loop、別置きtest=rittenhouse | F³準拠にするにはhorseをucity_big_loop（grayscale InternImage疑似ラベル）に変え、2系列を一緒に評価。学習は1組おき、20ms中央窓>200Kの評価13,501件。既報は非準拠。共通3train・2valの新経路は実装済み（下記）。 |
| M3ED Depth | F³車載昼間3train/2val。既報26,611例は独自採点 | F³公式ローダ・metric採点（論文5,387件）への別経路を用意。実データ一致・スコアは未確認。 |

参照: [GEP設定](https://github.com/uzh-rpg/generative_event_pretraining/blob/b07c78c8124ab60682d2aa440a2aa52911de9c36/src/config.py)、[DSEC disparity](https://dsec.ifi.uzh.ch/uzh/disparity-benchmark/)、[DSEC Flow規則](https://dsec.ifi.uzh.ch/uzh/dsec-flow-optical-flow-benchmark/)、[Flow提出形式](https://dsec.ifi.uzh.ch/optical-flow-submission-format/)、[F³補足・Semantic](https://arxiv.org/pdf/2509.25146v2#page=35)、[F³補足・Depth](https://arxiv.org/pdf/2509.25146v2#page=39)。
両DSEC benchmarkで全test系列に同じ設定を使い、testを事前学習・調整に使わない。Flowのみ論文の最終ablationとして複数提出可。

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
