# Semantic・Depthの共通イベント活動5分位評価

2026-11-10実装。現在のtask優先順位はSemantic segmentation・Depth、その次に可能ならFlow。
Detectionは低優先度。既存の学習済みheadを使い、活動分位別の追加評価を行う。

## 完了後の数値回収（GPU不要）

2026-11-15続報：以下の回収結果20件とmanifest2件を受領・保存し、台帳§85へ全体/群別の比較表を追記した。
同task/dataset内5条件のsample/活動割当/GT関連hashが一致し、全120群のsupportも整合。
Semanticは両datasetとも低活動群でE1のE0比改善が最大だが、M3ED Depth最低活動群はE0が最良。
M3ED Depthはpooled全体MAEの最良が新型DDP、F³式frame平均全体MAEの最良がE0へ反転する。
今後の比較でも両集計を残す。数値本体は`docs/evaluation_records/20261115_semantic_depth_activity_metrics.json`。
以下のコマンドは保存結果を再回収する場合の手順で、評価を再実行する必要はない。

2026-11-15、`outputs/semantic_depth_activity_20261110/summary.json`のSemantic10件・Depth10件の最終参照一覧を受領（台帳§84）。
一括評価の最終集計まで到達済み。次は保存済みJSONから群別の数値を回収する。`clear`で端末表示を消しても結果ファイルは残る。
以下は既存JSONを読み、共有manifestを重複除去して、全体＋実際の活動群の全指標・件数・hashをJSONLへ保存/表示するだけ。
DepthのF³式frame平均も保持し、巨大なframe単位配列は取り出さない。再学習・再評価・新しいコードのpullは不要。

```bash
cd /home/iASL/Arata_repo/EventState
source env/bin/activate
python - <<'PY' | tee outputs/semantic_depth_activity_20261110/metrics_for_review.jsonl
import json
from pathlib import Path

summary = json.loads(Path("outputs/semantic_depth_activity_20261110/summary.json").read_text())
seen = set()
def emit(value):
    print(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False))

emit({"type": "index", "counts": {task: len(summary[task]) for task in ("semantic", "depth")}})
for task in ("semantic", "depth"):
    for condition, filename in summary[task].items():
        activity = json.loads(Path(filename).read_text())["activity_evaluation"]
        manifest = activity.pop("activity_manifest")
        sha = manifest["manifest_sha256"]
        if sha not in seen:
            emit({"type": "manifest", "identity": manifest})
            seen.add(sha)
        emit({"type": "result", "task": task, "condition": condition,
              "report": filename, "manifest_sha256": sha, **activity})
PY
```

出力は`outputs/semantic_depth_activity_20261110/metrics_for_review.jsonl`にも保存される。
SemanticのmIoUとDepthのδはJSON本来の0–1表記。通常SqRel（誤差²/GT）とF³式frame平均SqRel（誤差²/GT²）は別欄を維持する。
分位境界が同値なら統合されるので、実際の群をそのまま回収する。評価splitの各群が20%ずつとは限らない。

## 先に全体スコアを回収

2026-11-15、common Depthの全10条件・最終summaryとDSEC提出用ZIPの生成完了をユーザーログで確認。
まずsig-gpu5で既存の全体スコアと提出条件を確認する。

```bash
cd /home/iASL/Arata_repo/EventState
cat outputs/stage2_common_depth_vkitti_20261110/summary.json
cat outputs/stage2_common_depth_vkitti_20261110/submission/manifest.json
```

DSECの`mae_px`は公開val視差MAEで、公式hidden testスコアではない。
ZIPは生成済みだが公式checker・実提出・公式採点は未確認。旧F³用shellのdsec-checkは出力配置が異なるため流用しない。

## Semantic・Depthの活動5分位をまとめて評価

全体スコアを保存した後、sig-gpu5で以下を実行する。既存の一括CLIで対応でき、head再学習は不要。
既定はDSEC/M3ED × Semantic/Depth × 基本3条件・E1旧新DDPの計20評価。
DSEC Semanticは既存final headによる公式test3系列、M3ED Semanticと両datasetのDepthは公開valを使う。

```bash
cd /home/iASL/Arata_repo/EventState
source env/bin/activate

# head・学習時config・cache・targetsの存在とDepth完了を確認。推論・出力作成は行わない。
python tools/run_semantic_depth_activity.py \
  --depth-run outputs/stage2_common_depth_vkitti_20261110 \
  --preflight

# 上の確認が通ったら、manifest生成 → Semantic → Depth追加評価をGPU 0で直列実行。
CUDA_VISIBLE_DEVICES=0 python -u tools/run_semantic_depth_activity.py \
  --depth-run outputs/stage2_common_depth_vkitti_20261110 \
  --output outputs/semantic_depth_activity_20261110
```

`--depth-run`には実際に学習したrun rootを指定する。基本3条件だけを学習した場合は、
確認・実行の両方に`--baselines-only`を追加する。
Semanticのcache・ラベルrootは既存headの`config.json`から復元する。
cacheを手動削除した場合は事前確認で止まるため、対応するStage 1重みからの再抽出が必要。
新しいhead学習は起動しない。Depthはval特徴を一時的に再抽出するため、時間・scratch空き容量が必要。

出力は`outputs/semantic_depth_activity_20261110/`の`manifests/`、`semantic/`と`summary.json`。
Depthの詳細JSON/CSVと`activity_summary.json`は元のDepth run配下に保存する。
同じコマンドを再実行すると、入力が一致する完了済み評価を再利用する。
一括summaryは詳細reportへの参照一覧で、数値比較には各JSON/CSVとDepthのactivity_summary.jsonを使う。
この一括評価には速度別集計は含まれない。活動5分位と自車速度の分析は別に扱う。
この一括入口のstdlibテスト7件を含め、Macの関連stdlibテスト63件が通過。
実データ/GPU実行の最終index・群別数値・hashは2026-11-15に受領・比較済み（台帳§84/85）。

## 共通の区分

- DSECは共通train35、M3EDは共通train3の全prepared RGB時刻から、それぞれ20/40/60/80%境界を固定する。
  同じdatasetのSemantic/Depth・全モデルで同じ活動manifestを使う。両datasetへ同じevent/s閾値を流用するものではない。
- 指標はイベント数/実RGB入力間隔秒数。DSECはrectified480×640全体、M3EDはDAGR downsampling・rectification後の
  360×640全体のcountを使う（モデル用中央352行cropの前）。活動画素率や生イベントの間引き率ではない。
- `p00_20`〜`p80_100`を評価し、同値境界は統合する。評価splitの各群が20%ずつになるとは限らない。
  先頭RGBには過去の入力区間がないため、従来通り評価対象に含めない。
- Semanticは群ごとに混同行列を合算してmIoU・クラス別IoUを算出する。
  Depthは有効GT画素の誤差和を集計し、MAE/RMSE/AbsRel/SqRel/RMSElog/δ1–δ3を算出する。
  この既存欄はframe別mIoUやRMSEの単純平均ではない。Depth再評価では下記のF³式frame平均も併記する。
  既存の全体・低活動/継続時間別指標も保持する。
- 各群のframe数・有効GT画素数・系列別件数・実活動率範囲をJSON/CSVへ保存する。
  空群・有効GTなし群の指標はnull。推論後の採点だけを区分し、系列の状態継承方針は変えない。

## Depthの計算式とframe平均（2026-11-10追加）

ユーザーの指摘を受け、Depthの再評価CLIは同一予測・同一GT maskから2通りの集計を出力する。
既存`metrics`と活動群内の`metrics`は全有効画素をpoolした値として保持する。
追加`metrics_frame_mean_f3`は、各frameでF³公式の式を計算し、有効GTのあるframeを等重みで平均した値。
全体・低活動/継続時間別はrootの`metrics_frame_mean_f3.<group>`、活動5分位は
`activity_evaluation.groups.<group>.metrics_frame_mean_f3`に保存する。
比較表でF³と計算・集計を揃える際はこちらの欄を使用し、旧集計は来歴を確認する参考値として残す。
SemanticのmIoUは従来通り群内の混同行列から算出する。

| 項目 | 既存の画素pool欄 | 追加のF³式frame平均欄 |
|---|---|---|
| frameの重み | 有効GT画素数に比例 | 有効GTのあるframeごとに同じ重み |
| RMSE | 全画素のMSEの平方根 | 各frameのRMSEを平均 |
| SqRel | 平均`(予測−GT)² / GT` | 各frameの平均`(予測−GT)² / GT²`を平均 |
| その他 | MAE・AbsRel・RMSElog・delta1–3 | MAE・AbsRel・RMSElog・d1–3・log10・silog |

数式の参照は[F³固定版eval_depth](https://github.com/grasp-lyrl/fast-feature-fields/blob/833e2c65233f9787449ce0675ae41e3f6e9c193a/src/f3/tasks/depth/utils/utils.py)、
集計は[同評価loop](https://github.com/grasp-lyrl/fast-feature-fields/blob/833e2c65233f9787449ce0675ae41e3f6e9c193a/src/f3/tasks/depth/evaluate.py)。
公式のSqRelは分母GT²、RMSElogは自然対数、silogは係数0.5・100倍なし、d1–3は割合0–1。
ここでの「公式」はF³公式実装を指す。分母GTの式はDAv2/Monodepth2で使われる通常のSqRelであり、
旧欄が誤りという意味ではない。分母GT²のF³互換値は別の量なので、同じSqRelとして数値を混ぜない。
通常SqRelとF³互換SqRelを区別して報告する。
MAEは公式関数の返却欄にない追加指標。固定revisionとsource SHAを`depth_frame_metric_protocol`に記録する。
内部計算はfloat64なので、公式float32とbit単位の一致を主張しない。

追加欄は計算・集計だけを揃え、RGB時刻・resize/crop・0.1<GT≤80m・低活動を含む評価集合を維持する。
有効GTゼロのframeは平均の分母から除外し、scored frame数と有効画素数を併記する。
F³の有効10点以上・中心20ms内20万イベント超の選別を追加していないため、F³論文と完全同条件のスコアではない。
学習中のlossやbest headの選択指標は既存のpooled val MAEを維持し、再学習は起動しない。

前述の一括コマンドで両方式を同時出力する。CSVの追加列は`f3_frame_mean_`で始まり、旧列と区別する。
旧`activity_continuous.json`に追加統計がない場合は上書きせず、同headの
`activity_continuous_frame_mean.json`/CSVへ再評価し、`activity_summary.json`が実reportを参照する。

## サーバーでのテスト

以下はsig-gpu5の既存env用。Macではstdlibと構文確認だけを行う。

```bash
cd /home/iASL/Arata_repo/EventState
source env/bin/activate
python -m pytest -q tests/test_activity_quantiles_stdlib.py \
  tests/test_m3ed_activity_quantiles_stdlib.py tests/test_activity_reporting_stdlib.py \
  tests/test_semantic_activity_stdlib.py tests/test_dense_activity_stdlib.py \
  tests/test_common_depth_runner.py \
  tests/test_semantic_depth_activity_runner_stdlib.py \
  tests/test_depth_frame_metrics_stdlib.py \
  tests/test_semantic_activity_server.py tests/test_dense_activity_server.py \
  tests/test_depth_frame_metrics_server.py
```

## Datasetごとにmanifestを一度生成

```bash
python tools/prepare_dsec_activity_quantiles.py \
  --event-cache-dir /home/iASL/Arata_repo/dataset/DSEC_cache/events/gep_rgb \
  --output outputs/dsec_activity_quantiles_20261110.json

python tools/prepare_m3ed_activity_quantiles.py \
  --prepared-root /home/iASL/Arata_repo/dataset/m3ed_cache/half_dagr \
  --sequences car_urban_day_rittenhouse car_urban_day_ucity_small_loop \
  --output outputs/m3ed_activity_quantiles_20261110.json
```

`--sequences`は追加する系列であり、各datasetの参照train系列は常に含む。
既存出力は上書きしない。以前作ったDSEC manifestのv1はそのまま使える。M3EDは専用contractのv2。
metadataと時刻・countを読み、重いevent tensorの展開や新しい教師推論は行わない。
初版は既存GEP RGBの完全なRGB間隔のみ。固定窓や短縮窓の混用は拒否する。

## DSEC Semanticの既存headを再評価

`--checkpoint`、`--feature-cache-dir`、`--labels-root`は対応する既存head・cache・Semanticラベルrootを指定する。
E0 Encoder（z蒸留）は`--feature z`、E1 Encoder＋LSTM（h蒸留）とE2 Encoder＋LSTM（z+h蒸留）は`--feature h`。

```bash
CUDA_VISIBLE_DEVICES=0 python tools/evaluate_dsec_semantic.py \
  --checkpoint outputs/stage2_dsec_semantic_linear_20261109/E1/final/last.pt \
  --feature-cache-dir /path/to/matching/dsec_semantic_features \
  --labels-root /path/to/dsec_semantic_labels \
  --feature h --role test \
  --activity-manifest outputs/dsec_activity_quantiles_20261110.json \
  --condition-label 'E1 Encoder＋LSTM（h蒸留）・単GPU' \
  --output outputs/semantic_activity/E1_dsec_test.json
```

headの上記パスは例であり、実際の配置は未確認。開発valなら対応する開発headと`--role val`を使う。
既存全体スコアだけから層別値は復元できないため、headを再推論する。

## M3ED Semanticの既存headを再評価

再学習を伴わないstandalone評価入口を追加した。既定は共通3train/2公開valのmanifest。
`--target-cache-dir`にはRittenhouse/UCITYのSemantics付きtargetsが揃うroot、または既存runのtargets viewを指定する。

```bash
CUDA_VISIBLE_DEVICES=0 python tools/evaluate_m3ed_semantic.py \
  --checkpoint outputs/stage2_m3ed_semantic_linear_20261109/E1/h/best.pt \
  --feature-cache-dir /path/to/matching/m3ed_semantic_features \
  --target-cache-dir /path/to/semantic_targets_view \
  --prepared-root /home/iASL/Arata_repo/dataset/m3ed_cache/half_dagr \
  --feature h \
  --activity-manifest outputs/m3ed_activity_quantiles_20261110.json \
  --condition-label 'E1 Encoder＋LSTM（h蒸留）・単GPU' \
  --output outputs/semantic_activity/E1_m3ed_val.json
```

既存common cacheの`identity.json`にある元prepared metadata hash、または新cacheに埋めたmetadataを照合する。
元入力の来歴がない古いgeneric cacheは受け付けない。必要なら既存Stage 1重みからcacheを再抽出する。
新規`run_common_m3ed_semantic.py`にも同じ`--activity-manifest`/`--condition-label`を渡せる。
従来の`continuous_groups.json`と、新しい`activity_groups.json`/CSVを出力する。

## Depth完了後、同じrunのheadだけを追加評価

現在の学習が完了してから、元のrun rootを`--output`へ指定する。

```bash
CUDA_VISIBLE_DEVICES=0 python tools/run_common_depth.py \
  --output outputs/stage2_common_depth_vkitti_20261110 \
  --dsec-activity-manifest outputs/dsec_activity_quantiles_20261110.json \
  --m3ed-activity-manifest outputs/m3ed_activity_quantiles_20261110.json \
  --activity-only
```

単GPU基本3条件だけなら`--baselines-only`を追加する。片方のdatasetだけを追加評価する場合は、そのmanifestだけ渡す。
この経路は完了済み`best.pt`と`_SUCCESS`を要求し、val特徴だけを一時的に再抽出する。
元の`continuous.json`/`summary.json`を保持し、各headの`activity_continuous.json`/CSVと
run rootの`activity_summary.json`へ保存する。元runのlockを共有するため、同じrunの学習中には並行実行しない。

既存val特徴cacheが残っている場合は直接評価も可能：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/evaluate_dense_probe.py \
  --checkpoint /path/to/completed/depth_head/best.pt \
  --cache /path/to/matching/val_feature_cache \
  --activity-manifest outputs/dsec_activity_quantiles_20261110.json \
  --condition-label 'E1 Encoder＋LSTM（h蒸留）・単GPU' \
  --output outputs/depth_activity/E1_dsec_val.json
```

同じ入口でDepth/Disparityの公開GTを採点できる。GT非公開testのPNG exportとの併用はしない。
公式testの全体スコアと公開valの活動別指標は評価先を区別する。

## 出力の読み方と検証範囲

群別は`activity_evaluation.groups.<群名>.metrics`。SemanticのmIoUは0–1、Depthは従来のm単位。
活動manifest identityは同一dataset内の全taskで共有するが、GT時刻や有効画素はtask別に維持する。
条件間比較では各task内の入力/GT・活動割当digestを確認する。

Macのstdlib・構文確認のみ。実データのmanifest生成・head再推論・GPU採点はサーバー未検証。
新しい精度結果や学習完了を示す記録ではない。

## 5分位を絶対イベント数でも確認する（2026-11-16）

Q1〜Q5の20%刻みはtrainの**イベント率の順位**であり、イベント数そのものの割合ではない。
既存境界・評価スコアを維持し、各群の実イベント数・RGB区間長・events/sを併記する。
`event_rate_hz = event_count × 1,000,000 / window_us`なので、区間長が違えば同じ件数でもrateは違う。
50ms相当の件数はrate×0.05で換算できるが、実際の区間長を確認せず実測件数として扱わない。

受領済みtrain境界（百万events/s）：

| 境界 | DSEC | M3ED |
| --- | ---: | ---: |
| p20 | 5.933583 | 1.549875 |
| p40 | 9.054970 | 3.296850 |
| p60 | 12.461568 | 5.084100 |
| p80 | 19.043645 | 8.072375 |

DSECは480×640のrectification後、M3EDは360×640のDAGR 2×2縮小＋rectification後に数える。
M3EDの中央352行crop前の値であり、両者を同条件の生センサーイベント数として比較しない。
画像の着色量・非ゼロ画素数からカウントを推定しない。

以下はサーバーの既存manifestだけを読む。GPU・再学習・head再評価・特徴cache生成は不要。
ユーザーが今回の変更をpushした後に実行する。

```bash
cd /home/iASL/Arata_repo/EventState
source env/bin/activate
git pull --ff-only

for dataset in dsec m3ed; do
  for task in semantic depth; do
    python tools/summarize_activity_counts.py \
      --activity-manifest "outputs/semantic_depth_activity_20261110/manifests/${dataset}.json" \
      --dataset "$dataset" --task "$task" \
      --output "outputs/activity_counts_20261116/${dataset}_${task}.json" || break 2
  done
done
```

既存出力は上書きしない。再取得時は新しい出力先を指定する。
train・task評価系列別に、全体／既存群／系列の件数・区間長・rateの分布、時間加重rate、群の時間占有率、
Q1/Q5の連続時間をJSONへ保存する。連続時間は系列境界・欠落区間・250ms超の区間を跨がない。
長い区間のcountは全体分布には残し、連続時間の計算からだけ除く。

この集計の母集団は**指定系列の全prepared入力区間**。GT有効の評価frameだけではないので、
既報mIoU/Depthのsupport件数と同一とは限らない。新しい精度スコアや共通の絶対量binによる再評価ではない。
「多い状態が続く／激しく変わる」は分位点だけで断定せず、系列別の低・高活動の連続秒数と実動画を確認する。
動画にも実イベント数と区間長を表示し、`frames.csv`に`event_count`・`window_us`を追加する。
