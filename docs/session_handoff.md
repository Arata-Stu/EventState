# EventState セッション引き継ぎメモ

最終更新: 2026-09-27

新しい会話セッションは、最初にこの文書と `docs/experiment_results.md` を読む。
数値の正本は `experiment_results.md` であり、この文書は研究状況を素早く復元するための要約である。

## 1. 研究の中心

イベントカメラの現在特徴 `z` と、LSTMの時系列状態 `h` にRGB基盤モデルの知識を蒸留し、
物体検出・Semantic Segmentationなどへ転用可能な汎用表現を学習する。

現在の中心仮説は次のとおり。

- `z` は現在イベントが活発な領域を表現する。
- `h` は過去の観測を保持し、現在イベントが少ない領域を補完する。
- Random clipだけでなくStream/TBPTTを混ぜるHybrid学習で状態継承を学ばせる。
- ただし現結果ではHybridの改善は小さく、長期記憶が獲得された証拠はまだ弱い。
- 次の主実験はactivity-awareな `z/h` 複合蒸留である。

ScaleEvent（CVPR 2026）は高activity領域だけをRGB教師へ蒸留する。EventStateではこれを拡張し、
`z`をactive領域、`h`をinactive領域へ蒸留する。ただし一度も観測されていない領域への
無理な蒸留を避けるため、将来的には「現在inactiveかつ過去Kフレームでactive」の
history-aware maskを検討する。

詳細仕様は `docs/scale_event_distillation.md` を参照する。

## 2. 学習サーバーの固定パス

```text
repository:
/home/iASL/Arata_repo/EventState

DSEC root:
/home/iASL/Arata_repo/dataset/DSEC

DSEC GEP-RGB event cache:
/home/iASL/Arata_repo/dataset/DSEC_cache/events/gep_rgb

DSEC DINOv3 teacher cache:
/home/iASL/Arata_repo/dataset/DSEC_cache/dinov3_vits16

DINOv3 ViT-S/16 checkpoint:
/home/iASL/Arata_repo/models/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth

M3ED prepared event cache:
/home/iASL/Arata_repo/dataset/m3ed_cache/half_dagr

M3ED DINOv3 teacher cache:
/home/iASL/Arata_repo/dataset/m3ed_cache/dinov3_vits16_640x352

M3ED downstream targets:
/home/iASL/Arata_repo/dataset/m3ed_cache/m3ed_downstream
```

GPUはTesla V100 32 GBが3台で、通常 `CUDA_VISIBLE_DEVICES=0,1,2` を使う。

## 3. 事前学習条件の呼称

| 条件 | 蒸留 | 時系列 | dropout | sampling |
|---|---|---|---|---|
| E0 | `z` | なし | なし | Random |
| E1 | `h` | LSTM | なし | Random |
| E2 | `z+h` | LSTM | なし | Random |
| E3 | `h` | LSTM | あり | Random |
| E4 | `z+h` | LSTM | あり | Random |
| Hybrid | `h` | LSTM | 条件による | Random + Stream |

Randomはclipごとに状態を初期化してclip内BPTT、Streamは状態をclip間で継承し境界でdetachするTBPTT。
Hybridは両方のmicro-batchを同じoptimizer stepで学習する。

## 4. DSEC分割と漏洩方針

事前学習は必ず `dataset=dsec_det_train41` を用いる。DSEC-Det公式train 41系列のみで
100,000 stepのfinal fitを行い、事前学習validationは無効。公式Detection val/testと
Semantic testを事前学習へ入れない。

- Detection: train 41 / val 6 / test 13。
- Semantic開発: 公式train 8系列をtrain 6 / val 2に分けてepoch選択。
- Semantic最終: 8系列でheadを再学習し、公式test 3系列で評価。
- Semantic開発valの `zurich_city_07_a`, `08_a` はラベルなし事前学習には含まれるが、
  公式trainに属する。公式testは未使用。

正確な系列名は次を参照する。

- `tools/manifests/dsec_det_official_split.yaml`
- `tools/manifests/dsec_semantic_split.yaml`

## 5. 主要結果の要約

詳細値と出力パスは `docs/experiment_results.md` を正本とする。

### DSEC Detection（Frozen backbone）

- E1 `h` Random: mAP `0.38693`
- Hybrid `h` Random+Stream: mAP `0.39209`
- Hybridは対応するE1より `+0.00516`。
- Scratch no-memory: `0.25065`、Scratch LSTM: `0.26290`。
- 全層Fine-tuningはFrozenを上回らなかった。

### DSEC Semantic（Frozen）

- E1 Linear+CE: mIoU `0.60383`
- Hybrid Linear+CE: mIoU `0.60843`
- Hybrid GEP patch+CE+Dice: mIoU `0.61462`
- 改善の中心はDiceであり、GEP patch単体では改善しなかった。

### M3ED Semantic validation（Frozen Hybrid h）

- Random no-dropout, Linear+CE: `0.36889`
- Hybrid no-dropout, Linear+CE: `0.37388`
- Hybrid dropout, Linear+CE: `0.37611`
- Hybrid dropout, Linear+CE+Dice: **`0.38392`**
- Hybrid dropout, GEP patch+CE: `0.37552`
- Hybrid dropout, GEP patch+CE+Dice: `0.38280`

M3EDの最高値はLinear+CE+Diceの38.392%。異なる18クラス／RGB-event融合などの
公開手法とはプロトコルが異なるので直接SOTA比較しない。E2VIDの39.40%は参考値に留める。

## 6. 現在の実装・直近の作業

activity-aware蒸留と下流activity weightingを実装済み。作業ツリーにはユーザー／別セッションの
未コミット変更があるため、上書き・reset・checkoutをしない。

主比較はGPU 0/1/2で次の3条件を走らせる。共通の学習設定を使い、3番目は損失形式も変える追加比較である。

1. 従来E2相当の `z+h` 全領域蒸留
2. `activity_dual`: activeを`z`、inactiveを`h`へ従来cosine+MSEで蒸留
3. `scale_event_dual`: 同じ領域分担にScaleEvent型構造損失を追加

100 step smokeはユーザーから完了報告あり。保存先は
`/home/iASL/Arata_repo/EventState/outputs/dsec_activity_smoke_20260922_231607`。
続いてユーザーが `--stage full --skip-tests` で100,000 stepの3条件を起動した。
2026-09-24にユーザー提供ログで3条件とも100,000 step到達と
`checkpoints/step_00100000.pt`（各277M）の存在を確認。重みのロード検証は未実施。
最終stepのloss・勾配は有限、optimizer skipは0（全期間の集計ではない）。
次は従来の分割・下流損失でFrozen評価を行う。新しい重みで下流特徴を再抽出する。
ただし先にサーバーストレージを整理する。2026-09-24のユーザー提供容量一覧では
`/home`は98%使用・空き77G。DSEC下流特徴はDetection 671G、Semantic 53G、
M3ED Semantic特徴は65G。旧特徴キャッシュの内訳・再生成元を確認中で削除は未実施。
DSEC GEP RGB入力270Gとデータ本体は次の特徴抽出でも必要。
M3ED関連は別PCでの再生成が必要なため、全キャッシュを保持し削除対象から除外する。
旧DSEC Detection/Semantic特徴の削除をユーザーが実施予定（削除完了・空き容量は未確認）。
下流はまず3条件のh特徴で従来E2と同じFrozen Detectionを比較する。
キャッシュは新しい専用ディレクトリへ作成し、Semanticとz/concat評価は別段階で進める。
その後ユーザーから3条件のFrozen Detection hの完了とtest結果を受領。
seed 0のmAPはbaseline_e2=0.37778238、activity_only=0.37342058、
scale_event_full=0.37427887。下流activity重みなし。詳細は実験台帳§14。
出力は `outputs/dsec_detection_activity_20260923_h/<条件名>/seed_0/test_metrics.json`。
Semantic h（Frozen Linear+CE）も3条件完了し、test JSONを受領。
mIoUはbaseline_e2=0.60053234、activity_only=0.58973430、scale_event_full=0.58832850。
全条件790,169,600 pixels、seed 0、開発6/2→全8系列で再学習→test 3系列。
出力は `outputs/dsec_semantic_activity_20260923_h/<条件名>/seed_0/test_metrics.json`。
詳細は実験台帳§15。次は予定済みのz/concat比較。
次の起動手順はSemanticの3条件×z/concat（計6 head）。GPUごとに1条件を担当し、
zとhを一度に新規 `semantic_features/activity_20260923_zh/<条件名>` へ保存する。
同じcacheを使ってz→concatを順次学習する。既存runnerは要求featuresの完全一致でcacheを
検査するため、この共用手順ではcache生成とhead学習を個別CLIで実行する。
出力予定は `outputs/dsec_semantic_activity_20260923_zh/<条件名>/<z|concat>/seed_0`。
Frozen Linear+CE、seed 0、batch 8、開発50 epoch、6/2→8再学習→test 3を維持。
2026-09-25に全3条件のz/concat完了表示と6実験のtest JSONを受領（台帳§16）。
z/concatのmIoUはbaseline=0.59591928/0.59959334、
activity_only=0.59787214/0.59868795、scale_event_full=0.59579199/0.59930571。
activity zはbaseline z比+0.195 pointだが、concatでは両条件ともbaseline未満。
単一seedであり、activity特有の相補性・低活動領域改善は未確認。
Semanticの予定したh/z/concat比較は完了。Detection z/concatと低活動領域評価は未実施。
事前学習3条件の`--suite h-relaxation`は、ユーザー提示ログでsmoke・fullとも全条件のcomplete表示を確認。
smoke出力は `outputs/dsec_activity_h_relaxation_smoke_20260925_103516`、
full出力は `outputs/dsec_activity_h_relaxation_full_20260925_182735`。
ユーザー提示ログで全条件100,000 stepと最終checkpointの存在を確認（262M/277M/277M）。
最終lossはz-only=0.10806、h-all=0.20348、h-soft=0.20148。ロード検証は未実施。
2026-09-26の空き容量は268G（/home使用率92%）。次はSemanticのtrain/valだけ特徴抽出し、
Frozen Linear+CE・seed0・50 epoch・batch8で開発6/2評価する。z-onlyはzのみ、他2条件はh/z/concat。
新しいcache予定先は `DSEC_cache/semantic_features/h_relaxation_20260925_182735/<条件名>`、
出力予定先は `outputs/dsec_semantic_h_relaxation_20260925_182735/<条件名>/<feature>/seed_0`。
2026-09-27、上記Semanticの全3条件のcomplete表示を受領。提示済み手順では計7 headの
開発学習とbest checkpointのvalidation評価まで完了したことを示す。
7件のval JSONを受領（全633,036,800 pixels）。mIoUはz-only/z=0.59365503、
h-allのz/h/concat=0.59764789/0.58961109/0.60040464、
h-softのz/h/concat=0.59887872/0.59001120/0.60152263。best epochは未受領。
softの上回り幅は小さく単一seed。次は既存baseline_e2/activity_onlyの同じval結果を確認し、
旧hard maskからの回復を比較する。既存test値とは直接比較しない。
この追加比較での公式train8再学習・test評価はまだ実施していない。
active_z_only（h蒸留なし）、active_z_h_all（h全領域）、active_z_h_soft（h inactive=1/active=0.5）。
全条件cosine+二乗L2、従来と同じtrain41・100k・batch8・clip8・seed0。
LSTM構成は揃えるがz-onlyのtemporal/h projectorは未学習なので下流はzのみ評価する。
既存hard maskはalpha=0のまま維持し、新条件は下流validationで比較する。
ローカルの構文・dry-run検査済み。サーバーpreflightの個別テスト結果は未受領。本番は--skip-testsで実行。
concatの入力次元増加を考慮し、同じfeature同士で事前学習条件を比較する。
単一seedのFrozen hでは検出・Semanticとも改善は確認できていない。
本番の出力先は `outputs/dsec_activity_full_20260923_000937`、
各条件のコンソールログはその配下の `logs/<条件名>.log`。
サーバーでは `source env/bin/activate` を使い、依存関係はuvで管理する。
起動ツールは
`tools/run_dsec_activity_comparison.sh`。正確な仕様と検証条件は
`docs/scale_event_distillation.md`を読む。

## 7. activity-aware実験の解釈上の注意

### 2026-09-27の継続方針

最新のユーザー意図は次の2本柱へ具体化された（以下の過去の提案より優先）。

1. 信号待ちシーンがあるM3EDで学習・評価する。
2. 低アクティビティシーンを抽出したデータセットで評価する。

狙いは停止前に観測した情報をhが停止中に保持し、発進後に更新できるかの検証。
M3EDの実際の対象系列・停止区間・利用可能ラベル・既存splitとの対応はこれから確認する。
既存のtrain/val/test分割を先に固定し、評価区間を学習へ混ぜない。低活動抽出は評価subsetとして扱い、
低活動区間だけで事前学習する指示とは解釈しない。低活動評価の対象をDSEC/M3ED双方にするかは未確定。
信号待ち/自車停止という場面ラベルと、イベント活動量の数値によるsubsetは区別する。
区間選定はモデル結果を見ずに行い、停止前・停止中・発進後を記録する。
continuous評価では停止前から状態を継承し、resetとの対照を用意する。区間の独立サンプル化で
履歴を失わないmanifest方式を検討。M3EDキャッシュは引き続き保持、学習起動・抽出実装は未実施。

ユーザーは比較の余地がある間はDSECを継続し、その後M3EDへ移る可能性を示した。
M3EDへの移行・新しい学習はまだ未起動。以下は次の検証の提案順序。

ユーザーが検証1を実施する方針を承認。baseline_e2/activity_only × h/z/concatの6評価用
サーバーコマンドを提示する。hは `dsec_semantic_activity_20260923_h/<条件>/seed_0/development/best.pt`、
z/concatは `dsec_semantic_activity_20260923_zh/<条件>/<feature>/seed_0/development/best.pt` を使用。
対応するactivity_20260923_h/zhキャッシュを再利用し、GPU0=h、1=z、2=concatで各2条件を順次評価。
保存先は `outputs/dsec_semantic_activity_validation_20260927/<条件>/<feature>/validation_metrics.json`。
全6評価の完了・JSONを受領（台帳§19）。baselineのh/z/concatは
0.58854159/0.59643383/0.59942271、hardは0.58259270/0.59838694/0.60128082。
hの制約緩和でh単独は約+0.70〜0.74 point回復するが、concatのsoft−hardは+0.02418 point。
次はbaseline/hard/softのconcatでhead seed1,2を追加し、既存seed0と比較する提案。未起動。
ユーザーの訂正により、下流headはseed1を3 GPUで実行し全条件完了後にseed2を3 GPUで実行する。
GPU0=baseline_e2、GPU1=activity_only、GPU2=active_z_h_soft。各GPUは常に1ジョブ。
Frozen concat Linear+CE、開発6/2、50 epoch、batch8、FP16を維持。既存特徴cacheを再利用。
出力予定は `outputs/dsec_semantic_concat_seeds_20260927/<条件>/seed_<1|2>`。
seed0や事前学習を再実行せず、testも評価しない。
ユーザー提示ログでseed1→seed2の各3条件、全6ジョブのcomplete表示を確認（台帳§20）。
出力ルートは `outputs/dsec_semantic_concat_seeds_20260927`。
6件のvalidation JSONを受領、best epoch番号は未受領。3 head seedのconcat mIoU平均±標本SDは
baseline=59.86771±0.06484%、hard=60.09768±0.02698%、soft=60.09124±0.05400%。
hard/softは全head seedでbaselineを上回るが、soft−hard平均は−0.00643 pointでsoft優位は未確認。
次は同じ二値マスクでactive/inactive別のval評価を行う提案。事前学習seedは固定であり効果の再現性全体は未検証。

20ch事前学習の参考実装としてユーザーが `/Users/at/project/competition/JetPilot` を指定。
読み取り調査のみ実施。ROS2の `jetpilot_e2e_inference/src/event_tensor_cuda_backend.cu` に
CUDA rolling ringのイベント集計があり、Python側 `e2e_learning/data/event_tensor.py` にCPU参照実装がある。
既定は10時間bin×正負2極性、40ms窓・4ms stride、212×120、polarity_major（正10→負10）、
時間補間なし、float32。CUDAは新規イベントをatomicAddでリングへ集計しsnapshotで出力・正規化する。
窓は[start,end)、snapshotはイベント起点のbin境界整列が必要。DSECの既存窓・教師時刻との整合は未検証。
ユーザーの明確化: リングはオンライン用であり学習前処理への移植は不要。
揃える対象は20ch生成アルゴリズムとモデル入力値（dtype、正規化、極性順、bin境界、座標処理）。
JetPilotのrosbag_extractor.pyではFP32集計後FP16保存、dataset.pyではFP32へ戻して
(tensor-mean)/stdを適用。CUDA側はFP32集計から同式を適用し、mean/std既定は0/1だが設定可能。
保存時FP16丸めと推論時FP32集計の差、実際のmean/std設定の一致は未検証。
移植時は同一イベント・窓・幾何変換・正規化係数を入力し、集計後と正規化後をそれぞれ比較する。
リング実装の共通化や全量キャッシュ方針の決定は目的ではない。移植・学習は未着手。
その後のユーザー依頼で、DSEC voxelに `channel_layout=polarity_major` と
前処理 `--event-cache-dtype float16` を追加。既定time_major/FP32と旧cache互換性は維持。
学習側のrepresentation生成へlayoutを渡し、manifestの順序・保存dtypeを検査してFP32へ読み込む。
JetPilotとの時間補間・窓境界・正規化の完全互換化と20ch学習起動は未実施。
Macでは構文・CLI引数のstdlib検査のみ通過。追加Torchテストは未実行。
ユーザー依頼により、20chの固定正規化経路も追加。
`compute_dsec_event_normalization.py` は公式train41の未正規化voxel cacheのみから、
ゼロ画素を含むチャネル別mean/stdをFP64逐次集計してJSON保存する。
`dataset=dsec_det_train41_voxel20` と `dataset.representation.fixed_normalization_file` で適用。
train.pyで係数と仕様をconfigへ埋め込んでからcheckpoint/logを作成し、学習・評価で固定使用。
JetPilotへは同じ20個の係数を渡す。実際の統計生成・20ch学習は未実施。
stdlibの契約検証3件を実行して通過。Torch/OmegaConfの数値・移植テストは追加済みだが未実行。

1. 既存baseline_e2/activity_only（必要ならscale_event_fullも）の開発bestと既存cacheを使い、
   同一Semantic valでh/z/concatを評価。train8で再学習したfinal headをdev val評価に使わない。
2. baselineと有力候補を絞り、固定backboneでhead seedを0/1/2へ増やす。
   これはhead学習のばらつき検証であり、pretrain seedの再現性検証とは区別する。
3. 同じ二値活動マスクでactive/inactive別のval指標と画素数・クラス別supportを測る。
   baselineと候補の両方でcontinuous/reset評価を揃え、時系列状態の寄与を調べる。
   resetやgapを使う場合は特徴を再計算し、既存continuous特徴の置換で済ませない。
4. 対応baselineと候補をDSEC Detection valでも比較。新規特徴cache容量を確認し条件を絞る。
5. 有力差があれば事前学習seedも追加して確認し、条件固定後にSemantic train8再学習→test、
   Detection val選択→testを実施。改善がなければその結果を記録してDSECを一区切りにする。

下流activity weightingの追加比較は上記の事前学習比較が整理できた後の別実験とする。
細かな活動量定義は引き続き今後の展望。M3EDへ進む場合は既存splitとcache互換性を確認し、
baseline＋候補に絞る。M3EDキャッシュは別PCでの再生成が必要なため保持する。

- 「event activityでlossを変える」一般概念自体は新規ではない。
- EventDAM（ICCV 2025）とScaleEvent（CVPR 2026）は高activity領域を重視する。
- EventStateの差分は、時系列状態`h`を低activity領域へ割り当てる点。
- 現在inactiveでも過去にactiveだった領域へ限定するhistory-aware maskはまだ今後の候補。
- active/inactive率、各枝のloss、低activity区間の下流性能を必ず記録する。
- 通常mAP/mIoUだけでは長期記憶を証明できない。continuous/reset、gap長、低activity subset、
  静止物体の評価が必要。

## 8. 新しい結果を受け取ったとき

次の情報を `docs/experiment_results.md` に追記する。

1. 条件名と変更点
2. dataset/split
3. seed
4. checkpoint stepまたはbest epoch
5. mAP/mIoUなどの正式な評価値
6. `test_metrics.json`または`validation_metrics.json`の出力パス
7. smoke、途中値、単一seedである場合の注記

会話だけに数値を残さない。

## 9. DSEC論文用の単一フレームPNG出力

`tools/export_dsec_detection_frame.py`で、DSEC-Detの1フレームからRGB、GEP Event、
PCA特徴、Detection結果、manifestを個別PNG/JSONとして出力できる。

ポスター素材として使用した既知のフレーム:

```text
sequence: zurich_city_14_b
timestamp: 55314607586
visualizer frame: 414 / 576
model: Hybrid no-augmentation cached-teacher
feature: h
```

サーバー上での再生成例:

```bash
cd ~/Arata_repo/EventState
source env/bin/activate

DET_DIR="outputs/dsec_detection_frozen_hybrid_noaug_cache/HYBRID/seed_0"
FEATURE_CACHE="/home/iASL/Arata_repo/dataset/DSEC_cache/detection_features/hybrid_noaug_cache_step100000/dsec_det/HYBRID"
EXPORT_DIR="$DET_DIR/publication/zurich_city_14_b_timestamp_55314607586"

CUDA_VISIBLE_DEVICES=0 python tools/export_dsec_detection_frame.py \
  --source "HYBRID:$DET_DIR/best.pt:$FEATURE_CACHE:h" \
  --event-cache-dir /home/iASL/Arata_repo/dataset/DSEC_cache/events/gep_rgb \
  --labels-root /home/iASL/Arata_repo/dataset/DSEC/dsec_det_labels \
  --dataset-root /home/iASL/Arata_repo/dataset/DSEC \
  --role test \
  --sequence zurich_city_14_b \
  --timestamp 55314607586 \
  --output-dir "$EXPORT_DIR" \
  --device cuda \
  --precision fp16 \
  --score-threshold 0.25 \
  --detection-background rgb
```

現在のスクリプトは既定でrectification paddingを全画像共通の有効領域へcropする。
paddingを意図的に残す場合だけ `--keep-padding` を付ける。PCAは選択フレームだけでfitせず、
対象シーケンス全体からfitする。Macへコピーした素材の既知の保存先は次である。

```text
/Users/at/Library/Mobile Documents/com~apple~CloudDocs/プレゼン/プレゼン/中間発表/自分/素材/
```

### M3ED activity事前学習の起動対応（2026-09-27、未実行）

ユーザーの次の希望はM3EDで損失を変えた事前学習と、信号待ち／低activity区間の評価。
確認時にはactivityがDSEC限定でM3ED loaderがmaskを返していなかったため、今回対応を追加した。
`tools/run_m3ed_activity_comparison.sh` でGPU順にbaseline_e2 / activity_only（hard）/
active_z_h_soft（alpha=0.5）を起動する。既存のh-only M3ED checkpointの流用ではなく、
3条件とも同じDINO初期値からz+hを学習する。Random、clip16、batch4、seed0、GEP RGB、
augmentation/dropoutなし。既存train4/validation1固定、validation有効。full100k、smoke100step。
20ch入力への変更は混ぜない。下流比較では同じ最終step checkpointを使う。

M3ED loaderはDSECと同じPairedSequenceTransformで正規化前GEPからevent_activityを生成する。
prepared/teacher cacheを読み取り専用で使い、raw不要。dataset.rootにもprepared_rootを渡せる
（M3ED loaderはraw rootを参照しない）。大きな新規入力cacheは作らないがcheckpointは増える。
正規化は既存prepared_root/event_statistics.jsonを既定で読み、train4だけの統計であることを
検証する。欠落・別系列を含む場合は起動しない。実サーバーの統計ファイルは未確認なので、
エラー時は由来を確認し、train4から計算した統計を指定する。過去runの漏洩を断定しない。

Mac検証: AST4ファイル、shell構文、stdlibによるsmoke/full dry-runと不正統計拒否が通過。
Torch/Hydraテストおよび実データ学習は未実行。サーバー同期後、まずsmokeのpreflightを実行。
