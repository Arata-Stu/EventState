# EventState セッション引き継ぎメモ

最終更新: 2026-10-01

新しい会話セッションは、最初にこの文書と `docs/experiment_results.md` を読む。
数値の正本は `experiment_results.md` であり、この文書は研究状況を素早く復元するための要約である。

## 最新状況（2026-10-01）

Hybrid蒸留対象4条件のfull起動後、ランチャーが子プロセスの非ゼロ終了を検出して停止した報告あり。
出力: `/home/iASL/Arata_repo/EventState/outputs/hybrid_targets_full_20261001_230705`。
`--stage full --skip-tests`、DSEC z-only/z+h・M3ED z-onlyの起動表示まで確認。
提示tracebackは最終RuntimeError本文がなく、失敗条件・終了コード・学習側原因は未確認。
ランチャーは1条件失敗時に他の実行中プロセスも終了させる設計。全4条件完了とは扱わない。
次は同出力のlogs/*.log末尾を取得し原因を特定する。修正・再実行・下流起動は未実施。

### 今後の研究方針（ユーザー指定）

ユーザーが事前学習後の下流一括shと、成功したモデルの新規下流cache自動削除を明示承認。
`tools/run_hybrid_target_downstream.sh`（stdlib Python本体あり）を追加。
必須: --pretrain-root <hybrid_targets_full実出力> --output-root <新規結果> --cache-root <新規専用scratch>。
標準は今回4モデル、DSEC Semantic/Detection・M3ED Semanticの計12 headを順次実行。
既存比較モデルも含める場合 --dsec-h-checkpoint と --m3ed-zh-checkpoint を追加し計21 head。
既存h-onlyの設定整合は別途確認。既存M3ED z+hの結果は取得済みで、再実行は必須ではない。
デフォルトGPU0、--gpuで変更。容量優先のため同時に1モデルのみ。z-onlyはz、他はz/h共用→z/h/concat。
Frozen、seed0、50epoch、FP16、Semantic Linear+CE/batch8、Detection dsec-det/batch16。
DSECはtrain/valのみ（Semantic6/2、Detection41/6）、M3ED train4/validation1。testは実行しない。
Detectionはrectified→公式座標へ変換し評価。モデル成功後に専用cache全体を削除。
所有marker・run identity・symlink拒否で削除範囲を固定し、全headのbest/last ZIP CRCと
有限なvalidation指標を確認してから削除する。完全なTorchロード保証ではない。
結果/ログ/重みはoutput-rootへ保持。M3ED入力・教師・ラベルや既存特徴cacheは削除対象外。
--resumeは同一設定・入力checkpoint hashを検査して完了モデル/工程をskipし、headはlast.ptから再開。
既存trainerの再開方式を使うため、中断なし実行との乱数列の完全一致は保証しない。
失敗時はそのモデルのcacheを保持。DSEC1モデル分のrectified/公式座標特徴等のピーク容量は必要。
Macではstdlib4テスト（CLI計画、削除保護、結果検証、失敗→再開の模擬処理）とshell構文を検証。
実データ・GPU実行は未実施。事前学習完了後に新規専用パスを指定して起動する。

ユーザーが4条件のaugmentation/dropoutなしを承認し、実行準備を依頼。
`tools/run_hybrid_target_ablation.py` を追加（既定smoke、--stage full、--dry-run、--skip-tests）。
GPU0=DSEC z-only、GPU1=DSEC z+h、GPU2=M3ED z-only→M3ED h-only（同GPU順次）。
Hybrid、seed0、全域cosine+二乗L2、DSEC Random4+Stream4/clip8、M3ED2+2/clip16、FP16。
100k step、warmup1000。smokeは100step/warmup10で、fullへ重みを継続しない。
DSEC train41・val無効、M3ED train4/val1・train4統計を検証してコピー、val周期1000。
モデル構成は全てLSTMで揃え、新しいz_distill_lstm presetはh損失無効。z-onlyのhは下流不使用。
全領域のz+hは既存h_distill_lstm_zloss、h-onlyはh_distill_lstm presetを使う。
出力は `outputs/hybrid_targets_<smoke|full>_<日時>/<dsec_z_only|dsec_z_h|m3ed_z_only|m3ed_h_only>`。
ログはルートlogs、起動コマンド全体はlaunch.json。保存は1万stepごと、最終step_00100000.ptを下流へ使う。
既存出力上書き拒否。起動前20GiB空きを要求。下流特徴の容量はこの予算に含めない。
事前学習のみを起動するランチャーであり、下流抽出は自動起動しない。完了後に最終重みと容量を確認し、
DSEC Detection/Semantic、M3ED Semanticの不足probeを実施する。
Macではstdlibテスト2件（両stageの計画/不正値拒否・模擬子プロセスでGPU順次実行/上書き拒否）と
AST検査が通過。Hydra/Torchの設定・勾配テストを追加し、サーバーsmoke前にpreflightで実行する。
実学習・ML依存テストは未実行。サーバーでは変更ファイル一式を同期しsmoke→fullの順に実施。

ユーザーは不足4条件＋下流タスクを進める案を提示：DSEC Hybrid z-only/z+h、
M3ED Hybrid z-only/h-only。基準batch（DSEC8、M3ED4）、全域蒸留、事前学習augmentation/dropoutなしで
蒸留対象を比較する方針案。DSEC既存Hybrid h-onlyは設定照合後再利用、M3ED既存Hybrid baseline_e2も再利用。
新規4条件の起動は未実施。z-onlyもsamplingを合わせるが状態保持を学ぶ条件ではない。
下流はDSEC Detection/Semantic、M3ED SemanticでFrozen probe、z-only→z、他→z/h/concat。
ユーザーからaugmentationの効果について質問あり。台帳§7ではDSEC augの下流完了報告はあるが
mAP未転記のため効果未判定。サーバーのoutputs/dsec_detection_frozen_hybrid_aug_online配下の
test_metrics.jsonを取得して比較する。既知の事前学習loss悪化のみでaug無効とは判断しない。
event dropoutの小幅改善と空間augmentationは別要因として扱う。

今後の時系列モデル事前学習はHybridを標準とする。Random-onlyの新規比較は原則追加せず、
既存結果をablationとして保持する。Hybrid内のRandom枝は維持し、pure Streamへの変更ではない。
系列境界のreset、clip境界のdetachも維持する。これは研究方針であり既存再現用configの既定値は未変更。
ユーザーは初期DSEC-Det実験のHybridでの再検証と、3 GPUを1学習に使う大batch検証を検討している。
過去値はE0 z=37.852%、E1 h=38.693%、E2 h=36.869%、Hybrid h-only=39.209% mAP。
従ってh-only Hybridは既評価だが、z+h/活動分担の結論をHybridへ一般化する検証は残っている。
提案順序：まず従来batch/clip/splitでHybrid h-only・全域z+h・activity候補を比較、
次に有力条件を固定してeffective batchを比較する。全て未起動。
現在のtrain.py/training経路にはDDP・distributed sampler実装なし。3 GPUの既存運用は独立3実験。
gradient_accumulationは実装済みだが、Streamでは連続clipを同じ更新に蓄積するため、
独立系列を増やすDDP batch拡大と同一ではない。DDPには系列をrankごとに分けるsampler、
状態のrank内保持、全rankの更新同期・保存/評価制御が必要。
batch拡大とHybrid導入を同時に変えず、総観測frame数とoptimizer更新数を併記する。
既存testを繰り返し設定選択に使わず、Detection valを用いる。低活動continuous/reset評価も残す。

### 論文ablationの計画（2026-10-01、未起動）

ユーザーはDSEC/M3EDを基盤として、(1)global batch 1倍/3倍と1 GPU/3 GPU、
(2)Random/Hybrid、(3)蒸留対象z/h/z+h、(4)タスク、(5)Scratch/Fine-tuning/Frozenを整理したい。
特にh-only蒸留後のzと、z+h蒸留後のz/h/concatを調べる意向。
Hybridを主設定、Randomはsampling ablationとし、条件の揃った既存結果を優先利用する。
Randomの劣位・Frozenの優位・M3EDでHybrid効果が大きいことは予想であり、全設定への確定事項ではない。

中心となる表は事前学習蒸留対象×下流入力。z-only→z、h-only→z/h/concat、
z+h→z/h/concatの計7 headをdataset/taskごとに比較する。
z-onlyの未学習temporalをh/concatの主比較に入れない。h-onlyでもh lossからencoderへ勾配が流れ、
raw zは学習される。z projectorは未学習なのでその出力をz probeとして使わない。
concatは次元/パラメータ数が増えるので、同じ入力同士で比較し容量差にも注意する。
全域蒸留の対象比較と、activity hard/softの領域分担比較は別の軸として扱う。

追加実験の優先順位案:
1. 既存h-only Random/Hybridのcheckpoint・正規化・splitを照合し、まずz probe、必要ならconcatを追加。
   DSECではh-only hとHybrid hが既評価。M3EDの旧h-onlyは新train4統計と異なり得るため、
   最新z+h結果との厳密比較には設定照合後に必要なh-onlyだけ再学習する。
2. Hybrid・基準batchで蒸留対象×probeの不足欄を埋める。DSEC Detection/Semantic、M3ED Semanticを主対象。
   同じpretrain checkpointを複数タスクへ転用し、下流FTしたbackboneを別タスクへ混ぜない。
3. 代表条件でglobal batchを比較。従来基準はDSEC8/M3ED4で、3倍候補は24/12（下流batch8と区別）。
   GPU数とglobal batchを同時に変えた比較は複合効果。DDP動作確認は可能な範囲で同一global batchも確認。
   同じ100k stepでは3倍の入力を消費するので、総frame予算を揃える比較と最大性能探索を分ける。
   学習率・schedule・総更新数・総frame数・Random/Stream比・系列数・GPU時間を記録する。
   M3ED train4では1系列1streamの現samplerでstream batch6を構成できない。
   系列重複、区間分割、勾配累積はそれぞれ独立性/履歴/更新間隔を変えるので単純な3倍DDPとは呼ばない。
   この設計とDDP実装が確定するまでは3 GPUの本学習を起動しない。
4. 有力条件のhead seed追加、低活動subsetと同じheadのcontinuous/reset対照を実施する。

Hybrid効果は同条件の絶対差（point）を主とし、相対改善率100*(Hybrid-Random)/Randomも併記。
dataset間では共通Semantic・同じhead/lossで比較するが、ラベル品質/分布等も違うため、
差を停止頻度だけに帰属しない。停止区間や活動率別の検証が必要。
Scratch/FT/FrozenはDSEC Detectionで既存証拠あり、追加優先度は低い。
最終HybridやM3EDへの一般化を主張する場合のみ代表条件で追加確認する。
過去スコアは条件付きで有効。大batchで改善なしでも普遍的な最高値とはしない。

### M3ED下流結果の詳細

ユーザーの --resume-cache 実行ログでRandom3条件→Hybrid3条件の全completeを確認。
Frozen Semantic h/z/concat計18 headが完了した報告（詳細は台帳§22）。
出力: `outputs/m3ed_semantic_activity_random_hybrid_20260930/<random|hybrid>/<条件>/<h|z|concat>/seed_0/validation_metrics.json`。
seed0・Linear+CE・50 epoch・batch8・FP16・continuous・既存train4/validation1。
18件のJSONを受領し台帳§23へ記録。全件3,218,555,646評価pixels、11クラスIoU平均も整合。
mIoU（%、h/z/concat）はRandom baseline=34.62334/35.46450/35.22061、
Random hard=35.84324/36.23875/36.30305、Random soft=35.29993/35.97274/35.95024、
Hybrid baseline=34.97230/35.66049/35.51019、Hybrid hard=35.94215/36.54401/36.50717、
Hybrid soft=36.08211/36.99275/36.78864。
Hybridは同条件Randomを全9比較で上回る。最高はHybrid soft z=36.99275%、
同じHybrid baseline z比+1.33227 point。単一pretrain/head seed・単一val系列で有意差未検証。
hは全モデルでz未満、concatのz超えはRandom hardのみ。長期記憶改善はまだ証明されない。
best epochはJSONの0始まりを+1で解釈。hは6件中5件がepoch50、concatは5件がepoch5。
次の提案は低活動subset・同じheadによるcontinuous/reset評価、必要に応じhead seed追加。
resetは特徴再計算が必要。追加評価は未起動。
下記の停止・削除・再開待ちは過去の経緯。削除実績・現在空き容量は未確認。

### 2026-09-30の容量整理・再開経緯

ユーザーの整理dry-runで指定4組・計12条件の中間checkpoint 1,080個、290.75 GiB（logical size）が
削除候補と確認できた。各条件90個、通常24.28 GiB、z-only22.99 GiB、Hybrid各24.50 GiB。
まだdry-runで削除未確認。次は同じ4組へ--apply→dfで回収確認→容量に応じ下流--resume-cache。

ユーザーが過去cache/checkpointの取捨選択・削除を許可。まず完了済み事前学習の中間重みを整理する。
`tools/prune_intermediate_checkpoints.py` を追加（既定dry-run、--applyで削除）。
対象提案はdsec_activity_full_20260923_000937、dsec_activity_h_relaxation_full_20260925_182735、
m3ed_activity_full_20260927_192110、m3ed_activity_hybrid_full_20260929_131030の4組。
各checkpoints内に100k最終重みのZIP構造・data.pklがある場合のみ、100k未満のstep重みを
1万step間隔へ間引く。10k刻み・最終・最新・best・非stepファイル・symlinkを保持する。
ZIP確認はTorchロード保証ではない。ログ/指標/下流head/cacheには触れない。
削除候補と結果をoutputs/checkpoint_prune_<timestamp>.jsonlへ保存。
一時fixtureでdry-run不変、削除対象、未完了run/cache/best保護とauditをstdlib検証済み。
サーバー削除・回収容量は未確認。容量確認後に--resume-cacheで下流抽出を再開する。

追加ログで容量枯渇を確認：特徴のtorch.save中にiostream error、/home空き831M（100%）、
inode使用1%。新規Random特徴cacheは各23G、合計68G。削除・移動は未実施。
次は別filesystemの空きと事前学習checkpoint群の容量を確認し、容量確保後に再開する。
ランチャーに --resume-cache を追加（途中特徴再利用・ログ追記、既存head出力は上書き拒否）。
系列forwardは状態を再構築するため先頭から実施する。構文・dry-run通過、実サーバー再開は未実施。
以下の「原因未確定」は追加ログ受領前の記録。

下流ランチャーをユーザーが起動したが、Random全3条件がfailedで終了しHybridへ未移行。
起動前/homeは空き68G・使用率98%。容量不足はユーザーの推測で、例外ログ未受領のため未確定。
次は `outputs/m3ed_semantic_activity_random_hybrid_20260930/logs/random_<条件>.log` 末尾と
停止後df（容量/inode）・新規cacheのduを確認。再実行は既存ルート拒否になるため診断後に準備する。
入力・教師・target等のM3ED既存cacheは削除しない。下記「下流未起動」は起動前の記録。

Hybrid全3条件もユーザー提示ログでcompleteを確認。出力は
`outputs/m3ed_activity_hybrid_full_20260929_131030`。最終step・重みの存在とロードは未確認。
Randomは最終100k・各277M確認済み。ユーザーの指示で下流Semanticへ移行する（台帳§22）。
`tools/run_m3ed_semantic_activity_comparison.sh` を準備：全6最終checkpointの存在確認、
Random3条件→Hybrid3条件、各GPU1ジョブ。各モデルのcontinuous z/hを共用しh/z/concat計18 head。
Frozen Linear+CE、seed0、50 epoch、batch8、FP16、既存train4/validation1・11クラス疑似ラベル。
test不使用。出力prefixは `outputs/m3ed_semantic_activity_random_hybrid_20260930`、
cacheは `m3ed_cache/semantic_features/activity_random_hybrid_20260930`。空き容量確認後サーバーで実行。
exporterのtrain抽出が事前学習split検査に拒否される問題を修正（元split検証後に抽出loaderを構築）。
Macの構文・dry-runは通過、ML依存検証・下流起動は未実施。以下の未起動記述は過去の経緯。

### 2026-09-29までの経緯

M3ED activityの本番事前学習について、ユーザー提供ログでbaseline_e2 / activity_only /
active_z_h_softの全3条件のcomplete表示を確認。出力ルートは
`/home/iASL/Arata_repo/EventState/outputs/m3ed_activity_full_20260927_192110`。
`--event-statistics /home/iASL/Arata_repo/dataset/m3ed_cache/half_dagr/event_statistics_train4.json
--stage full --skip-tests` を指定。詳細は実験台帳§21。
後続ログで全条件100k stepと最終checkpoint各277Mの存在、launch.txtのbatch4・seed0を確認。
Random clip16、train4/validation1。保存config・ロード・下流精度は未確認。
最終loss・勾配は有限だが、末尾にbaseline/soft各1件のoverflowあり。詳細は台帳§21。
ユーザーは長い状態保持を学ばせるためHybridでも学習する方針を表明。
`tools/run_m3ed_activity_comparison.sh --sampling hybrid` を追加。同じ3条件をGPU0/1/2で、
Random2＋Stream2、clip16、等重み、100k、同じtrain4統計・split・DINO初期値から新規学習する。
Streamはclip間で状態を保持しdetachする。勾配の範囲は16フレームのまま。
出力prefixは `m3ed_activity_hybrid`。既定Randomは維持。Hybridは未起動。
Macでは構文・両samplingのsmoke/full dry-run・不正引数拒否を確認。
Torch/Hydra・streamテストはサーバーpreflightで実施する。
次はサーバー同期→Hybrid smoke→full。その後同じ最終stepでFrozen Semantic比較、
低活動subset・continuous/reset評価へ進む。下流起動報告はない。
ユーザーがRandom/Hybridの両事前学習完了後にSemantic Segmentation評価へ移行する方針を明示。
Hybrid Stream経路をコード確認：各系列の最初の有効frame1から16刻みの非重複clipを読み、
系列名ごとに状態を保持・detach。系列の選択順だけshuffleし、系列内順序は維持する。
新系列・epoch境界でreset、Random側は独立clipでStream状態を変更しない。
frame0は先行イベント窓がなく対象外。Streamでは末尾16未満の端数を除外するため、
全フレームを必ず消費する仕様ではない。学習max_steps到達時も系列途中で終了し得る。
stdlibだけで実samplerクラスを抽出し100 seed×3 epochの順序・被覆・lane継続を検証、通過。
実データ・Torchによる状態継承の実行確認は今回未実施。コード上、想定した連続読み込みを妨げる不具合は見つからなかった。
以下の2026-09-27時点の「M3ED未起動」「統計再計算→smoke」は過去の経緯であり、
現在の状態はこの最新状況を優先する。M3EDキャッシュは引き続き保持する。

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

### M3ED smoke前の正規化統計で停止（2026-09-27）

ユーザーのサーバーで既存event_statistics.jsonの系列情報がtrain4と一致せず、
normalization validation failedで学習開始前に停止した。実際のJSON内容は未確認なので、
どの系列を含むか／過去実験への影響はまだ不明。
`tools/compute_m3ed_event_normalization.py`を追加。既存prepared GEPをtrain4だけ読み、
従来prepare_m3edと同じ中央352x640（rows4:356）で全画素のmean/stdをfloat64集計し、
別名event_statistics_train4.jsonへ保存する。rawも教師cacheも不要、既存ファイル上書き拒否。
次の手順はサーバー同期→統計再計算→--event-statisticsで新JSON指定→smoke。
MacはAST/--helpのみ確認、数値テストはtests/test_m3ed_activity.pyに追加しサーバーで実行する。
