# EventState 実験結果台帳

最終更新: 2026-10-01

この文書を、会話セッションに依存しない実験結果の正本とする。数値を追加するときは、
比較条件、seed、評価split、対応する出力ディレクトリを併記する。特記がない結果は
`seed=0` の単一試行であり、平均・標準偏差や統計的有意差を示すものではない。

## 1. 事前学習条件

| 条件 | 蒸留対象 | 時系列モデル | Event dropout | サンプル形式 |
|---|---|---|---|---|
| E0 | 現在特徴 `z` | なし | なし | Random |
| E1 | 時系列状態 `h` | 1層 LSTM | なし | Random |
| E2 | `z` と `h` | 1層 LSTM | なし | Random |
| E3 | `h` | 1層 LSTM | あり | Random |
| E4 | `z` と `h` | 1層 LSTM | あり | Random |
| Hybrid | `h` | 1層 LSTM | なし | Random + Stream |

- Random: clipごとにLSTM状態を初期化し、clip内部でBPTTする。
- Stream: 時系列順に状態を継承し、clip境界で勾配を切断する（TBPTT）。
- Hybrid: RandomとStreamのmicro-batchを同じoptimizer stepで学習する。
- DSECの主要checkpointは100,000 stepの最終重み。

ポスター主表で使用している事前学習lossは次のとおり。

| 蒸留対象 | サンプル形式 | Pretrain loss |
|---|---|---:|
| `z` | Random | `z: 0.1170` |
| `h` | Random | `h: 0.1010` |
| `z/h` | Random | `z: 0.1175`, `h: 0.1126` |
| `h` | Random + Stream | `h: 0.0998` |

Pretrain lossは目的関数が異なる条件間の主指標にはせず、下流タスクのmAP/mIoUを主比較にする。

## 2. DSEC-Detection: Frozen backbone

EventState backboneを固定し、YOLOX型検出headのみを学習した公式test結果。

| 条件 | 使用特徴 | mAP | AP50 | AP75 | AP car | AP pedestrian |
|---|---|---:|---:|---:|---:|---:|
| E0 | `z` | 0.37852 | 0.67123 | 0.36834 | 0.51080 | 0.24624 |
| E1 | `h` | **0.38693** | **0.68502** | **0.39052** | 0.51324 | **0.26063** |
| E2 | `h` | 0.36869 | 0.67147 | 0.36311 | 0.49900 | 0.23839 |
| E3 | `h` | 0.38595 | 0.68138 | 0.38337 | **0.52552** | 0.24639 |
| E4 | `h` | 0.38436 | 0.67750 | 0.38250 | 0.51501 | 0.25370 |
| Hybrid | `h` | **0.39209** | **0.69416** | 0.38749 | 0.51973 | **0.26444** |

Hybridと直接対応するRandom条件はE1である。

- mAP: `0.38693 -> 0.39209`（`+0.00516`, 約 `+0.52` point）
- AP50: `0.68502 -> 0.69416`（`+0.00914`）
- AP pedestrian: `0.26063 -> 0.26444`（`+0.00381`）

主な出力:

- `outputs/dsec_detection_frozen_step100000/{E0,E1,E2,E3,E4}/seed_0/test_metrics.json`
- `outputs/dsec_detection_frozen_hybrid_noaug_cache/HYBRID/seed_0/test_metrics.json`

## 3. DSEC-Detection: Scratch / Fine-tuning

### Scratch

| 条件 | mAP | AP50 | AP75 | AP car | AP pedestrian | best epoch |
|---|---:|---:|---:|---:|---:|---:|
| No memory | 0.25065 | 0.44250 | 0.24755 | 0.38762 | 0.11368 | 30 |
| 1層 LSTM | **0.26290** | **0.46985** | **0.26461** | **0.40235** | **0.12346** | 25 |

出力: `outputs/dsec_detection_end_to_end/scratch/*/seed_0/test_metrics.json`

### 全層Fine-tuning

| 条件 | Frozen mAP | Fine-tune mAP | 差 | Fine-tune AP50 | Fine-tune AP75 |
|---|---:|---:|---:|---:|---:|
| E0 | 0.37852 | 0.30931 | -0.06921 | 0.48738 | 0.32946 |
| E2 | 0.36869 | 0.35721 | -0.01148 | 0.55658 | 0.38129 |
| E4 | 0.38436 | 0.35955 | -0.02481 | 0.53222 | 0.39603 |

この設定では全層Fine-tuningによるmAP改善は確認されなかった。特にpedestrian APが低下した。
出力: `outputs/dsec_detection_end_to_end/finetune/*/seed_0/test_metrics.json`

### E4 temporal部分 + headのみ更新

Event encoderを固定し、LSTMと検出headを更新した条件。

| 条件 | mAP | AP50 | AP75 | AP car | AP pedestrian |
|---|---:|---:|---:|---:|---:|
| E4 temporal + head | 0.38181 | 0.67657 | 0.38323 | 0.51081 | 0.25280 |
| E4 temporal + head, dropout学習 | 0.38313 | 0.67946 | 0.38342 | 0.53988 | 0.22638 |

出力:

- `outputs/dsec_detection_temporal_head/finetune/E4/seed_0/test_metrics.json`
- `outputs/dsec_detection_temporal_head_dropout/finetune/E4/seed_0/test_metrics.json`

## 4. DSEC-Detection: Event gap評価

評価中に1 / 2 / 4 / 8フレームのevent入力を欠落させたときのoverall mAP。

| 条件 | gap 0 | gap 1 | gap 2 | gap 4 | gap 8 |
|---|---:|---:|---:|---:|---:|
| Frozen E4 | 0.3842 | 0.3830 | 0.3799 | 0.3734 | 0.3555 |
| Temporal FT | 0.3818 | 0.3805 | 0.3759 | 0.3628 | 0.3369 |
| Temporal FT + dropout | 0.3831 | 0.3821 | 0.3802 | 0.3728 | 0.3550 |

gap 8での低下量:

- Frozen E4: `-0.0286`
- Temporal FT: `-0.0449`
- Temporal FT + dropout: `-0.0281`

Temporal fine-tuningだけでは欠落耐性が悪化したが、downstream学習時のdropoutでFrozen E4相当まで
回復した。gap内mAPや回復直後のsubsetはフレーム数が少ないため、overall mAPより慎重に解釈する。

出力:

- `outputs/dsec_detection_gap_e4/summary.csv`
- `outputs/dsec_detection_gap_e4_temporal_dropout/summary.csv`

## 5. DSEC-Semantic: Frozen Linear + CE

11クラス、640 x 440、EventState backbone固定、1x1 Linear head + CE。6系列train / 2系列valで
epoch数を選択し、8系列でheadを再学習して公式3 test系列を評価した。全条件で評価できたのは
2,806 frames（790,169,600 pixels）。

| 条件 | 使用特徴 | mIoU | Pixel Accuracy | Mean Class Accuracy |
|---|---|---:|---:|---:|
| E0 | `z` | 0.59614 | 0.91561 | 0.67852 |
| E1 | `h` | 0.60383 | 0.91817 | 0.69234 |
| E2 | `h` | 0.60057 | 0.91687 | 0.68718 |
| E3 | `h` | 0.60389 | 0.91889 | 0.69253 |
| E4 | `h` | 0.60148 | 0.91764 | 0.68723 |
| Hybrid | `h` | **0.60843** | **0.92018** | **0.69661** |

HybridとE1の直接比較:

- mIoU: `0.60383 -> 0.60843`（`+0.00460`, 約 `+0.46` point）

DetectionとSemanticの両方で、Random + Streamが対応するRandom条件を小幅に上回った。

出力:

- `outputs/dsec_semantic_frozen_linear/{E0,E1,E2,E3,E4}/seed_0/test_metrics.json`
- `outputs/dsec_semantic_frozen_linear/HYBRID_noaug_cache/seed_0/test_metrics.json`

## 6. DSEC-Semantic: Head / Loss 2 x 2 ablation

表現をHybrid `h`に固定し、headとlossのみを変更した。Backboneは全条件でFrozen。
`GEP patch`はGEPリポジトリの単一スケール`BaselineSegHead`相当であり、多段特徴を使う
UPerNet完全再現ではない。

| Head | Loss | mIoU | Pixel Accuracy | Mean Class Accuracy | Linear+CEとの差 |
|---|---|---:|---:|---:|---:|
| Linear | CE | 0.60843 | 0.92018 | 0.69661 | 0.00000 |
| Linear | CE + Dice | 0.61224 | 0.91878 | **0.71579** | +0.00382 |
| GEP patch | CE | 0.60617 | **0.92179** | 0.68786 | -0.00226 |
| GEP patch | CE + Dice | **0.61462** | 0.92016 | 0.71252 | **+0.00619** |

要因別のmIoU差:

- LinearでDice追加: `+0.00382`
- CEでGEP patchへ変更: `-0.00226`
- GEP patchでDice追加: `+0.00845`
- Diceの平均主効果: `+0.00613`
- Headの平均主効果: `+0.00006`
- Head x Diceの相互作用: `+0.00464`

解釈:

- 改善の中心はDice lossである。
- GEP patch head単独では改善しなかった。
- GEP patchとDiceを併用した条件が最高mIoUとなった。
- GEP patch + DiceはLinear + CEに対して、pole `+0.04655`、traffic sign `+0.01634`、
  sidewalk `+0.01289`、car `+0.01252`、person `+0.00835`改善した一方、wallは
  `-0.03161`低下した。
- 表現間の主比較には引き続きLinear + CEを用い、このablationはdecoder/lossの追加検証として扱う。

出力:

- `outputs/dsec_semantic_frozen_head_loss_ablation/HYBRID_noaug_cache/*/seed_0/test_metrics.json`

## 7. DSEC Hybrid / Augmentation事前学習

| 条件 | step | 直近train loss | h projected cosine | throughput |
|---|---:|---:|---:|---:|
| Hybrid, no augmentation, cached teacher | 100,000 | 0.09978 | 0.9667 | 9.963 sample/s |
| Hybrid, no augmentation, online teacher | 100,000 | 0.09981 | 0.9667 | 5.989 sample/s |
| Hybrid, augmentation, online teacher | 100,000 | 0.1164 | 0.9612 | 6.014 sample/s |

cached/online teacherのno-augmentation条件はほぼ一致した。augmentation条件はpretrain lossとcosineでは
悪化したが、異なるaugmentation条件間のlossだけで有効性を判断せず、下流評価を用いる。

既知の下流結果:

- Hybrid no-augmentation cached teacher Frozen Det: mAP `0.39209`
- Hybrid augmentation online teacher Frozen Det: 実行完了報告あり。ただし数値はこの台帳へ未転記。

出力: `outputs/dsec_hybrid_aug/`

## 8. M3ED事前学習・Semantic validation

M3EDの5系列を使った`h`蒸留・1層LSTMの途中結果。

| 条件 | step | 直近train loss | 最新validation loss | best validation |
|---|---:|---:|---:|---:|
| Hybrid + dropout | 100,000 | 0.06842 | 0.5128 | 0.4748 @ 3,000 |
| Hybrid, no dropout | 100,000 | 0.06490 | 0.5143 | 0.4713 @ 3,000 |
| Random + dropout | 84,180時点 | 0.07116 | 0.5184 @ 84,000 | 0.4856 @ 2,000 |

Random + dropoutは89,000 step checkpointから100,000 stepへ再開した。最終checkpointの存在は
確認済みだが、100,000 step時点の集約値はこの台帳へ未転記。

出力: `outputs/m3ed_state_dropout_factorial_20260920_145338/`

### Frozen Linear + CE

M3ED InternImage疑似ラベルの11クラスを用い、4系列でheadを学習し、
`car_urban_day_ucity_small_loop`をvalidationとして評価した。EventState backboneはFrozen。

| 事前学習条件 | mIoU | Pixel Accuracy | Mean Class Accuracy | best epoch |
|---|---:|---:|---:|---:|
| Random, no dropout | 0.36889 | 0.73351 | 0.47826 | 50 |
| Hybrid, no dropout | 0.37388 | 0.73799 | 0.48405 | 25 |
| Hybrid + dropout | **0.37611** | **0.73815** | 0.48372 | 40 |

出力: `outputs/m3ed_semantic_frozen_linear/*/seed_0/validation_metrics.json`

### Hybrid + dropout: Head / Loss 2 x 2 ablation

| Head | Loss | mIoU | Pixel Accuracy | Mean Class Accuracy | Linear+CEとの差 | best epoch |
|---|---|---:|---:|---:|---:|---:|
| Linear | CE | 0.37611 | 0.73815 | 0.48372 | 0.00000 | 40 |
| Linear | CE + Dice | **0.38392** | 0.73902 | 0.49336 | **+0.00781** | 50 |
| GEP patch | CE | 0.37552 | **0.73934** | 0.47908 | -0.00059 | 10 |
| GEP patch | CE + Dice | 0.38280 | 0.73510 | **0.49895** | +0.00669 | 25 |

M3EDでもDiceが有効だった。最高mIoUはLinear + CE + Diceであり、GEP patch単体の改善は
確認されなかった。E2VIDの公開値39.40%との差は約1.01 pointだが、クラス数、split、入力、
学習方式が異なるため直接比較には用いない。

出力:
`outputs/m3ed_semantic_frozen_head_loss_ablation/hybrid_dropout/*/seed_0/validation_metrics.json`

## 9. Event activity: DSEC / M3ED

prepared event tensor上のevent density分布。

| Dataset | frames | sequences | p1 | p5 | p10 | median | p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DSEC | 78,284 | 60 | 2,448,252.5 | 8,948,289.7 | 12,382,362.6 | 32,600,281.9 | 76,238,557.3 |
| M3ED | 34,009 | 5 | 70,980.9 | 159,440.1 | 341,666.7 | 12,748,697.9 | 38,957,291.7 |

M3EDはDSECより低event activityの裾が大幅に広く、長い低event区間も含む。そのため、LSTM状態保持の
評価にはDSECよりM3EDの方が適している可能性がある。

出力: `outputs/event_activity/dsec_vs_m3ed_density/`

## 10. 現時点の主要な観察

1. `h`蒸留は`z`蒸留よりFrozen Detection / Semanticで良好な傾向を示した。
2. `z/h`同時蒸留は、現設定では`h`単独蒸留を上回らなかった。
3. Random + Stream事前学習は、Random単独よりDetectionとSemanticの両方で小幅に改善した。
4. 全層Fine-tuningはFrozen backbone評価を上回らなかった。
5. Event gap耐性はdownstream dropout学習で改善した。
6. SemanticではDice lossが有効で、GEP patch + Diceが最高mIoUとなった。
7. DSECは低event区間が比較的少なく、状態保持の優位性を測るにはM3EDや実機RCカー評価が重要である。

## 11. 未完了・追記待ち

- [ ] M3ED Random + dropout 100,000 stepの最終集約値を転記
- [ ] M3ED特徴可視化（Random continuous / Hybrid continuous / Hybrid clip reset）
- [x] M3ED downstream Semantic Linear+CE（Random/Hybrid/dropout）
- [x] M3ED downstream Semantic Head/Loss 2 x 2（Hybrid+dropout）
- [ ] M3ED Linear+CE+DiceでRandom/Hybrid/dropoutの事前学習要因を再比較
- [ ] Hybrid augmentation checkpointのDSEC Frozen Detection数値転記
- [ ] DSEC Semantic head/loss ablationの複数seed評価
- [ ] DSEC以外のdomain transfer評価
- [ ] RCカーopen-loop / closed-loop評価
- [ ] JEPA teacherとの融合・比較

## 12. 更新ルール

- test結果を追加するときは、対応する`test_metrics.json`のパスを必ず残す。
- 単一seedと複数seed平均を混同しない。
- Frozen、Fine-tuning、Scratchを同じ欄で暗黙に比較しない。
- 異なるhead/lossの結果を、表現そのものの優劣として扱わない。
- 未完了値、途中step、smoke testは明示する。

## 13. DSEC activity比較のsmokeと本番起動（2026-09-23記録）

ユーザーから100 step smokeの完了報告を受領。条件は `baseline_e2`、
`activity_only`、`scale_event_full`。DSEC-Det train 41系列で事前学習し、
validation/testは使用しない。seedは起動既定値の0（保存configは未照合）。
共通設定はclip長8、batch size 8。出力先:

`/home/iASL/Arata_repo/EventState/outputs/dsec_activity_smoke_20260922_231607`

提示されたScaleEvent型損失のstep 100ログ（ファイル名はログ断片に含まれず、
損失項から `scale_event_full` 相当と判断）:

| 指標 | 値 |
|---|---:|
| loss | 0.31568 |
| h_distill_loss | 0.065251 |
| z_distill_loss | 0.25043 |
| active_patch_fraction | 0.78654 |
| grad_norm | 0.29122 |
| optimizer_step_skipped | 0 |
| samples_per_second | 8.8763 |
| gpu_memory_mb | 20524 |

これは動作確認の値であり、下流mAP/mIoUや手法の改善を示す結果ではない。
他2条件の最終数値は未受領。

続いて同じ3条件をGPU 0/1/2で `--stage full --skip-tests` により起動したとの報告あり。
本番は100,000 step、出力は `outputs/dsec_activity_full_<起動日時>`。
正確な出力ディレクトリ名と学習進行・最終結果は未確認。

2026-09-24追記: ユーザーから本番学習が終了したようだとの報告あり。
条件・split・seedは上記の起動条件。3条件の100,000 step到達、最終checkpoint、
最終数値はまだ未照合であり、正常終了確認済みとは扱わない。

### 本番100,000 step到達確認（2026-09-24追加）

後続のユーザー提供ログで、全3条件のstep 100000と最終checkpointの存在を確認。
出力ルートは `/home/iASL/Arata_repo/EventState/outputs/dsec_activity_full_20260923_000937`。
ログは `logs/<条件名>.log`、重みは `<条件名>/checkpoints/step_00100000.pt`（各277M）。
checkpointのロード検証は未実施。splitはDSEC-Det train41、事前学習val/testなし、
seedは起動既定値0（保存config未照合）、clip長8、batch size 8。

| 条件 | 最終step loss | h loss | z loss | h projected cosine | z projected cosine | active率 | grad norm | samples/s | GPU memory MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_e2 | 0.18805 | 0.091518 | 0.096530 | 0.96949 | 0.96782 | 未記録 | 0.44905 | 9.5659 | 17065 |
| activity_only | 0.18862 | 0.072590 | 0.116030 | 0.96253 | 0.96330 | 0.75991 | 0.47226 | 10.324 | 16952 |
| scale_event_full | 0.083002 | 0.018364 | 0.064638 | 0.95669 | 0.96261 | 0.75084 | 0.10510 | 6.1957 | 20524 |

各条件の提示された最後3回の記録では `optimizer_step_skipped=0`。全期間のskip集計ではない。
数値は最後の学習ログであり、全データ平均ではない。マスク対象・損失形式が異なるため
lossの大小で優劣を判断しない。最終stepのevent_countも条件間で異なるため、同一batch比較ではない。
次は同じ下流条件でFrozen Detection / Semanticを評価する。下流mAP/mIoUは未取得。

## 14. Activity事前学習: Frozen DSEC Detection h

ユーザー提供の `test_metrics.json` を記録。事前学習は§13の100,000 step重み、
下流はseed 0、continuous h、backbone固定、YOLOX型headのみ学習。
提示した起動条件はbatch 16・50 epoch・FP16・従来損失（activity重み1/1）。
公式41/6/13分割のvalでbest.ptを選択し、testで評価。
`protocol=dsec-det`、`coordinate_space=dsec_det_distorted`。best epochは未受領。

| 条件 | mAP | AP50 | AP75 | AP car | AP pedestrian | AP small | AP medium | AP large |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_e2 | 0.37778238 | 0.67310797 | 0.37677932 | 0.50850079 | 0.24706397 | 0.13867482 | 0.39766928 | 0.59484951 |
| activity_only | 0.37342058 | 0.67779599 | 0.35627772 | 0.50928362 | 0.23755755 | 0.13820554 | 0.39278423 | 0.59727616 |
| scale_event_full | 0.37427887 | 0.67851461 | 0.35585001 | 0.51983709 | 0.22872064 | 0.15040354 | 0.39414030 | 0.56004745 |

出力（サーバー）:
`/home/iASL/Arata_repo/EventState/outputs/dsec_detection_activity_20260923_h/{baseline_e2,activity_only,scale_event_full}/seed_0/test_metrics.json`

特徴キャッシュ:
`/home/iASL/Arata_repo/dataset/DSEC_cache/detection_features/activity_20260923_h/<条件名>/dsec_det`

同時実行したbaselineに対するmAP差はactivity_onlyが−0.43618 point、
scale_event_fullが−0.35035 point（0–100表記）。AP50は上昇したが、AP75と
pedestrian APは低下した。単一seedであり有意差は未検証。
過去のE2 0.36869とは区別し、今回の比較基準は0.37778238とする。
この結果だけでz/hの相補性、低活動領域での性能、Semanticの効果は判断できない。
予定済みのSemanticおよびz/concat評価を進め、testを用いた重み・閾値の調整は行わない。

## 15. Activity事前学習: Frozen DSEC Semantic h

ユーザー提供の公式test JSONを記録。§13の100,000 step重み、seed 0、continuous h、
Frozen backbone、Linear head + CE、下流activity重みなし。
提示した起動条件はbatch 8、開発50 epoch、FP16。公式trainの6/2分割でepochを選択し、
headを初期化し直して全8系列で再学習、公式test 3系列を評価する既存runnerを使用。
選択epoch数は未受領。全条件のevaluated_pixelsは790,169,600で一致。

| 条件 | mIoU | Pixel accuracy | Mean class accuracy | baseline比mIoU (point) |
|---|---:|---:|---:|---:|
| baseline_e2 | 0.6005323404125538 | 0.9168447950920916 | 0.6870124260848327 | 0 |
| activity_only | 0.5897342979648699 | 0.9150212156985033 | 0.6777412867096996 | −1.07980 |
| scale_event_full | 0.588328497455661 | 0.9155564311762943 | 0.6719785038555767 | −1.22038 |

| Class IoU | baseline_e2 | activity_only | scale_event_full |
|---|---:|---:|---:|
| background | 0.93925240 | 0.93609110 | 0.93740680 |
| building | 0.82943686 | 0.82306586 | 0.82694177 |
| fence | 0.24227367 | 0.23281210 | 0.22842223 |
| person | 0.32195141 | 0.27222079 | 0.25678517 |
| pole | 0.15524223 | 0.13985178 | 0.14917611 |
| road | 0.93503312 | 0.93787378 | 0.93655054 |
| sidewalk | 0.66460490 | 0.67571522 | 0.66567616 |
| vegetation | 0.84206051 | 0.83564423 | 0.83692761 |
| car | 0.82050617 | 0.81152706 | 0.81683195 |
| wall | 0.42468745 | 0.42209278 | 0.41470152 |
| traffic_sign | 0.43080702 | 0.40018258 | 0.40219362 |

出力（サーバー）:
`/home/iASL/Arata_repo/EventState/outputs/dsec_semantic_activity_20260923_h/{baseline_e2,activity_only,scale_event_full}/seed_0/test_metrics.json`

特徴キャッシュ:
`/home/iASL/Arata_repo/dataset/DSEC_cache/semantic_features/activity_20260923_h/<条件名>`

Semanticラベル: `/home/iASL/Arata_repo/dataset/DSEC/task_labels/semantic`。
両activity条件は11クラス中9クラスでbaselineを下回り、roadとsidewalkで上回った。
personの低下はそれぞれ−4.97306 / −6.51662 point。
Frozen hではDetection/Semanticの両方で改善を確認できなかったが、単一seedで有意差は未検証。
hへの直接蒸留領域を限定した影響は原因仮説であり、クラス別activity測定なしに断定しない。
z/concatおよび低活動領域での評価は未実施。次は予定済みのz/concat比較を行う。

## 16. Semantic z/concat: Frozen Linear+CE

2026-09-25にbaseline_e2 / activity_only / scale_event_fullの全条件について
z / concatの完了表示と、後続で計6実験のtest JSONを受領。選択epochは未受領。
提示コマンドでは§13の100,000 step重み、Frozen Linear+CE、seed 0、batch 8、
開発50 epoch、6/2でepoch選択→全8系列でhead再学習→公式test 3系列。
下流activity重みなし。z/hを一度に抽出した共通cacheを使用。

出力（サーバー）:
`/home/iASL/Arata_repo/EventState/outputs/dsec_semantic_activity_20260923_zh/<条件名>/<z|concat>/seed_0/test_metrics.json`

キャッシュ:
`/home/iASL/Arata_repo/dataset/DSEC_cache/semantic_features/activity_20260923_zh/<条件名>`

全6条件でrole=test、evaluated_pixels=790,169,600。

| 条件 | feature | mIoU | Pixel accuracy | Mean class accuracy |
|---|---|---:|---:|---:|
| baseline_e2 | z | 0.595919275782201 | 0.9149983244103544 | 0.6791772736658802 |
| baseline_e2 | concat | 0.5995933364506295 | 0.9165051085235372 | 0.6833152562051312 |
| activity_only | z | 0.5978721433045646 | 0.9161309192355667 | 0.6820371420335752 |
| activity_only | concat | 0.5986879464900309 | 0.9168519378624538 | 0.6836284479653612 |
| scale_event_full | z | 0.5957919884934975 | 0.9153792730067064 | 0.6780036755768604 |
| scale_event_full | concat | 0.5993057068798803 | 0.9177549794373259 | 0.6810438682149224 |

| Class IoU | baseline z | baseline concat | activity z | activity concat | scale z | scale concat |
|---|---:|---:|---:|---:|---:|---:|
| background | 0.93509156 | 0.93682284 | 0.93467207 | 0.93679622 | 0.93520471 | 0.93874515 |
| building | 0.82530405 | 0.82847793 | 0.82519969 | 0.82752573 | 0.82500331 | 0.83096521 |
| fence | 0.22837901 | 0.23074692 | 0.22379475 | 0.22538436 | 0.22642701 | 0.22639580 |
| person | 0.32536316 | 0.32811784 | 0.32349545 | 0.31845643 | 0.31876352 | 0.31435709 |
| pole | 0.16016384 | 0.16285425 | 0.16404143 | 0.16159233 | 0.16427867 | 0.16478901 |
| road | 0.93409611 | 0.93536836 | 0.93717756 | 0.93760418 | 0.93431008 | 0.93664460 |
| sidewalk | 0.65865215 | 0.66469434 | 0.67370747 | 0.67474153 | 0.66283802 | 0.67169574 |
| vegetation | 0.83785621 | 0.84087855 | 0.83853796 | 0.84086304 | 0.83835369 | 0.84271702 |
| car | 0.81629096 | 0.82087597 | 0.82004821 | 0.82020399 | 0.82020779 | 0.82531374 |
| wall | 0.40976228 | 0.41736508 | 0.41355420 | 0.41681214 | 0.40825305 | 0.41860086 |
| traffic_sign | 0.42415271 | 0.42932462 | 0.42236479 | 0.42558746 | 0.42007202 | 0.42213854 |

同じfeatureのbaseline比mIoUはactivity z +0.19529、activity concat −0.09054、
scale z −0.01273、scale concat −0.02876 point。
concat−zはbaseline +0.36741、activity +0.08158、scale +0.35137 point。
activity条件のconcatはh単独の低下を大きく回復するが、同じconcatのbaselineを超えない。
baselineにもconcat−zの改善があり、concatの入力次元・headパラメータ数も増えるため、
これだけでactivity特有の相補性や長期記憶の改善とは結論できない。
単一seedで小差の有意性は未検証。低活動領域の性能は依然未測定。

## 17. h活動制約の緩和：事前学習の完了報告（2026-09-26受領）

ユーザー提示のランチャーログで、以下の全3条件についてsmoke・fullのcomplete表示を確認。
本番コマンドは `run_h_relaxation --stage full --skip-tests`。
設定上はseed=0、DSEC-Det train41、validationなし、100,000 step、batch=8、clip=8。
追加のユーザー提示ログで全条件100,000 step到達と最終checkpointの存在を確認。
checkpointのロード検証は未実施、下流評価値は未受領。

| 条件 | z蒸留 | h蒸留 | 本番GPU |
|---|---|---|---|
| active_z_only | activeのみ | 無効 | 0 |
| active_z_h_all | activeのみ | 全領域、重み1 | 1 |
| active_z_h_soft | activeのみ | inactive=1、active=0.5 | 2 |

全条件cosine＋二乗L2。hは重み付き平均であり、ScaleEventのL1/Gram損失ではない。
z-onlyはtemporal/h projectorが未学習なので下流はzのみを対象とする。

サーバー上の出力ルート:

- smoke: `/home/iASL/Arata_repo/EventState/outputs/dsec_activity_h_relaxation_smoke_20260925_103516`
- full: `/home/iASL/Arata_repo/EventState/outputs/dsec_activity_h_relaxation_full_20260925_182735`

各ルート配下の `logs/<条件名>.log` にログ、`<条件名>/checkpoints/` にcheckpointを保存する設定。
最終checkpointは各条件の `checkpoints/step_00100000.pt`。

| 条件 | 最終loss | h distill（診断値） | z distill | active fraction | checkpointサイズ |
|---|---:|---:|---:|---:|---:|
| active_z_only | 0.10806 | 3.0572（目的に不使用） | 0.10806 | 0.75991 | 262M |
| active_z_h_all | 0.20348 | 0.097399 | 0.10608 | 0.76279 | 277M |
| active_z_h_soft | 0.20148 | 0.094694 | 0.10678 | 0.76279 | 277M |

提示された末尾3 stepでは全条件optimizer_step_skipped=0、loss・勾配は有限。
z-onlyのtemporal/h projectorのgrad_norm=0はh蒸留無効の設定通り。
h-all/h-softの最終h projected cosineは0.96753/0.96747。
損失の対象・枝数が異なるためtotal lossを条件間の性能比較に使わない。
サーバーの/homeは3.5T中3.1T使用、空き268G（92%使用）。
次は従来のSemantic開発6/2分割、Frozen Linear+CE・seed0・batch8・50 epochでvalidation比較。
testはこの開発段階では使用しない。

## 18. h活動制約の緩和：Semantic開発評価の完了報告（2026-09-27）

ユーザーからactive_z_only、active_z_h_all、active_z_h_softの全complete表示を受領。
提示済みコマンドではtrain/val特徴抽出、開発head学習、best checkpointのval評価を実行。
条件はFrozen Linear+CE、下流activity重みなし、seed0、batch8、50 epoch、従来のtrain6/val2。
対象はactive_z_onlyのz、他2条件のh/z/concat（計7 head）。事前学習は§17の最終100,000 step。

出力ルート（ユーザー確認済み）:
`/home/iASL/Arata_repo/EventState/outputs/dsec_semantic_h_relaxation_20260925_182735`

指標の保存先は `<条件名>/<feature>/seed_0/validation_metrics.json`、
選択checkpointは同じディレクトリの `development/best.pt`。
2026-09-27に7件のJSONを受領。best epochは未受領。公式train8での再学習・test評価は未実施。
既存のtest mIoUと今回のval mIoUを直接比較しない。

全7件でrole=val、evaluated_pixels=633,036,800。

| 条件 | feature | mIoU | Pixel accuracy | Mean class accuracy |
|---|---|---:|---:|---:|
| active_z_only | z | 0.5936550348768449 | 0.9139634978566807 | 0.6666766395201381 |
| active_z_h_all | z | 0.5976478936737156 | 0.915098237574814 | 0.6749755879087276 |
| active_z_h_all | h | 0.5896110949675109 | 0.9139837146908363 | 0.6636134446003449 |
| active_z_h_all | concat | 0.6004046401519081 | 0.915991784363879 | 0.6789537606022189 |
| active_z_h_soft | z | 0.5988787156129507 | 0.9155894033332659 | 0.677557200136638 |
| active_z_h_soft | h | 0.5900111973447252 | 0.9143810817949288 | 0.6646528956543013 |
| active_z_h_soft | concat | 0.6015226328610124 | 0.9165077227737787 | 0.6811112558438454 |

クラス順: background, building, fence, person, pole, road, sidewalk, vegetation, car, wall, traffic_sign。
各行のclass_IoU（元JSONの精度を保持）:

- active_z_only/z: `[0.9360979234002375, 0.811270060054022, 0.24968635311096482, 0.27756004541221097, 0.18545733825366464, 0.9311316590545105, 0.6999662969725184, 0.8170628303863302, 0.8007221377749288, 0.4577051461379717, 0.3635455930879333]`
- active_z_h_all/z: `[0.9391977698016432, 0.8123044231538853, 0.2778889555772978, 0.2755216416623556, 0.18853271638510946, 0.9330807274416958, 0.7071161948363351, 0.8182893957140804, 0.7998619517201754, 0.4593131501361804, 0.36301990398211315]`
- active_z_h_all/h: `[0.9413831996294401, 0.8117125340598353, 0.259858724907728, 0.25630682647065545, 0.16968037782885634, 0.931928479647213, 0.6940432303346913, 0.8170045268252063, 0.7918067971527415, 0.4534537693196394, 0.3585435784666131]`
- active_z_h_all/concat: `[0.9412133147552542, 0.8154690469794122, 0.2841760625052086, 0.27785715115401144, 0.19074775660285928, 0.9333524760825888, 0.7057371144083374, 0.821391368035048, 0.7998546883216833, 0.4669071430160013, 0.3677449198105855]`
- active_z_h_soft/z: `[0.9400321180918663, 0.8139770221460821, 0.2878758664528983, 0.2741650538434817, 0.18867001015937548, 0.9336432774746328, 0.7088978603837243, 0.8191646366319894, 0.7987897736713654, 0.459945887784805, 0.3625043651022366]`
- active_z_h_soft/h: `[0.9423163337546286, 0.8126379621382159, 0.26216163783940394, 0.2507862962806603, 0.1690579914568633, 0.9327901970257662, 0.6964496580189236, 0.8170713180693923, 0.7925353849085202, 0.4570748582475157, 0.3572415330520869]`
- active_z_h_soft/concat: `[0.9422767404864723, 0.8172686146389362, 0.2928824548316412, 0.276149046707599, 0.19109060673235415, 0.9338397526860112, 0.7072823183437482, 0.8223461178000963, 0.7990631583304209, 0.46749548594430734, 0.3670546649695495]`

同じzでz-onlyに対しh-allは+0.39929、h-softは+0.52237 point。
concat−zはh-all +0.27567、h-soft +0.26439 point。
soft−allはz +0.12308、h +0.04001、concat +0.11180 point。
h単独は各条件のz未満。h目的追加による共有encoderへの影響と整合するが、因果機構や長期記憶の証明ではない。
単一seedで小差の有意性は不明。baseline_e2と旧hard maskの同じval結果がないため、活動制約緩和による回復はまだ判定しない。
次は既存baseline_e2/activity_onlyの開発best checkpointを同じvalで比較する。test値をこの比較へ混ぜない。

## 19. 従来baseline・hard maskのSemantic validation再評価（2026-09-27）

既存の開発用best headと特徴cacheによる6件の評価完了・JSONを受領。追加学習なし。
Frozen Linear+CE、seed0、開発train6/val2、50 epochからbest選択。best epoch番号は未受領。
全件role=val、evaluated_pixels=633,036,800。下流activity重みなし。事前学習は§13のtrain41・100,000 step。

出力: `/home/iASL/Arata_repo/EventState/outputs/dsec_semantic_activity_validation_20260927/<条件>/<feature>/validation_metrics.json`。
hのheadは `outputs/dsec_semantic_activity_20260923_h/<条件>/seed_0/development/best.pt`、
z/concatは `outputs/dsec_semantic_activity_20260923_zh/<条件>/<feature>/seed_0/development/best.pt`。
キャッシュは `/home/iASL/Arata_repo/dataset/DSEC_cache/semantic_features/activity_20260923_<h|zh>/<条件>`。

| 条件 | feature | mIoU | Pixel accuracy | Mean class accuracy |
|---|---|---:|---:|---:|
| baseline_e2 | h | 0.5885415879336299 | 0.9140888270634503 | 0.6616996918758157 |
| baseline_e2 | z | 0.5964338328835226 | 0.9145161734673245 | 0.6760015569427753 |
| baseline_e2 | concat | 0.5994227128651685 | 0.916260323254509 | 0.6782709467828725 |
| activity_only | h | 0.5825927012599785 | 0.913406778247331 | 0.6579835623770273 |
| activity_only | z | 0.5983869417449444 | 0.9159347876774304 | 0.6789321521764847 |
| activity_only | concat | 0.6012808220147776 | 0.9170934485957214 | 0.6817710396285337 |

クラス順: background, building, fence, person, pole, road, sidewalk, vegetation, car, wall, traffic_sign。
class_IoU（元JSONの精度を保持）:

- baseline_e2/h: `[0.9405627134507942, 0.8128041329557507, 0.2494364704501468, 0.25439572981230135, 0.171728161418259, 0.9318197493488624, 0.6926168102209304, 0.8183171563145719, 0.7910141576169959, 0.4560662508855379, 0.3551961347957774]`
- baseline_e2/z: `[0.9391972699744885, 0.813879935884694, 0.2830207674146338, 0.2745171810464948, 0.18775045920543676, 0.9324426035366091, 0.7011285552779966, 0.8184651480453127, 0.7934416186886906, 0.454402190862604, 0.3625264317817883]`
- baseline_e2/concat: `[0.9412323939103008, 0.8170886640382562, 0.28004360851833776, 0.27530516957339624, 0.18915816879423697, 0.9339096714124603, 0.7069638098291856, 0.8221465626936814, 0.7983900267094476, 0.46473829051029014, 0.3646734755272609]`
- activity_only/h: `[0.9412020326272454, 0.8122714934500557, 0.3093448460034503, 0.20887690177302717, 0.146156156478747, 0.9339782050696167, 0.7012974880504388, 0.8113243754053354, 0.780686289485497, 0.4525644094196982, 0.31081751609665215]`
- activity_only/z: `[0.9409424742078872, 0.8144474986560141, 0.2998386850501468, 0.26846430861491943, 0.18602124818831486, 0.9354477241190492, 0.7142541370960908, 0.8181646242652423, 0.7970809104968626, 0.45713758968307094, 0.3504571588167913]`
- activity_only/concat: `[0.9434938483925662, 0.8179500266323908, 0.3074495989054775, 0.27163065813470166, 0.18784080124743335, 0.9359233930904939, 0.7139652348046926, 0.8215318362903758, 0.7961791287338715, 0.46429542420178443, 0.35382909172876637]`

§18と同一valで比較すると、hard maskからh-all/h-softへの変更でh mIoUは+0.70184/+0.74185 point。
一方concatはhard=60.12808%、all=60.04046%、soft=60.15226%であり、soft−hardは+0.02418 pointのみ。
baseline concat=59.94227%に対しhard +0.18581、all +0.09819、soft +0.20999 point。
非活動領域限定を緩めるとh単独が回復する仮説を支持する単一seedの結果だが、concatの優位や低活動領域での記憶改善は未証明。
次の提案はbaseline/hard/softのconcatでhead seed1,2を追加（seed0は既存）し、3 seed比較。
事前学習seedは固定のため、この比較はheadのばらつきのみを検証する。

## 20. Semantic concatのhead seed追加：完了報告（2026-09-27）

ユーザー提示ログでbaseline_e2、activity_only（hard）、active_z_h_softの
seed1・seed2、計6ジョブのcomplete表示を確認。seed1を3 GPUで実行し、全条件完了後seed2を実行。
提示コマンドではFrozen concat Linear+CE、通常の下流損失、50 epoch、batch8、FP16、
従来のtrain6/val2でhead学習し、development/best.ptでvalidation評価を実施。
事前学習checkpointは従来のseed0・100,000 stepを固定。追加事前学習・test評価なし。

保存先:
`/home/iASL/Arata_repo/EventState/outputs/dsec_semantic_concat_seeds_20260927/<条件>/seed_<1|2>/validation_metrics.json`

6件のJSONを受領。best epoch番号は未受領。全件feature=concat、role=val、evaluated_pixels=633,036,800。

| 条件 | head seed | mIoU | Mean class accuracy | Pixel accuracy |
|---|---:|---:|---:|---:|
| baseline_e2 | 1 | 0.5982448118518318 | 0.6761172894068253 | 0.9164632703817535 |
| baseline_e2 | 2 | 0.5983638131214635 | 0.6748146178059966 | 0.9166613805074207 |
| activity_only | 1 | 0.600766018521001 | 0.6794716500371876 | 0.9180572503841798 |
| activity_only | 2 | 0.600883428667169 | 0.6803142984120325 | 0.9178686736695244 |
| active_z_h_soft | 1 | 0.6004964631040933 | 0.6792303297197196 | 0.9167582484936104 |
| active_z_h_soft | 2 | 0.6007181576123003 | 0.6789579844997969 | 0.9171617432035547 |

class_IoU順: background, building, fence, person, pole, road, sidewalk, vegetation, car, wall, traffic_sign。

- baseline_e2/seed1: `[0.9412119559531955, 0.8168952154705887, 0.2716243024082978, 0.2696233256320197, 0.18645424691860257, 0.9340882085176158, 0.7098949493302907, 0.82273521874614, 0.8010625508693785, 0.4646658890995534, 0.36243706742446613]`
- baseline_e2/seed2: `[0.9411863810019895, 0.8168720155164946, 0.27209561714449554, 0.272047061934034, 0.184624753381827, 0.9344019103368864, 0.7105468746194359, 0.8222887842871537, 0.8005526047552579, 0.46326390523289984, 0.36412203612562466]`
- activity_only/seed1: `[0.9441668230089059, 0.818480242129556, 0.29503780364420795, 0.26697320161700533, 0.18771872850104476, 0.9372161916668019, 0.7221829982144412, 0.8227782180282956, 0.7999849838428384, 0.4630320536927894, 0.35085495938512473]`
- activity_only/seed2: `[0.9440562263843674, 0.8182966955922754, 0.29694917204318555, 0.2682818843945952, 0.186101499233349, 0.9371437571731918, 0.7212950659569625, 0.8225243199473332, 0.7998206169859785, 0.4610924832634389, 0.3541559943641831]`
- active_z_h_soft/seed1: `[0.9423280629816023, 0.8172708434187967, 0.2854310510815133, 0.2702843826287848, 0.18804586763206907, 0.9340125140462334, 0.710195456744523, 0.823034677683639, 0.8017574600547362, 0.46766920415599766, 0.3654315737171322]`
- active_z_h_soft/seed2: `[0.9429216173298384, 0.8167647066444306, 0.2782241986555508, 0.27110860949670784, 0.18992680858917643, 0.9350697363464991, 0.7146494654707031, 0.8231639074730496, 0.8026086686208093, 0.464717242493485, 0.36874477261505284]`

§18・§19のseed0と合算したmIoU（%、標準偏差は標本標準偏差、ddof=1）:

| 条件 | seed0 | seed1 | seed2 | 平均 ± SD |
|---|---:|---:|---:|---:|
| baseline_e2 | 59.94227 | 59.82448 | 59.83638 | 59.86771 ± 0.06484 |
| activity_only | 60.12808 | 60.07660 | 60.08834 | 60.09768 ± 0.02698 |
| active_z_h_soft | 60.15226 | 60.04965 | 60.07182 | 60.09124 ± 0.05400 |

hard/softは各head seedでbaselineを上回る。平均差は+0.22996/+0.22353 point。
soft−hardはseed0 +0.02418、seed1 −0.02696、seed2 −0.01653 point、平均−0.00643 point。
この3 head seedではsoft優位は確認できず、concatは実質同程度。h単独回復のseed0結果とは区別する。
pretrainは条件ごとに単一seedのcheckpointで固定。事前学習seed・独立系列での再現性や低活動領域の記憶効果は未検証。
次はbaseline/hard/softを同じ二値マスクでactive/inactive別に評価する提案。通常のval全体指標も併記。

## 21. M3ED activity事前学習：本番完了報告（2026-09-29）

ユーザー提示の `run_m3ed_activity --event-statistics .../event_statistics_train4.json
--stage full --skip-tests` のログで、GPU 0/1/2の全3条件のcomplete表示を確認。
ローカルのランチャー実装ではcompleteは各学習プロセスの終了コード0を意味する。
最終stepのログ、checkpointの存在・ロード、保存config、下流指標は未確認。
smoke/preflightの成功ログは今回提示されていない。

| GPU | 条件 | z蒸留 | h蒸留 |
|---|---|---|---|
| 0 | baseline_e2 | 全領域 | 全領域 |
| 1 | activity_only | activeのみ | inactiveのみ（hard） |
| 2 | active_z_h_soft | activeのみ | inactive=1、active=0.5（soft） |

現行ランチャーの設定は全条件cosine＋正規化二乗L2、GEP RGB、Random clip16、
batch4、seed0、augmentation/dropoutなし、100,000 step。seed/batchは既定値であり、
サーバー上の関数定義・launch.txt・保存configとは未照合。ScaleEventのGram損失は使わない。
trainはcity_hall / horse / penno_big_loop / penno_small_loopの4系列、
validationはucity_small_loop（いずれも `car_urban_day_` 接頭辞）。test不使用。
正規化統計の指定先は
`/home/iASL/Arata_repo/dataset/m3ed_cache/half_dagr/event_statistics_train4.json`。
現行ランチャーはtrain4由来を検査し、統計を出力ルートへコピーする。

確認済み出力ルート:
`/home/iASL/Arata_repo/EventState/outputs/m3ed_activity_full_20260927_192110`

設定上の保存先:

- ログ: `<出力ルート>/logs/<条件名>.log`
- 最終重み: `<出力ルート>/<条件名>/checkpoints/step_00100000.pt`（存在未確認）
- 起動情報・統計: `<出力ルート>/launch.txt`、`<出力ルート>/event_statistics.json`

今回の完了は事前学習であり、下流Semantic mIoUや低活動区間での改善を示すものではない。
旧M3ED h-only/Hybrid結果とは蒸留枝・sampling・正規化が異なり得るため、今回の3条件を
同じ最終step・下流条件で比較する。次は最終ログ・重みを確認し、新しい重みから特徴を抽出して
Frozen Semantic validationを行う提案。その後、低活動subsetとcontinuous/resetの対照で
状態保持を検証する。下流評価の起動は今回報告されていない。

### 最終ログ確認とHybrid継続方針（2026-09-29追記）

後続のユーザー提供ログで全条件100,000 stepと最終重み各277Mの存在を確認。
ロード・下流評価は未実施。launch.txtでseed0、batch4、max_steps100000、GPU0/1/2、
revision `af1b19b46b42b6d1bac1479990adb34bd6043578` を確認。保存configは未照合。
split・出力パスは本節の通り。以下は最終stepの値であり全期間平均ではない。

| 条件 | train loss | train h loss | train z loss | grad norm | val loss | val h projected cosine | val z projected cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_e2 | 0.13234 | 0.063316 | 0.069024 | 0.2718 | 1.1097 | 0.81284 | 0.81727 |
| activity_only | 0.12911 | 0.050691 | 0.078415 | 0.25835 | 0.87647 | 0.81772 | 0.83322 |
| active_z_h_soft | 0.15159 | 0.070427 | 0.081165 | 0.45117 | 0.88558 | 0.81536 | 0.8273 |

最終stepのloss・勾配は有限、skip=0。ただし提示末尾にはbaseline step99343、soft step99761の
train_overflowが各1件ある。全期間のskip数は未集計。現行実装ではoverflow時にglobal_stepを進めない。
最終trainのactive率はhard=0.55755、soft=0.26628、valは両方0.19479。
最終trainのevent_countはbaseline/hard=116850、soft=47223であり同一batch比較ではない。
対象領域・重みが異なるためval lossの大小を性能順位として扱わない。
val低event群は534 frames、low_max=2877（中群533、高群533）。
低群projected cosine（z/h）はbaseline=0.77346/0.76687、hard=0.79415/0.77293、
soft=0.78584/0.77009。教師との整合の診断値であり長期記憶・下流精度の証明ではない。

ユーザーは次にHybridでも学習する方針を表明。Randomの3条件を保持し、対応する
baseline/hard/softをRandom2＋Stream2のHybridで新規学習する準備を追加。未起動。

## 22. M3ED Hybrid activity完了報告とSemantic移行（2026-09-30）

ユーザー提示の `--sampling hybrid --gpus 0,1,2 --stage full --skip-tests` ログで
baseline_e2 / activity_only / active_z_h_softの全complete表示を確認。
出力: `/home/iASL/Arata_repo/EventState/outputs/m3ed_activity_hybrid_full_20260929_131030`。
設定上はseed0、train4/validation1（§21と同じ）、Random2＋Stream2、clip16、100k step、
augmentation/dropoutなし、train4正規化統計。保存launch/configは未受領。
Hybridの最終stepログ・checkpoint存在とロード・学習指標は未確認。正常終了報告と区別する。
設定上の最終重みは `<出力>/<条件>/checkpoints/step_00100000.pt`、ログは `logs/<条件>.log`。

ユーザーはRandom/Hybridの完了を受けSemantic Segmentationへ移行する意向。
`tools/run_m3ed_semantic_activity_comparison.sh` を準備。全6 checkpointの存在を起動前に確認し、
Randomの3条件をGPU0/1/2で実行→全完了後Hybridの3条件を実行。各GPUは1ジョブ。
各モデルのtrain/validationでcontinuous z/hを一度抽出し、Frozen Linear+CEのh/z/concatを
順次学習する（計18 head）。seed0、50 epoch、batch8、FP16、5 epochごとval選択。
既存4系列train / ucity_small_loop validation、InternImage疑似ラベル11クラス。test不使用。

予定出力: `/home/iASL/Arata_repo/EventState/outputs/m3ed_semantic_activity_random_hybrid_20260930/<random|hybrid>/<条件>/<h|z|concat>/seed_0/validation_metrics.json`。
予定cache: `/home/iASL/Arata_repo/dataset/m3ed_cache/semantic_features/activity_random_hybrid_20260930/<random|hybrid>/<条件>`。
新しい専用cacheへ保存し、既存cacheは保持。実行前に空き容量を確認する。
下流起動・結果は未報告。Macでは構文とdry-run（抽出12コマンド・head18コマンド）を確認。

特徴抽出の修正: 旧exporterはtrain抽出時に事前学習val設定を書き換えてからruntimeを作り、
M3ED activityの固定split検査に拒否される。checkpointの元splitを検証・ロードした後に、
独立した抽出configで対象roleのloaderを作るよう変更した。事前学習split検査は維持。
ML依存の実行検証はサーバーで未実施。

### 下流初回起動の失敗（2026-09-30）

ユーザーが上記ランチャーを実行し、Randomのbaseline_e2 / activity_only /
active_z_h_softの全3ジョブでfailed表示。Hybrid下流へは進んでいない。
直前のdfは/home 3.5T、使用3.2T、空き68G、使用率98%。ユーザーは容量不足を推測しているが、
各ログの例外・停止時点・停止後容量は未受領のため原因未確定。条件・seed・split・保存先は本節通り。
新規特徴cacheの作成量とhead開始有無も未確認。次は各logs/random_<条件>.log末尾、
df -h / df -i、特徴cacheのduを確認する。既存M3ED入力・教師・target cacheは保持する。
現ランチャーは既存出力/cacheルートへの再実行を拒否するので、そのまま再起動しない。
exporterには同一checkpointの保存済み特徴を保持する処理があるが、continuous状態再構築のため
系列のforwardは先頭から必要。原因と容量を確認後、既存cacheを活かす再開手順を用意する。

後続ログで特徴保存中のtorch.save iostream error / unexpected posを受領。
/homeは空き831M・100%使用、inodeは1%使用。新規Random cacheは各条件23G、合計68G。
ディスク容量枯渇による保存失敗と判断。各条件の抽出完了率・head開始有無は未確認。
既存cacheは削除せず、次は他filesystemの空きと事前学習checkpoint群の容量を確認する。
削除・移動は未実施。ランチャーへ --resume-cache を追加：保存済み特徴の再利用とログ追記を許可、
既存head出力があれば上書きせず拒否。headの学習再開を行うオプションではない。
shell構文とresume-cache dry-runを確認。容量確保前には再実行しない。

### 再開後の全6モデル完了（2026-10-01）

ユーザーが `bash tools/run_m3ed_semantic_activity_comparison.sh --resume-cache` を実行し、
Randomのbaseline_e2/activity_only/active_z_h_soft、続いてHybridの同3条件の全complete表示を受領。
現行ランチャーでは各モデルのtrain/validation特徴抽出とh/z/concat学習が成功した後にcompleteとなるため、
計18 headの学習・validation評価完了報告として扱う。各結果JSONの実在・数値はまだ未受領。
条件は本節通りFrozen Linear+CE、seed0、50 epoch、batch8、FP16、continuous特徴、
train4/validation1、InternImage疑似ラベル11クラス。test評価ではない。
確認済み出力ルート: `/home/iASL/Arata_repo/EventState/outputs/m3ed_semantic_activity_random_hybrid_20260930`。
指標の保存先: `<ルート>/<random|hybrid>/<条件>/<h|z|concat>/seed_0/validation_metrics.json`。
次は18件のmIoU・クラス別IoU・選択epoch・評価画素数を受領し、同じfeatureでRandom/Hybridと
baseline/hard/softを比較する。checkpoint削除の実施件数と再開後の空き容量は未確認。

## 23. M3ED Semantic：Random/Hybrid × baseline/hard/soft（2026-10-01）

ユーザーから18件のvalidation JSONを受領。§22のFrozen Linear+CE、head seed0、
事前学習seed0・最終100k、50 epoch、batch8、FP16、continuous特徴。
trainはcar_urban_day_city_hall / horse / penno_big_loop / penno_small_loop、
validationはcar_urban_day_ucity_small_loop。InternImage疑似ラベル11クラス、test不使用。
全18件のrole=validation、evaluated_pixels=3,218,555,646が一致。
class_IoUの11クラス平均とmIoUが一致することもstdlibで確認した。
JSONのepochは0始まりなので下表のbest epochは+1した実epoch数。

出力: `/home/iASL/Arata_repo/EventState/outputs/m3ed_semantic_activity_random_hybrid_20260930/<sampling>/<条件>/<feature>/seed_0/validation_metrics.json`。
checkpoint・cacheのパスは§22参照。

| sampling | 条件 | feature | mIoU | Pixel accuracy | Mean class accuracy | best epoch (1始まり) |
|---|---|---|---:|---:|---:|---:|
| random | baseline_e2 | h | 0.346233425154836 | 0.7131907940919907 | 0.45464351862291985 | 50 |
| random | baseline_e2 | z | 0.35464504749873754 | 0.714333222064168 | 0.4617298878760465 | 20 |
| random | baseline_e2 | concat | 0.35220612553045777 | 0.7131635054539617 | 0.46104075521978216 | 15 |
| random | activity_only | h | 0.35843235233754467 | 0.7244152577873435 | 0.46271725022930305 | 50 |
| random | activity_only | z | 0.3623875334485287 | 0.7240797644422644 | 0.4663442039128403 | 10 |
| random | activity_only | concat | 0.3630304787322097 | 0.725604868414321 | 0.46710889840620945 | 5 |
| random | active_z_h_soft | h | 0.3529992579536549 | 0.7227853574292373 | 0.4562809751513182 | 50 |
| random | active_z_h_soft | z | 0.3597273822907423 | 0.7209223043521678 | 0.4672949001212505 | 20 |
| random | active_z_h_soft | concat | 0.35950239313266835 | 0.7228010809417585 | 0.46374350898190336 | 5 |
| hybrid | baseline_e2 | h | 0.34972301880105966 | 0.7145226032857597 | 0.45949114548780884 | 30 |
| hybrid | baseline_e2 | z | 0.3566048867887892 | 0.7151048722306291 | 0.46263251155021623 | 15 |
| hybrid | baseline_e2 | concat | 0.3551018601828252 | 0.715984547249925 | 0.46042584508228257 | 5 |
| hybrid | activity_only | h | 0.3594215297552264 | 0.7265497422442265 | 0.46120961265837934 | 50 |
| hybrid | activity_only | z | 0.3654400596307395 | 0.7282230673603212 | 0.4709193852879444 | 15 |
| hybrid | activity_only | concat | 0.36507168731407946 | 0.7291186749921428 | 0.46791695385609233 | 5 |
| hybrid | active_z_h_soft | h | 0.36082106313670104 | 0.7265439850033899 | 0.46371289246468805 | 50 |
| hybrid | active_z_h_soft | z | 0.36992754794408944 | 0.7299022615686663 | 0.47552601951149853 | 15 |
| hybrid | active_z_h_soft | concat | 0.36788638905939036 | 0.7301374673824732 | 0.4709178166416849 | 5 |

クラス順: background, building, fence, person, pole, road, sidewalk, vegetation, car, wall, traffic_sign。元JSON精度のclass_IoU:

- random/baseline_e2/h: `[0.5539377006830357, 0.5897763721033066, 0.05424547945109671, 0.32599924838571204, 0.12782696156369946, 0.6867215409703079, 0.2922102856582706, 0.6387768479942117, 0.45885846341138026, 0.015204766301899129, 0.06501001018027616]`
- random/baseline_e2/z: `[0.5480802787158376, 0.5859568189060569, 0.06344474653533555, 0.37472218877941055, 0.13989059837591128, 0.6841349204092796, 0.30200389580980563, 0.6382516238529438, 0.4588627793627046, 0.01778675601664331, 0.08796091572218435]`
- random/baseline_e2/concat: `[0.548146091667829, 0.5866252990052059, 0.06204123194596753, 0.36670577602502924, 0.13270949132601653, 0.6836829691668437, 0.29565365107594227, 0.6401730018827434, 0.4614285231762107, 0.010601380055184514, 0.08649996550806294]`
- random/activity_only/h: `[0.5651307935872888, 0.5921078174013648, 0.05107928821652886, 0.35299525376359464, 0.13710485894437968, 0.7003652334604398, 0.3316668549099134, 0.6428378889025528, 0.49269755647308977, 0.008529018667799494, 0.06824131138603938]`
- random/activity_only/z: `[0.5535938451219542, 0.5927476304335791, 0.05766952410569487, 0.3864373647775539, 0.13864324907853764, 0.6996482253498063, 0.32837169628708357, 0.6462731599034652, 0.4847796295182717, 0.014351296456696134, 0.08374724690117273]`
- random/activity_only/concat: `[0.5613181861891857, 0.592272658958151, 0.052806227988518625, 0.38153356624375767, 0.13686800186872647, 0.701312972348649, 0.3304061705632138, 0.6486881316420103, 0.4907384761438781, 0.017115947805572927, 0.08027492630264353]`
- random/active_z_h_soft/h: `[0.5587502882824038, 0.6002324314076458, 0.05033806480958254, 0.3377227810063309, 0.1329029600145955, 0.6846840883379565, 0.3037598587306056, 0.6529998328965232, 0.47500022227664773, 0.015332199466984976, 0.07126911026092748]`
- random/active_z_h_soft/z: `[0.5516596036495389, 0.5907753790769248, 0.06401215407238085, 0.3669787629411692, 0.1446352027184064, 0.687683495328123, 0.32081979104712033, 0.6484247042107926, 0.4739085221115424, 0.015905987877901432, 0.09219760216426535]`
- random/active_z_h_soft/concat: `[0.556782894026568, 0.5952019029774878, 0.053917308724475574, 0.3687337746077054, 0.14117630140337872, 0.685946357402928, 0.3168173955807512, 0.6518035716303907, 0.4739783296632421, 0.018695445929922515, 0.09147304251250209]`
- hybrid/baseline_e2/h: `[0.5477430358942722, 0.5863837111511526, 0.05555806729413254, 0.33052018870311584, 0.13441424968502877, 0.684322820150576, 0.2991202757649396, 0.6394673426324576, 0.46984440384104575, 0.028002994789491167, 0.07157611690544437]`
- hybrid/baseline_e2/z: `[0.54728283978749, 0.5799374510718395, 0.062038178526078784, 0.38223280376291324, 0.14422105921101736, 0.6828414946854603, 0.3088036775602149, 0.6404494641631011, 0.4610383496430148, 0.020796591292670173, 0.09301184497288124]`
- hybrid/baseline_e2/concat: `[0.5475944374445831, 0.5826887520340491, 0.055463693065063865, 0.3738684861251969, 0.1418334327403245, 0.6821029723607834, 0.30394910609617154, 0.6428921164565222, 0.4646060229657306, 0.02390454814256078, 0.08721689458009083]`
- hybrid/activity_only/h: `[0.5494887095625839, 0.5919384271889626, 0.052846023675947644, 0.3565187055795304, 0.13501260033639514, 0.7104470259261206, 0.3453403660944058, 0.6443926797229256, 0.48671256243348754, 0.01502523776680396, 0.0659144890203273]`
- hybrid/activity_only/z: `[0.5509055698721503, 0.5937216146537315, 0.05971810219453367, 0.3751301724049805, 0.1458486759129526, 0.7108916640851404, 0.3507425094734795, 0.6502311920540854, 0.4820039581712898, 0.01574561239738267, 0.08490158471840872]`
- hybrid/activity_only/concat: `[0.5515155680092113, 0.5920571474193055, 0.05323305011684323, 0.3774018209251854, 0.1402997211366461, 0.713799441463051, 0.35565168012431536, 0.6489771457530416, 0.48579207613725345, 0.01871024049395662, 0.07835066887606448]`
- hybrid/active_z_h_soft/h: `[0.5650033715437996, 0.5816214524030683, 0.056417931128455785, 0.35280571037346964, 0.12902480483008436, 0.7043750066268153, 0.3547243446993504, 0.6470849739041322, 0.5063422959938111, 0.00973245833431458, 0.061899344666409936]`
- hybrid/active_z_h_soft/z: `[0.5514619646550966, 0.5868956075097722, 0.06343896587948546, 0.38359465166650253, 0.14483992414160146, 0.7128104304232715, 0.37957223096114645, 0.6493021467391482, 0.491040350132392, 0.013985741605623134, 0.09226101367094396]`
- hybrid/active_z_h_soft/concat: `[0.5583790941905197, 0.5840368573032295, 0.056721471623028735, 0.37981579533576654, 0.13975196021895578, 0.7093309272032544, 0.372072123077503, 0.6516862066612522, 0.4984322934056045, 0.015865694838222335, 0.08065785579595702]`

解釈（単一pretrain/head seed・単一validation系列）:

- Hybridは対応するRandomを9比較すべてで上回った。h/z/concatの改善幅は
  baseline +0.34896/+0.19598/+0.28957、hard +0.09892/+0.30525/+0.20412、
  soft +0.78218/+1.02002/+0.83840 point。因子間の独立反復ではなく有意差未検証。
- 全18件の最高はHybrid soft z=36.99275%。同じHybrid baseline zに対し+1.33227 point、
  Hybrid hard zに対し+0.44875 point。Randomではhardが各featureでsoftを上回るが、Hybridでは逆転。
- hは全6モデルでz未満。concatがzを上回るのはRandom hardのみ（+0.06429 point）。
  共有encoderへの学習効果の可能性はあるが、hからの追加情報・長期記憶効果は未証明。
- Hybrid soft zのbaseline比はsidewalk +7.07686、road +2.99689、car +3.00020 pointが大きい。
  wallとtraffic_signは低下。全体改善から低activity改善を推定しない。
- hの6条件中5条件はbest epoch50、concatは6条件中5条件で最初の検証epoch5。
  headごとに学習曲線が異なる可能性があり、終点の改善継続や過学習はmetrics.jsonlで確認する。
- 旧M3ED h-only/Hybrid/dropout/CE+Dice結果とは条件・正規化が異なり得るため直接順位付けしない。

次の提案は既存validationの低活動subset評価とcontinuous/reset対照。
同じh/headと同じ評価フレームで状態だけを変える。resetは特徴再計算が必要。
全体精度の再現性にはhead seed1/2を追加し、pretrain seed検証とは区別する。
追加評価・学習は未起動。

## 24. Hybrid蒸留対象4条件の起動失敗報告（2026-10-01）

- コマンド: `python tools/run_hybrid_target_ablation.py --stage full --skip-tests`。
- 出力: `/home/iASL/Arata_repo/EventState/outputs/hybrid_targets_full_20261001_230705`。
- GPU0 DSEC z-only、GPU1 DSEC z+h、GPU2 M3ED z-onlyの起動表示後、
  ランチャーexecute内で子プロセスの非ゼロ終了によるRuntimeErrorに到達。
  最終例外本文・子プロセスログは未受領で、失敗条件・原因・到達stepは不明。
- 起動スクリプトの既定計画: seed0、Hybrid、augmentation/dropoutなし、
  DSEC train41・validation無効、M3ED train4/validation1。
  実行時launch.json・保存configは未確認。M3ED h-onlyの起動表示は未受領。
- 1条件の失敗時にランチャーは残りの実行中プロセスを終了させる。
  成功結果や精度の報告ではない。次はlogs/*.log末尾の取得・原因確認。

追加ログによる原因確定:
- DSEC z+hはHydraの `Could not override 'dataset.activity_mask'` で設定合成に失敗。
  dsec_det_train41とh_distill_lstm_zlossには同キーがなく、通常overrideが拒否された。
- DSEC z-onlyはtrain41のthun_00_aのcache検証、M3ED z-onlyはruntime検証開始まで確認。
  学習step・checkpoint生成は未確認。残りプロセスはランチャーの失敗時停止対象。
- 起動指定を `++dataset.activity_mask=false` に修正。既存キーの上書きと未定義キーの
  追加を両立させ、他条件と同じactivity無効を維持する。容量不足を示すログではない。
- サーバーでpreflight付きsmokeを再実行し、成功後fullを新規出力へ実行する。
  修正後のサーバー実学習は未確認。

## 25. Hybrid蒸留対象4条件のfull正常終了報告（2026-10-04受領）

- コマンド: `python tools/run_hybrid_target_ablation.py --stage full --skip-tests`。
- 出力: `/home/iASL/Arata_repo/EventState/outputs/hybrid_targets_full_20261001_231617`。
- DSEC z-only/z+h、M3ED z-only/h-onlyの全completeおよびall completeをユーザー提供ログで確認。
  GPU0=DSEC z-only、GPU1=DSEC z+h、GPU2=M3ED z-only→h-onlyの順次実行。
- ランチャー既定条件: seed0、Hybrid、全域蒸留、augmentation/dropoutなし、100k step。
  DSEC train41・validation無効、batch8/clip8、M3ED train4/validation1・batch4/clip16。
  実保存config/launch.jsonは未受領であり、条件は起動コードに基づく。
- プロセス正常終了の確認であり、最終step・checkpointロード・lossの有限性・下流精度は未確認。
  次は最終重みと容量を確認し、準備済み下流shで今回4モデルの12 headを学習・validation評価。
  下流実行報告はまだない。230705 runのHydra失敗とは別run。

追加のls/df報告（2026-10-04）:
- 全4条件の`checkpoints/step_00100000.pt`存在を確認。
  DSEC z+h=280M、DSEC z-only=265M、M3ED h-only=275M、M3ED z-only=265M（ls -lh表示）。
  ファイル名と存在の確認であり、内部step・Torchロード・lossの確認ではない。
- `/dev/sdb1` (/home): 3.5T中3.2T使用、空き91G、使用率98%。
  下流cacheのピーク必要量は未測定。現ランチャーはDSECのSemanticとDetection変換前後の
  cacheを同一モデルの全head成功まで保持するため、容量が足りるとはまだ判断しない。

追加du報告: DSEC_cache計1.2T（教師410G、events275G、Semantic特徴112G、
Detection特徴activity_20260923_h 372G）。M3ED Semantic特徴計323G、
うちactivity_random_hybrid_20260930が258G（random/hybrid各129G）、
旧hybrid_dropout/hybrid_no_dropout/random_no_dropoutが各22G。
評価済みM3ED activity特徴258Gを整理候補とするが、削除・容量回収は未確認。
DSEC Detectionの単一条件容量は未確認。入力・教師・target・結果・checkpointは保持する。

Detection内訳の追加確認: baseline_e2/activity_only/scale_event_fullは各124G、
それぞれrectified約64G＋dsec_det約61G（du丸め表示）。hのみの既存cacheを基準に、
同じ特徴次元・対象frame数なら今回z+hのDetectionは約248–250Gと概算する。
Semantic特徴・head重み等を加味し、開始時空き330G以上を運用上の目安とする
（厳密な必要量や上限ではない）。M3ED特徴258G削除後の想定空き349Gなら収まる見込みだが、
削除実績・現在dfは未受領。他ジョブによる容量増加もこの概算に含まない。

続報df: `/dev/sdb1`は3.5T中3.0T使用、空き348G、使用率90%を確認。
容量回復は確認済み。削除対象自体の一覧は未受領。
前記概算では下流12 headを開始可能な範囲。下流起動・実ピーク容量・完了は未確認。

## 26. Hybrid蒸留対象4モデルの下流評価完了（2026-10-06受領）

- 事前学習: `outputs/hybrid_targets_full_20261001_231617`。
- 出力: `/home/iASL/Arata_repo/EventState/outputs/hybrid_target_downstream_20261004`。
- 実行: `tools/run_hybrid_target_downstream.sh`、追加比較モデルのオプションなし。
  専用cache: `/home/iASL/Arata_repo/dataset/downstream_scratch/hybrid_targets_20261004`。
- DSEC z-only: Semantic/Detection各z（2 head）、DSEC z+h: 各z/h/concat（6 head）、
  M3ED z-only: Semantic z（1 head）、M3ED h-only: Semantic z/h/concat（3 head）。
  全12 head、4モデルのcomplete and cache removed、およびall completeを提供ログで確認。
- スクリプト条件: Frozen、head seed0、50 epoch、FP16。Semantic Linear+CE/batch8、
  Detection dsec-det/batch16。DSEC Semantic train6/val2、Detection train41/val6、
  M3ED train4/validation1、test不使用。実保存configは未受領。
- 完了ログはランチャー所定のbest/last checkpoint ZIP CRC、有限なvalidation指標等の検査と
  モデル単位の専用cache削除を通過したことを示す。サーバー成果物の直接検証は未実施。
- 精度JSONは未受領で、mAP/mIoU・best epoch・条件間優劣は未判定。
  次は`<model>/<task>/<feature>/seed_0/validation_metrics.json`全12件を取得する。
  既存DSEC h-onlyとM3ED z+hの9 headは今回の実行に含まれない。


## 27. Hybrid蒸留対象×下流入力の評価数値（2026-10-06）

§26の全12 validation JSONをユーザー添付から受領。
元の全桁数・AP内訳・クラスIoUを含むJSONは
[保存した評価JSON](evaluation_records/hybrid_target_downstream_20261004_metrics.json)に保存。
サーバー出力ルート: `/home/iASL/Arata_repo/EventState/outputs/hybrid_target_downstream_20261004`。
各JSON: `<model>/<task>/<feature>/seed_0/validation_metrics.json`。
事前学習seed0/Hybrid/no augmentation/no dropout、Frozen head seed0・50 epoch、
Semantic Linear+CE/batch8、Detection dsec-det/batch16、FP16（§26の起動設定）。
DSEC Semantic train6/val2、Detection train41/val6、M3ED train4/validation1。
実保存config未照合。test値と混ぜない。

Semantic全8件の11クラスIoU平均とmIoUが一致（誤差1e-12未満）。
DSEC Semanticの評価pixelsは全4件633,036,800、M3EDは全4件3,218,555,646。
Detection全4件はrole=val/protocol=dsec-det/coordinate_space=dsec_det_distorted。
値は百分率、best epochはJSONの0始まりに1を加算。DSEC JSONにepochはない。

| モデル | タスク | 下流入力 | mAPまたはmIoU (%) | best epoch |
|---|---|---|---:|---:|
| dsec_z_h | detection | concat | 37.06807168 | 未記載 |
| dsec_z_h | detection | h | 35.55525783 | 未記載 |
| dsec_z_h | detection | z | 34.57714747 | 未記載 |
| dsec_z_h | semantic | concat | 60.09713701 | 未記載 |
| dsec_z_h | semantic | h | 59.39195553 | 未記載 |
| dsec_z_h | semantic | z | 59.70888412 | 未記載 |
| dsec_z_only | detection | z | 34.97486703 | 未記載 |
| dsec_z_only | semantic | z | 59.55236888 | 未記載 |
| m3ed_h_only | semantic | concat | 36.38302655 | 15 |
| m3ed_h_only | semantic | h | 36.17935854 | 50 |
| m3ed_h_only | semantic | z | 36.09033648 | 20 |
| m3ed_z_only | semantic | z | 34.85107752 | 15 |

M3EDの全域z+h既存Hybrid baseline_e2（§23）を加えた基本表:

| 蒸留対象 | z mIoU (%) | h mIoU (%) | concat mIoU (%) |
|---|---:|---:|---:|
| z-only | 34.85108 | 対象外 | 対象外 |
| h-only | 36.09034 | 36.17936 | 36.38303 |
| z+h（既存§23） | 35.66049 | 34.97230 | 35.51019 |

既存z+hは同じtrain4/validation1、評価pixels、Hybrid、seed0、最終100k、Linear+CE、
50epoch/batch8/FP16/continuousの記録。別runの再利用であり保存configの完全照合は未実施。
DSEC h-onlyは今回未評価、既存の同val条件の指標を取得・照合するまで欠測扱い。

解釈（差はpercentage point）:
- M3ED h-onlyのzはz-onlyのz比+1.23926。hだけの蒸留でも共有encoderのzに有用な特徴が
  学習されることと整合的。単一seedなので再現性・有意差は未検証。
- 同じh-only内のh−zは+0.08902、concat−zは+0.29269。z側の改善に比べ小さく、
  長期状態保持の寄与そのものはcontinuous/reset対照などで検証が必要。
- M3ED h-onlyは既存全域z+hを同じ入力の3比較すべてで上回る。
  activity soft z=36.99275（§23）は今回h-only最高concat=36.38303を+0.60973上回り、
  同じz入力でもh-only z比+0.90242。activity soft hは今回h-only hより低い。
  全域z+h蒸留が常に最良という結論にはならない。
- DSEC Detection: z+h concat=37.06807、z-only z=34.97487との差+2.09320。
  z+hのz単独は34.57715でz-only比−0.39772。concatはz+hのh比+1.51281。
- DSEC Semantic: z+h concat=60.09714で今回最高、z-only z比+0.54477。
  z+h zはz-only z比+0.15652、同モデルhはzより低い。
- concatは入力次元とheadパラメータ数が増える。concatの利得をそのまま長期記憶の効果と呼ばない。
  dataset/taskごとに適した蒸留対象・下流入力が異なる兆候であり、普遍的優位は主張しない。

次の優先事項: DSEC既存Hybrid h-onlyのcheckpoint/設定/validation結果を照合して
両タスクのz/h/concatの不足欄を埋める。単一seedの小差を踏まえて有力条件のseed追加を検討。
activity重みablationは別軸として保留、今回数値から自動で追加学習を開始しない。

## 28. DSEC Hybrid h-only既存重みの比較条件確認（2026-10-06）

ユーザー提示のlsと.hydra/config.yamlから、
`outputs/dsec_hybrid_aug/hybrid_noaug_cache/checkpoints/step_00100000.pt`（276M）の存在を確認。
seed0、h損失cosine/MSE各1、z目的なし、mixed Random4+Stream4・等重み、batch8/clip8、
100k、FP16、gradient accumulation1、augmentation/event dropout無効、LSTM dropout0。
DSEC train41の系列リスト・GEP正規化・448x640・DINO教師/cache・384次元LSTMと
optimizer AdamW lr1e-4/weight decay0.05/encoder lr mult0.1・warmup1000は
今回の新規DSEC比較条件のローカル起動設定と一致。
保存周期は旧1000、新10000で異なる。学習コードrevision・重み内部stepの一致は未検証。
dfで空き348G・使用率90%を再確認。新規事前学習はせず同重みをFrozen評価に再利用する。
予定: `--only dsec_h_only --dsec-h-checkpoint <上記重み>`でSemantic/Detection各z/h/concat、
seed0、同じvalidation条件の6 headを実行。
出力: `outputs/hybrid_h_only_downstream_20261006`、専用cache:
`/home/iASL/Arata_repo/dataset/downstream_scratch/hybrid_h_only_20261006`。
既存4モデルの下流runと分離。実行・結果はまだ未報告。

## 29. DSEC Hybrid h-only追加6 headの評価完了（2026-10-08受領）

- ユーザー実行: `tools/run_hybrid_target_downstream.sh --only dsec_h_only`、
  事前学習重みは§28の`outputs/dsec_hybrid_aug/hybrid_noaug_cache/checkpoints/step_00100000.pt`。
- 出力: `/home/iASL/Arata_repo/EventState/outputs/hybrid_h_only_downstream_20261006`。
  専用cache: `/home/iASL/Arata_repo/dataset/downstream_scratch/hybrid_h_only_20261006`。
- DSEC Semantic/Detection各z/h/concatの学習・validation評価工程、
  `complete and cache removed: dsec_h_only`と`all complete`を提供ログで確認。
  ランチャー所定の結果検査と専用モデルcache削除を通過。成果物の直接検証は未実施。
- 設定: Frozen、head seed0、50epoch、FP16、Semantic Linear+CE/batch8/train6-val2、
  Detection dsec-det/batch16/train41-val6。事前学習seed0/Hybrid4+4/clip8/no aug/no dropout。
  testは実行しない。評価JSONの数値は未受領で、mIoU/mAP・優劣は未判定。
- 次は`dsec_h_only/<semantic|detection>/<z|h|concat>/seed_0/validation_metrics.json`
  全6件を取得して§27のDSEC両タスクの表を補完する。
