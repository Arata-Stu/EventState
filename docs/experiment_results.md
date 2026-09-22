# EventState 実験結果台帳

最終更新: 2026-09-22

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
