# Patchwise LSTM対ConvLSTM：Semantic・Depth比較結果

2026-11-18受領。今回のStage 1と2dataset×2taskの学習・評価実行、および全体・活動5分位の数値回収が揃った。

## 主比較の全体結果

比較条件はE1 Encoder＋LSTM（h蒸留）・Patchwise LSTM対E1 Encoder＋LSTM（h蒸留）・ConvLSTM。seed0、既存のdataset/task内recipe・split・GT・共通活動manifestを維持する。

| dataset/task | 指標 | E1 Encoder＋LSTM（h蒸留）・Patchwise LSTM | E1 Encoder＋LSTM（h蒸留）・ConvLSTM | ConvLSTMの変化 |
| --- | --- | ---: | ---: | ---: |
| DSEC Semantic | mIoU[%] ↑ | 60.482540 | 60.594520 | +0.111980pt |
| M3ED Semantic | mIoU[%] ↑ | 38.700944 | 38.601202 | -0.099742pt |
| DSEC Depth | pooled MAE[m] ↓ | 2.445824 | 2.487786 | 誤差+1.715653% |
| M3ED Depth | pooled MAE[m] ↓ | 3.678082 | 3.318846 | 誤差-9.766929% |

SemanticはDSEC微増/M3ED微減、DepthはDSEC悪化/M3ED改善とdataset/task依存。全条件に共通する勝者とは結論しない。
SemanticはDSEC公式test3、M3ED公開val2（InternImage疑似11class）。DepthはDSEC公開val4、M3ED公開val2。
異なるdataset/taskのスコアを平均して総合点を作らない。

## Depth全体の指標

変化率は `(ConvLSTM / Patchwise − 1) × 100`。誤差指標は負が改善、deltaは高い方がよい。

### DSEC Depth

| 指標 | E1 Encoder＋LSTM（h蒸留）・Patchwise LSTM | E1 Encoder＋LSTM（h蒸留）・ConvLSTM | 変化 |
| --- | ---: | ---: | ---: |
| pooled MAE[m] ↓ | 2.445824 | 2.487786 | +1.715653% |
| pooled RMSE[m] ↓ | 4.088795 | 4.162017 | +1.790790% |
| AbsRel ↓ | 0.104564 | 0.106697 | +2.040084% |
| pooled SqRel ↓ | 0.607859 | 0.634274 | +4.345580% |
| RMSElog ↓ | 0.155085 | 0.157225 | +1.379474% |
| delta1[%] ↑ | 88.145181 | 87.612788 | -0.532394pt |
| delta2[%] ↑ | 98.010433 | 97.953140 | -0.057293pt |
| delta3[%] ↑ | 99.549667 | 99.541956 | -0.007712pt |
| frame平均 MAE[m] ↓ | 2.590480 | 2.641458 | +1.967870% |
| frame平均 RMSE[m] ↓ | 4.145879 | 4.218360 | +1.748265% |
| frame平均 AbsRel ↓ | 0.107302 | 0.109881 | +2.404091% |
| F³式frame平均 SqRel ↓ | 0.028912 | 0.031007 | +7.243139% |
| 公開val視差 MAE[px] ↓ | 1.870813 | 1.888776 | +0.960203% |

### M3ED Depth

| 指標 | E1 Encoder＋LSTM（h蒸留）・Patchwise LSTM | E1 Encoder＋LSTM（h蒸留）・ConvLSTM | 変化 |
| --- | ---: | ---: | ---: |
| pooled MAE[m] ↓ | 3.678082 | 3.318846 | -9.766929% |
| pooled RMSE[m] ↓ | 6.766504 | 6.236755 | -7.828993% |
| AbsRel ↓ | 0.215305 | 0.189963 | -11.770324% |
| pooled SqRel ↓ | 2.323548 | 1.843164 | -20.674624% |
| RMSElog ↓ | 0.290663 | 0.271894 | -6.457353% |
| delta1[%] ↑ | 72.571850 | 75.267108 | +2.695257pt |
| delta2[%] ↑ | 88.414227 | 89.792684 | +1.378457pt |
| delta3[%] ↑ | 95.361440 | 95.941486 | +0.580046pt |
| frame平均 MAE[m] ↓ | 4.121049 | 3.716064 | -9.827234% |
| frame平均 RMSE[m] ↓ | 7.184567 | 6.665799 | -7.220588% |
| frame平均 AbsRel ↓ | 0.251039 | 0.220894 | -12.007931% |
| F³式frame平均 SqRel ↓ | 0.272129 | 0.212782 | -21.808530% |

pooledは全有効画素をまとめた値。frame平均は各frame指標を等重みで平均する。
SqRelはpooled側が誤差²/GT、F³式frame側が誤差²/GT²であり、単なる集計重み違いではない。F³全プロトコルの完全再現という意味ではない。
両ConvLSTM Depth headは50epoch予算内からval pooled MAE最小の8epoch目を採用した（1始まり）。
DSEC視差は公開valの内部評価であり、公式hidden-test serverのスコアではない。

## 活動5分位のDepth MAE

### DSEC：pooled MAE[m]

| 群 | E1 Encoder＋LSTM（h蒸留）・Patchwise LSTM | E1 Encoder＋LSTM（h蒸留）・ConvLSTM | 誤差変化 |
| --- | ---: | ---: | ---: |
| 全体 | 2.445824 | 2.487786 | +1.715653% |
| p00_20 | 2.196312 | 2.246397 | +2.280390% |
| p20_40 | 2.263113 | 2.288793 | +1.134745% |
| p40_60 | 2.413834 | 2.415825 | +0.082482% |
| p60_80 | 2.687870 | 2.777262 | +3.325769% |
| p80_100 | 3.050876 | 3.143762 | +3.044565% |

### DSEC：frame平均 MAE[m]

| 群 | E1 Encoder＋LSTM（h蒸留）・Patchwise LSTM | E1 Encoder＋LSTM（h蒸留）・ConvLSTM | 誤差変化 |
| --- | ---: | ---: | ---: |
| 全体 | 2.590480 | 2.641458 | +1.967870% |
| p00_20 | 2.233590 | 2.277846 | +1.981381% |
| p20_40 | 2.407605 | 2.441883 | +1.423727% |
| p40_60 | 2.551977 | 2.558961 | +0.273655% |
| p60_80 | 2.848041 | 2.945591 | +3.425149% |
| p80_100 | 3.088253 | 3.194104 | +3.427516% |

### M3ED：pooled MAE[m]

| 群 | E1 Encoder＋LSTM（h蒸留）・Patchwise LSTM | E1 Encoder＋LSTM（h蒸留）・ConvLSTM | 誤差変化 |
| --- | ---: | ---: | ---: |
| 全体 | 3.678082 | 3.318846 | -9.766929% |
| p00_20 | 5.119266 | 4.563695 | -10.852559% |
| p20_40 | 3.611621 | 3.437420 | -4.823359% |
| p40_60 | 3.152041 | 2.868305 | -9.001657% |
| p60_80 | 3.026193 | 2.701145 | -10.741158% |
| p80_100 | 2.798025 | 2.502655 | -10.556381% |

### M3ED：frame平均 MAE[m]

| 群 | E1 Encoder＋LSTM（h蒸留）・Patchwise LSTM | E1 Encoder＋LSTM（h蒸留）・ConvLSTM | 誤差変化 |
| --- | ---: | ---: | ---: |
| 全体 | 4.121049 | 3.716064 | -9.827234% |
| p00_20 | 5.336390 | 4.755376 | -10.887784% |
| p20_40 | 3.374407 | 3.210148 | -4.867791% |
| p40_60 | 3.032343 | 2.783192 | -8.216426% |
| p60_80 | 2.935276 | 2.657157 | -9.475068% |
| p80_100 | 2.723778 | 2.487394 | -8.678540% |

DSECは全5群でMAEが悪化、M3EDは全5群でMAEが改善し、この主比較ではpooled/frame平均の両方で同じ方向。

## M3EDの改善と残る留保

M3ED全体では、新ConvLSTMは旧E0/E1/E2/旧DDP/新DDPの全報告指標を上回る。
ただしE0との比較は蒸留先/readoutもz→hへ変わるため、空間・時間モデルだけの効果ではない。
- 全体pooled MAE：旧最良 3.556542m → 3.318846m（誤差-6.683334%）。
- 全体frame平均 MAE：旧最良 3.898916m → 3.716064m（誤差-4.689829%）。

最低活動Q1をE0 Encoder（z蒸留）と比べると、pooled MAEは4.664813→4.563695mで改善するが、frame平均は4.659016→4.755376m（誤差+2.068243%）で依然悪い。
したがって「低活動での問題が全面的に解消した」とはしない。主比較のPatchwise E1に対しては両集計で改善している。

## 比較対象と数値整合の監査

Depth両datasetとも以下6種類の受領hashが既存Patchwise E1と完全一致：

- 活動manifest
- 評価sample
- 活動群割当
- validity support
- sample＋GT
- validity mask

全体と全5群のframes、valid_frames、frame割合、有効画素数、系列数/系列別frame数、event rate min/maxも一致。
群別supportの合計は全体と一致。pooled指標は画素重み（RMSE系は平方してから）、frame平均はframe数重みで再構成でき、浮動小数点誤差内で整合した。
現行ds_f3 protocol digestとも一致。これは受領値・hash文字列の照合であり、サーバーのraw GT/predictionをこちらが独立に再計算したものではない。
旧20件抽出recordにはhead/backbone SHAがないので、新旧モデル重みSHAの差をこの正本だけで照合したとはしない。
Semantic2件は群別support件数まで一致を確認。Semanticのsample/GT/割当hashは未受領。

## 解釈の範囲と次の段階

単一seedの結果で、有意差・統計的同等性・普遍的な方式優位・記憶機序の因果は主張しない。
時間モデル＋出力projectionのparameter数はPatchwise1,330,560、ConvLSTM10,766,208（約8.09倍）。総モデル容量/実測latencyが8倍という意味ではない。
M3ED Depthで追加容量に対応する利益は見られる一方、SemanticとDSEC Depthでは一律の利益は得られなかった。
今回の種類比較の数値整理は完了。次の研究段階は従来計画のLSTM配置比較だが、この結果だけで一つの方式を全taskに最適と決めない。

## 正本と来歴

- Depth受領全量：`docs/evaluation_records/20261118_temporal_depth_convlstm_metrics.json`
- 差分・hash/support/再集計監査：`docs/evaluation_records/20261118_temporal_comparison_audit.json`
- 旧Patchwise正本：`docs/evaluation_records/20261115_semantic_depth_activity_metrics.json`
- Semantic新値：`docs/evaluation_records/20261117_m3ed_semantic_convlstm_activity_best.json`、`20261118_dsec_semantic_convlstm_activity_best.json`
- サーバー新Depth：`outputs/stage2_temporal_remaining_20261117/heads/{dataset}_E1_convlstm/stages/activity_continuous.json`
- サーバーroot成功marker=Trueを今回再確認。raw受領ログはGit非管理で保持する。
- Macはstdlib読取・数値/hash整合確認のみで、追加学習・推論・依存導入は行っていない。
