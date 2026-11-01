# DSEC公開splitの系列重複監査（2026-10-31）

## EventVGGT追加確認（2026-10-31）

- 原論文v2: https://arxiv.org/html/2603.09385v2 （2026-07-08）。
- 公開実装: https://github.com/yinruiRen/EventVGGT 。
- 論文§4.1–4.3の評価対象はEventScape/MVSEC/DENSE。DSEC/M3EDを用いたsplitや評価は提示されていない。
- EventScapeのin-domain実験はtraining Town1→test split。Town1–3の学習データ多様性ablationもある。
  完全なtrain/val/testファイル一覧は公開学習コードが未公開のため今回照合できない。
- MVSEC in-domain: Day2 train、Day1/Night1/Night2/Night3で評価。系列名レベルでtrainと評価は分離。
- zero-shot: EventScapeのみで学習しMVSEC/DENSEへ直接評価。対象イベントdatasetレベルでは分離。
- READMEは公開checkpoint.ptをEventScapeのみで学習したと明記。training/evaluation codeはTODO。
  したがって公開記載に従えば、この重みのstudent学習にDSEC系列は含まれない。
  ただしVGGT教師の元学習データ、実checkpointの全履歴を監査したものではない。
- 公開infer.pyはseq_length枚をまとめて推論し既定seq_length=1。複数frame比較時は、
  LSTMの因果的履歴入力と同じ情報範囲になるよう因果性を別途確認する。
- DSECで比較するなら、既存の公開DSEC評価splitを使う追加実験となり、EventVGGT論文splitの再現とは呼ばない。
  この確認を理由に既存F3 splitを変更/再学習する作業は開始していない。

ユーザー・EventState独自splitは対象外。公式配布と先行研究公開設定を、系列名の完全一致で比較した。
判定対象はtrainと評価集合の系列重複。生フレーム重複、同一収録の切出し、基盤モデル由来の重複、実checkpointの学習履歴は未監査。
別タスクのtrainに評価系列が存在するだけでは、単独実験の漏洩ではない。そのtrainで学習した共有モデルを対象評価に転用するときに混入になる。

## 公開split一覧

| 出典・protocol | train | val | test | trainと評価系列の重複 |
|---|---:|---:|---:|---|
| DSEC_official | 41 | 定義なし | 12 | 0 |
| DSEC_disparity_official | 41 | 定義なし | 12 | 0 |
| DSEC_Det_official | 41 | 6 | 13 | 0 |
| DSEC_Semantic_official | 8 | 定義なし | 3 | 0 |
| DSEC_Flow_official | 18 | 定義なし | 7 | 0 |
| F3_Depth | 37 | 4 | 別途定義なし | 0 |
| DepthAnyEvent_Depth | 31 | 10 | 別途定義なし | 0 |

F3の公開ファイルはtrain/val、DepthAnyEvent公開コードはtrain/validationという名称。独立testを付け足さない。
公式Depthの行はDSEC disparity benchmarkのtrain/test。単眼Depthの各研究が公式train内を再分割する設定と区別する。
Flowは全DSEC train41中、公式forward flow GTが配布されている18系列。公式test対象は7系列のみ。

## 各trainをモデル学習に使った場合の、他protocol評価集合との重複数

| train集合 / 評価集合（val ∪ test） | DSEC_official | DSEC_disparity_official | DSEC_Det_official | DSEC_Semantic_official | DSEC_Flow_official | F3_Depth | DepthAnyEvent_Depth |
|---|---:|---:|---:|---:|---:|---:|---:|
| DSEC_official | 0 | 0 | 0 | 0 | 0 | 4 | 10 |
| DSEC_disparity_official | 0 | 0 | 0 | 0 | 0 | 4 | 10 |
| DSEC_Det_official | 0 | 0 | 0 | 0 | 0 | 4 | 10 |
| DSEC_Semantic_official | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| DSEC_Flow_official | 0 | 0 | 0 | 0 | 0 | 2 | 7 |
| F3_Depth | 0 | 0 | 0 | 0 | 0 | 0 | 9 |
| DepthAnyEvent_Depth | 0 | 0 | 0 | 0 | 0 | 3 | 0 |

公式5 protocolだけなら全交差が0。DepthAnyEvent/F3の評価は元DSEC公式train内にあるため、公式train41事前学習からの転用は別判定。
公式train41 ∩ DepthAnyEvent validation は10系列、公式train41 ∩ F3 val は4系列。
F3 train37 ∩ DepthAnyEvent validation は9系列。両研究のDepth分割を同じものとして扱えない。

## 公開系列リスト（全件）

### DSEC_official

**train**

```text
interlaken_00_c
interlaken_00_d
interlaken_00_e
interlaken_00_f
interlaken_00_g
thun_00_a
zurich_city_00_a
zurich_city_00_b
zurich_city_01_a
zurich_city_01_b
zurich_city_01_c
zurich_city_01_d
zurich_city_01_e
zurich_city_01_f
zurich_city_02_a
zurich_city_02_b
zurich_city_02_c
zurich_city_02_d
zurich_city_02_e
zurich_city_03_a
zurich_city_04_a
zurich_city_04_b
zurich_city_04_c
zurich_city_04_d
zurich_city_04_e
zurich_city_04_f
zurich_city_05_a
zurich_city_05_b
zurich_city_06_a
zurich_city_07_a
zurich_city_08_a
zurich_city_09_a
zurich_city_09_b
zurich_city_09_c
zurich_city_09_d
zurich_city_09_e
zurich_city_10_a
zurich_city_10_b
zurich_city_11_a
zurich_city_11_b
zurich_city_11_c
```

**val**

```text
(defined set unavailable / no separate split)
```

**test**

```text
interlaken_00_a
interlaken_00_b
interlaken_01_a
thun_01_a
thun_01_b
zurich_city_12_a
zurich_city_13_a
zurich_city_13_b
zurich_city_14_a
zurich_city_14_b
zurich_city_14_c
zurich_city_15_a
```

### DSEC_disparity_official

**train**

```text
interlaken_00_c
interlaken_00_d
interlaken_00_e
interlaken_00_f
interlaken_00_g
thun_00_a
zurich_city_00_a
zurich_city_00_b
zurich_city_01_a
zurich_city_01_b
zurich_city_01_c
zurich_city_01_d
zurich_city_01_e
zurich_city_01_f
zurich_city_02_a
zurich_city_02_b
zurich_city_02_c
zurich_city_02_d
zurich_city_02_e
zurich_city_03_a
zurich_city_04_a
zurich_city_04_b
zurich_city_04_c
zurich_city_04_d
zurich_city_04_e
zurich_city_04_f
zurich_city_05_a
zurich_city_05_b
zurich_city_06_a
zurich_city_07_a
zurich_city_08_a
zurich_city_09_a
zurich_city_09_b
zurich_city_09_c
zurich_city_09_d
zurich_city_09_e
zurich_city_10_a
zurich_city_10_b
zurich_city_11_a
zurich_city_11_b
zurich_city_11_c
```

**val**

```text
(defined set unavailable / no separate split)
```

**test**

```text
interlaken_00_a
interlaken_00_b
interlaken_01_a
thun_01_a
thun_01_b
zurich_city_12_a
zurich_city_13_a
zurich_city_13_b
zurich_city_14_a
zurich_city_14_b
zurich_city_14_c
zurich_city_15_a
```

### DSEC_Det_official

**train**

```text
thun_00_a
interlaken_00_c
interlaken_00_d
interlaken_00_e
interlaken_00_f
interlaken_00_g
zurich_city_00_a
zurich_city_00_b
zurich_city_01_a
zurich_city_01_b
zurich_city_01_c
zurich_city_01_d
zurich_city_01_e
zurich_city_01_f
zurich_city_02_a
zurich_city_02_b
zurich_city_02_c
zurich_city_02_d
zurich_city_02_e
zurich_city_03_a
zurich_city_04_a
zurich_city_04_b
zurich_city_04_c
zurich_city_04_d
zurich_city_04_e
zurich_city_04_f
zurich_city_05_a
zurich_city_05_b
zurich_city_06_a
zurich_city_07_a
zurich_city_08_a
zurich_city_09_a
zurich_city_09_b
zurich_city_09_c
zurich_city_09_d
zurich_city_09_e
zurich_city_10_a
zurich_city_10_b
zurich_city_11_a
zurich_city_11_b
zurich_city_11_c
```

**val**

```text
zurich_city_16_a
zurich_city_17_a
zurich_city_18_a
zurich_city_19_a
zurich_city_20_a
zurich_city_21_a
```

**test**

```text
thun_01_a
thun_01_b
thun_02_a
interlaken_00_a
interlaken_00_b
interlaken_01_a
zurich_city_12_a
zurich_city_13_a
zurich_city_13_b
zurich_city_14_a
zurich_city_14_b
zurich_city_14_c
zurich_city_15_a
```

### DSEC_Semantic_official

**train**

```text
zurich_city_00_a
zurich_city_01_a
zurich_city_02_a
zurich_city_04_a
zurich_city_05_a
zurich_city_06_a
zurich_city_07_a
zurich_city_08_a
```

**val**

```text
(defined set unavailable / no separate split)
```

**test**

```text
zurich_city_13_a
zurich_city_14_c
zurich_city_15_a
```

### DSEC_Flow_official

**train**

```text
thun_00_a
zurich_city_01_a
zurich_city_02_a
zurich_city_02_c
zurich_city_02_d
zurich_city_02_e
zurich_city_03_a
zurich_city_05_a
zurich_city_05_b
zurich_city_06_a
zurich_city_07_a
zurich_city_08_a
zurich_city_09_a
zurich_city_10_a
zurich_city_10_b
zurich_city_11_a
zurich_city_11_b
zurich_city_11_c
```

**val**

```text
(defined set unavailable / no separate split)
```

**test**

```text
interlaken_00_b
interlaken_01_a
thun_01_a
thun_01_b
zurich_city_12_a
zurich_city_14_c
zurich_city_15_a
```

### F3_Depth

**train**

```text
interlaken_00_c
interlaken_00_d
interlaken_00_e
interlaken_00_f
thun_00_a
zurich_city_00_a
zurich_city_00_b
zurich_city_01_a
zurich_city_01_b
zurich_city_01_c
zurich_city_01_d
zurich_city_01_e
zurich_city_01_f
zurich_city_02_a
zurich_city_02_b
zurich_city_02_c
zurich_city_02_d
zurich_city_03_a
zurich_city_04_a
zurich_city_04_b
zurich_city_04_c
zurich_city_04_d
zurich_city_04_e
zurich_city_04_f
zurich_city_05_a
zurich_city_05_b
zurich_city_06_a
zurich_city_07_a
zurich_city_08_a
zurich_city_09_a
zurich_city_09_b
zurich_city_09_c
zurich_city_09_d
zurich_city_10_a
zurich_city_10_b
zurich_city_11_a
zurich_city_11_b
```

**val**

```text
interlaken_00_g
zurich_city_02_e
zurich_city_09_e
zurich_city_11_c
```

**test**

```text
(defined set unavailable / no separate split)
```

### DepthAnyEvent_Depth

**train**

```text
interlaken_00_c
interlaken_00_d
interlaken_00_e
zurich_city_00_a
zurich_city_00_b
zurich_city_01_a
zurich_city_01_b
zurich_city_01_c
zurich_city_01_d
zurich_city_01_e
zurich_city_01_f
zurich_city_02_a
zurich_city_02_b
zurich_city_02_c
zurich_city_02_d
zurich_city_02_e
zurich_city_03_a
zurich_city_04_a
zurich_city_04_b
zurich_city_04_c
zurich_city_04_d
zurich_city_04_e
zurich_city_04_f
zurich_city_09_a
zurich_city_09_b
zurich_city_09_c
zurich_city_09_e
zurich_city_10_a
zurich_city_11_a
zurich_city_11_b
zurich_city_11_c
```

**val**

```text
interlaken_00_f
interlaken_00_g
thun_00_a
zurich_city_05_a
zurich_city_05_b
zurich_city_06_a
zurich_city_07_a
zurich_city_08_a
zurich_city_09_d
zurich_city_10_b
```

**test**

```text
(defined set unavailable / no separate split)
```

## 出典

- official: https://dsec.ifi.uzh.ch/dsec-datasets/download/
- det: https://raw.githubusercontent.com/uzh-rpg/dsec-det/master/config/train_val_test_split.yaml
- flow_test: https://dsec.ifi.uzh.ch/optical-flow-submission-format/
- disparity_test: https://dsec.ifi.uzh.ch/disparity-submission-format/
- f3_train: https://raw.githubusercontent.com/grasp-lyrl/fast-feature-fields/833e2c65233f9787449ce0675ae41e3f6e9c193a/confs/monocular_depth/full_dsec_gt_train.yml ; SHA256=baf0e0fd0d292a90ccb58a5caa85f5e048547cca6de196ba5c0737d9a9c722eb
- f3_val: https://raw.githubusercontent.com/grasp-lyrl/fast-feature-fields/833e2c65233f9787449ce0675ae41e3f6e9c193a/confs/monocular_depth/full_dsec_gt_val.yml ; SHA256=2533b3cf282744599617cace699d91641d227edee4a3271bffee45a5e41ab7d6
- depth_anyevent: https://raw.githubusercontent.com/bartn8/depthanyevent/09247b13e52c2d7050630202a413e6a410a21eb0/dataset/dsec_dataset/constants.py ; SHA256=3ddd12e5bc7440c0b89caffb1c7213575896a249d63b5b4b0865bb20cbba073d

公式サイトのtrain/test見出しとラベル配布リンクから系列を抽出。Detectionは公式YAML、F3/DepthAnyEventは上記固定revisionをHTTPで再取得して照合した。
EventStateのmanifestや独自SemSeg開発6/2、共通事前学習35案は集合算出の入力に使っていない。

## 先行研究の事前学習コードは別問題

ScaleEventの上記固定revisionのdsec.txtは42系列で、元DSEC公式testと10系列重複する。
Dsec→DistillDataset→pretrain_ddpのサンプリング経路ではこれらを選択可能。未見test評価の条件では入力集合重複あり。
ただし実論文重みがこのままのリストで生成されたかは未確認。ラベルありtest学習をしたとの断定ではない。
GEP公開configはDSEC trainをtrain_dataset、DSEC testをvalid_datasetへ設定。testを勾配学習に使うこととは区別し、モデル選択への使用歴はcheckpoint別に確認が必要。
DepthAnyEventを継承するScaleEvent Depthの正確な実行系列は、公開downstream実行設定が揃わないため未確定。ここではDepthAnyEventの公開31/10を、その著者の設定としてのみ掲載。
これらは公式splitそのもののtrain/test混入を意味しない。学習コード設定、実際のcheckpoint、論文評価条件は別途確認する。
