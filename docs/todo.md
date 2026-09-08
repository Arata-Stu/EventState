# TODO

## DSEC-Detection event-only benchmark

- [ ] E0を`dsec_det_train41`、batch size 8、100,000 stepで学習し、最終重みを採用する。
- [ ] E2を1層LSTMとし、E0と同じsplit・batch size・step数で学習する。
- [ ] E4を1層LSTMとし、E0/E2と同じ条件でevent dropoutを有効にして学習する。
- [ ] E0/E2/E4の100,000 step最終checkpointから検出特徴を別々のcacheへ書き出す。
- [ ] **Frozen detection**: E0/E2/E4の100,000 step最終重みを固定し、公式DSEC-Detection
  41 train / 6 validation / 13 test splitと`dsec-det` protocolで共通のYOLOX型headだけを
  学習してCOCO mAP@[.50:.95]を報告する。
- [ ] **Fine-tune detection**: 各事前学習済みencoder（E2/E4は1層LSTMも含む）とYOLOX型headを、
  eventとbox annotationだけでend-to-end fine-tuneする。教師RGB/DINOv3は使用しない。
- [ ] **Scratch detection**: Fine-tuneと同じarchitecture、dataloader、augmentation、optimizer、
  step数で、event encoder・LSTM・headをランダム初期化して学習する。E2/E4は同一architecture
  なので、Scratchはnon-recurrent版と1層LSTM版の2条件とする。
- [ ] Frozen / Fine-tune / Scratchの各protocolで3 seedを実行し、mAPの平均と標準偏差を報告する。
- [ ] E1（1層LSTMの`h`のみをDINOv3へ蒸留）の100,000 step学習と検出評価を追加する。
- [ ] 1層で時系列効果を確認した後、2層LSTMをdepth ablationとして比較する。

E1は初回のE0/E2/E4比較が完了するまで後回しとする。本番の表現学習は公式train 41 sequence
すべてを使い、validationによるcheckpoint選択は行わない。100,000 stepの最終重みを採用し、
公式validation/testは表現学習で使用しない。検出headでは公開されている公式41/6/13 splitを使う。
Frozenは表現品質、Fine-tuneは事前学習込みの最終性能、Scratchは事前学習利得を測る主比較とする。
