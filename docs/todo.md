# TODO

## DSEC-Detection event-only benchmark

- [ ] E0を`dsec_det_train41`、batch size 8、100,000 stepで学習し、最終重みを採用する。
- [ ] E2を1層LSTMとし、E0と同じsplit・batch size・step数で学習する。
- [ ] E4を1層LSTMとし、E0/E2と同じ条件でevent dropoutを有効にして学習する。
- [ ] E0/E2/E4のbest checkpointから検出特徴を別々のcacheへ書き出す。
- [ ] 公式DSEC-Detection 41 train / 6 validation / 13 test splitと`dsec-det`
  protocolでYOLOX型headを比較し、COCO mAP@[.50:.95]を報告する。
- [ ] E1（1層LSTMの`h`のみをDINOv3へ蒸留）の100,000 step学習と検出評価を追加する。
- [ ] 1層で時系列効果を確認した後、2層LSTMをdepth ablationとして比較する。

E1は初回のE0/E2/E4比較が完了するまで後回しとする。本番の表現学習は公式train 41 sequence
すべてを使い、validationによるcheckpoint選択は行わない。100,000 stepの最終重みを採用し、
公式validation/testは表現学習で使用しない。検出headでは公開されている公式41/6/13 splitを使う。
