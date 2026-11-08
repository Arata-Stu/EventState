# 4層特徴を使うDepth / Disparity

E0 Encoder（z蒸留）は既存重みからDINOv3ブロック3/6/9/12の正規化済みパッチ特徴を読む。
Encoder＋4段LSTM（最終h蒸留）は各ブロック群の後で位置別LSTMを更新し、
投影後のhをdecoderと次のブロック群へ直接渡す。CLS・storage tokenは同じ時刻のViT経路を通す。
各段の(h,c)を系列内で引き継ぎ、Stage 1の教師損失は最終hに適用する。

学習サーバーで `python -m pytest -q tests/test_four_stage_depth.py` を実行。
新構成のStage 1は従来の `python tools/run_stage1_common.py` に
`--four-stage-lstm` と新規 `--output-root` を指定する。まず `--stage smoke`。
単GPU比較なら `--phase baseline`、3 GPU DDPなら `--phase ddp`。
既存の最終段LSTM checkpointはこの構成と異なる。

下流は従来のdense probeのtarget indexと統計を使い、以下のオプションで切り替える。

1. `tools/cache_dense_features.py` に `--four-level` を追加し、各splitを新規ディレクトリに抽出。
   3/6/9/12層の特徴をFP16で保存するので、単一特徴の約4倍の特徴容量を使う。
2. `tools/train_dense_probe.py --feature stages` にそのtrain cacheを渡す。
   F3はval cacheも渡す。両backboneで同じDPT型head（新規初期化）を学習する。
3. `tools/evaluate_dense_probe.py` にhead checkpointと同じ構成のcacheを渡す。
   公式DSEC disparity用に学習したheadは公式test cacheに対し `--export` で出力する。

このheadは4解像度に再配置した特徴を深い段から残差畳み込みで融合し、各画素の正のスカラーを返す。
単位は訓練ターゲットに従う（depthはm、disparityはpx）。ScaleEventの学習済みdecoder重みは使用しない。

配置比較では `--lstm-from-stage N` でSNからS4までを有効にする（N=1,2,3,4）。
有効段はh、無効段はzを同じ順序でdecoderに渡す。事前学習時の損失は全条件で最終段のみ。
全段なしは既存E0 Encoder（z蒸留）、S4のみは既存E1 Encoder＋LSTM（h蒸留）を再利用できる。
各配置の出力・特徴cacheは別ディレクトリに保存する。
