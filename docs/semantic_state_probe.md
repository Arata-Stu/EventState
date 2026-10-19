# 状態保持と活動別Semantic評価

対象は既存Hybrid h-only/z+h、DSEC development valとM3ED validation。
backboneと学習済みbest headを固定する。新しい学習・testによる条件選択はしない。

## 比較と出力

- continuous: 各系列の最初の利用可能なイベントフレームから処理し、系列境界だけで状態を初期化。
- frame: 全フレームでLSTMのh/cを初期化。clip境界だけのresetではない。
- 未ラベルフレームも処理し、評価は両方で同じラベル付きフレームのみ。
- z/h/concatそれぞれで全画素・active・inactive・低活動フレーム・その他を集計。
- 活動判定は既存ScaleEvent準拠GEPマスク。DSECはpatch strideで展開して440行へcrop。
  M3EDは学習時transformのマスクを同じstrideで展開。低活動は停止の真値ではない。
- 低活動フレームは評価画像領域の活動画素率が10%以下。ignore画素も活動率の分母に含めるが、
  精度には含めない。これは事前固定の閾値で、train分位点推定ではない。
  `--low-activity-threshold`で変更可能だがval精度を見て最適化しない。
- mIoUは各群の累積confusion matrixから計算。クラスのunionが0ならそのクラスは平均から除外。
  空群のmIoUはJSON null。群間のクラス構成が異なるので、同じ群のcontinuous−frameを主比較にする。
- 各JSONにconfusion matrix、class support、画素数、フレーム数、head/backbone SHA256を保存。
  `summary.tsv/json`のmIoUは0–1、delta_pointsのみ百分率ポイント。
- paired summaryはサンプル・マスク・重み・class supportの一致を確認。z特徴の完全一致も検査し、
  不一致時は比較を成功扱いにしない（summary生成で停止）。

4モデル×2方策×3head=24評価。特徴抽出は8回で、各回のcacheを3headで共有。
validationのみのcacheを一つずつ作り、評価成功後は所有権マーカーが一致するcacheだけ削除する。
全量train cacheは不要。必要容量は実データに依存するため、起動前にdfで確認する。

## サーバー実行

ローカルの変更をサーバーへ同期した後、GPUサーバーで実行する（MacでTorch/GPUテストは未実施）。

```bash
cd /home/iASL/Arata_repo/EventState
source env/bin/activate
python -m pytest -q tests/test_semantic_state_probe.py tests/test_m3ed_downstream.py tests/test_state_probe_runner_stdlib.py
df -h /home/iASL/Arata_repo/dataset

# 既存重みとheadのパスを表示。実行・cache作成はしない。
bash tools/run_semantic_state_probe.sh \
  --output-root outputs/semantic_state_probe_20261019 \
  --cache-root /home/iASL/Arata_repo/dataset/downstream_scratch/state_probe_20261019 \
  --dry-run

bash tools/run_semantic_state_probe.sh \
  --output-root outputs/semantic_state_probe_20261019 \
  --cache-root /home/iASL/Arata_repo/dataset/downstream_scratch/state_probe_20261019 \
  --gpu 0
```

再開は同じ引数に`--resume`を追加。途中の方策はcacheから再検査して再評価し、完了方策はskip。
コード・重み・閾値・出力先が変わった場合は新しいoutput/cache rootを指定する。
1モデルだけ試すなら新しいrootで`--only dsec_h_only`（他はdsec_z_h/m3ed_h_only/m3ed_z_h）。

既定モデルパスは`tools/run_semantic_state_probe.py::default_models`に明記。
異なる保存先は`--models-json FILE`で置換する。形式は次のリスト。

```json
[{"name":"dsec_h_only","dataset":"dsec","checkpoint":"/absolute/backbone.pt","heads":"/absolute/semantic"}]
```

heads直下に`z/seed_0/best.pt`、`h/seed_0/best.pt`、`concat/seed_0/best.pt`を要求する。
全head/teacher/backboneの存在を起動前に検査。headとbackboneの意味上の組合せはmanifestの責任であり、
古いheadにbackboneハッシュがないため自動的に由来を証明することはできない。
ログは`OUTPUT/MODEL/POLICY/{cache,z,h,concat}.log`、結果は同ディレクトリのJSON。
continuousの全領域値を過去の同一headのvalidation値と照合してから差を解釈する。

## 解釈の範囲

固定headのreset劣化は学習済みモデルの履歴依存を示す。resetは学習時分布からの介入でもあり、
状態なしで再学習したモデルに対する優位性を示す実験ではない。
単一seedで有意差は判定しない。低活動区間の長さ別・物理的停止の分類は今回未実装。

## M3ED全領域蒸留とα=1の比較（2026-10-19）

reset介入は履歴利用の確認であり、蒸留損失の優位性とは分ける。
新規α1はcontinuousだけ実行し、全領域baselineの既存continuous JSONを再利用する。
各モデル専用の学習済みheadを使う。単一seedのモデル間比較であり損失変更の因果効果を確定しない。

```bash
bash tools/run_semantic_state_probe.sh \
  --models-json tools/manifests/m3ed_alpha1_state_probe.json \
  --policies continuous \
  --output-root outputs/m3ed_alpha1_activity_probe_20261019 \
  --cache-root /home/iASL/Arata_repo/dataset/downstream_scratch/m3ed_alpha1_probe_20261019 \
  --gpu 0

python tools/compare_semantic_activity_probes.py \
  --baseline outputs/semantic_state_probe_20261019/m3ed_z_h/continuous \
  --candidate outputs/m3ed_alpha1_activity_probe_20261019/m3ed_alpha_1/continuous \
  --output outputs/m3ed_alpha1_activity_probe_20261019/vs_full_region.json
```

学習なし、抽出1回・3head評価。成功後cache削除。`--resume`は新runの同じ引数に追加。
新ランチャーはpoliciesをrun identityに含むため、旧version1のrunをこの版でresumeしない。
旧runの結果JSONを読む比較ツールは互換。comparisonの差はα1−全領域。
クラスIoU/supportと両head/backboneハッシュも比較JSONへ保存する。
サンプル・マスク・閾値・クラスsupportが異なれば停止する。異なる重みのSHA一致は要求しない。
新規結果の全体zが以前の37.147%を再現するか、まず確認する。
