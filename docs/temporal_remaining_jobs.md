# 残り3ジョブ：ConvLSTMのSemantic・Depth

2026-11-17。M3ED SemanticのE1 Encoder＋LSTM（h蒸留）・ConvLSTMは採用epoch20、best mIoU38.601202%まで受領済み。
その出力を保持し、以下の3ジョブだけを既存Patchwiseと同じ学習条件で実行する。

| job | 条件 | 内容 |
| --- | --- | --- |
| `dsec_semantic` | E1 Encoder＋LSTM（h蒸留）・ConvLSTM | 開発6/2で50epoch→最良epoch数でtrain8再学習→公式test3＋活動5分位 |
| `m3ed_E1_convlstm` | E1 Encoder＋LSTM（h蒸留）・ConvLSTM | M3ED Depth train3/val2、4段特徴→VKITTI初期化DPT 50epoch→全体＋活動5分位 |
| `dsec_E1_convlstm` | E1 Encoder＋LSTM（h蒸留）・ConvLSTM | DSEC Depth train37/val4、同じDPT条件→全体＋活動5分位＋公開val視差 |

DSEC Semanticは2回のhead fittingを含むので、3ジョブ内に合計4回の学習処理がある。
Stage 1を学習し直す処理や、GT補完・loss変更は含まない。Depthは従来通りlog-L1、batch4、FP32、seed0。
Semanticはh384ch、linear、CE、batch8、FP16、seed0。両方式で同じ既存活動manifestを使う。

2026-11-18、既定runの最終`Completed: .../summary.json`を受領。3ジョブと追加評価はすべて終了した（台帳§98）。
結果回収は保存JSONを読むだけでよく、以下の起動コマンドを再実行する必要はない。

## サーバーでの実行

追加したコードをユーザー自身がpushした後、sig-gpu5の別ターミナルで実行する。
Macでは学習や実データの依存テストを実行しない。

```bash
cd /home/iASL/Arata_repo/EventState
source env/bin/activate

git pull --ff-only &&
python tools/run_temporal_downstream.py --gpus 0,1,2 --preflight &&
python -u tools/run_temporal_downstream.py --gpus 0,1,2 --execute
```

`--gpus`は物理GPU番号。指定したGPUのうち、`nvidia-smi`で既存のcompute processがないものだけを使う。
3台空いて容量も足りれば3並列。2台なら2並列で残り1ジョブを待機し、1台なら順次実行する。
Xorg等のgraphics processだけのGPUはcompute使用中とは扱わない。空き0台／GPU照会失敗なら学習を始めず停止する。
空きは起動前と各worker開始時に再確認するが、GPUを他ユーザーに対して予約する仕組みではない。

GPU1・2だけを候補にする例：

```bash
python tools/run_temporal_downstream.py --gpus 1,2 --preflight &&
python -u tools/run_temporal_downstream.py --gpus 1,2 --execute
```

親schedulerがscratchの未割当分も含めて合計予約するため、空きGPUがあっても容量待ちになることがある。
OMP/MKL/OpenBLAS/NumExprのCPUスレッドはworkerで1に固定する。SemanticのDataLoader worker数は以前の設定を維持する。
これは以前のCPU過剰使用を抑えるためで、共有diskのI/O競合や実速度はサーバーのログで確認する。

## 入力とpreflight

既定の最終重みは次の2つ。Stage 1の親`_SUCCESS`を待つための追加処理はなく、実際の最終checkpointを使う。

```text
outputs/stage1_temporal_full_20261115/dsec_E1_convlstm_w1/checkpoints/step_00099999.pt
outputs/stage1_temporal_full_20261115/m3ed_E1_convlstm_w1/checkpoints/step_00099999.pt
```

Semanticのラベルroot・head recipeは既存Patchwise E1の
`outputs/stage2_dsec_semantic_linear_20261109/E1/final/config.json`から復元する。
DepthのGT indexは`outputs/stage2_common_depth_vkitti_20261110/targets/`のtrain/valをそのまま新runへコピーする。
統計は`outputs/stage1_common_20261105/statistics/`、活動manifestは
`outputs/semantic_depth_activity_20261110/manifests/{dsec,m3ed}.json`を再利用する。

preflightは新規出力の確認、旧split/GT/統計/ラベル/source metadata、GTと入力のhash、公開val視差用rawファイル、
GPU空きとscratch予約を確認する。元GTをhash照合するため、ファイル数・容量によって時間が掛かる。
この段階ではTorch、モデル、GPU推論を使わず、出力やcacheを作らない。
worker開始時にはcheckpoint内のConvLSTM・最終段1層・幅384・E1 h蒸留・seed0/full99999設定も確認する。
最初のRGB時刻は過去event intervalがないため、既存extractorと同じくその時刻のGTだけを特徴抽出から除外する。

パスを変えていた場合は、両コマンドへ同じ`--stage1-root`、`--depth-run`、`--statistics`等を指定する。
新runはfresh-only。既存の出力rootやSemantic cacheがある場合は、上書きや自動再学習をせず停止する。
途中で失敗した場合も生成済みcache・head・ログを残す。同じrootを削除して再起動せず、まずログと完了済み成果物を確認する。

## 出力と進捗

既定rootは`outputs/stage2_temporal_remaining_20261117/`。

```text
plan.json                              入力SHA・3ジョブ・recipe
logs/dsec_semantic.log                  Semantic workerの段階ログ
logs/m3ed_E1_convlstm.log                M3ED Depth学習/評価ログ
logs/dsec_E1_convlstm.log                DSEC Depth学習/評価ログ
dsec_semantic/logs/train.log            Semantic開発headのepoch進捗
dsec_semantic/logs/final_fit.log        Semantic train8再学習のepoch進捗
dsec_semantic/test_metrics.json         Semantic公式test全体
dsec_semantic/activity_test.json        Semantic全体＋Q1〜Q5
heads/m3ed_E1_convlstm/stages/           M3ED Depth head・全体/活動スコア
heads/dsec_E1_convlstm/stages/           DSEC Depth head・全体/活動/視差スコア
summary.json                           全3ジョブが成功した後の集約
_SUCCESS                               全3ジョブ・追加評価の成功後のみ生成
```

別ターミナルでの簡単な確認：

```bash
cd /home/iASL/Arata_repo/EventState
tail -F outputs/stage2_temporal_remaining_20261117/logs/*.log
```

Depthは有効画素pooledとF³式frame平均の両方を同じbest headで出す。
学習checkpoint選択は従来通りval pooled MAE最小。Depthの成功時は当該workerが作った一時特徴cacheだけを削除し、
head・評価・GT index・ログは保持する。失敗時は一時cacheも保持する。Semantic cacheは成功後も保持する。
公式testへのuploadや他者への送信は行わない。

## 検証範囲

Macではstdlibの合成fixture／mockで、計画、旧recipe、GPU振分け、容量予約、worker dispatch、入力改変拒否、
lock継承、既存出力保護を検証し、新規22件＋既存scheduler18件＋Stage1 planner12件の計52件が通過した。
CLI/AST確認も通過。
実サーバーのモデル・GT・CUDA実行は、上のpreflightと本実行で確認する。新しい精度値はこの実装時点ではない。
