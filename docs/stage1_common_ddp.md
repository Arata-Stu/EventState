# 共通Stage 1と3 GPU DDP（2026-11-04）

## 目的と固定条件

データセット×条件×seedごとにStage 1を一度学習し、同じ最終重みをSemantic（必須）、
Depth、将来のFlow/MVSEC転移に利用する。タスクごとにStage 1やcheckpoint選択を変えない。
DINOv3 ViT-S/16固定教師、同じDINO初期値、GEP RGB間隔、Hybrid、augmentation/dropoutなし。
E0はEncoderのみ・z蒸留、E1はLSTM・h蒸留、E2はLSTM・z+h蒸留。
V-JEPA教師比較とDSEC+M3ED混合は今回の8runには含めない。

## 共通train（公式task splitそのものではない）

- DSEC: F3 train37からSemantic開発valの07_a/08_aを除いた35系列。
  Semantic公式test3、F3 Depth val4は混入しない。Detection追加val/testもtrain35外。
  Flow候補の開発valは既に除外済みの02_e/07_a/08_a/11_cを利用する案。
  これは公式Flow valではない。別protocolのFlow holdoutを選ぶなら起動前に再監査する。
- M3ED: F3 train3（city_hall/penno_big_loop/penno_small_loop）。
  ucity_small_loop/rittenhouseはStage 1に使わない。horseも今回の共通poolには含めない。
  下流Semanticは既存train4/validation ucity/test rittenhouse、DepthはF3 train3/val2を維持できる。
  M3ED SemanticはInternImage pseudo-labelであり、人手GTと同一視しない。
- JSON正本: `tools/manifests/{dsec,m3ed}_stage1_common_v1.json`。
- Stage 1の正規化は共通trainだけから新しいJSONへ算出。旧train41/train4統計は流用しない。
  既存の未正規化イベント/RGB教師cacheは読み取り専用で共有する。
- 基盤教師自身の元学習集合の完全監査を意味しない。

## 8runとbatch定義

| dataset | 条件 | GPU数 | local Hybrid microbatch | 蓄積回数 | global clips/update | full updates |
|---|---|---:|---|---:|---:|---:|
| DSEC | E0/E1/E2 各1run | 1 | Random1+Stream1 | 4 | 8 | 99,999 |
| DSEC | E1 | 3 | 各rank Random1+Stream1 | 4 | 24 | 33,333 |
| M3ED | E0/E1/E2 各1run | 1 | Random1+Stream1 | 2 | 4 | 99,999 |
| M3ED | E1 | 3 | 各rank Random1+Stream1 | 2 | 12 | 33,333 |

clip長はDSEC8、M3ED16。全full runで予定入力は6,399,936 frame。
基準100kを3で割り切れる99,999に揃える。warmupも999/333更新、LRは1e-4固定
（encoder倍率0.1）、cosine/min ratio0.01。seed0。
batch定義はクリップ数であり、時刻数・パッチ数ではない。
本条件はVRAM最大化実験ではなく実効batch/分散方式の比較。物理microbatchは小さく、
3GPUによる速度改善は実測するまで未確認。毎microbatchでDDP同期する初期実装。

系列をrankへ固定配分し、各rankで状態保持・clip境界detachする。Random側は
DistributedSamplerで分割。Streamは各shardを独立に周回し、rank間で系列を重複させない。
M3EDはrankあたり1系列。これは単GPUと系列提示順・反復頻度が変わるため、結果を
「batchだけの因果効果」と呼ばず、単GPU/3GPU学習方式比較として報告する。
最小限の追加比較としてまずE1に限定。E0/E2のDDP比較は今回未指定。
FP16 overflowで更新がskipされると実入力数が増える。training_summary.jsonの
input_frames_seen/optimizer_attempts/optimizer_skipsを必ず確認する。

## 実行（sig-gpu5のみ）

変更をGitでcommit/pushし、サーバーのrepoでpull後、既存envを使う。
今回の同期はtar転送ではなくGitを使用する。Macで実学習しない。

```bash
cd /home/iASL/Arata_repo/EventState
git pull --ff-only origin main
source env/bin/activate
tmux new -s eventstate-stage1
# tmux内でもenvを有効化してから実行
source env/bin/activate
bash tools/start_stage1_common.sh "outputs/stage1_common_$(date +%Y%m%d_%H%M%S)"
```

この1コマンドは合成CPU/Glooテスト→train-only統計→8runのGPU smoke→全成功後に8runのfull。
smokeは単GPU99更新/3GPU33更新で、fullはsmoke重みから継続せず初期重みから新規学習。
6本の単GPU jobを3GPUに分配し、それらの完了後にDDP2本を全3GPUで順次実行する。
失敗時は同launcherが起動したprocess groupを停止し、成果物を保持。自動削除・自動再試行なし。
出力rootの再使用は拒否する。GPU上の他jobは事前にnvidia-smiで確認する。

個別操作は `python tools/run_stage1_common.py --help`。
--executeなしは計画表示のみ。--phase baseline/ddpで部分実行可能。
--skip-preflightは同コードのサーバーテスト通過後だけ使用。
中断runの再開にはplan.jsonの元commandを使用しtraining.resumeだけ該当checkpointへ変更する。
DDP world size、rank割当seed、data配置/clip構成を変えない。停止済みを確認してから行う。

## 実装・検証境界

DDP.forward内にencoder/LSTM/有効projectorを含め、診断値と持越し状態はdetachする。
unused projectorは勾配なし。各rankの実batch数で勾配を補正。rank0だけが重み/ログを書き、
全rankのiterator/state bank/RNGを集約してcheckpointへ保存。重みはmodule prefixなし。
DDPは現時点でtrain-only・clip境界TBPTTに限定。内部detach_state_everyは拒否。

Mac: stdlib21テスト、構文/CLI/bash確認。PyTorch/Gloo/Hydra/NCCL/実データのテストは未実行。
サーバーpreflightに3rank合成勾配一致、Hybrid持越し状態、checkpoint rank状態復元、
8commandのHydra合成/runtime guardを含む。本学習開始を示す実行結果は未受領。

DDP API参照: https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
