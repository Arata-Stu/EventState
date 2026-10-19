# EventState 実験一覧

作成日: 2026-10-19。根拠は `experiment_results.md` と `session_handoff.md`、受領済み実行ログ。
サーバーを直接確認した一覧ではない。数値の正本は `experiment_results.md`。
サーバーの出力rootは `/home/iASL/Arata_repo/EventState/outputs/`。

## 読み方

- 「結果あり」: 数値が台帳に記録済み。「一部未転記」: 完了報告と数値の揃い方が異なる。
- 「途中経過」: 実行報告あり、最終完了未確認。「準備済み」: コードやコマンドだけでは実施済みと数えない。
- 数値はmAP/mIoUの百分率。testとvalidation、head/lossや事前学習条件が異なる値を横断ランキングしない。
- 原則seed0。複数seedと明記した項目だけ追加試行あり。
- 以下は実験系列の一覧であり、独立した学習job数ではない。既存checkpoint/headの再利用や再評価を含む。
- z=event encoder特徴、h=LSTM状態、concat=zとhの連結。蒸留対象と下流head入力は別の軸。

## 実施した実験系列

| ID | データ・タスク | 比較した内容 | 状況・代表的な結果 | 台帳 |
|---|---|---|---|---|
| E01 | DSEC事前学習 | RandomのE0=z、E1=h、E2=z+h、E3=h+dropout、E4=z+h+dropout | 下流結果あり。主checkpointは100k | §1–3 |
| E02 | DSEC Detection | E0〜E4＋Hybrid、frozen backbone＋YOLOX型head | test。Random E1=38.693、Hybrid h=39.209 | §2 |
| E03 | DSEC Detection | Scratch、memoryなし／LSTMあり | test 25.065／26.290 | §3 |
| E04 | DSEC Detection | 全層fine-tuning対frozen（E0/E2/E4） | test。今回の全層FTは対応するfrozenを超えず | §3 |
| E05 | DSEC Detection | E4のtemporal＋headのみ更新、下流dropout有無 | test 38.181／38.313 | §3 |
| E06 | DSEC Detection | 評価時event gap=0/1/2/4/8、frozen／temporal FT／FT+dropout | gap8 overall mAP=35.55／33.69／35.50 | §4 |
| E07 | DSEC Semantic | E0〜E4＋Hybrid、frozen Linear+CE | test。Random E1=60.383、Hybrid h=60.843 | §5 |
| E08 | DSEC Semantic | Linear/GEP patch × CE/CE+Dice、Hybrid h固定 | test 4条件。GEP patch+Dice=61.462 | §6 |
| E09 | DSEC事前学習 | Hybridのaugmentation有無、cached/online教師 | 3条件100k。noaug cached/onlineのlossほぼ一致。aug下流Detは完了報告あり・値未転記 | §7 |
| E10 | M3ED事前学習 | Random/Hybrid、event dropoutの要因比較 | 複数100k重み。Random+dropout最終集約値は未転記 | §8 |
| E11 | M3ED Semantic | Random no-dropout／Hybrid no-dropout／Hybrid dropout、Linear+CE | validation 36.889／37.388／37.611 | §8 |
| E12 | M3ED Semantic | Linear/GEP patch × CE/CE+Dice、Hybrid dropout固定 | validation 4条件。Linear+Dice=38.392 | §8 |
| E13 | DSEC/M3EDデータ解析 | event activity分布、低活動区間 | 結果あり。M3EDに広い低activityの裾。性能評価ではない | §9 |
| E14 | DSEC activity事前学習・下流 | Random baseline_e2／activity_only／scale_event_full | 3条件100k＋Det hのtest、Semantic h/z/concatのtest。Det h=37.778／37.342／37.428 | §13–16 |
| E15 | DSEC h蒸留領域 | active z-only／active z+h-all／active z+h-soft | 3条件100k＋Semantic val計7 head | §17–18 |
| E16 | DSEC Semantic再評価 | baseline/hard maskのh/z/concatを同じdevelopment valで比較 | 既存headの6件再評価。追加学習ではない | §19 |
| E17 | DSEC Semantic head seed | baseline/hard/softのconcat、head seed0/1/2 | val、追加6 job。平均59.86771／60.09768／60.09124。pretrain seedは固定 | §20 |
| E18 | M3ED activity | Random/Hybrid × baseline/hard/soft × head z/h/concat | 6事前学習モデル・下流18欄。Hybrid soft z=36.99275（val） | §21–23 |
| E19 | Hybrid蒸留対象 | DSEC/M3ED、z-only／h-only／z+h事前学習 × head z/h/concat | 3タスク各7有効欄=21欄。新規4事前学習＋既存重み再利用、追加DSEC h-only6 headを含む | §25–30 |
| E20 | Hybrid activity α | DSEC/M3ED、h活動重みα=0/0.5/1 × head z/h/concat | 新規4事前学習＋下流21 head、既存M3ED6件と合わせ27欄。全数値受領 | §31–33 |
| E21 | DSEC 20ch encoder | 時間線形補間、160×120→padding160×128、固定RGB教師からz蒸留 | 63,100step途中結果。loss約0.20–0.21、projector後cosine約0.93。最終完了未確認 | §34–35 |
| E22 | DSEC 20ch encoder | 時間補間なし、40ms/10bin、160×120、count集計 | 初回fullは事前テスト失敗。境界修正/v2統計手順済み、修正後smoke/full完了ログ未受領 | §36 |

## 最近の主比較の結果

いずれもvalidation、単一seed。蒸留対象の基本表とα比較を区別する。

| 系列 | DSEC Detection mAP | DSEC Semantic mIoU | M3ED Semantic mIoU |
|---|---|---|---|
| Hybrid蒸留対象×入力（E19）内の最高 | z+h蒸留→concat: 37.06807 | h-only蒸留→concat: 60.23975 | h-only蒸留→concat: 36.38303 |
| Hybrid α比較（E20）内の最高 | α1→z: 36.91390 | α0.5→concat: 60.32212 | α1→z: 37.14701 |

αはhの活動領域に対する重み、非活動領域は1。zは全α条件で活動領域のみ。
α1は全領域z+h蒸留とは別条件。α1がDet/M3EDで有望だが、DSEC Semanticのα0.5−α0は
僅か0.02777 point。複数pretrain seedでの優位性は未確認。

**37.14701はプロジェクト全体のM3ED最高値ではない。** E11の37.611、E12の38.392などが
別系列に存在する。dropout、head/loss、既存runの条件を揃えて比較する必要がある。
またDSEC test39.209とvalidation37.06807も同じ表のランキングにはしない。

## 出力先索引

以下はサーバー出力rootからの相対パス。実ファイルの現在の存在は未確認。
キャッシュ削除済みでもhead重み・指標・ログを残す運用である。

| ID | 代表的な出力ディレクトリ |
|---|---|
| E01 | `v100_dsec_det_e0_e2_e4_20260908_020523`、`v100_dsec_det_e1_e3_20260915_160809` |
| E02 | `dsec_detection_frozen_step100000`、`dsec_detection_frozen_hybrid_noaug_cache` |
| E03–04 | `dsec_detection_end_to_end` |
| E05 | `dsec_detection_temporal_head`、`dsec_detection_temporal_head_dropout` |
| E06 | `dsec_detection_gap_e4`、`dsec_detection_gap_e4_temporal_dropout` |
| E07 | `dsec_semantic_frozen_linear` |
| E08 | `dsec_semantic_frozen_head_loss_ablation` |
| E09 | `dsec_hybrid_aug`、`dsec_detection_frozen_hybrid_aug_online` |
| E10 | `m3ed_state_dropout_factorial_20260920_145338`、`m3ed_e0_e1_random_noaug` |
| E11–12 | `m3ed_semantic_frozen_linear`、`m3ed_semantic_frozen_head_loss_ablation` |
| E13 | `event_activity/dsec_vs_m3ed_density` |
| E14 | `dsec_activity_full_20260923_000937`、`dsec_detection_activity_20260923_h`、`dsec_semantic_activity_20260923_h`、`dsec_semantic_activity_20260923_zh` |
| E15 | `dsec_activity_h_relaxation_full_20260925_182735`、`dsec_semantic_h_relaxation_20260925_182735` |
| E16–17 | `dsec_semantic_activity_validation_20260927`、`dsec_semantic_concat_seeds_20260927` |
| E18 | `m3ed_activity_full_20260927_192110`、`m3ed_activity_hybrid_full_20260929_131030`、`m3ed_semantic_activity_random_hybrid_20260930` |
| E19 | `hybrid_targets_full_20261001_231617`、`hybrid_target_downstream_20261004`、`hybrid_h_only_downstream_20261006` |
| E20 | `hybrid_activity_alpha_full_20261009_021017`、`hybrid_activity_alpha_downstream_20261011` |
| E21 | `voxel20_encoder_160_full_20261014_181246`、`voxel20_encoder_160_smoke_20261014_174109` |
| E22 | 新規full run名未受領。prefix `voxel20_encoder_160_histogram_full_` |

## これからの項目・未確認事項

| 項目 | 記録上の状態 |
|---|---|
| 1GPU/batch1倍 vs 3GPU/batch3倍 | 未実施。過去の3GPU並列実験は各GPUで独立jobを実行したもの |
| Hybrid主比較／α比較のpretrain seed追加 | 未実施。E17はhead seed追加だけ |
| 同じモデルでcontinuous/reset比較 | 計画・可視化ツールあり、数値比較結果は台帳未記録 |
| active/inactive別の下流評価 | 提案段階、数値未受領 |
| 20ch 320×240 | linear/histogram設定準備済み、学習実行報告なし |
| 20ch linear vs histogramの性能比較 | 両条件の最終結果未確認 |
| 20ch特徴動画・実重みのkart一致検証 | ツール準備済み、実行結果未受領 |
| kart EVS制御head学習・ONNX/TensorRT・実車評価 | 移植準備中、性能/走行結果なし |
| augmentation下流Det、M3ED Random+dropout最終ログ | 完了報告のある項目だが台帳数値の補完が必要 |

`outputs/1mpx`、`paper_figures`、各visualization等のディレクトリ名だけでは
実験条件・完了・評価値を確定できないため、独立した完了実験には数えない。
失敗run、smoke、cache生成、再評価も独立した本学習数へ加算しない。
