# 実装上の決定事項

## 1. 最初のデータセットは DSEC

Phase 0/1 は DSEC を使う。RGB と event の同期、event の公式 rectification map、
さらに下流の flow / semantic / detection 評価を同じデータセット群で構成できるためである。
M3ED adapter は、DSEC 上で E0/E1/E2 の比較条件と学習系が固まった後に追加する。

## 2. MVP の event 入力は3チャネル

最初の E0/E1/E2 はすべて GEP と同系統の3チャネル表現を使う。

- 白背景
- negative event は赤、positive event は青
- positive / negative の count を別々に非ゼロ画素の90 percentileでclip・正規化
- 同じ画素に両極性がある場合は強い側だけを残し、同値ならpositiveを優先
- GEPで学習時に観測されるDSEC event統計で正規化

GEP実装では一度OpenCVでPNG保存してからPILで読むため、作成時の配列に対してR/Bが反転する。
本実装はPNGを介さず、その**学習時の実効RGB**を直接float tensorとして作る。

20チャネル（10 temporal bins × 2 polarity）は削除せず、追加実験
`h_distill_lstm_voxel20` として切り替え可能にする。これにより、temporal modelの効果を
検証した後で入力表現の容量を上げられる。

## 3. 時間窓は blueprint を優先

3チャネルの見た目はGEP互換だが、event windowはGEPの中点partitionを再現しない。
各RGB時刻 `T_t` に対して、厳密に次を使う。

```text
T_(t-1) < event timestamp <= T_t
```

GEP参考実装の中点partitionには境界を越えたtimestamp groupを旧frame側へ含める挙動があり、
blueprintの仮説検証条件と一致しない。したがって E0 は「GEP-style alignment baseline」であり、
既存GEP checkpointとのbitwise reproductionではない。

## 4. 空間整列

- raw event座標はDSEC公式 `rectify_map.h5` でevent rectified座標へ移す。
- RGBはcalibrationから得たrotation-only homographyで同じevent rectified座標へwarpする。
- その後のcrop / resize / horizontal flipは、clip内の全時刻・両modalitiesへ同じ係数を使う。
- 入力は448×640とし、DINOv3 patch size 16で割り切れない解像度を拒否する。

RGB/event camera間のtranslationとscene depthを無視するため、homographyは無限遠平面近似である。
この残差は将来のdepth-aware alignmentの検討事項とする。

## 5. Teacher cache と augmentation

teacher patch token cacheは固定された448×640 geometryに対応する。したがって、cache利用時に
random crop / flipを有効化すると対応が壊れるため、Dataset初期化時にエラーにする。
random spatial augmentationを使う実験ではonline frozen teacherへ切り替える。

## 6. 参考リポジトリからの独立

`reference_repo/` は調査専用であり、import path、設定path、submodule pathとして利用しない。
実装は仕様を確認したうえで新規に記述している。特に次の問題は引き継がない。

- hard-coded dataset / checkpoint path
- 設定上20chでも実体が3chのままになるDINOv3 patch embedding
- CLS/register tokenを含むhook出力とpatch tokenの混同
- event/RGBファイルをbasename照合せず独立sortするpairing
- worker例外の握り潰し
- frame単位のsplitまたはsequence boundary越え

DINOv3本体はvendor copyせず、公式packageまたは明示したtorch hub repositoryを外部依存として使う。
これにより `reference_repo/` を削除しても本プロジェクトは影響を受けない。

## 7. Cache は条件と完了状態を検証する

aligned RGB、event tensor、teacher tokenは、同じshapeであっても生成条件が異なれば混用しない。
sequenceごとに入力file、timestamp列、representation、geometry、DINOv3 source/weightを
SHA-256で識別し、各payloadをmanifestへ結び付ける。全frameの生成が完了した時点でだけ
`_SUCCESS`を作成し、読み込み時には期待する全outputの内容hashを再計算する。

metadataがない既存outputを名前やshapeだけで現行cacheとして採用することはしない。
このstrictな検証は大規模datasetやnetwork filesystemではI/O costを伴うが、初期研究段階では
速度よりstale/partial cacheを使わないことを優先する。

## 8. Validation はfull-sequence streaming

trainingはsequence境界を越えないrandom short clipを使う。validationは各sequenceを時系列順に
一度だけ走査し、chunk間でLSTM stateを維持し、sequence境界で必ずresetする。末尾の短いchunkを
含むmetricはclip数ではなくframe数で重み付けする。

checkpointには解決済みconfigを保存する。resume時はmodel、representation、loss、optimizer等の
研究条件を照合し、pathやworker数などの実行環境だけ変更可能とする。単独評価ではcheckpoint側の
E0/E1/E2条件を復元してから、評価対象dataset/cache/outputのruntime指定だけを上書きする。

## 9. DSEC-Detectionを評価するときは追加sequenceをpretrainingから隔離する

DSEC-Detectionの公式splitはtrain 41 / validation 6 / test 13である。追加された7 sequenceは、
`zurich_city_16_a`〜`zurich_city_21_a`がvalidation、`thun_02_a`がtestに割り当てられる。
物理archive上では前者が`train/`、後者が`test/`に入るため、directory名だけで学習対象を
決めてはならない。

標準的なinductive比較では、representation pretrainingも公式train 41件だけから作る。その41件を
sequence単位で内部train/validationへ分け、公式validation/testはcache作成、正規化統計、
checkpoint選択を含めて未使用にする。追加7件を含む全60件でのself-supervised pretrainingは
別の`transductive / test-exposed` ablationとして明記する。

取得scriptは追加7件をoriginal DSECへ自動mergeせず、`dsec_det_extra/`へ隔離する。公式split名は
`tools/manifests/dsec_det_official_split.yaml`に固定している。

## 10. Benchmark-clean internal validation splitを実験前に固定する

標準E0/E1/E2では`dataset=dsec_benchmark_clean`を使う。DSEC-Detection公式train 41件のうち、
recording group全体として`interlaken_00_{c..g}`（5件）と`zurich_city_11_{a..c}`（3件）を
internal validationへhold outし、残り33件だけをoptimizationに使う。この分割は、本学習結果を
見る前に地理的多様性と同一recording group内の近接sequence漏洩を避ける目的で固定した。

公式validation 6件と公式test 13件はpretraining、early stopping、checkpoint選択、cache統計に
使用しない。GEP比較・alignment開発でoriginal testを使う設定は標準結果と混ぜない。

## 11. DINOv3 attributionと配布物

本repositoryはDINOv3 source codeとpretrained weightを再配布せず、公式sourceの固定revisionと
利用者が別途取得したweight pathだけを参照する。READMEには「Built with DINOv3」を表示し、
`THIRD_PARTY_NOTICES.md`に使用箇所、公式project、license、paperを記録する。

DINOv3初期値を含む学習済みcheckpointを第三者へ配布する前には、weight取得時に同意したlicenseを
改めて確認し、必要なagreementとattributionを配布物へ同梱する。論文ではDINOv3利用を明記・引用する。
