# DSEC GEP固定20ms蒸留

RGB時刻tに対し、イベント(t−20,000µs, t]をGEP 3chへ変換する。
RGBフレーム間隔やモデル更新時刻は変更しない。event_window_fraction=0.4ではなく固定幅。
RGB間隔が20ms未満なら黙って窓を短くせずエラーにする。

2条件: baseline（z/h全領域）、alpha1（z activeのみ、h全領域）。
Hybrid 4+4、batch8、clip8、seed0、augmentation/dropoutなし、100k step。
smokeは100step。正規化は既存DSEC train41係数を維持し、窓変更以外の変更を避ける。
新統計は生成しない。GEPのpercentile正規化も既存と同じ。
生イベントHDF5から逐次GEPを生成するため新規全量event cacheは作らない。
RGB時刻・空間変換を維持するため既存教師cacheを再利用する。
I/Oは既存GEP cache利用時より重くなる可能性がある。

```bash
cd /home/iASL/Arata_repo/EventState
source env/bin/activate
python tools/run_dsec_gep20_pretrain.py --stage smoke --gpus 0,1
# smoke完了とloss/finiteを確認後
python tools/run_dsec_gep20_pretrain.py --stage full --gpus 0,1
```

ランチャーはサーバーの境界テストを実行してから2GPUで独立2jobを起動。
DDPではない。fullはsmokeの継続ではなく初期重みから開始する。
出力outputs/dsec_gep20_{smoke|full}_TIMESTAMP、ログlogs/{baseline|alpha1}.log。
--dry-run、--data-root、--teacher-checkpoint、--output-root、--seedに対応。
中間checkpointはfullでは10k毎。重みと設定にevent_window_us=20000を保存。

## 検証と制限

2026-10-21受領: サーバー2テスト通過、full baseline/alpha1正常終了。
出力outputs/dsec_gep20_full_20261019_235316。最終loss・checkpoint内容・下流精度は未確認。
2026-10-21: Semantic/Detectionのオンライン20ms抽出と下流suiteを追加。
固定窓checkpointは--raw-event-rootを要求し、旧event-cache-dirとの混用を拒否する。
下流のTorch/実データ検証はサーバー待ち。
旧約50ms cacheで20ms学習モデルを評価して同条件と扱わない。
M3EDや20chは変更対象外。
短縮した窓の間には未入力の期間が残る。20ms毎の連続入力・50Hz推論を実装したものではない。


## 20ms下流学習・評価（2026-10-21実装）

同じ事前学習datasetのraw読み込み/rectification/固定窓を再利用する。
Semantic: crop440行→padding、Detection: 従来のresize→DSEC-Det distorted座標warp。
各系列先頭から全イベントフレームを処理してhを保持。旧windowのevent cacheは一切読まない。
既存プロトコルどおりfrozen backbone、head seed0/50epoch、Semantic Linear+CE development val、
Detection official val。2モデル×2タスク×3入力=12head。追加でSemantic活動別評価6件。

```bash
cd /home/iASL/Arata_repo/EventState
source env/bin/activate
python -m pytest -q tests/test_gep_fixed_window.py tests/test_gep20_downstream.py tests/test_gep20_downstream_stdlib.py tests/test_hybrid_downstream_stdlib.py

df -h /home/iASL/Arata_repo/dataset
bash tools/run_hybrid_target_downstream.sh \
  --suite gep20 \
  --pretrain-root outputs/dsec_gep20_full_20261019_235316 \
  --output-root outputs/dsec_gep20_downstream_20261021 \
  --cache-root /home/iASL/Arata_repo/dataset/downstream_scratch/dsec_gep20_20261021 \
  --gpu 0
```

実行計画だけなら同じコマンドに--dry-run。再開なら--resume。
まず1モデルだけの場合は新rootで--only dsec_gep20_baselineを指定する。
モデルごとに全評価成功・結果確認後、所有権付きの一時cacheだけ削除。
中断時はcache/ログを保持。大きなDetection特徴cacheは必要で、オンラインGEP生成は
「イベント画像cache追加なし」を意味し、下流特徴cacheが不要という意味ではない。
baseline→alpha1を逐次実行し、同時蓄積を避ける。

出力:
- `summary.json`: 12条件のmIoU/mAP（0–1）
- `activity_comparison.json`: 20ms α1−baselineの活動別比較（差はpoint）、class IoU/support付き
- `{dsec_gep20_baseline,dsec_gep20_alpha1}/semantic_activity/{z,h,concat}.json`: 各モデルの活動別結果
- 各task/feature/seed_0以下: head best/last、validation_metrics.json

活動別比較は同じ20ms窓・同じmaskの両モデル間で実施する。
約50msと20msでは活動mask自体が異なるため、異なる窓の活動別指標を同じ母集団の比較と扱わない。
全体の同一split指標を先に比較する。実サーバーの下流完了/精度は未受領。
