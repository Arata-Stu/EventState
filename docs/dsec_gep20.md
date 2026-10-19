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

MacではAST/dry-run/diffのみ。Torchおよび実DSEC/GPUは未実施。
事前学習用のオンライン経路を実装した段階。
既存Semantic/Detection cache exporterはRGB間隔のGEP cacheを読むため、
20ms checkpointを明示的に拒否する。下流評価には同じ20msを生成する抽出経路の追加が必要。
旧約50ms cacheで20ms学習モデルを評価して同条件と扱わない。
M3EDや20chは変更対象外。
短縮した窓の間には未入力の期間が残る。20ms毎の連続入力・50Hz推論を実装したものではない。
