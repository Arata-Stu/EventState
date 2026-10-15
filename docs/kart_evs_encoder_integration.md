# kart EVS E2Eへの20ch encoder移植計画

2026-10-15、ユーザーが学習完了後にkartのRGB E2Eを参考にEVS E2Eへ導入する方針を指定。
kart `/Users/at/project/competition/kart` を読み取り調査。kart/OpenEBの変更やROS/GPU実行は今回行っていない。

## 現行実装の確認

- kart_e2e/kart_e2e/model.py: DINOv3 backbone＋384→128→操舵/スロットルhead。
  steer_only/steer_throttle、frozen encoder、strict checkpoint loadあり。
- contract.py: RGB専用SPEC、ONNX入力image=[1,3,120,212]を厳密検査。
  固定DINOv3 commitは6876159a11b4df116f30f667f8c9888617df0751。
- inference.launch.py: TensorRTNode→C++ ControlDecoderNode。走行時Torchなしの方針。
- sensors/openeb_tensor_cuda_async.yaml: 10 bins×2極性、polarity_major、時間補間none、
  212×120、40ms窓/4ms stride、channel_mean0/std1、tensor_name=input_tensor。
  sensors/README.mdでは /event_camera/tensor はisaac_ros_tensor_msgs/msg/TensorList、
  FP32 NCHW [1,20,120,212]、250Hz。これは設定/文書の確認で実機観測ではない。
- RGB前処理は正規化後zero padding。今回のEVS学習はcountsのzero padding後に固定正規化。

## 接続方針

OpenEB raw→20ch tensor→学習時と同じ前処理→event encoder→patch token集約→制御head
→ONNX→TensorRTNode→既存C++ decoder/制御gate。
RGB用画像encoderノードはEVS経路には不要。tensor名/形状/データ型/時刻を別SPECとして検証する。
最初は蒸留対象のpatch tokensをmean poolingして384次元とし、既存制御head構成を使う案。
RGBのbackbone.forwardをそのまま流用せず、x_norm_patchtokensを明示して集約する。
制御headにはkartの手動走行データと操作教師による追加学習が必要。蒸留済み重みだけでは走行policyにならない。

## 移植前に合わせる点

1. DINOv3版: EventStateはadc254450203739c8149213a7a69d8d905b4fcfa。
   RGBの既存commitへ無条件ロードしない。EVS専用版を保持するか、同じ入力で出力一致を検証して統一する。
2. 表現: EventStateはVGA rectifiedイベントで時間線形補間voxel化→bilinear縮小。
   OpenEB既定の補間なし低解像度histogramへ解像度だけ合わせても一致しない。
   RAWからの同一窓テストでchannel順・bin境界・空間量子化/縮小・rectificationを比較する。
3. 時間: DSECはRGB間隔、OpenEBは40ms/4ms stride。学習元と実運用の差として記録し、
   実際のkart RAWを学習/推論共通実装で再生して検証する。
4. 空間/正規化: 160×120 content→上下4px raw-zero padding→160×128→固定mean/std。
   portable loaderのprepare_countsはVGA voxel入力用。OpenEB既定tensorへ直接適用できない。
5. ROS契約: input_tensorとRGBのimageは異なる。EVS用ONNX SPEC/launch/configを追加する。
   tensor時刻を出力まで保持し、既存gateの鮮度判定と整合させる。
6. 周期: tensor250HzはViT推論250Hzを意味しない。Jetsonで遅延・queue・入力間引きを計測する。

## 成果物と検証順

EventStateのevent_encoder.pt/.json/_model.pyを移植元とする。
kart側のmodel/contract/data/export/bringupをEVS modality対応する際、package READMEとdocs/policies/e2e.mdも更新する。
既存方針のRGB必須/イベント非対応をEVSの明示選択へ変更し、RGB経路を保持する。

1. 学習完了とexportの確認。
2. 同一tensorでEventStateとkartのtoken出力一致、同一RAWで学習時/推論時前処理一致。
3. frozen encoder＋制御head学習。
4. ONNX checker/ORT一致→Jetson TensorRT比較。
5. 保存データ再生・ROS時刻/出力検証→実機での確認。

現時点では接続計画。kartへの実装、学習完了、Jetson推論は未確認。

## 2026-10-15 固定版の統一

ユーザー指示でkartのe2e.reposとcontract.pyをEventStateと同じ
adc254450203739c8149213a7a69d8d905b4fcfaへ変更。kart policy/READMEに移行条件を追記。
旧RGB SPEC拒否とソースpin一致を含むstdlib contractテスト3件通過。
kartローカルにpython_ws/dinov3 checkoutがないため実モデルロードは未検証。
既存checkpoint/ONNXは削除せず、旧版metadataは拒否を維持。
上記のcommit差異は調査時点の記録で、この設定変更により解消。入力表現の差は残る。

## 2026-10-15 補間なしの両側対応

ユーザー指定でdriverは補間なしを維持。EventState histogram presetを追加し、
時間bin=40ms/10、境界[start,end)、空間集計=sum poolingを採用。旧linear重みとの自動互換は認めない。
kart `evs_contract.py` がモデルJSONから解像度を取り、元のfull OpenEB設定を保存して
幅/高さ・40ms/4ms・補間none・raw count mean0/std1を設定した新YAMLを出力する。
`evs_model.py`は既存の固定版backbone builder＋vendored event_encoder_model.pyでstrict loadする。
content tensorをpadding/固定正規化してpatch特徴へ変換するPC用APIを追加。
モデルwrapperをROS上のTorchとして起動する変更は行っていない。

kart側での操作（PyYAMLとkart_e2eが利用可能な環境）:

```bash
cd /path/to/kart
export PYTHONPATH="$PWD/ros2_ws/src/kart_e2e${PYTHONPATH:+:$PYTHONPATH}"
python -m kart_e2e.evs_contract \
  --model-json /path/to/histogram/event_encoder.json \
  --base-config ros2_ws/src/kart_bringup/config/sensors/openeb_tensor_cuda_async.yaml \
  --output /path/to/openeb_histogram_model.yaml
```

ROSのセンサmodule `openeb.launch.py` の新規 `evs_tensor_config` 引数に生成した絶対パスを渡す。
既定は空文字で元設定を維持。mission側への引数転送は未追加。
モデルのcontent解像度を出力し、driver側ではpadding/正規化しない。
backendを替える場合はそのbackendのbase-configから生成する。

検証: kart入力契約2テスト通過。Python構文・shell構文・diff確認通過。
kart既存EVSテストはMacのPyYAML不足で起動不可。Torch/ROS/CUDA/実重み比較は未実施。
EventStateのtests/test_histogram_encoder.pyでbin境界・極性・空窓と、OpenEB式floor直接集計対
sum poolingを両解像度で検証するサーバーテストを追加。実重み比較コマンドはvoxel20_encoder_320.md参照。
DSECはrectified、kart driverはrawなので、カメラ座標の分布差は残る。
