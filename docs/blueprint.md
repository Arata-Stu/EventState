# EventState Pretraining — Implementation Blueprint

## 1. 目的

本プロジェクトでは、イベントカメラから得られる短時間のイベント特徴と、時間方向に統合されたpersistent visual stateを分離して学習する。

基本pipelineは以下とする。

```text
raw events
    ↓
event representation
    ↓
Event Encoder
    ↓
z_t
    ↓
Temporal Model (LSTM / Mamba)
    ↓
h_t
```

ここで、

* `z_t`: Event Encoderが出力する短時間・局所的なevent representation
* `h_t`: 過去のevent sequenceを統合したtemporal / persistent representation

と定義する。

中心仮説は以下。

$$
z_t = \text{event-native local representation}
$$

$$
h_t = \text{temporally integrated persistent visual state}
$$

そして、

* `z_t`にはEvent固有の学習信号
* `h_t`にはRGB Vision Foundation Modelからの教師信号

を与える。

最初のRGB teacherには **Frozen DINOv3** を用いる。

最終的な主学習目標は、

$$
P_h(h_t) \approx \operatorname{sg}(\mathrm{DINOv3}(I_t))
$$

である。

GEPではEvent Encoderのfeature自体をDINO系image featureへalignしているが、本研究ではそのalignment targetを`z_t`ではなく`h_t`へ移す。

---

# 2. 研究上の中心的な問い

本実装では以下を検証する。

### RQ1

単一event representationから得られる`z_t`を直接RGB VFMにalignするより、

$$
z_1,z_2,\dots,z_t
\rightarrow
h_t
$$

と時間統合した後の`h_t`をRGB VFMへalignした方が良いrepresentationになるか。

### RQ2

Temporal ModelはEvent Encoderだけでは保持できない、

* static object
* scene structure
* semantic context
* object persistence
* temporary occlusion

を保持できるか。

### RQ3

`z_t`にRGBとは異なるEvent-native supervisionを与えることで、

$$
z_t
$$

と

$$
h_t
$$

に役割分担が形成されるか。

---

# 3. 最初の実装範囲

最初から完全なJEPA、CMax、flow、multi-teacherを実装しない。

MVPでは以下を完成させる。

```text
Events
  ↓
Event Representation
  ↓
Event Encoder
  ↓
z_t
  ↓
Temporal Model
  ↓
h_t
  ↓
Projection Head
  ↓
DINOv3 patch features
```

RGB branch:

```text
RGB_t
  ↓
Frozen DINOv3
  ↓
teacher patch tokens r_t
```

主loss:

$$
L_h =
L_{\mathrm{distill}}
(
P_h(h_t),
r_t
)
$$

`z_t`用lossはinterfaceだけ最初から用意し、複数方式を切り替え可能にする。

初期実験では、

```yaml
z_loss: none
```

も正式なbaselineとして扱う。

その後、

```yaml
z_loss:
  - temporal_prediction
  - flow
  - cmax
```

を順番に追加する。

---

# 4. 想定データセット

## Primary dataset: DSEC

最初のpretrainingにはDSECを使う。

理由：

* event stream
* synchronized RGB
* calibration
* optical flow GT
* semantic segmentation
* disparity
* object detection annotations

を同じデータセット上で利用できるため。

DSECのeventsは`x, y, t, p`としてHDF5に保存され、event timestampsをimage timestampと同一clockへ変換する` t_offset`も提供されている。

RGB cameraは20 Hzであるため、RGB frame間隔は基本的に約50 ms。

DSEC公式配布にはtraining events、images、optical flow、semantic segmentation、disparityなどが含まれている。

DSEC Detectionにはobject detection labelとtrack IDも提供されている。

したがって、

```text
Pretraining:
DSEC events + RGB

Evaluation:
DSEC Flow
DSEC Semantic
DSEC Detection
```

という一貫した実験が可能。

---

# 5. データ単位

RGB image timestampを基準にeventを切る。

RGB timestampを

$$
T_t
$$

とする。

基本のevent inputは、

$$
E_t =
\{e_i \mid T_{t-1}<t_i\le T_t\}
$$

とする。

DSECではRGBが20 Hzなので、おおよそ

$$
50\text{ ms}
$$

のeventsとなる。

最初のbaselineではこの方式を採用する。

```text
T_(t-1) ---------------- T_t
       events ~50ms
                         RGB_t
```

これにより、

```text
event_repr_t ↔ RGB_t
```

の対応が明確になる。

将来的には、

* 10 ms
* 20 ms
* 50 ms
* fixed event count

もablationする。

---

# 6. Event Representation

## Baseline

GEPとの比較を容易にするため、まずGEP-compatibleなevent representationを使用する。

現在のGEP configではEvent Encoder入力として20 channelsを設定できるようになっている。

```python
event_channels = 20
```

したがって最初は、

```text
events
 ↓
temporal bins
 ↓
20-channel event tensor
```

とする。

例：

```text
10 temporal bins × 2 polarity
=
20 channels
```

shape:

```python
event_repr.shape
# [B, 20, H, W]
```

初期入力解像度：

```text
H = 448
W = 640
```

DINOv3のpatch sizeが16なので、両方を16の倍数にする。

GEP側もDSEC + DINOv3の場合に448×640へ設定する分岐を持っている。

---

# 7. Sequence Dataset

新規Dataset classを実装する。

例：

```python
class DSECSequenceDataset(Dataset):
```

返り値：

```python
{
    "events": Tensor[T, C, H, W],
    "images": Tensor[T, 3, H, W],
    "timestamps": Tensor[T],
    "sequence_name": str,
}
```

例えば、

```yaml
sequence_length: 8
```

なら、

```text
E_t-7
E_t-6
E_t-5
...
E_t

RGB_t-7
...
RGB_t
```

を返す。

最初のモデルではRGB teacherを全時刻に計算してよい。

つまり、

$$
L_h
=
\frac1T
\sum_{\tau=1}^{T}
D(h_\tau,r_\tau)
$$

とする。

これは最後のframeだけをsuperviseするよりTemporal Modelへ多くのsignalを与えられる。

---

# 8. Data Split

sequence単位でtrain/validationを分割する。

絶対にframe単位でrandom splitしない。

理由：

同一driving sequenceの隣接frameがtrainとvalidationへ混ざるとtemporal leakageが発生する。

設定例：

```yaml
dataset:
  name: dsec
  train_sequences: [...]
  val_sequences: [...]
```

最初はGEPが使用しているsplitを再利用できるならそれを優先する。

---

# 9. RGB Teacher

## Model

最初は、

```text
DINOv3 ViT-S/16
```

を使用する。

理由：

* embedding dim = 384
* patch size = 16
* ViT-Bより軽い
* dense patch featuresを利用可能

DINOv3 ViT-S/16は384-dimensional embedding、patch size 16を持つ。DINOv3自体はdepth、semantic segmentation、correspondence、trackingなどのdense taskでfrozen feature利用を想定したモデルである。

Teacherは完全freezeする。

```python
for p in teacher.parameters():
    p.requires_grad = False

teacher.eval()
```

RGB branchにはgradientを流さない。

---

# 10. DINOv3 Teacher Feature

CLS tokenではなく、

```text
patch tokens
```

を使う。

DINOv3 ViTはclass token、register tokens、patch tokensを返す。

teacher feature:

```python
r_t.shape
# [B, N, 384]
```

448×640の場合、

```text
28 × 40 = 1120 patches
```

なので、

```python
r_t.shape = [B, 1120, 384]
```

を想定する。

必要なら、

```python
[B, 1120, 384]
→
[B, 384, 28, 40]
```

へreshapeできるようにする。

---

# 11. Teacher feature cache

DINOv3はfrozenなので、毎epoch RGBをDINOv3へ通す必要はない。

以下のprecompute commandを実装する。

```bash
python tools/cache_dinov3_features.py \
    dataset=dsec \
    teacher=dinov3_vits16
```

保存：

```text
cache/
└── dinov3_vits16/
    └── sequence_name/
        ├── 000000.pt
        ├── 000001.pt
        └── ...
```

各`.pt`：

```python
{
    "patch_tokens": Tensor[N, D],
    "timestamp": int,
}
```

通常trainingではcacheを利用。

debug時のみonline teacher inferenceを許可する。

```yaml
teacher:
  cache_features: true
```

---

# 12. Event Encoder

## Baseline architecture

GEPに合わせてDINOv3 ViT-S/16 architectureをEvent Encoderとして使用する。

```text
Event tensor
[B,20,H,W]

↓ patch embedding

ViT-S/16

↓

z_t
[B,N,384]
```

Event Encoderはtrainable。

重要：

RGB DINOv3 weightsをEvent Encoderの初期値として使うかどうかをconfigで切り替える。

```yaml
event_encoder:
  architecture: dinov3_vits16
  init:
    type: dinov3
```

20 channelsなのでpatch embedding weightだけshapeが一致しない。

それ以外のTransformer block weightsはDINOv3から初期化可能。

patch embeddingは、

* random initialization
* RGB weight average → replicate
* polarity/bin-aware initialization

を選択可能にする。

最初はsimpleにrandom initでよい。

---

# 13. z_t の定義

Event Encoderの最終normalized patch tokensを、

$$
z_t
$$

とする。

```python
z_t.shape
# [B, N, D]
```

基本：

```text
N = 1120
D = 384
```

`z_t`はglobal poolingしない。

Detection / segmentation / flowへ使えるdense spatial representationを維持する。

---

# 14. Temporal Model

Temporal modelのinterfaceを統一する。

```python
class TemporalBackbone(nn.Module):

    def forward(
        self,
        z: Tensor,
        state=None,
    ):
        ...
```

入力：

```python
z.shape
# [B, T, N, D]
```

出力：

```python
h.shape
# [B, T, N, D]
```

必ずspatial patch dimension `N` を保持する。

---

# 15. MVP Temporal Model: LSTM

最初はLSTMから実装する。

ただしpatch tokenごとに独立したtemporal sequenceとして処理する。

```python
[B,T,N,D]
→
[B*N,T,D]
→
LSTM
→
[B*N,T,D]
→
[B,T,N,D]
```

つまり各patch locationについて、

$$
z_1(x,y),z_2(x,y),...,z_T(x,y)
$$

をtemporal LSTMへ入力する。

初期設定：

```yaml
temporal:
  type: lstm
  input_dim: 384
  hidden_dim: 384
  num_layers: 1
  dropout: 0.0
```

出力projection：

```python
hidden_dim → 384
```

とする。

---

# 16. LSTMについての注意

単純patch-wise LSTMは最初のbaselineとして使う。

ただしcamera motionやobject motionにより、

```text
物体が同じpatch locationに留まるとは限らない
```

という問題がある。

そのため、このモデルを最終モデルとは考えない。

将来的には、

* ConvLSTM
* spatial Mamba
* token mixing
* deformable temporal correspondence
* flow-guided memory

へ拡張する。

しかし最初の目的は、

> temporal stateを入れるだけでDINO alignmentが改善するか

を検証することなのでpatch-wise LSTMで十分。

---

# 17. Mamba

最初のrepoではinterfaceだけ実装する。

```yaml
temporal:
  type:
    - none
    - lstm
    - mamba
```

Mamba実装はPhase 2。

最初の論理的比較は、

```text
none
vs
LSTM
```

でよい。

LSTMで仮説が成立した後に、

```text
LSTM
vs
Mamba
```

を比較する。

---

# 18. h_t の定義

Temporal Model出力を、

$$
h_t
$$

とする。

```python
h_t.shape
# [B,N,384]
```

`h_t`は、

* current event information
* past event information
* persistent scene state

を含むことを期待する。

---

# 19. h Projection Head

DINOv3 feature spaceへ直接LSTM出力を固定しすぎないため、projection headを置く。

```text
h_t
 ↓
Linear
 ↓
GELU
 ↓
Linear
 ↓
DINO feature dimension
```

例：

```python
class TeacherProjection(nn.Module):
    Linear(384, 768)
    GELU()
    Linear(768, 384)
```

alignmentは、

```text
P_h(h_t)
```

と、

```text
DINOv3 patch tokens
```

の間で行う。

fine-tuning時にはprojection headを捨てられるようにする。

---

# 20. Main Loss: h → DINOv3 alignment

最初はGEPと比較しやすいように、

### Cosine loss

$$
L_{\cos}
=
1-
\cos
(
P_h(h_t),
r_t
)
$$

### MSE

$$
L_{\mathrm{MSE}}
=
\|
\operatorname{Norm}(P_h(h_t))
-
\operatorname{Norm}(r_t)
\|_2^2
$$

を実装する。

config：

```yaml
loss:
  h_distill:
    enabled: true
    cosine_weight: 1.0
    mse_weight: 1.0
```

patchごとの平均を取る。

---

# 21. z_t の学習signal

ここは研究上の拡張pointとして明示的にmodule化する。

interface：

```python
class ZObjective(nn.Module):
    def forward(
        self,
        z,
        events,
        metadata,
    ):
        ...
```

以下をconfigで差し替え可能にする。

---

## Z0. None

```yaml
z_objective:
  type: none
```

`z_t`には`h_t`経由のgradientだけが入る。

これは重要なbaseline。

---

## Z1. Direct DINO alignment

GEP-style baseline。

$$
P_z(z_t)
\approx
r_t
$$

とする。

これは本研究の本命ではないが、非常に重要な比較対象。

```text
GEP-like:
z → DINO

Ours:
z → Temporal → h → DINO
```

これを必ず実装する。

---

## Z2. Temporal latent prediction

未来のevent latentを予測する。

```text
z_t
 ↓
predictor
 ↓
z_(t+1)
```

ただしcollapse対策が必要なので、Phase 2扱いとする。

最初からprimary objectiveにしない。

---

## Z3. Optical Flow supervision

DSEC flow GTを利用する。

```text
z_t
 ↓
lightweight flow decoder
 ↓
flow
```

これはself-supervised pretrainingではないため、representationの評価・diagnosticとして使う。

目的：

> z_tにmotion informationが存在するか

をprobeする。

Event Encoderをfreezeしたlinear / shallow flow probeを基本とする。

---

## Z4. CMax

Event-native self-supervisionとしてPhase 3で追加する。

```text
z_t
 ↓
flow head
 ↓
event warping
 ↓
contrast maximization
```

最初のMVPでは実装しなくてよいが、loss registryにはplaceholderを用意する。

---

# 22. MVP Training Objective

最初の本命experimentは、

$$
\boxed{
L
=
L_{\mathrm{DINO}}(h)
}
$$

とする。

つまり、

```text
event
 ↓
z
 ↓
LSTM
 ↓
h
 ↓
DINOv3 alignment
```

だけ。

次に、

$$
L
=
L_{\mathrm{DINO}}(h)
+
\lambda_z L_z
$$

へ拡張する。

---

# 23. なぜ最初にz lossを入れないのか

`z`と`h`の役割分担を評価するには、

まず、

> hへのRGB supervisionだけでzがどの程度Event-native informationを保持するか

を見る必要がある。

最初からCMaxやfuture predictionを入れると、

Temporal Modelの効果とz supervisionの効果を分離できない。

そのため実験順は必ず、

```text
h supervision only
↓
+ z supervision
```

とする。

---

# 24. Training loop

sequence単位でforwardする。

pseudo-code:

```python
events, teacher_features = batch

state = None
loss = 0

for t in range(T):

    z_t = event_encoder(events[:, t])

    h_t, state = temporal_model(
        z_t,
        state
    )

    pred_t = h_projector(h_t)

    loss_h = distillation_loss(
        pred_t,
        teacher_features[:, t]
    )

    loss += loss_h

loss /= T

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

実装上は可能ならsequence dimensionをまとめてEvent Encoderへ投入する。

```python
[B,T,C,H,W]
→
[B*T,C,H,W]
→
event_encoder
→
[B,T,N,D]
```

その後Temporal Modelへ入れる。

---

# 25. Truncated BPTT

最初は、

```yaml
sequence_length: 8
```

とする。

DSEC 20 Hzなら約400 ms相当。

その後、

```text
T = 1
T = 2
T = 4
T = 8
T = 16
```

を比較。

長sequenceを扱う場合にはtruncated BPTTを実装する。

```yaml
temporal:
  detach_state_every: null
```

を最初からconfigに用意する。

---

# 26. State reset

sequence boundaryでは必ずstate reset。

```python
state = None
```

同一sequence中ではstateを維持可能なevaluation modeも作る。

training:

```text
random short clips
```

evaluation:

```text
full sequence streaming
```

の両方に対応する。

---

# 27. Optimizer

初期設定：

```yaml
optimizer:
  type: adamw
  lr: 1e-4
  weight_decay: 0.05

event_encoder_lr_mult: 0.1
temporal_lr_mult: 1.0
projector_lr_mult: 1.0
```

DINO pretrained initializationをEvent Encoderに使う場合はEvent EncoderのLRを小さくする。

Teacherはoptimizerへ入れない。

---

# 28. Precision / efficiency

以下をサポートする。

```yaml
training:
  amp: bf16
  gradient_clip: 1.0
  gradient_accumulation: 1
```

teacher cache使用時、GPUに載せるのは、

* Event Encoder
* LSTM/Mamba
* projector

だけにする。

---

# 29. Checkpoint

保存：

```python
{
    "event_encoder": ...,
    "temporal_model": ...,
    "h_projector": ...,
    "z_projector": ...,
    "optimizer": ...,
    "scheduler": ...,
    "step": ...,
    "config": ...,
}
```

Teacher weightはcheckpointへ含めない。

---

# 30. 実装すべきモデルclass

```text
models/
├── event_encoder.py
├── temporal/
│   ├── base.py
│   ├── identity.py
│   ├── lstm.py
│   └── mamba.py
├── teacher/
│   └── dinov3.py
├── projectors.py
├── objectives/
│   ├── distillation.py
│   ├── z_none.py
│   ├── z_distill.py
│   ├── z_future.py
│   └── cmax.py
└── event_state.py
```

main model:

```python
class EventStateModel(nn.Module):

    def encode_event(self, event):
        return z

    def update_state(self, z, state=None):
        return h, new_state

    def forward_sequence(self, events):
        return {
            "z": z_seq,
            "h": h_seq,
        }
```

Teacherは`EventStateModel`の内部へ組み込まない。

Teacherはtraining system側に置く。

---

# 31. Repo structure

```text
event-state/
├── README.md
├── pyproject.toml
├── configs/
│   ├── dataset/
│   │   └── dsec.yaml
│   ├── model/
│   │   ├── no_memory.yaml
│   │   ├── lstm.yaml
│   │   └── mamba.yaml
│   ├── teacher/
│   │   └── dinov3_vits16.yaml
│   └── experiment/
│       ├── gep_baseline.yaml
│       ├── h_distill_lstm.yaml
│       └── h_distill_lstm_zloss.yaml
├── event_state/
│   ├── data/
│   ├── models/
│   ├── losses/
│   ├── training/
│   └── evaluation/
├── tools/
│   ├── prepare_dsec.py
│   ├── cache_dinov3_features.py
│   └── visualize_features.py
├── train.py
├── evaluate.py
└── tests/
```

Hydra/OmegaConf形式を推奨。

hard-coded pathは禁止。

GEPは現在一部pathをconfig / preprocessing scriptへ直接設定する構成なので、新repoではその部分を改善する。GEP READMEもdataset pathなどをconfig側で変更する前提になっている。

---

# 32. 最重要比較実験

## Experiment A: GEP-style baseline

```text
Event
 ↓
Encoder
 ↓
z
 ↓
DINOv3 alignment
```

Temporal Modelなし。

$$
L=L_{\mathrm{DINO}}(z)
$$

---

## Experiment B: No alignment pretraining

```text
Event
 ↓
Encoder
```

random / DINO initialized encoder。

事前学習なし。

下流taskのbaseline。

---

## Experiment C: Ours — h alignment

```text
Event
 ↓
Encoder
 ↓
z
 ↓
LSTM
 ↓
h
 ↓
DINOv3
```

$$
L=L_{\mathrm{DINO}}(h)
$$

これが最初の本命。

---

## Experiment D: z + h alignment

```text
                ┌→ z → DINO
Event → Encoder
                └→ LSTM → h → DINO
```

$$
L
=
\lambda_zL_{\mathrm{DINO}}(z)
+
\lambda_hL_{\mathrm{DINO}}(h)
$$

これで、

> zまでRGBへ寄せた方が良いか

を調べる。

---

## Experiment E: h alignment + Event-native z objective

最終的な狙い。

```text
Event
 ↓
Encoder
 ↓
z ─── Event-native objective
 ↓
LSTM
 ↓
h ─── DINOv3 alignment
```

$$
L
=
L_{\mathrm{DINO}}(h)
+
\lambda L_{\mathrm{event}}(z)
$$

---

# 33. 最初に絶対に実行するablation

最低限以下の4本。

| Experiment | z supervision | Temporal | h supervision |
| ---------- | ------------- | -------- | ------------- |
| A          | DINOv3        | none     | none          |
| B          | none          | none     | none          |
| C          | none          | LSTM     | DINOv3        |
| D          | DINOv3        | LSTM     | DINOv3        |

最重要比較は、

```text
A vs C
```

つまり、

$$
z\rightarrow DINO
$$

と、

$$
z\rightarrow LSTM\rightarrow h\rightarrow DINO
$$

の比較。

---

# 34. Temporal length ablation

本命Cについて、

```text
T = 1
T = 2
T = 4
T = 8
T = 16
```

を比較。

DSEC 20 Hzの場合、概ね、

```text
1  → 50 ms
2  → 100 ms
4  → 200 ms
8  → 400 ms
16 → 800 ms
```

となる。

これにより、

> RGB representationを再現するためにどの程度event historyが必要か

を解析できる。

---

# 35. Downstream Evaluation

pretraining後、

```text
Event Encoder
+
Temporal Model
```

をbackboneとして評価する。

projection headは捨てる。

---

## Evaluation 1: Semantic Segmentation

DSEC Semanticを使用。

比較：

```text
z only
vs
h
```

つまり、

```text
seg_head(z_t)
```

と、

```text
seg_head(h_t)
```

を比較。

metric：

```text
mIoU
```

---

## Evaluation 2: Optical Flow

DSEC optical flow GTを使う。

DSEC training dataにはoptical flow GTが提供される。

主目的：

```text
z
```

と

```text
h
```

のどちらがmotionを保持しているか調べる。

metric：

```text
EPE
```

重要な仮説：

$$
z
$$

の方がflowに適し、

$$
h
$$

の方がsemanticsに適する可能性がある。

この差が出れば、z/h分離の強い証拠になる。

---

## Evaluation 3: Detection

DSEC Detectionを使用。

DSECにはobject detection labelsとtrack IDsが公開されている。

比較：

```text
Detector(z)
vs
Detector(h)
```

さらに、

```text
Detector(concat[z,h])
```

も評価。

これは非常に重要。

最終的には、

$$
[z_t,h_t]
$$

が最も汎用的である可能性がある。

---

# 36. Frozen evaluationを優先する

representation qualityを見るため、

最初はbackboneをfreezeする。

```text
Frozen z/h
+
small downstream head
```

を評価。

その後fine-tuning。

評価を、

```text
Linear / shallow probe
Frozen backbone
Full fine-tuning
```

に分ける。

これにより、

> pretrainingで本当にfeatureが改善したのか

と、

> downstream fine-tuningが強かっただけなのか

を分離する。

---

# 37. Representation diagnostics

数値性能だけでなくfeatureを解析する。

実装：

```bash
python tools/visualize_features.py
```

最低限、

### PCA visualization

```text
z PCA
h PCA
DINOv3 PCA
```

を同じsceneで比較。

### Cosine similarity map

指定patchをqueryとして、

```text
z similarity
h similarity
DINO similarity
```

を表示。

### Temporal feature stability

同一patchについて、

$$
\cos(f_t,f_{t+\Delta})
$$

を測る。

仮説：

```text
z: temporal variation large
h: temporal variation smaller
```

---

# 38. Static / low-event analysis

本研究では特に重要。

各frameについてevent countを計測。

```text
high event
medium event
low event
```

にbinningする。

それぞれで、

$$
D(h_t,r_t)
$$

と、

$$
D(z_t,r_t)
$$

を測る。

仮説：

low-event領域では、

```text
z → DINO similarity decreases strongly
h → DINO similarity remains relatively high
```

となる。

これが、

> Temporal MemoryがEventから消えたscene informationを保持する

という主張を直接検証する。

---

# 39. Occlusion / persistence evaluation

DSEC Detection track IDを将来的に利用する。

objectが一時的にeventを出さない、あるいは部分的にoccludeされた区間で、

```text
z object feature
h object feature
```

の継続性を測る。

Phase 1には不要だが、研究として非常に重要な追加評価候補。

---

# 40. Mamba比較

LSTMで仮説が確認できた後、

```text
LSTM
vs
Mamba
```

を行う。

条件：

* Event Encoder同一
* DINOv3 teacher同一
* sequence length同一
* embedding dimension同一

比較metric：

* downstream accuracy
* training memory
* streaming latency
* parameter count
* throughput
* long-sequence performance

---

# 41. DINOv2 vs DINOv3

RGB teacher comparison。

```text
DINOv2
vs
DINOv3
```

GEPとの比較上重要。

ただし最初の実装teacherはDINOv3のみでよい。

DINOv2 adapterを後から追加できるinterfaceにする。

---

# 42. Phase別実装計画

## Phase 0 — GEP reproduction

目的：

新repoのdata pipelineが正しいことを確認。

実装：

```text
DSEC
→ Event Representation
→ Event Encoder
→ z
→ DINOv3 alignment
```

Temporalなし。

これはGEP Stage 1のDINOv3版baselineに相当する。

GEP公式Stage 1はEvent Encoderとpretrained image encoderのcross-modal alignmentを行う。

---

## Phase 1 — Temporal state

実装：

```text
Event
→ Encoder
→ z
→ LSTM
→ h
→ DINOv3 alignment
```

これが最初の研究モデル。

---

## Phase 2 — z objectives

追加：

```text
direct DINO
future prediction
flow probe
```

---

## Phase 3 — Event-native learning

追加：

```text
CMax
feature warp
motion consistency
```

---

## Phase 4 — Mamba

```text
LSTM → Mamba
```

---

# 43. Phase 1完了条件

Code Agentは以下がすべて成立した時点でPhase 1完成と判断する。

### Data

* DSEC sequence loaderが動く
* RGB timestampとevent windowが一致する
* RGB/eventへ同一spatial augmentationを適用できる
* sequence boundaryを跨がない

### Teacher

* Frozen DINOv3 ViT-S/16が動く
* patch tokensを取得できる
* cache生成・読み込み可能

### Model

* Event Encoderが20ch入力を処理できる
* `[B,T,N,D]` のzが取得できる
* LSTMが`[B,T,N,D]`を処理できる
* hがDINO patch gridと一致する

### Training

* DINO lossが減少する
* resume可能
* AMP training可能
* validation lossを記録
* checkpoint保存可能

### Evaluation

最低限、

```text
z-DINO cosine similarity
h-DINO cosine similarity
```

をvalidationで出す。

---

# 44. 必須logging

Weights & BiasesまたはTensorBoard。

記録：

```text
train/loss
train/h_cosine
train/z_cosine
val/loss
val/h_cosine
val/z_cosine
event_count
grad_norm/event_encoder
grad_norm/temporal
learning_rate
GPU_memory
samples_per_second
```

特に、

```text
z-DINO similarity
h-DINO similarity
```

の両方を必ず記録する。

hだけsuperviseしていてもzのsimilarityを測定する。

---

# 45. Unit tests

最低限：

```text
test_event_voxelization
test_timestamp_alignment
test_sequence_boundary
test_spatial_transform_alignment
test_dinov3_patch_shape
test_event_encoder_shape
test_lstm_shape
test_state_reset
test_teacher_frozen
test_loss_backward
test_checkpoint_resume
```

---

# 46. 最初に使うconfig

```yaml
experiment:
  name: dsec_dinov3_lstm_hdistill

dataset:
  name: dsec
  sequence_length: 8
  event_window: rgb_interval
  event_bins: 10
  polarity_split: true

input:
  height: 448
  width: 640

teacher:
  type: dinov3_vits16
  frozen: true
  cache_features: true
  feature: patch_tokens

event_encoder:
  type: dinov3_vits16
  in_channels: 20
  pretrained_init: true

temporal:
  type: lstm
  hidden_dim: 384
  num_layers: 1

loss:
  h_distill:
    cosine: 1.0
    mse: 1.0

  z_objective:
    type: none

training:
  optimizer: adamw
  lr: 1.0e-4
  event_encoder_lr_mult: 0.1
  weight_decay: 0.05
  sequence_length: 8
  precision: bf16
```

---

# 47. 最初に実行する3 experiment

まず以下だけを実行可能にする。

### E0 — GEP-like DINOv3

```text
Event → Encoder → z → DINOv3
```

### E1 — Ours

```text
Event → Encoder → z → LSTM → h → DINOv3
```

### E2 — Dual supervision

```text
Event → Encoder → z ─────────→ DINOv3
                  ↓
                 LSTM
                  ↓
                  h ─────────→ DINOv3
```

この3つを同一DSEC split・同一Event Encoder・同一teacherで比較する。

---

# 48. 最初の研究判断

最初に確認するべきなのは、

$$
\boxed{
\text{E1がE0より良いか}
}
$$

である。

特に、

```text
low-event frames
static objects
longer temporal context
```

でE1が有利になるかを見る。

もし有利なら、

> Event representationをRGB VFMへ直接alignするのではなく、temporal integration後のstateをalignすべき

という研究仮説を支持する。

その後初めて、

$$
z_t
$$

へEvent-native objectiveを追加する。

---

# 49. 実装上の基本方針

GEP repositoryを直接forkして大規模改造するより、

```text
GEP = reference implementation
DINOv3 = submodule / dependency
EventState = new clean repository
```

とする。

ただし以下はGEPから積極的に参考・再利用する。

* DSEC preprocessing
* Event representation
* paired Event/RGB transforms
* DINOv3 loading
* Stage 1 alignment loss
* downstream DSEC code

GEPは既にDSEC preprocessingとDINOv3 branchを持っているため、完全なzero-base実装は避ける。

---

# 50. 最終的なモデル思想

最終的に目指すrepresentationは、

$$
\boxed{
z_t =
\text{Event-native fast representation}
}
$$

$$
\boxed{
h_t =
\text{persistent, semantic, temporally integrated state}
}
$$

である。

最初の実装では、

$$
h_t
$$

をDINOv3によって形成する。

その後、

$$
z_t
$$

へCMax / flow / temporal predictionなどEvent固有のsignalを追加する。

したがって開発順序は必ず、

```text
GEP-style z→DINO baseline
        ↓
z→LSTM→h→DINO
        ↓
z objective追加
        ↓
LSTM→Mamba
        ↓
downstream / streaming evaluation
```

とする。

最初のmilestoneは、

$$
\boxed{
\text{DSEC}
\rightarrow
\text{Event Encoder}
\rightarrow
z
\rightarrow
\text{LSTM}
\rightarrow
h
\rightarrow
\text{Frozen DINOv3 dense feature}
}
$$

のend-to-end trainingを安定して成立させることである。
