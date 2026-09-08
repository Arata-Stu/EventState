# Downstream evaluation roadmap

## 目的

E0 / E2 / E4の100,000 step最終重みについて、DSEC-Detection公式41 train / 6 validation /
13 testを変更せず、次の順に事前学習の価値と時系列stateの価値を切り分ける。

1. Frozen detection
2. End-to-end fine-tuningとmatched scratch
3. State-sensitive detection
4. DINO + video-JEPA teacher fusion

公式testは設計選択に使わない。head構造、学習時間、seedをvalidation上で固定した後に一度だけ評価する。

## Phase 1: Frozen detection

E0は`z`、E2/E4は`h`を主比較とする。backboneをsequence先頭から連続実行し、特徴mapをDSEC-Detの
distorted 640x430座標へwarpして保存する。各条件で同一初期化のYOLOX型headだけを学習し、
3 seedのCOCO mAP@[.50:.95]を平均±標準偏差で報告する。

補助表として各checkpointの`z` / `h` / `concat[z,h]`も比較できるが、主比較を見た後に行う。
DAGRの320x215条件は計算量比較用の任意ablationとし、主結果には使わない。

## Phase 2: Fine-tuningとScratch

Fine-tuningは事前学習済みevent encoderと1層LSTMをYOLOX型headと同時に更新する。RGB、DINO teacher、
teacher cache、蒸留lossは使わない。Scratchはcheckpointからarchitecture設定だけを読み、event encoder、
LSTM、headをランダム初期化する。

Fine-tuningとScratchで、公式split、event表現、clip長、box filter、augmentation、optimizer、epoch数、
評価器を一致させる。ScratchはE0相当のnon-recurrent版と、E2/E4共通の1層LSTM版の2条件で十分である。

現在のend-to-end loaderは、学習時にはlabel frameで終わる固定長clipを返し、validation時には未ラベル
frameを含む全frameを時系列順に返す。したがってLSTM stateはsequence境界だけでresetされ、評価対象で
ないframeでも更新される。

## Phase 3: State-sensitive detection

再学習なしで、E4-hの同一headと同一validation streamに次の介入を適用する。

- stateを毎frame reset
- 連続するevent入力を1 / 2 / 4 / 8 frame欠落
- event蓄積窓を1 / 2、1 / 4、1 / 8へ短縮
- 欠落後の回復曲線を、欠落直前・欠落中・復帰後のframe別APまたは検出一致率で測る

通常mAPだけでは短い障害区間が平均化されるため、gap内mAP、復帰後1--8 frameのmAP、連続条件との差を
併記する。ここで差が出なければ、LSTMは現在のDSEC-Det条件では主要な性能源ではないと判断できる。

## Phase 4: DINO + video-JEPA fusion

JEPA実装を始める前に、teacher adapterが次の共通形式を返すようにする。

- `tokens`: `[B, T_teacher, N_teacher, D_teacher]`
- 空間patch中心と有効mask
- 各teacher tokenが覆う入力時刻区間
- teacher固有の正規化と入力sampling metadata

DINOは`T_teacher=1`の空間teacher、video-JEPAは複数frameを覆う時空間teacherとして扱う。異なるtoken数・
次元をstudentへ直接一致させず、teacherごとに独立したprojectorとtoken alignerを置く。projectorなしで
encoderをDINO学習後にfreezeし、LSTMだけをJEPAへ合わせる条件は有効なablationだが、主方式にはしない。
それではJEPAの座標系・表現基底をLSTMへ直接強制し、encoderとの共同適応を測れないためである。

最初の融合実験は、(a) DINOのみ、(b) JEPAのみ、(c) DINO空間loss + JEPA時間loss、
(d) DINO事前学習後encoder freeze + JEPA-LSTM、の4条件とする。Phase 1--3の評価系を固定してから着手し、
teacherを変えたことで下流protocolまで同時に変えない。

## 実装状態

- Frozen cache・DSEC-Det warp・YOLOX head学習: 実装済み
- E0/E2/E4 frozen一括実行: 実装済み
- Fine-tuning / matched Scratch: 実装済み、Linux V100での実データ確認待ち
- State-sensitive detection metric: 未実装
- 共通teacher adapter / JEPA token aligner / JEPA loss: 未実装

JEPAは公開checkpointの入力fps、clip長、patch/tubelet形状と利用条件を確定してから実装する。
