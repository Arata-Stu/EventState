# EventState project instructions

このリポジトリで作業を始めるときは、回答・実装・実験提案の前に必ず次を読む。

1. `docs/session_handoff.md` — サーバーパス、現状、次の作業、研究上の注意
2. `docs/experiment_results.md` — 実験結果の正本
3. activity-aware / ScaleEvent関連の作業では `docs/scale_event_distillation.md`

新しい実験結果をユーザーから受け取ったら、会話内だけに残さず
`docs/experiment_results.md`へ条件、seed、split、出力パスとともに追記する。
研究方針、サーバーパス、実行中／次に行う作業が変わった場合は
`docs/session_handoff.md`も更新する。

学習用サーバーとローカルMacを混同しない。重い学習や依存関係を使うテストは
ユーザーから明示的に指示されない限りMacでは実行せず、サーバー用コマンドとして提示する。

学習サーバーでは空きGPUを積極的に使い、独立した実験条件をGPUごとに並列実行する。
Depthの下流タスクはデータセット・条件を合わせて同時2ジョブまでとし、別に起動済みのDepthも数える。
残りの空きGPUにはSemanticなど他の独立したジョブを割り当ててよい。

長時間の実験を案内するときは、起動した端末で進捗を読めるコマンドを示す。バックグラウンド実行ではログを保存して追跡できるようにする。残り時間と日本時間の終了見込みは現在の処理速度か同条件に近い過去の実測から求め、並列数・対象工程・未計測の工程を明示する。

図表・凡例・進捗表示では条件IDと条件名を併記する。表記は「E0 Encoder（z蒸留）」「E1 Encoder＋LSTM（h蒸留）」「E2 Encoder＋LSTM（z+h蒸留）」とする。
