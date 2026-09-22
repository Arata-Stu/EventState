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

