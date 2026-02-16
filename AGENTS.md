# Project Rules (Codex / contributors)

1) Never hard-code dataset paths in code.
   - All dataset locations must come from YAML configs under configs/datasets/local/ (local-only).

2) Never commit datasets or audio files to GitHub.

3) Every dataset must be converted to a unified metadata schema:
   Columns:
   utt_id, path, dataset, split, label, speaker_id, attack_id, sr, duration_sec

4) Every new module should include unit tests in /tests.

5) Interventions must be deterministic where possible:
   - seed control
   - fixed transforms unless randomness is explicitly required

6) Results must be written to /results (ignored by git).
