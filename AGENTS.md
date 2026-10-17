# Team development rules

- Before **every commit and every push**, inspect the diff, fetch the latest remote refs, and check for concurrent work or conflicts. Integrate relevant remote changes locally and run the affected checks before pushing. Never force-push shared branches.
- Preserve the work of other team members. Backend owns `backend/`, `contracts/`, and infrastructure. Frontend owns `frontend/`. Coordinate changes to shared contracts.
- Do not commit datasets, uploaded videos, extracted frames, model weights, secrets, runtime databases, or local environment files. Use generated synthetic fixtures for automated tests.
- All video analysis must execute locally. Never send videos, frames, or their descriptions to external AI services. Do not implement face identification or audio analysis.
- Missing ML, timeouts, and exceptions must never become `no_violation`. Partial analysis (team decision 18.10.2026): violations found on the covered part are reported; with none found and coverage > 0 the result is `no_violation` flagged `partial_result` in diagnostics; nothing covered or coverage not reported stays `failed`.
- Keep the jury API compatible with the case. The jury key must never be exposed to frontend JavaScript. Public UI routes are separate.
