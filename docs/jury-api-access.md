# Jury API key

## Key policy

`FORTE_API_KEY` is a runtime secret for the protected `/api/v1` routes. It is read from the server environment or the ignored `.env` file and compared with `hmac.compare_digest`. The same key authorizes video uploads and job polling. The browser UI uses separate `/ui-api` routes and must never receive the jury key.

A key committed to an open-source repository, embedded in JavaScript, or shown on a public page is public to everyone. Do not use such a value as the jury credential. Generate a key for the running deployment and give it only to approved judges through a private channel. This repository has no deployed host or private judge-delivery channel configured.

## Shared key on the site

For the hackathon the jury takes one shared key from the site's **API** page. `GET /ui-api/key` creates `fk_…` (`secrets.token_urlsafe(32)`) on the first request, stores it in the `public_api_key` table and returns the same value on every later request and after restarts, so reloading the page never changes it. The page shows it right away and puts it into the cURL / PowerShell / Python examples. `/api/v1` accepts it next to the private `FORTE_API_KEY`.

The key is public by design: anyone who opens the site can read it. To stop it after the event set `FORTE_PUBLIC_API_KEYS=false` and restart the API: the page stops showing it and `/api/v1` rejects it; only `FORTE_API_KEY` keeps working. To rotate it instead, delete the row from `public_api_key`; the next request creates a new key. Tests: `backend/tests/test_api.py` (`test_public_key_*`).

## Generate or rotate

From the repository root:

```powershell
python infra/generate_jury_key.py
```

The script creates `.env` from `.env.example` if needed, writes a 256-bit URL-safe key, and prints it once for the operator to copy. `.env` is ignored by Git. Running the command again does not replace a valid key. To rotate it:

```powershell
python infra/generate_jury_key.py --rotate
```

To display the already configured key locally before sharing it with judges:

```powershell
python infra/generate_jury_key.py --show
```

Treat terminal output as sensitive. Do not run these commands in CI logs, screenshots, public streams, or a shell transcript. Restart the API process after changing `.env`; all previously issued copies stop working after rotation.

If `FORTE_API_KEY` is set directly in the process environment or a deployment secret manager, update that value there too: process environment values take precedence over `.env`.

## Call the API

Replace `JURY_API_BASE_URL` with the HTTPS URL supplied by the deployment owner. Keep the key in a local environment variable and send it only in the `X-API-Key` header:

```powershell
$env:FORTE_API_KEY = Read-Host "Jury API key"
$base = "https://<JURY_API_BASE_URL>" # TODO(deployment): insert the actual host after it is provisioned.
curl.exe -fsS -H "X-API-Key: $env:FORTE_API_KEY" -F "file=@C:/path/example.mov" "$base/api/v1/analyze"
curl.exe -fsS -H "X-API-Key: $env:FORTE_API_KEY" "$base/api/v1/jobs/JOB_ID"
```

The first request returns a `job_id`; poll the second endpoint until `status` is `done` or `failed`. Never treat `failed` as a clean result. Full response schemas are in [`contracts/openapi.json`](../contracts/openapi.json).

For interactive use, open `https://<JURY_API_BASE_URL>/docs`, choose **Authorize**, and enter the key. Swagger UI will send it as `X-API-Key` for the protected jury routes. Use this only on the deployed HTTPS host; the repository does not contain a live host yet.

## Deployment TODOs

- [ ] Deployment owner: provision the HTTPS API host and replace `JURY_API_BASE_URL` in the judge handout. No host is configured in this repository yet.
- [ ] Deployment owner: store `FORTE_API_KEY` in the host's secret manager or ignored `.env`; do not put it in GitHub Actions output, frontend variables, or the public repository.
- [ ] Organizer: distribute the generated key through the official private channel. This implementation uses one shared jury key, so individual judges cannot be revoked separately; rotate the shared key after the event or if it leaks.
- [ ] Deployment owner: add ingress rate limits and access restrictions before exposing the service publicly. `/ui-api` intentionally has no API key, and the application queue limit alone is not a per-IP abuse limit.
