# Application: run, test and deploy

The web application serves **detector v2** (Notebooks 20-21): a LightGBM model on UNSW-NB15 flow features with calibrated confidence,
class-wise conformal prediction sets, exact TreeSHAP explanations and verified incident reports (local LLM, MITRE ATT&CK / NIST / CISA).

| Part | Technology | Folder |
|---|---|---|
| Backend API | FastAPI, SQLAlchemy (PostgreSQL / SQLite), JWT authentication | `backend/` |
| Frontend | React 19 + TypeScript, Vite, Tailwind CSS, Recharts, TanStack Query | `frontend/` |
| Model bundle | JSON preprocessing + LightGBM text model + calibration + knowledge base (no pickles) | `serving/bundle/` (built, not in git) |
| Reports LLM | Ollama with `qwen2.5:7b` (optional; without it reports are evidence-only) | - |
| CI/CD | GitHub Actions: lint, tests, build on every push; images to GHCR on version tags | `.github/workflows/` |
| Deployment | docker-compose (one machine) or Kubernetes | `docker-compose.yml`, `deployment/` |

## 1. Build the model bundle

```bash
.venv/Scripts/python.exe scripts/build_serving_bundle.py
```

It writes `serving/bundle/` (about 16 MB) and **stops unless the serving code reproduces the research detector v2 exactly** on the 17,534
model-selection flows and the 82,332 official test flows (the manifest records the check). The bundle is not in git. Zip it and attach
it to each GitHub release as `serving-bundle.zip`:

```powershell
Compress-Archive -Path servingundle\* -DestinationPath serving\serving-bundle.zip -Force
```

On another machine, put that zip in `serving/`: `setup-app.cmd` unpacks it.

## 2. Run locally (development)

**Windows, one click:**
1. Run `setup-app.cmd` once. It:
   - installs the packages;
   - unpacks or builds the bundle;
   - asks for the first administrator;
   - writes `backend/.env` with a random JWT secret.
2. Run `start-app.cmd`. It opens a backend window and a frontend window, then opens http://localhost:5173. Close both windows to stop.

**By hand (any OS):**
1. Copy `backend/.env.example` to `backend/.env` and set `JWT_SECRET` and `ADMIN_PASSWORD`. The first administrator is created on the first start.
2. Backend: `.venv/Scripts/python.exe -m uvicorn app.main:app --app-dir backend --port 8000`. API docs are at http://localhost:8000/docs.
3. Frontend: run `npm --prefix frontend install` once, then `npm --prefix frontend run dev`. The app is at http://localhost:5173.
4. Optional reports LLM: start Ollama and run `ollama pull qwen2.5:7b`. The backend uses `http://127.0.0.1:11434`.

## 3. Tests

```bash
cd backend && ../.venv/Scripts/python.exe -m pytest      # 38 tests, synthetic model bundle (no real model needed)
cd frontend && npm run lint && npm test && npm run build
```

## 4. Deploy with Docker (one machine)

Requires Docker Desktop (Windows / macOS) or Docker Engine (Linux).

```bash
cp deployment/.env.example .env          # then fill in the secrets
docker compose build                     # or use the images published by the release workflow
docker compose up -d                     # http://localhost:8080
docker compose --profile llm up -d       # with the local LLM; then: docker compose exec ollama ollama pull qwen2.5:7b
```

The bundle in `./serving/bundle` is mounted read-only into the backend; staged uploads and generated PDF reports are kept in the `app-data` volume (`DATA_DIR=/data`). The LLM needs about 8 GB of GPU or system memory.

## 5. Release (CD)

```bash
git tag v1.0.0 && git push origin v1.0.0
```

The `Release` workflow builds and publishes `ghcr.io/swaraj08tech/botnet-attack-detection-backend` and `-frontend`. Then create the GitHub release
for the tag and attach `serving-bundle.zip`. A server updates with `docker compose pull && docker compose up -d`.

## 6. Kubernetes

`deployment/kubernetes/app.yaml` contains the namespace config, PostgreSQL, backend and frontend. Create the `app-secrets` secret and put the
bundle on the `model-bundle` volume first (instructions at the top of the file). Run Ollama separately if reports should use the LLM.

## Security notes

- Passwords are hashed with bcrypt; API access uses signed JWT tokens. The first administrator is created from `ADMIN_EMAIL` / `ADMIN_PASSWORD`.
- Failed logins are rate-limited per client address; logins, analyses and reports are written to an internal audit log.
- `ENVIRONMENT=production` refuses to start without a `JWT_SECRET`. Never commit `.env` files (they are gitignored).
- Uploads are limited to 50 MB and 200,000 rows and are validated before analysis (required features, values); raw errors are logged, never shown.
- The backend runs as one replica: analysis and report jobs run inside the API process.

## Honest scope

- Input must be **completed flows in the UNSW-NB15 42-column format**. Turning live packets into these columns needs a flow extractor
  (for example Argus or Zeek) and is outside this project.
- On the official test file the detector catches 98.2% of attacks but flags about 30% of normal flows (a shift between the official files).
  Re-calibrate on labelled traffic from the target network before relying on it.
