# Deployment Runbook

## Triggers
- **CI (PR)**: runs syntax check on pull requests to `main`/`deploy`.
- **CD (push)**: deploys automatically on push/merge to `main`.

## Secrets (GitHub Actions)
- `EC2_HOST`, `EC2_USER`, `EC2_SSH_KEY`, `EC2_PORT`

## Deploy Flow (CD)
1. GitHub Actions SSH into EC2.
2. `git reset --hard origin/main`
3. `docker compose up -d --build`
4. Health check:
   - `http://127.0.0.1/nginx-health`
   - `http://127.0.0.1/api/v1/health`

## Rollback (Manual)
Use either option below.

### Option A) Roll back to a known commit
```bash
cd /opt/callact-backend
git fetch --all
git reset --hard <commit_sha>
sudo docker compose -f docker-compose.prod.yml -f docker-compose.prod.override.yml up -d --build
```

### Option B) Revert via PR
1. Create a PR that reverts the bad commit(s).
2. Merge to `main` (CD will deploy automatically).

## Notes
- `.env.prod` is untracked and stays on the server during deploys.
- EC2 must allow SSH (port 22) from GitHub Actions to run CD.
