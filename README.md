# TG Anti-Collision Bot V1.7

GitHub + Railway deployment package.

## Core behavior
- Group/supergroup only; private chat is disabled.
- Exact image detection first, followed by perceptual/local/visual similarity checks.
- Historical import rule: **one imported photo = one customer**.
- A historical photo does not need name/age/job/profile text to become a customer record.
- Exact duplicate photos are not inserted twice.
- Customer name is auxiliary data and is not the sole collision criterion.

## Railway
Set `BOT_TOKEN` and `DATA_DIR=/data`, then mount a Railway Volume at `/data`.
Keep one replica when using Telegram long polling.

See `DEPLOY_RAILWAY.txt` for step-by-step deployment.
