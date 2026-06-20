# Deploying the Air India Chatbot (Docker + AWS EC2)

The app is containerized. The Docker image already contains the **pre-built index**
(Chroma + BM25) and the **reranker model**, so the server starts and answers with no
ingestion step. The only thing supplied at runtime is your **API key**.

> **Important:** the search index (`chroma_db/`, `data/bm25.pkl`) is **not in GitHub**
> (it's git-ignored). So you can't just `git clone` on the server and build there —
> it would have no index. The clean path is: **build the image locally (where the
> index exists) → push to a registry → pull on EC2.**

---

## 1. Test the image locally first
```bash
docker build -t air-india-chatbot .
docker run --rm -p 8000:8000 --env-file .env air-india-chatbot
# open http://127.0.0.1:8000
```
- `--env-file .env` passes your `GOOGLE_API_KEY` at runtime (never baked into the image).
- `-p 8000:8000` maps host port 8000 to the container.

---

## 2. Push the image to a registry
Pick one.

**Docker Hub (simplest):**
```bash
docker tag air-india-chatbot <your-dockerhub-username>/air-india-chatbot:latest
docker login
docker push <your-dockerhub-username>/air-india-chatbot:latest
```

**Amazon ECR (AWS-native):**
```bash
aws ecr create-repository --repository-name air-india-chatbot
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <acct>.dkr.ecr.<region>.amazonaws.com
docker tag air-india-chatbot <acct>.dkr.ecr.<region>.amazonaws.com/air-india-chatbot:latest
docker push <acct>.dkr.ecr.<region>.amazonaws.com/air-india-chatbot:latest
```

---

## 3. Launch the EC2 instance
- **AMI:** Ubuntu 22.04 (or Amazon Linux 2023)
- **Instance type:** **`t3.small` (2 GB) minimum, `t3.medium` (4 GB) recommended.**
  Do NOT use `t2.micro`/`t3.micro` (1 GB) — torch + the reranker will run out of memory.
- **Security Group (inbound rules):**
  - `22` (SSH) — **only from your IP**
  - `8000` (the app) — from anywhere (`0.0.0.0/0`) for a public demo, or your IP to keep it private
- Give it a small bit of disk (e.g., 16–20 GB) for the image.

---

## 4. Install Docker on the instance
```bash
ssh ubuntu@<EC2_PUBLIC_IP>
sudo apt-get update && sudo apt-get install -y docker.io
sudo usermod -aG docker $USER && newgrp docker
```

---

## 5. Put the API key on the server (NOT in git)
Create the env file directly on the instance:
```bash
echo "GOOGLE_API_KEY=YOUR_REAL_KEY" > ~/.env
chmod 600 ~/.env
```
(Alternative: pass `-e GOOGLE_API_KEY=...` on `docker run`, or use AWS Secrets Manager.)

---

## 6. Pull and run
```bash
docker pull <your-image-ref>            # from step 2
docker run -d --name chatbot \
  -p 8000:8000 \
  --env-file ~/.env \
  --restart unless-stopped \
  <your-image-ref>
```
Now open **http://<EC2_PUBLIC_IP>:8000**.

`--restart unless-stopped` keeps it running across reboots/crashes.

---

## 7. Verify / operate
```bash
curl http://localhost:8000/health        # -> {"status":"ok"}
docker logs -f chatbot                    # watch logs
docker restart chatbot                    # restart (e.g., after changing the key file)
```

---

## Notes & gotchas
- **RAM:** confirmed need is ~1.5–2.5 GB → that's why `t3.small`/`t3.medium`.
- **Gemini free quota:** ~20 generations/day on the free tier. A public app will hit
  this fast — enable billing on the Gemini API for real use. Built-in **rate limiting**
  (12 req/min per IP) and a **1000-char input cap** protect against spam/abuse.
- **Chat history persistence:** SQLite lives inside the container and is lost on
  redeploy. To persist it, mount a volume:
  `-v /home/ubuntu/chatdata:/app/data`.
- **HTTPS (later):** for a real domain, put the app behind an HTTPS reverse proxy
  (Caddy/Nginx) or an AWS ALB + ACM certificate, and route 443 → 8000.
- **Updating the app:** rebuild locally → push → on EC2 `docker pull` + `docker restart`.
- **Managed alternative:** the same image runs on **AWS App Runner** (no server to
  manage) — point it at the ECR image, set the key as a secret, port 8000.
