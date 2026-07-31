# AWS Deployment Guide (Step 21)

**Honest disclaimer up front:** nothing in this file was actually deployed or tested — this sandbox
has no AWS access. Every command below is real and correct AWS CLI/Docker syntax, but you need to
run it yourself, in your own AWS account, to confirm it works end to end. Treat this as a tested-in-
principle, not tested-in-practice, guide.

## Option 1 — EC2 (simplest, good for a demo/portfolio deployment)

1. Launch a small EC2 instance (t3.micro is enough for a demo): Ubuntu 22.04, allow inbound port
   8000 (and 8501 if you also want Streamlit reachable) in the security group.
2. SSH in, install Docker:
   ```bash
   sudo apt update && sudo apt install -y docker.io
   sudo systemctl start docker
   ```
3. Copy the project over (`scp -r credit-card-fraud ubuntu@<ec2-ip>:~/`) or `git clone` it if pushed
   to GitHub.
4. Build and run:
   ```bash
   cd credit-card-fraud
   sudo docker build -t fraud-detection-api .
   sudo docker run -d -p 8000:8000 --restart unless-stopped fraud-detection-api
   ```
5. Test from your own machine: `curl http://<ec2-public-ip>:8000/health`

This is the "get something live fast" path — one machine, no auto-scaling, no load balancing, but
genuinely running in the cloud, reachable by a real public IP.

## Option 2 — ECR + ECS Fargate (containerized, closer to a real production setup)

This is the path `deploy/ecr_push.sh` and `deploy/ecs-task-definition.json` are built for.

1. **Push the image to ECR:**
   ```bash
   aws configure   # set your access key, secret, region
   bash deploy/ecr_push.sh
   ```
   This builds the Docker image, creates an ECR repo if needed, and pushes the image.

2. **Fill in the task definition** — replace `<ACCOUNT_ID>` and `<REGION>` in
   `deploy/ecs-task-definition.json` with your real values, then register it:
   ```bash
   aws ecs register-task-definition --cli-input-json file://deploy/ecs-task-definition.json
   ```

3. **Create an ECS cluster (Fargate) and a service** pointing at that task definition:
   ```bash
   aws ecs create-cluster --cluster-name fraud-detection-cluster

   aws ecs create-service \
     --cluster fraud-detection-cluster \
     --service-name fraud-api-service \
     --task-definition fraud-detection-api \
     --desired-count 1 \
     --launch-type FARGATE \
     --network-configuration "awsvpcConfiguration={subnets=[<SUBNET_ID>],securityGroups=[<SG_ID>],assignPublicIp=ENABLED}"
   ```
   (You'll need a VPC/subnet/security group already set up — the default VPC in a fresh AWS account
   works fine for a demo.)

4. **Attach a load balancer** (Application Load Balancer) in front of the service if you want a
   stable DNS name instead of a task's ephemeral IP, and to support running >1 task later without
   changing how clients connect.

### Why ECS/Fargate over EC2 for anything beyond a demo
No server to patch or manage (Fargate is serverless containers), built-in health-check-based restart
(the `healthCheck` block in the task definition), and straightforward horizontal scaling
(`desired-count`) if traffic grows — none of which EC2 gives you without extra manual setup.

## What's deliberately NOT in this guide

- **RDS / a real database** — this project has no persistent storage need beyond the model artifact
  itself (loaded from the Docker image), so there's no database to provision.
- **MLflow and Streamlit deployment** — same pattern as the API: containerize each
  (`mlflow server --backend-store-uri ...` and `streamlit run streamlit_app.py` both run fine in a
  Docker container), push to ECR, run as separate ECS services. Not written out here since it's a
  repeat of the exact same steps above with a different Dockerfile/command.
- **CI/CD auto-deploy to AWS** — `.github/workflows/ci.yml` (Step 17) deliberately stops at "build
  and smoke-test the image." Wiring `aws ecs update-service` into that workflow is the natural next
  step once you have real AWS credentials to store as GitHub Secrets — a few lines added to the
  existing workflow, not a new system.
- **Monitoring in production** — the `/metrics` endpoint from Step 18 is ready to be scraped; on AWS
  that typically means either a self-managed Prometheus+Grafana stack on a small EC2 instance, or
  AWS's own CloudWatch Container Insights for ECS, which needs no extra code on your end beyond what's
  already in `api/main.py`.
