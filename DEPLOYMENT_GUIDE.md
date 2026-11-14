# ModelCloud – Zero-Cost AWS Deployment & Operations Guide
> **Cloud-Based ML Model Serving Platform**  
> *Designed for College Student Projects with Strict $0 Free-Tier Controls*

---

## 1. Zero-Cost Architecture Overview

To ensure the deployment remains **completely free** (0.00 USD), ModelCloud uses this lightweight, production-ready stack:

| Component | Service | Free Tier Allowance | Cost |
| :--- | :--- | :--- | :--- |
| **Backend Compute** | Amazon EC2 (`t2.micro` or `t3.micro`) | 750 hours/month (enough for 24/7 run for 12 months) | **$0.00** |
| **Model & CSV Storage** | Amazon S3 + Local Caching | 5 GB standard storage, 20,000 GET, 2,000 PUT/month | **$0.00** |
| **Database** | SQLite on EC2 EBS Storage | 30 GB EBS General Purpose (gp2/gp3) storage included | **$0.00** |
| **Frontend Hosting** | Vercel / Netlify (or Nginx on EC2) | 100 GB bandwidth/month, unlimited free SSL | **$0.00** |
| **Security & Auth** | AWS IAM Roles for EC2 | Always Free | **$0.00** |

> **Why No Amazon RDS?**  
> Amazon RDS (managed MySQL/Postgres) frequently incurs unexpected charges from backup storage, multi-AZ toggles, or exceeding IOPS. Storing the SQLite database (`ml_platform.db`) directly on the EC2 root EBS volume is 100% free, requires zero database management, persists across reboots, and easily handles thousands of student demo predictions.

---

## 2. Step 1: Set Up AWS Cost Controls First (Must Do!)

Before launching any resources, set up AWS Budgets to alert you if spending ever exceeds $1:

1. Log in to the [AWS Management Console](https://console.aws.amazon.com/).
2. In the top search bar, type **AWS Budgets** and select it.
3. Click **Create budget** -> Choose **Cost budget (Recommended)**.
4. Set:
   - **Budget Name**: `ModelCloud-ZeroCost-Budget`
   - **Period**: Monthly
   - **Budgeted amount**: `$1.00`
5. Configure Alerts:
   - Alert 1: When actual cost reaches **50% ($0.50)** -> Enter your email.
   - Alert 2: When actual cost reaches **80% ($0.80)** -> Enter your email.
   - Alert 3: When forecasted cost reaches **100% ($1.00)** -> Enter your email.
6. Click **Confirm budget**.

---

## 3. Step 2: S3 Bucket Setup (Model Artifact Storage)

1. Open **Amazon S3** in the AWS Console.
2. Click **Create bucket**.
   - **Bucket name**: e.g., `modelcloud-storage-[yourname]` (must be globally unique).
   - **AWS Region**: Choose closest to you (e.g., `us-east-1` or `ap-south-1`).
   - **Block Public Access**: Keep **all blocked** (the bucket stays completely private).
3. Click **Create bucket**.

---

## 4. Step 3: Launch the EC2 Backend Server

1. Open **Amazon EC2** -> Click **Launch Instance**.
2. **Name**: `ModelCloud-Backend`
3. **OS Image**: `Ubuntu Server 24.04 LTS (HVM), SSD Volume Type` (Free tier eligible).
4. **Instance Type**: `t2.micro` (or `t3.micro` depending on region eligibility).
5. **Key pair**: Create new key pair `modelcloud-key.pem` and download it.
6. **Network / Security Group**:
   - Allow **SSH traffic** from `Anywhere` (port 22) or `My IP`.
   - Allow **Custom TCP** on port `8000` from `Anywhere` (0.0.0.0/0) for FastAPI.
7. **Storage**: Keep default `8 GiB gp3` (up to 30 GiB is free).
8. Click **Launch Instance**.

---

## 5. Step 4: Attach IAM Role to EC2 (No Hardcoded Keys!)

1. Go to **IAM Console** -> **Roles** -> **Create role**.
2. Select **AWS service** -> Use case: **EC2** -> Click **Next**.
3. In Permissions, attach: `AmazonS3FullAccess` (or create a policy limited to your bucket).
4. Role name: `ModelCloud-EC2-S3-Role` -> Click **Create role**.
5. Back in **EC2 Console**:
   - Right-click your instance -> **Security** -> **Modify IAM role**.
   - Select `ModelCloud-EC2-S3-Role` and click **Update IAM role**.

*Your backend can now upload and download from S3 without putting any AWS Access Keys or Secret Keys in the code!*

---

## 6. Step 5: Deploy the Backend on EC2

Connect to your EC2 instance via SSH:
```bash
ssh -i "modelcloud-key.pem" ubuntu@<YOUR_EC2_PUBLIC_IP>
```

Run the following commands on the server:

```bash
# 1. Update system packages
sudo apt update && sudo apt install -y python3-pip python3-venv git

# 2. Clone or copy your project folder
git clone <YOUR_GITHUB_REPO_URL> dpti
cd dpti/backend

# 3. Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
nano .env
```

In `.env`, configure:
```ini
APP_NAME=ML Model Serving Platform
STORAGE_BACKEND=s3
AWS_S3_BUCKET_NAME=modelcloud-storage-[yourname]
AWS_REGION=us-east-1
# Leave AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY empty (handled by EC2 IAM Role)
```

### Enable Automatic Startup (Systemd Service)
```bash
# Copy systemd service file
sudo cp modelcloud.service /etc/systemd/system/

# Reload systemd, start and enable service
sudo systemctl daemon-reload
sudo systemctl start modelcloud
sudo systemctl enable modelcloud

# Check backend status
sudo systemctl status modelcloud
```

Test your backend in browser or curl:
`http://<YOUR_EC2_PUBLIC_IP>:8000/health` -> `{"status": "healthy"}`

---

## 7. Step 6: Deploy Frontend (100% Free on Netlify or Vercel)

### Option A: Free Vercel Deployment (Recommended)
1. Push your repository to GitHub.
2. Sign in to [Vercel](https://vercel.com) using your GitHub account.
3. Click **Add New Project** -> Select your repository.
4. Set **Root Directory** to `frontend`.
5. Under **Environment Variables**, add:
   - `VITE_API_URL` = `http://<YOUR_EC2_PUBLIC_IP>:8000`
6. Click **Deploy**.
7. Your app is now live at `https://your-project.vercel.app`!

### Option B: Free Netlify Deployment
1. Sign in to [Netlify](https://netlify.com).
2. Click **Add new site** -> **Import an existing project**.
3. Select your repository.
4. Base directory: `frontend`
5. Build command: `npm run build`
6. Publish directory: `frontend/dist`
7. Environment variables: `VITE_API_URL` = `http://<YOUR_EC2_PUBLIC_IP>:8000`
8. Click **Deploy site**.

---

## 8. How to Stop / Delete AWS Resources (Prevent Any Billing)

When your college presentation or testing is finished:

1. **Stop the EC2 Instance** (Stops all compute hours):
   - EC2 Console -> Select `ModelCloud-Backend` -> **Instance state** -> **Stop instance**.
   - *Note:* Stopped instances incur zero compute charges. Only a tiny fraction of a cent per month for the 8GB disk if kept indefinitely.
2. **Delete the S3 Bucket**:
   - S3 Console -> Select your bucket -> **Empty** -> Then **Delete**.
3. **Terminate EC2 Instance** (Complete deletion):
   - Instance state -> **Terminate instance** (deletes instance and its EBS volume completely).
