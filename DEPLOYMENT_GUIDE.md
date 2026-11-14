# ModelHub – Zero-Credit-Card Free Cloud Deployment & Demo Guide
> **ModelHub: Web-Based ML Model Training and Prediction Platform**  
> *Production-Ready Multi-User Architecture with JWT Authentication & Model Ownership*

---

## 1. Architecture & Security Highlights

Your platform is completely updated with enterprise-grade user authentication and access control:
- **Authentication**: Users create accounts and log in with their **Email** and **Password**.
- **Cryptographic Hashing**: Passwords are saved with salted PBKDF2-SHA256 (100,000 hashing rounds), never in plain text.
- **Authorization**: Stateless **JSON Web Tokens (JWT)** protect all endpoints.
- **Access Control & Ownership**:
  - Anonymous users are completely blocked (HTTP 401 Unauthorized) from listing models, training datasets, and running predictions.
  - Model uploads and CSV training sessions are tagged with the creator's `user_id` and `user_email`.
  - Predictions are private: users only see their own prediction history and personalized analytics summary.

---

## 2. 100% Free Cloud Deployment (No Credit Card Required)

To deploy live on the web with a custom public URL for your project review without needing a credit card, we use **Render** (for the FastAPI backend) and **Vercel** (for the React Vite frontend).

### Step 1: Push Your Code to GitHub (Takes 1 Minute)

1. Open your browser and go to [github.com](https://github.com) (sign in or create a free account).
2. Click the **+** icon in the top right -> **New repository**.
3. Name it: `modelhub` (choose **Public** or **Private**) -> Click **Create repository**.
4. In your PowerShell terminal in `c:\dpti`, run:
   ```powershell
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/modelhub.git
   git branch -M master
   git push -u origin master
   ```
*(Replace `<YOUR_GITHUB_USERNAME>` with your actual GitHub username).*

---

### Step 2: Deploy Backend to Render (Free Tier, No Card)

1. Go to [render.com](https://render.com) and click **Sign Up** -> **Continue with GitHub**.
2. From the Render Dashboard, click **New +** -> **Web Service**.
3. Click **Build and deploy from a Git repository** -> Select your `modelhub` repository.
4. Render will automatically read the included `render.yaml` file!
   - **Name**: `modelhub-backend`
   - **Root Directory**: `backend`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Select **Free ($0/month)**
5. Click **Create Web Service**.
6. Render will build and deploy your API in ~2 minutes. Once finished, copy your free backend URL:
   `https://modelhub-backend-xxxx.onrender.com`

---

### Step 3: Deploy Frontend to Vercel (Free Tier, No Card)

1. Go to [vercel.com](https://vercel.com) and click **Sign Up** -> **Continue with GitHub**.
2. Click **Add New...** -> **Project**.
3. Select your `modelhub` repository from the list and click **Import**.
4. In the Project Configuration:
   - **Framework Preset**: Vite (detected automatically).
   - **Root Directory**: Click *Edit* and select `frontend`.
   - Expand **Environment Variables**:
     - Key: `VITE_API_URL`
     - Value: `https://modelhub-backend-xxxx.onrender.com` *(your Render backend URL from Step 2)*.
5. Click **Deploy**.
6. In ~45 seconds, your live frontend is ready with free SSL:
   `https://modelhub-xxxx.vercel.app`

🎉 **You now have a live, cloud-deployed, authenticated ML platform ready for examiners!**

---

## 3. Local College Demo Mode (Offline / Zero Internet)

If the examination or presentation is conducted offline or on a campus intranet, you can run everything locally with zero cloud dependencies:

### 1. Start the Backend API:
Open PowerShell:
```powershell
cd c:\dpti
c:\dpti\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
*Backend runs at: `http://127.0.0.1:8000` (Interactive API docs at `http://127.0.0.1:8000/docs`)*

### 2. Start the Frontend Dashboard:
Open a second PowerShell window:
```powershell
cd c:\dpti\frontend
npm run dev
```
*Frontend runs at: `http://localhost:5173`*

---

## 4. How to Present and Demo to Examiners

Here is the recommended script for your viva/presentation:

1. **User Authentication & Protection**:
   - Open `http://localhost:5173` (or your live Vercel URL).
   - Show that anonymous access is prohibited: no models, predictions, or datasets can be viewed without logging in.
   - Switch to **Create Account**: Register a new user (e.g. `alex@college.edu`).
   - Log in and point out the user profile badge in the sidebar displaying the authenticated user.
2. **Automated ML Training from CSV**:
   - Navigate to **Train from CSV**.
   - Upload any CSV (e.g., student grades, house prices, salary data, or iris).
   - The platform automatically inspects columns, identifies numeric and categorical features, suggests the task (classification vs regression), and trains a scikit-learn pipeline with preprocessing (imputation, scaling, one-hot encoding).
   - Show the held-out validation metrics (Accuracy, F1-Score, R², RMSE).
3. **Model Registry & Access Ownership**:
   - Navigate to **Model Registry**.
   - Show the newly trained model: note the **"My Model"** ownership badge and the owner's email address.
4. **Real-Time Interactive Prediction**:
   - Go to **Run Prediction**.
   - The dynamic form automatically generates form inputs according to the model's feature schema.
   - Enter input values and click **Run Prediction**.
   - Show the instantaneous prediction result and model latency.
5. **Prediction History & Analytics Isolation**:
   - Navigate to **Prediction History**: Show that predictions are saved with timestamp and model ID.
   - Open **Analytics**: Show charts of model usage, success rate, and prediction distribution.
6. **Sign Out & Multi-Tenancy Proof**:
   - Click **Sign Out**.
   - Register a second user (e.g., `examiner@college.edu`).
   - Show that the second user cannot see the previous user's private prediction history, proving complete multi-tenant isolation.
