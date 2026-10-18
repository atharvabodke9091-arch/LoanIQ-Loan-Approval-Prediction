# LoanIQ — Loan Approval Prediction Dashboard

An interactive Streamlit dashboard for exploring loan application data and comparing machine-learning classifiers.

## Features
- Dataset upload or automatic loading of `loan_approval_data.csv`
- Application overview and interactive charts
- Data exploration and missing-value inspection
- Applicant-level prediction form
- Logistic Regression, K-Nearest Neighbors (KNN), and Gaussian Naive Bayes comparison
- Accuracy, weighted precision, weighted recall, weighted F1-score, and confusion matrix

## Dataset
Place `loan_approval_data.csv` in the project root, or upload a CSV from the dashboard sidebar. The target column should be named `Loan_Approved` (common variations such as `Approved` are also recognized).

The app discovers input columns from the uploaded dataset. An `Applicant_ID`, `Application_ID`, or `ID` column is excluded from model training to reduce identifier leakage.

## Run locally
```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Upload to GitHub
1. Create a new repository on GitHub, for example `loan-approval-prediction`.
2. Put `app.py`, `requirements.txt`, `README.md`, and `.gitignore` in your project folder.
3. Add your dataset only if you are allowed to share it and it contains no private/personal information.
4. From Terminal in the project folder:
```bash
git init
git add app.py requirements.txt README.md .gitignore
git commit -m "Build LoanIQ loan approval dashboard"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/loan-approval-prediction.git
git push -u origin main
```
Replace `YOUR-USERNAME` with your GitHub username. If you choose to include a public dataset, add it separately after checking its license and privacy.

## Deploy on Streamlit Community Cloud
1. Push the repository to GitHub.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Select the repository, branch `main`, and main file `app.py`.
4. Deploy. If the dataset is not committed to the repository, upload it through the app sidebar.

## Notes
- The dashboard retrains models when it starts or when the page reruns; this is suitable for a class project and small datasets.
- Metrics are evaluated on a held-out split. They are not a guarantee of future performance.
- This project is for learning and demonstration, not real credit underwriting.
