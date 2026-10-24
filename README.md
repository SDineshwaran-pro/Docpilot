# Multi-Agent Document Assistant (Google ADK + Gemini)

A personal RAG assistant built with Google's Agent Development Kit (ADK) and Gemini. Upload PDFs or notes, then ask by text, screenshot or voice. Runs on the Gemini API or Vertex AI, and deploys to Cloud Run.

## Flow

```
Question (text / voice / image)
        |
 [Multimodal step]   voice -> transcript, image -> extracted text (Gemini)
        |
 [planner agent]     ADK LlmAgent: routes the question
     |                         |
 [retrieval_agent]       [general_agent]
  tool: search_docs       answers directly
  (Gemini embeddings,
   cosine top-5)
        |
 Answer with [file, page] citations  ->  Streamlit chat UI
```

## Run locally

```bash
git clone https://github.com/<your-username>/doc-assistant.git
cd doc-assistant
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add GOOGLE_API_KEY, or switch to the Vertex AI block
streamlit run app.py
```

Free API key: https://aistudio.google.com/apikey. For Vertex AI locally, run `gcloud auth application-default login` and set the three Vertex variables in `.env`.

## Zero-cost setup (recommended)

GitHub (public repo) + Streamlit Community Cloud + a free Gemini API key from AI Studio. No billing account or card needed. Limits: the free Gemini tier is rate limited (the app retries automatically and shows a friendly message), and submitted content may be used by Google to improve its products, so use only non-sensitive sample documents.

## Deploy option 1: Streamlit Community Cloud (free)

1. Push the repo to GitHub (`.gitignore` keeps `.env` out).
2. At https://share.streamlit.io choose **New app**, pick the repo, branch `main`, file `app.py`.
3. Under **Advanced settings > Secrets** add `GOOGLE_API_KEY = "your-key"`.

## Deploy option 2 (optional, needs a billing account): Google Cloud Run + Vertex AI (no API key stored)

Cloud Run has a monthly free tier, but a billing account is required and Vertex AI is billed per use. Add `--max-instances 1` and a budget alert if you try it.

```bash
export PROJECT_ID=<your-project-id>
export REGION=asia-south1
gcloud config set project $PROJECT_ID
gcloud services enable run.googleapis.com aiplatform.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

# dedicated service account with only the Vertex AI permission it needs
gcloud iam service-accounts create doc-assistant-sa
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:doc-assistant-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

gcloud run deploy doc-assistant \
  --source . \
  --region $REGION \
  --service-account doc-assistant-sa@$PROJECT_ID.iam.gserviceaccount.com \
  --set-env-vars GOOGLE_GENAI_USE_VERTEXAI=TRUE,GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GOOGLE_CLOUD_LOCATION=us-central1 \
  --memory 1Gi \
  --no-allow-unauthenticated
```

`--no-allow-unauthenticated` keeps the service private. To open it to yourself in a browser, run `gcloud run services proxy doc-assistant --region $REGION`, or grant `roles/run.invoker` to specific users. Use `--allow-unauthenticated` only for a public demo.

## Notes

- The vector index lives in memory per browser session and is rebuilt when you upload files again.
- To add Google Drive as a source, create a second ADK tool using the Drive API with a read-only OAuth scope.
- Model names are configurable through `GEMINI_MODEL` and `EMBED_MODEL`.
