"""
streamlit_rag_app.py
Minimal, safe Streamlit RAG app (FAISS + Sentence-Transformers).
- Loads promotions CSV (promotions_6000.csv by default) or allows upload.
- Build/Reload FAISS index with a button (deferred heavy work).
- Query box to retrieve top-k documents.
- Optional OpenAI call to synthesize answers from retrieved docs.
"""
import streamlit as st
import os, time, pickle, io
import pandas as pd
import numpy as np
# streamlit_rag_app.py — top of file
try:
    import faiss
    USE_FAISS = True
except Exception:
    USE_FAISS = False

if USE_FAISS:
    # existing FAISS-based index init
    pass
else:
    # fallback: chroma in-memory
    import chromadb
    from chromadb.config import Settings
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=None))
    collection = client.get_or_create_collection(name="docs")


st.set_page_config(page_title="Promotions RAG", layout="wide")
st.title("Retail Promotions RAG (FAISS)")

# config
DEFAULT_CSV = "promotions_6000.csv"
INDEX_FILE = "faiss_promos.index"
DOCS_FILE = "rag_docs.pkl"
EMB_FILE = "emb_matrix.npy"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# helper: lazy imports for heavy libs (done inside button)
def safe_load_df(uploaded_file):
    if uploaded_file:
        try:
            return pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            return pd.read_excel(uploaded_file)
    else:
        if os.path.exists(DEFAULT_CSV):
            return pd.read_csv(DEFAULT_CSV)
        return None

st.sidebar.header("Data & Index")
uploaded = st.sidebar.file_uploader("Upload promotions CSV (optional)", type=["csv","xlsx"])
df = safe_load_df(uploaded)

if df is None:
    st.sidebar.warning(f"No dataset loaded and default file '{DEFAULT_CSV}' not found. Upload a CSV to proceed.")
    st.stop()
else:
    st.sidebar.success(f"Loaded dataset with {len(df)} rows")

# Show a small sample
if st.sidebar.checkbox("Show sample rows", value=True):
    st.subheader("Data sample")
    st.write(df.head(5))

# Build rag_text if missing
def ensure_rag_text(df):
    if "rag_text" not in df.columns:
        def make_summary(r):
            try:
                return (
                    f"Promo {r.promo_id} | {r.product_name} | {r.discount_type} {r.discount_value} | "
                    f"{r.start_date}—{r.end_date} | sales_before:{int(r.sales_before)} sales_during:{int(r.sales_during)} | "
                    f"lift_pct:{float(r.get('lift_pct',0)):.2f}% | incremental_profit:{float(r.get('incremental_profit',0)):.2f} | roi:{float(r.get('roi',0)):.2f}"
                )
            except Exception:
                return str(r.to_dict())
        df["rag_text"] = df.apply(make_summary, axis=1)
    return df

df = ensure_rag_text(df)

# Index build button (deferred heavy work)
if st.sidebar.button("Build / Rebuild Index"):
    with st.spinner("Building embeddings and FAISS index (this may take a while)..."):
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            # prepare docs
            documents = df["rag_text"].tolist()
            model = SentenceTransformer(EMBED_MODEL_NAME)
            # batch encode
            BATCH = 256
            embs = []
            for i in range(0, len(documents), BATCH):
                batch = documents[i:i+BATCH]
                e = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
                embs.append(e)
            emb_matrix = np.vstack(embs).astype("float32")
            faiss.normalize_L2(emb_matrix)
            d = emb_matrix.shape[1]
            index = faiss.IndexFlatIP(d)
            index.add(emb_matrix)
            # persist
            faiss.write_index(index, INDEX_FILE)
            with open(DOCS_FILE, "wb") as f:
                pickle.dump(documents, f)
            np.save(EMB_FILE, emb_matrix)
            st.success("Index built and saved.")
        except Exception as e:
            st.error("Index build failed: " + str(e))

# Try to load existing index
index = None
documents = None
if os.path.exists(INDEX_FILE) and os.path.exists(DOCS_FILE):
    try:
        import faiss
        with open(DOCS_FILE,"rb") as f:
            documents = pickle.load(f)
        index = faiss.read_index(INDEX_FILE)
    except Exception as e:
        st.warning("Failed to load saved index (will need rebuild): " + str(e))
        index = None

# If no index available, offer to build with small button
if index is None:
    st.info("No FAISS index loaded. Click 'Build / Rebuild Index' in the sidebar to create it.")
else:
    st.success(f"FAISS index loaded with {index.ntotal} vectors.")

# Query UI
st.header("Query the promotions RAG")
query = st.text_input("Enter a natural language question (e.g., 'Which discount worked best by ROI?')")

col1, col2 = st.columns([3,1])
with col2:
    k = st.number_input("Top-k", min_value=1, max_value=20, value=5)
    use_openai = st.checkbox("Use OpenAI to synthesize answer", value=False)
    if use_openai:
        openai_key = st.text_input("OpenAI API key (optional)", type="password")
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key

def retrieve_faiss(index, model, query, k=5):
    qv = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(qv)
    D, I = index.search(qv, k)
    hits = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0: continue
        hits.append({"score": float(score), "doc": documents[idx], "idx": int(idx)})
    return hits

if st.button("Search") and query.strip():
    if index is None:
        st.error("No index available. Build it first.")
    else:
        with st.spinner("Retrieving..."):
            try:
                from sentence_transformers import SentenceTransformer
                import faiss
                model = SentenceTransformer(EMBED_MODEL_NAME)
                hits = retrieve_faiss(index, model, query, k=k)
            except Exception as e:
                st.error("Retrieval failed: " + str(e))
                hits = []

        st.subheader("Retrieved promos")
        for i, h in enumerate(hits, 1):
            st.markdown(f"**Rank {i} — score {h['score']:.3f}**")
            st.text(h["doc"])

        # optional LLM synthesis (OpenAI)
        if use_openai and os.getenv("OPENAI_API_KEY"):
            try:
                import openai
                openai.api_key = os.getenv("OPENAI_API_KEY")
                context = "\n".join([h["doc"] for h in hits])
                prompt = f"You are an analyst. Use ONLY the CONTEXT below to answer the QUESTION.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nREPLY: short answer + top 3 promos (promo_id and numeric KPIs)."
                resp = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user","content":prompt}],
                    temperature=0.0,
                    max_tokens=400,
                )
                st.subheader("LLM Answer")
                st.write(resp["choices"][0]["message"]["content"].strip())
            except Exception as e:
                st.error("OpenAI call failed: " + str(e))
        elif use_openai:
            st.warning("OpenAI key not provided. Set OPENAI_API_KEY env var or enter key in sidebar.")
