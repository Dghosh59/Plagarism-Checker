import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os

# Constants
PERSIST_DIR = "db"
COLLECTION_NAME = "research_papers"
SIMILARITY_THRESHOLD = 0.8
TOP_K = 3  # Top results per chunk

# App layout
st.set_page_config(page_title="Plagiarism Checker", layout="wide")
st.title("📄 AI PDF Plagiarism Checker (Page-Level)")

# Load vector DB
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_model,
    collection_name=COLLECTION_NAME
)
st.success(f"✅ Vector store loaded with {vectordb._collection.count()} documents")

mode = st.sidebar.radio("Select Mode", ["🔍 Check Plagiarism", "📥 Upload to Database"])

if mode == "📥 Upload to Database":
    st.subheader("📥 Uploading to Chroma Vector DB...")

# Upload a PDF
    uploaded_file = st.file_uploader("📎 Upload a PDF to add to Database", type=["pdf"])

    if uploaded_file is not None:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        st.info("📄 .....PDF uploaded....")

        # Load and chunk
        loader = PyMuPDFLoader(tmp_path)
        raw_docs = loader.load()

        vectordb.add_documents(raw_docs)
        vectordb.persist()

        st.success(f"✅ Successfully uploaded and indexed {uploaded_file.name} with {len(raw_docs)} pages.")

elif mode == "🔍 Check Plagiarism":
    st.subheader("🚨 Running Plagiarism Check (Page-Level)")
    uploaded_file = st.file_uploader("📎 Upload a PDF to to run Plagarism Check", type=["pdf"])


    if uploaded_file is not None:
    # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        st.info("📄 .....PDF uploaded....")

        # Load and chunk
        loader = PyMuPDFLoader(tmp_path)
        raw_docs = loader.load()


        st.write(f"📚 Split into {len(raw_docs)} Pages.")

        if st.button("🚨 Run Plagiarism Check"):
            with st.spinner("🔍 Checking for similarity..."):
                for idx, chunk in enumerate(raw_docs):
                    st.markdown(f"## 🔍 Page {idx+1}")
                    st.markdown(f"📄 **Content Preview**:\n```\n{chunk.page_content[:300]}...\n```")

                    try:
                        result = vectordb.similarity_search_with_relevance_scores(chunk.page_content, k=TOP_K)
                        results = []
                        for i in result:
                            if(i[1] > 0.6) :
                                results.append(i)

                        if not results:
                            st.info("✅ No similar content found for this chunk.")
                        else:
                            for i, (doc, score) in enumerate(results):
                                is_plag = score > SIMILARITY_THRESHOLD
                                label = "⚠️ Potential Plagiarism" if is_plag else "✅ Likely Original"
                                color = "red" if is_plag else "green"

                                st.markdown(f"""
                                    <div style="border:1px solid #ccc; padding:10px; border-radius:10px; margin-bottom:10px;">
                                        <b>🔢 Match {i+1}</b><br>
                                        <b>📄 Source:</b> {doc.metadata.get('source', 'N/A')}<br>
                                        <b>🧮 Similarity Score:</b> {round(score, 3)}<br>
                                        <b style='color:{color}'>{label}</b><br><br>
                                        <details>
                                            <summary>📑 View Matched Content</summary>
                                            <pre style="white-space:pre-wrap;">{doc.page_content[:1200]}</pre>
                                        </details>
                                    </div>
                                """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"❌ Error comparing chunk {idx+1}: {e}")

        # Cleanup
        os.remove(tmp_path)
