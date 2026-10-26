"""Build a local FAISS index from the PDF policies in the documents folder."""

from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_chain import get_embeddings


BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"


def load_pdf_documents():
    """Load all PDFs and keep each page's source and page metadata."""
    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF documents found in {DOCUMENTS_DIR}. "
            "Add bank policy PDFs before building the vector store."
        )

    documents = []
    print(f"Loading documents from {DOCUMENTS_DIR}...")
    for pdf_path in pdf_files:
        print(f"  Loading {pdf_path.name}")
        pages = PyPDFLoader(str(pdf_path)).load()
        for page in pages:
            # Keep the loader metadata but use a project-relative source name.
            page.metadata["source"] = pdf_path.name
        documents.extend(pages)

    if not documents:
        raise ValueError("The PDF files did not contain any readable pages.")
    return documents


def build_vector_store():
    """Split the loaded pages, embed the chunks, and save the FAISS index."""
    load_dotenv(BASE_DIR / ".env")
    documents = load_pdf_documents()

    print("Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("No text could be extracted from the PDF documents.")

    print("Creating embeddings...")
    embeddings = get_embeddings()

    print("Creating FAISS vector store...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(VECTORSTORE_DIR))
    print(f"Vector store created successfully at {VECTORSTORE_DIR}")


if __name__ == "__main__":
    try:
        build_vector_store()
    except Exception as error:
        raise SystemExit(f"Document ingestion failed: {error}") from error
