from unittest.mock import MagicMock
from langchain_core.documents import Document
from .midprj_retriever import HybridRetriever

def test_hybrid_retriever_manual_filter():
    print("Starting verification test for HybridRetriever manual filter...")
    
    # 1. Setup mock retrievers
    mock_bm25 = MagicMock()
    mock_faiss = MagicMock()
    
    doc1 = Document(page_content="교육 내용 1", metadata={"agency": "서울특별시"})
    doc2 = Document(page_content="IT 내용 1", metadata={"agency": "다른기관"})
    doc3 = Document(page_content="교육 내용 2", metadata={"agency": "서울특별시"})
    
    mock_bm25.invoke.return_value = [doc1, doc2]
    mock_faiss.invoke.return_value = [doc2, doc3]
    mock_faiss.search_kwargs = {"k": 5}
    
    retriever = HybridRetriever(
        bm25_retriever=mock_bm25,
        faiss_retriever=mock_faiss,
        weights=[0.5, 0.5]
    )
    
    # 2. Define a filter
    def metadata_filter(metadata):
        return metadata.get("agency") == "서울특별시"
    
    # 3. Call get_relevant_documents
    print("Calling get_relevant_documents with filter...")
    results = retriever.get_relevant_documents("교육", filter=metadata_filter)
    
    # 4. Assertions
    print(f"Results found: {len(results)}")
    for i, doc in enumerate(results):
        print(f" Result {i+1}: {doc.page_content} ({doc.metadata})")
        assert doc.metadata["agency"] == "서울특별시", f"Document {doc.page_content} should have been filtered out!"
    
    assert len(results) > 0, "Should have found at least one document after filtering"
    
    # Verify mock calls
    mock_bm25.invoke.assert_called_once_with("교육")
    mock_faiss.invoke.assert_called_once_with("교육")
    
    print("Verification test passed successfully!")

if __name__ == "__main__":
    try:
        test_hybrid_retriever_manual_filter()
    except Exception as e:
        print(f"Verification test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
