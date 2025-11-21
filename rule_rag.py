import os
import json
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Global variables to hold the vector store
_VECTOR_STORE = None
_RULES_DATA = []

def initialize_rule_vector_store(json_path: str = "business_rules.json"):
    """
    Initializes the FAISS vector store with business rules from a JSON file.
    """
    global _VECTOR_STORE, _RULES_DATA
    
    if _VECTOR_STORE is not None:
        return _VECTOR_STORE

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Rule file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        _RULES_DATA = json.load(f)

    documents = []
    for rule in _RULES_DATA:
        # Create a document for each rule
        # Content combines title and content for better semantic search
        page_content = f"{rule['title']}\n{rule['content']}"
        metadata = {
            "category": rule["category"], 
            "title": rule["title"],
            "code_example": rule.get("code_example") # Store code example in metadata
        }
        documents.append(Document(page_content=page_content, metadata=metadata))

    # Use Google Generative AI Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Create FAISS index
    _VECTOR_STORE = FAISS.from_documents(documents, embeddings)
    print(f"Initialized FAISS vector store with {len(documents)} rules.")
    return _VECTOR_STORE

def retrieve_relevant_rules(query: str, category: str = None, k: int = 5) -> str:
    """
    Retrieves relevant business rules based on the query and optional category.
    Returns a formatted string of rules.
    """
    global _VECTOR_STORE
    
    if _VECTOR_STORE is None:
        initialize_rule_vector_store()

    # Search for relevant documents
    # We retrieve slightly more candidates to filter by category post-retrieval if needed
    # However, FAISS doesn't support pre-filtering easily without metadata filtering support in the wrapper
    # For a small number of rules, we can just retrieve top k*2 and filter.
    
    docs = _VECTOR_STORE.similarity_search(query, k=k*2)
    
    filtered_docs = []
    for doc in docs:
        rule_category = doc.metadata.get("category")
        # Include if category matches or is 'common'
        if category:
            if rule_category in ["common", category]:
                filtered_docs.append(doc)
        else:
            filtered_docs.append(doc)
            
        if len(filtered_docs) >= k:
            break
    
    # Format the output
    result_strings = []
    for doc in filtered_docs:
        content = doc.page_content.split(doc.metadata['title'], 1)[-1].strip()
        rule_text = f"* {doc.metadata['title']}\n{content}"
        
        # Check if there is a code example in the original data
        # Since we don't store code_example in metadata by default in the previous step,
        # we need to ensure it's stored or re-fetch. 
        # Better: Store it in metadata during initialization.
        if "code_example" in doc.metadata and doc.metadata["code_example"]:
            rule_text += f"\n\n[Code Example]\n```python\n{doc.metadata['code_example']}\n```"
            
        result_strings.append(rule_text)
    
    return "\n\n".join(result_strings)

if __name__ == "__main__":
    # Test retrieval
    print("Testing retrieval...")
    rules = retrieve_relevant_rules("How to calculate visit rate?", category="python")
    print("\n--- Retrieved Rules ---\n")
    print(rules)
