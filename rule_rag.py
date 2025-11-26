import os
import json
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Global variables to hold the vector stores
_BUSINESS_VECTOR_STORE = None
_PYTHON_VECTOR_STORE = None
_BUSINESS_RULES_DATA = []
_PYTHON_RULES_DATA = []

def initialize_rule_vector_store(json_path: str = "business_rules.json", rule_type: str = "business"):
    """
    Initializes the FAISS vector store with rules from a JSON file.
    
    Args:
        json_path: Path to the JSON file containing rules
        rule_type: Type of rules ("business" or "python")
    
    Returns:
        FAISS vector store
    """
    global _BUSINESS_VECTOR_STORE, _PYTHON_VECTOR_STORE, _BUSINESS_RULES_DATA, _PYTHON_RULES_DATA
    
    # Check if already initialized
    if rule_type == "business" and _BUSINESS_VECTOR_STORE is not None:
        return _BUSINESS_VECTOR_STORE
    elif rule_type == "python" and _PYTHON_VECTOR_STORE is not None:
        return _PYTHON_VECTOR_STORE
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Rule file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        rules_data = json.load(f)

    # Store rules data
    if rule_type == "business":
        _BUSINESS_RULES_DATA = rules_data
    else:
        _PYTHON_RULES_DATA = rules_data

    documents = []
    for rule in rules_data:
        # Create a document for each rule
        # Content combines title and content for better semantic search
        page_content = f"{rule['title']}\n{rule['content']}"
        metadata = {
            "category": rule["category"], 
            "title": rule["title"],
            "code_example": rule.get("code_example")  # Store code example in metadata
        }
        documents.append(Document(page_content=page_content, metadata=metadata))

    # Use Google Generative AI Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Create FAISS index
    vector_store = FAISS.from_documents(documents, embeddings)
    
    # Store in appropriate global variable
    if rule_type == "business":
        _BUSINESS_VECTOR_STORE = vector_store
    else:
        _PYTHON_VECTOR_STORE = vector_store
    
    print(f"Initialized FAISS vector store for {rule_type} rules with {len(documents)} rules.")
    return vector_store


def initialize_python_rule_vector_store(json_path: str = "python_rules.json"):
    """
    Initializes the FAISS vector store with Python rules from a JSON file.
    Convenience wrapper for initialize_rule_vector_store with rule_type="python".
    """
    return initialize_rule_vector_store(json_path, rule_type="python")


def retrieve_relevant_rules(query: str, category: str = None, k: int = 5, rule_type: str = "business") -> str:
    """
    Retrieves relevant rules based on the query and optional category.
    
    Args:
        query: Search query
        category: Optional category filter
        k: Number of rules to retrieve
        rule_type: Type of rules to retrieve ("business" or "python")
    
    Returns:
        Formatted string of rules
    """
    global _BUSINESS_VECTOR_STORE, _PYTHON_VECTOR_STORE
    
    # Initialize appropriate vector store if not already done
    if rule_type == "business":
        if _BUSINESS_VECTOR_STORE is None:
            initialize_rule_vector_store("business_rules.json", rule_type="business")
        vector_store = _BUSINESS_VECTOR_STORE
    elif rule_type == "python":
        if _PYTHON_VECTOR_STORE is None:
            initialize_rule_vector_store("python_rules.json", rule_type="python")
        vector_store = _PYTHON_VECTOR_STORE
    else:
        raise ValueError(f"Invalid rule_type: {rule_type}. Must be 'business' or 'python'.")

    # Search for relevant documents
    # We retrieve slightly more candidates to filter by category post-retrieval if needed
    docs = vector_store.similarity_search(query, k=k*2)
    
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
        
        # Add code example if available
        if "code_example" in doc.metadata and doc.metadata["code_example"]:
            rule_text += f"\n\n[Code Example]\n```python\n{doc.metadata['code_example']}\n```"
            
        result_strings.append(rule_text)
    
    return "\n\n".join(result_strings)


if __name__ == "__main__":
    # Test retrieval for both business and python rules
    print("=== Testing Business Rules Retrieval ===")
    business_rules = retrieve_relevant_rules("How to calculate visit rate?", category="common", rule_type="business")
    print("\n--- Retrieved Business Rules ---\n")
    print(business_rules)
    
    print("\n\n=== Testing Python Rules Retrieval ===")
    python_rules = retrieve_relevant_rules("How to implement Block Logic?", category="python", rule_type="python")
    print("\n--- Retrieved Python Rules ---\n")
    print(python_rules)

