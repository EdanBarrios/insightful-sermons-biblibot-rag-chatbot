"""
System verification script.
Tests all components before deployment.

Usage:
    python test_system.py
"""

import os
import sys
from dotenv import load_dotenv

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_test(name, passed, details=""):
    status = f"{Colors.GREEN}✓ PASS{Colors.END}" if passed else f"{Colors.RED}✗ FAIL{Colors.END}"
    print(f"{status} | {name}")
    if details:
        print(f"     {details}")

def test_environment():
    """Test environment variables"""
    print(f"\n{Colors.BLUE}=== Testing Environment ==={Colors.END}")
    
    load_dotenv()
    
    pinecone_key = os.getenv("PINECONE_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    print_test("PINECONE_API_KEY exists", bool(pinecone_key))
    print_test("GROQ_API_KEY exists", bool(groq_key))
    
    return bool(pinecone_key and groq_key)

def test_imports():
    """Test all required imports"""
    print(f"\n{Colors.BLUE}=== Testing Imports ==={Colors.END}")
    
    tests = [
        ("Flask", lambda: __import__('flask')),
        ("flask_cors", lambda: __import__('flask_cors')),
        ("Pinecone", lambda: __import__('pinecone')),
        ("sentence_transformers", lambda: __import__('sentence_transformers')),
        ("Groq", lambda: __import__('groq')),
        ("torch", lambda: __import__('torch')),
    ]
    
    all_passed = True
    for name, import_fn in tests:
        try:
            import_fn()
            print_test(name, True)
        except ImportError as e:
            print_test(name, False, str(e))
            all_passed = False
    
    return all_passed

def test_embeddings():
    """Test embedding generation"""
    print(f"\n{Colors.BLUE}=== Testing Embeddings ==={Colors.END}")
    
    try:
        from embeddings import embed
        
        test_text = "This is a test sentence"
        vector = embed(test_text)
        
        is_list = isinstance(vector, list)
        correct_dim = len(vector) == 384  # all-MiniLM-L6-v2 dimension
        has_values = all(isinstance(v, float) for v in vector[:5])
        
        print_test("Embedding function works", True)
        print_test("Returns list", is_list)
        print_test("Correct dimension (384)", correct_dim, f"Got {len(vector)}")
        print_test("Contains float values", has_values)
        
        return is_list and correct_dim and has_values
        
    except Exception as e:
        print_test("Embedding generation", False, str(e))
        return False

def test_pinecone():
    """Test Pinecone connection"""
    print(f"\n{Colors.BLUE}=== Testing Pinecone ==={Colors.END}")
    
    try:
        from pinecone import Pinecone
        
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index("sermon-index")
        
        # Get index stats
        stats = index.describe_index_stats()
        
        vector_count = stats.get('total_vector_count', 0)
        
        print_test("Pinecone connection", True)
        print_test("Index 'sermon-index' exists", True)
        print_test("Index has vectors", vector_count > 0, f"Count: {vector_count}")
        
        if vector_count == 0:
            print(f"     {Colors.YELLOW}⚠ Warning: No vectors in index. Run ingestion/scrape_and_embed.py{Colors.END}")
        
        return True
        
    except Exception as e:
        print_test("Pinecone connection", False, str(e))
        return False

def test_retrieval():
    """Test retrieval function"""
    print(f"\n{Colors.BLUE}=== Testing Retrieval ==={Colors.END}")
    
    try:
        from retrieval import retrieve
        
        test_query = "What is faith?"
        results = retrieve(test_query, top_k=3)
        
        has_results = len(results) > 0
        results_are_strings = all(isinstance(r, str) for r in results)
        
        print_test("Retrieval function works", True)
        print_test("Returns results", has_results, f"Got {len(results)} chunks")
        print_test("Results are strings", results_are_strings)
        
        if has_results:
            print(f"     Preview: {results[0][:100]}...")
        
        return has_results and results_are_strings
        
    except Exception as e:
        print_test("Retrieval", False, str(e))
        return False

def test_llm():
    """Test LLM generation"""
    print(f"\n{Colors.BLUE}=== Testing LLM ==={Colors.END}")
    
    try:
        from llm import generate_answer
        
        test_context = "Faith is trust in God. It means believing in what you cannot see."
        test_question = "What is faith?"
        
        answer = generate_answer(test_context, test_question)
        
        has_answer = bool(answer)
        is_string = isinstance(answer, str)
        reasonable_length = len(answer) > 20
        
        print_test("LLM function works", True)
        print_test("Returns answer", has_answer)
        print_test("Answer is string", is_string)
        print_test("Reasonable length", reasonable_length, f"{len(answer)} chars")
        
        if has_answer:
            print(f"     Preview: {answer[:150]}...")
        
        return has_answer and is_string and reasonable_length
        
    except Exception as e:
        print_test("LLM generation", False, str(e))
        return False

def test_server_imports():
    """Test server can import all modules"""
    print(f"\n{Colors.BLUE}=== Testing Server Imports ==={Colors.END}")
    
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        import server
        print_test("server.py imports successfully", True)
        
        has_app = hasattr(server, 'app')
        has_chat = hasattr(server, 'chat')
        
        print_test("Flask app exists", has_app)
        print_test("Chat endpoint exists", has_chat)
        
        return has_app and has_chat
        
    except Exception as e:
        print_test("Server imports", False, str(e))
        return False

def main():
    """Run all tests"""
    print(f"\n{Colors.BLUE}{'='*50}{Colors.END}")
    print(f"{Colors.BLUE}BibliBot System Verification{Colors.END}")
    print(f"{Colors.BLUE}{'='*50}{Colors.END}")
    
    results = {
        "Environment": test_environment(),
        "Imports": test_imports(),
        "Embeddings": test_embeddings(),
        "Pinecone": test_pinecone(),
        "Retrieval": test_retrieval(),
        "LLM": test_llm(),
        "Server": test_server_imports(),
    }
    
    # Summary
    print(f"\n{Colors.BLUE}=== Summary ==={Colors.END}")
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = f"{Colors.GREEN}✓{Colors.END}" if result else f"{Colors.RED}✗{Colors.END}"
        print(f"{status} {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print(f"\n{Colors.GREEN}🎉 All tests passed! System is ready.{Colors.END}")
        print(f"\nNext steps:")
        print(f"  1. Run: python server.py")
        print(f"  2. Open: http://localhost:5001")
        return 0
    else:
        print(f"\n{Colors.RED}❌ Some tests failed. Please fix issues above.{Colors.END}")
        return 1

if __name__ == "__main__":
    sys.exit(main())