"""Quick smoke test for the model factory."""
import os

if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = "test-key"
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    
    print("🔥 Starting test_factory.py...")
    
    from src.ragsynapse.llm.model_factory import get_llm, get_available_models, LLMProvider
    
    print("✅ Imports successful!")
    
    # Test 1: enum values
    print("🧪 Test 1: Checking enum...")
    assert LLMProvider.OPENAI == "openai"
    print("✓ LLMProvider enum works")
    
    # Test 2: available models
    print("🧪 Test 2: Checking models...")
    models = get_available_models()
    assert "openai" in models
    assert "anthropic" in models
    assert "ollama" in models
    print(f"✓ get_available_models() → {list(models.keys())}")
    
    # Test 3: bad provider raises clearly
    print("🧪 Test 3: Bad provider...")
    try:
        get_llm("notamodel")
        assert False, "Should have raised"
    except ValueError as e:
        print(f"✓ Bad provider raises ValueError: {e}")
    
    print("\n✅ All factory tests passed")
