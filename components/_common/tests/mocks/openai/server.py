from fastapi import FastAPI, HTTPException, Request
import numpy as np
import time

app = FastAPI(title="OpenAI Embeddings Mock API")

# Store the dimension sizes for different models
MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

# Cache to store previously generated embeddings for consistency
embedding_cache = {}

@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"Request path: {request.url.path}")
    print(f"Request method: {request.method}")
    response = await call_next(request)
    return response

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/embeddings")
async def create_embeddings_legacy(request: Request):
    """Legacy endpoint that redirects to v1/embeddings."""
    return await create_embeddings(request)

@app.post("/v1/embeddings")
async def create_embeddings(request: Request):
    """Mock OpenAI embeddings API endpoint."""
    try:
        # Parse the request body
        body = await request.json()
        
        # Extract input text(s) and model
        input_texts = body.get("input", [])
        model = body.get("model", "text-embedding-3-small")
        
        # Validate model
        if model not in MODEL_DIMENSIONS:
            return {
                "error": {
                    "message": f"Model {model} not found",
                    "type": "invalid_request_error",
                    "code": "model_not_found"
                }
            }
        
        # Handle single string or list input
        if isinstance(input_texts, str):
            input_texts = [input_texts]
            
        # Generate embeddings
        dimension = MODEL_DIMENSIONS[model]
        data = []
        
        for i, text in enumerate(input_texts):
            # Use cached embedding if available for consistency
            cache_key = str(text)  # Convert to string for consistent hashing
            if cache_key in embedding_cache:
                embedding = embedding_cache[cache_key]
            else:
                # Generate a deterministic embedding based on the text hash
                np.random.seed(hash(cache_key) % 2**32)
                embedding = np.random.normal(0, 1, dimension).tolist()
                # Normalize embedding vector to unit length
                norm = np.linalg.norm(embedding)
                embedding = [x / norm for x in embedding]
                embedding_cache[cache_key] = embedding
            
            data.append({
                "object": "embedding",
                "embedding": embedding,
                "index": i
            })
        
        # Simulate slight processing delay
        time.sleep(0.05)
        
        # Return response in OpenAI format
        return {
            "object": "list",
            "data": data,
            "model": model,
            "usage": {
                "prompt_tokens": sum(len(str(text).split()) for text in input_texts),
                "total_tokens": sum(len(str(text).split()) for text in input_texts)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 