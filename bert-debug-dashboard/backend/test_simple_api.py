from fastapi import FastAPI
import time

app = FastAPI(title="Simple Test API")

@app.get("/")
async def root():
    return {"message": "Simple test API is working", "timestamp": time.time()}

@app.get("/test")
async def test():
    return {"status": "ok", "test": "passed"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
