from fastapi import FastAPI

app = FastAPI(title="Research Paper Analyzer API")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
