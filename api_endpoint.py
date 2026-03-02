import fastapi

from spam_service import (
    generate_random_email,
)

app = fastapi.FastAPI()

# --- simple health/check endpoint ------------------------------------------------
@app.get("/test")
def test_message():
    return {"message": "Test message!!!"}

@app.get("/random-email")
def random_email():
    return {"email": generate_random_email()}

