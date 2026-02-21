import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
# import data_loader  <-- Permanently comment this out

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    if os.path.exists('simplified_model.pkl'):
        try:
            from train_simplified_model import SimplifiedLawyerModel
            model = SimplifiedLawyerModel()
            model.load_model('simplified_model.pkl')
            app.state.model = model
            print("✓ AI Engine Online: Model and Vectorizer loaded.")
        except Exception as e:
            print(f"❌ Load Error: {e}")
    yield

app = FastAPI(title="Lawyer Assignment API", lifespan=lifespan)

class CaseInput(BaseModel):
    case_name: str
    nature_of_suit: str
    summary: str

def get_model():
    if not app.state.model:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return app.state.model

@app.get("/")
def root():
    return {"status": "online", "model": "loaded" if app.state.model else "missing"}

@app.post("/predict")
def predict(case: CaseInput, model=Depends(get_model)):
    # This now works perfectly without the CSV!
    return model.predict(case.case_name, case.nature_of_suit, case.summary)

# Removed /train and /debug endpoints that required data_loader