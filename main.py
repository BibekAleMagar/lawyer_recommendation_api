import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import data_loader

# --- 1. Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown logic"""
    app.state.model = None
    app.state.model_type = None

    if os.path.exists('simplified_model.pkl'):
        try:
            # Import inside lifespan to prevent startup crashes
            from train_simplified_model import SimplifiedLawyerModel
            
            model = SimplifiedLawyerModel()
            model.load_model('simplified_model.pkl')
            app.state.model = model
            app.state.model_type = 'simplified'
            print("✓ Simplified model loaded successfully!")
        except ImportError:
            print("⚠ Warning: simplified_model.pkl exists but train_simplified_model.py was not found.")
    else:
        print("⚠ No model found. Please use the /train endpoint.")
    
    yield
    # Clean up on shutdown if necessary
    app.state.model = None

app = FastAPI(title="Lawyer Assignment API", lifespan=lifespan)

# --- 2. Schemas ---
class CaseInput(BaseModel):
    case_name: str
    nature_of_suit: str
    summary: str

# --- 3. Dependencies ---
def get_model():
    if app.state.model is None:
        raise HTTPException(
            status_code=400, 
            detail="Model not trained or loaded. Please call /train first."
        )
    return app.state.model

# --- 4. Endpoints ---

@app.get("/")
def root():
    return {
        "message": "Lawyer Assignment API (Simplified Categories)",
        "model_status": "loaded" if app.state.model else "not_trained",
        "model_type": app.state.model_type,
        "note": "Predicts main categories (Criminal, Civil, etc.)"
    }

@app.post("/train")
def train_model_endpoint():
    """Train the simplified model dynamically"""
    try:
        from train_simplified_model import SimplifiedLawyerModel
        new_model = SimplifiedLawyerModel()
        
        # Assume data_loader.dataset is already loaded via your data_loader script
        results = new_model.train(data_loader.dataset)
        new_model.save_model('simplified_model.pkl')
        
        # Update the app state so the new model is used immediately
        app.state.model = new_model
        app.state.model_type = 'simplified'
        
        return {
            "status": "success",
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/predict")
def predict(case: CaseInput, model=Depends(get_model)):
    """Predict main lawyer category"""
    try:
        prediction = model.predict(
            case.case_name,
            case.nature_of_suit,
            case.summary
        )
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/categories")
def get_categories(model=Depends(get_model)):
    """Get main categories the model predicts"""
    return {
        "main_categories": list(model.category_keywords.keys()),
        "category_mapping": model.category_keywords
    }

@app.get("/debug")
def debug():
    return {
        "total_cases": len(data_loader.dataset),
        "columns": list(data_loader.dataset.columns),
        "unique_specializations": data_loader.dataset['assigned_lawyer_specialization'].nunique()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)