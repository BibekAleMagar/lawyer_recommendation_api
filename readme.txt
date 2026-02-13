================================================================================
COLD CASES LAWYER ASSIGNMENT AI MODEL - README
================================================================================

PROJECT OVERVIEW
----------------
This AI system automatically predicts which lawyer specialization should be
assigned to a cold case based on the case details (name, nature of suit, and
summary). It uses machine learning to learn patterns from your existing cases.

================================================================================
FILES IN THIS PROJECT
================================================================================

1. sampled_cold_cases.csv
   - Your dataset with cold cases
   - Columns: case_name, nature_of_suit, summary, assigned_lawyer_specialization

2. data_loader.py
   - Loads the CSV file into memory
   - Creates both pandas and HuggingFace dataset objects

3. main.py
   - FastAPI web server
   - Provides REST API endpoints for training and predictions

4. train_model.py
   - Basic model trainer (simple version)
   - Uses Random Forest with 100 trees

5. train_improved_model.py
   - Advanced model trainer (recommended!)
   - Tests 3 different algorithms
   - Uses better text processing (bigrams, weighted fields)
   - Filters rare classes
   - Shows detailed metrics

6. requirements.txt
   - List of all Python packages needed

7. lawyer_assignment_model.pkl (created after training)
   - Saved trained model file

8. improved_model.pkl (created after training improved version)
   - Saved improved model file

================================================================================
INSTALLATION
================================================================================

Step 1: Install Python
----------------------
Make sure you have Python 3.8 or higher installed
Check with: python --version

Step 2: Install Dependencies
-----------------------------
Open terminal/command prompt in project folder and run:

    pip install -r requirements.txt

Or install manually:
    pip install fastapi uvicorn pandas scikit-learn joblib datasets

================================================================================
HOW TO USE - QUICK START
================================================================================

METHOD 1: TRAIN AND USE VIA API (RECOMMENDED FOR BEGINNERS)
------------------------------------------------------------

Step 1: Start the API Server
    python main.py

Step 2: Open your browser and go to:
    http://localhost:8000/docs

Step 3: Train the model
    - Find the "/train" endpoint
    - Click "Try it out"
    - Click "Execute"
    - Wait for training to complete (shows accuracy)

Step 4: Make predictions
    - Find the "/predict" endpoint
    - Click "Try it out"
    - Fill in the form:
        case_name: "State vs. John Doe"
        nature_of_suit: "Criminal Defense"
        summary: "Armed robbery and assault charges"
    - Click "Execute"
    - See the predicted lawyer specialization!


METHOD 2: TRAIN VIA COMMAND LINE (RECOMMENDED FOR BEST RESULTS)
----------------------------------------------------------------

Step 1: Train the improved model
    python train_improved_model.py

    This will:
    - Load your CSV data
    - Try 3 different ML algorithms
    - Pick the best one
    - Save the model
    - Show detailed accuracy metrics

Step 2: Start the API server
    python main.py

Step 3: Make predictions
    Go to http://localhost:8000/docs
    Use the /predict endpoint

================================================================================
API ENDPOINTS
================================================================================

GET /
    - Home page with API information

GET /debug
    - Shows dataset information
    - Number of rows, columns, column names
    - First row of data

GET /cases?limit=10
    - Get sample cases from your dataset
    - limit: how many cases to show (default: 10)

GET /specializations
    - List all unique lawyer specializations in your dataset

POST /train
    - Train the machine learning model
    - Returns accuracy and training statistics

POST /predict
    - Predict lawyer specialization for a new case
    - Body (JSON):
        {
          "case_name": "Your case name",
          "nature_of_suit": "Type of case",
          "summary": "Detailed description"
        }
    - Returns:
        {
          "predicted_specialization": "Criminal Defense",
          "confidence": 0.85,
          "top_5_predictions": {...}
        }

================================================================================
TESTING THE API
================================================================================

Option 1: Use Browser (Interactive Docs)
-----------------------------------------
1. Start server: python main.py
2. Go to: http://localhost:8000/docs
3. Try any endpoint by clicking and filling the form


Option 2: Use curl (Command Line)
----------------------------------
# Train the model
curl -X POST http://localhost:8000/train

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "case_name": "State vs. John Doe",
    "nature_of_suit": "Criminal Defense",
    "summary": "Armed robbery with assault charges"
  }'

# Get cases
curl http://localhost:8000/cases?limit=5

# Get specializations
curl http://localhost:8000/specializations


Option 3: Use Python
---------------------
import requests

# Make prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "case_name": "Murder Investigation",
        "nature_of_suit": "Homicide",
        "summary": "Cold case from 1995"
    }
)
print(response.json())


Option 4: Use Postman/Insomnia
-------------------------------
1. Create new POST request
2. URL: http://localhost:8000/predict
3. Body type: JSON
4. Paste JSON data
5. Send

================================================================================
HOW THE MODEL WORKS
================================================================================

1. DATA LOADING
   - Reads sampled_cold_cases.csv
   - Loads case_name, nature_of_suit, summary, assigned_lawyer_specialization

2. TEXT PROCESSING
   - Combines case_name + nature_of_suit + summary into one text
   - Gives extra weight to nature_of_suit and summary (more important)
   - Uses TF-IDF to convert text into numbers
   - Creates 2000 numerical features from the text

3. TRAINING
   - Splits data: 80% training, 20% testing
   - Trains Random Forest with 200 decision trees
   - Each tree learns different patterns
   - All trees vote to make final prediction

4. PREDICTION
   - Takes new case details
   - Converts to numbers using same TF-IDF
   - Runs through all 200 trees
   - Returns prediction + confidence score

5. EVALUATION
   - Tests on 20% of data model hasn't seen
   - Calculates accuracy (% correct predictions)
   - Shows confidence scores

================================================================================
UNDERSTANDING THE RESULTS
================================================================================

When you make a prediction, you get:

{
  "predicted_specialization": "Criminal Defense",
  "confidence": 0.85,
  "top_5_predictions": {
    "Criminal Defense": 0.85,
    "Criminal Law": 0.08,
    "Trial Advocacy": 0.04,
    "Appellate Practice": 0.02,
    "Civil Litigation": 0.01
  }
}

WHAT THIS MEANS:
- predicted_specialization: The model's best guess
- confidence: How sure the model is (0.85 = 85% sure)
- top_5_predictions: Alternative options if confidence is low

CONFIDENCE LEVELS:
- 0.80-1.00 (80-100%): Very confident, trust this prediction
- 0.60-0.79 (60-79%): Moderately confident, probably correct
- 0.40-0.59 (40-59%): Low confidence, review alternatives
- 0.00-0.39 (0-39%): Very uncertain, manual review needed

================================================================================
IMPROVING MODEL ACCURACY
================================================================================

If your model has low accuracy or confidence:

1. ADD MORE DATA
   - More examples = better learning
   - Aim for at least 20-30 examples per specialization

2. CLEAN YOUR DATA
   - Fill in missing summaries
   - Use consistent specialization names
   - Fix typos and formatting

3. BALANCE YOUR CLASSES
   - Make sure each specialization has similar number of examples
   - Model struggles if one has 100 cases and another has 2

4. USE BETTER FEATURES
   - Add more detailed summaries
   - Include case outcomes, evidence types, etc.

5. TRY DIFFERENT MODELS
   - The improved trainer tests 3 algorithms automatically
   - Run: python train_improved_model.py

================================================================================
TROUBLESHOOTING
================================================================================

ERROR: "TF-IDF vectorizer is not fitted"
SOLUTION: You need to train the model first
    - Run: python train_model.py
    - OR call: POST http://localhost:8000/train
    - Then restart the server

ERROR: "Method Not Allowed" on /predict
SOLUTION: /predict is POST not GET
    - Don't visit in browser (that's GET)
    - Use /docs page or curl command

ERROR: "Module not found"
SOLUTION: Install dependencies
    - Run: pip install -r requirements.txt

ERROR: Low confidence scores (below 40%)
SOLUTION: 
    1. Train the improved model: python train_improved_model.py
    2. Add more training data
    3. Balance your dataset

ERROR: "File not found: sampled_cold_cases.csv"
SOLUTION: Make sure CSV is in same folder as Python files

================================================================================
CUSTOMIZATION
================================================================================

To modify for your specific needs:

1. Change model parameters (train_improved_model.py):
   - n_estimators: Number of trees (more = better but slower)
   - max_features: Number of text features (more = captures more words)
   - ngram_range: (1,2) uses 1-word and 2-word phrases

2. Add more features:
   - Edit prepare_features() function
   - Include additional columns from your CSV

3. Change API endpoints (main.py):
   - Add new endpoints for your specific needs
   - Modify response format

================================================================================
PRODUCTION DEPLOYMENT
================================================================================

For deploying to production server:

1. Use production ASGI server:
   gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker

2. Set environment variables:
   export MODEL_PATH=/path/to/model.pkl
   export CSV_PATH=/path/to/data.csv

3. Add authentication:
   - Implement API keys
   - Add OAuth2/JWT tokens

4. Add logging:
   - Log all predictions
   - Monitor accuracy over time

5. Set up monitoring:
   - Track API response times
   - Monitor prediction confidence
   - Alert on errors

================================================================================
SUPPORT & DOCUMENTATION
================================================================================

For more information:
- FastAPI docs: https://fastapi.tiangolo.com
- Scikit-learn: https://scikit-learn.org
- Pandas: https://pandas.pydata.org

================================================================================
VERSION HISTORY
================================================================================

v1.0 - Basic model with Random Forest
v2.0 - Improved model with multiple algorithms and better text processing

================================================================================
LICENSE
================================================================================

This project is for internal use. All rights reserved.

================================================================================
END OF README
================================================================================