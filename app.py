
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
import pickle
import re
import pdfplumber  # For PDF extraction
from fastapi.templating import Jinja2Templates
from fastapi import Request
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import traceback

# Initialize FastAPI app
app = FastAPI()

# Set up Jinja2 templates for HTML rendering
templates = Jinja2Templates(directory="templates")

# Load the pre-trained models, label encoder, and vectorizer
model = pickle.load(open('resumeg.pkl', 'rb'))  # Load the trained model
le = pickle.load(open('resumele.pkl', 'rb'))   # Load the label encoder
vectorizer = pickle.load(open('resumecv.pkl', 'rb'))  # Load the vectorizer used during training

# NLTK setup for text preprocessing
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Text preprocessing function."""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenization and stop word removal
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def extract_text_from_pdf(file):
    """Extract text from PDF."""
    try:
        # Convert byte content into a file-like object
        file_like_object = io.BytesIO(file)
        with pdfplumber.open(file_like_object) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text from PDF: {str(e)}")

# Route to serve the HTML file
@app.get("/")
async def get_html(request: Request):
    """Serve the HTML file."""
    return templates.TemplateResponse("index.html", {"request": request})

# Define prediction route
@app.post("/predict")
async def predict(file: UploadFile = File(...), text_input: str = None):
    """Predict category from resume text."""
    try:
        # Check if a file was uploaded
        if file:
            # Read uploaded file
            content = await file.read()
            
            if file.filename.endswith(".txt"):
                text = content.decode('utf-8')  # Decode text file content
            elif file.filename.endswith(".pdf"):
                text = extract_text_from_pdf(content)  # Extract text from PDF file
            else:
                return JSONResponse(content={"error": "Unsupported file type"}, status_code=400)
        elif text_input:
            text = text_input  # Use the simple text input if provided
        else:
            return JSONResponse(content={"error": "No file or text input provided"}, status_code=400)
        
        # Preprocess text
        processed_text = preprocess_text(text)

        # Vectorize using the loaded vectorizer
        text_vectorized = vectorizer.transform([processed_text]).toarray()  # Use loaded vectorizer

        # Predict category using the trained model
        prediction = model.predict(text_vectorized)

        # Convert the numerical prediction to its corresponding label
        category = le.inverse_transform(prediction)[0]
        
        # Return the predicted category as a response
        return JSONResponse(content={"category": category})

    except Exception as e:
        # Log the error and provide a response with the error message
        traceback.print_exc()  # Prints the stack trace in the server logs for debugging
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Run the app
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5500)
