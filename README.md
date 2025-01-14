Creating a website that uses a Meta Open Source AI model, such as the LLaMA model, to answer questions from a PDF file can be accomplished by utilizing the Langchain framework along with some additional Python libraries. Below is the full guide and code to help you set up the application.

### Prerequisites:
- Python (>= 3.8)
- Flask (for creating a web interface)
- Langchain (for building AI-driven apps)
- PyPDF2 (to extract text from PDF files)
- OpenAI (or another LLM service) for answering questions
- Any additional packages for web serving (like Flask-Uploads)

### Step-by-step Instructions:

#### 1. Set up the Environment

Make sure you have Python installed and then install the required libraries:

```bash
pip install flask langchain pypdf2 openai flask-uploads
```

#### 2. Create the Python Script (`app.py`)

In this script, we will use Flask to handle the web server and Langchain to process the PDF and answer questions.

```python
import os
from flask import Flask, render_template, request, jsonify
from flask_uploads import UploadSet, configure_uploads, DOCUMENTS
import PyPDF2
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Initialize Flask app
app = Flask(__name__)

# Configure file upload
pdf_files = UploadSet("pdf", DOCUMENTS)
app.config["UPLOADED_PDF_DEST"] = "uploads"
configure_uploads(app, pdf_files)

# Initialize OpenAI and Langchain (ensure to set your OpenAI API key in environment variables)
# You should already have your OpenAI key configured.
openai_api_key = "your_openai_api_key"

# Initialize the OpenAI model
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_api_key)

# Path to save uploaded PDFs
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper function to read text from PDF
def extract_text_from_pdf(pdf_file_path):
    with open(pdf_file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
        return text

# Route for the homepage
@app.route("/")
def home():
    return render_template("index.html")

# Route for handling file upload
@app.route("/upload", methods=["POST"])
def upload_file():
    if "pdf" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["pdf"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file and file.filename.endswith(".pdf"):
        # Save the file to the server
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        # Extract text from the PDF
        pdf_text = extract_text_from_pdf(file_path)

        # Create embeddings for the PDF text and store in FAISS vector store
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        vector_store = FAISS.from_texts([pdf_text], embeddings)

        # Create the retriever and conversational chain
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})
        qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever)

        # Save the retriever for further use in the session
        session["qa_chain"] = qa_chain
        return jsonify({"message": "File uploaded successfully. You can now ask questions."}), 200
    return jsonify({"error": "Invalid file format. Only PDF files are allowed."}), 400

# Route for asking questions
@app.route("/ask", methods=["POST"])
def ask_question():
    question = request.form.get("question")
    if not question:
        return jsonify({"error": "Question is required"}), 400
    if "qa_chain" not in session:
        return jsonify({"error": "No document loaded. Please upload a PDF first."}), 400
    
    qa_chain = session["qa_chain"]
    response = qa_chain({"question": question})
    return jsonify({"answer": response['answer']}), 200

if __name__ == "__main__":
    app.run(debug=True)
```

#### 3. Create the HTML Template (`templates/index.html`)

This HTML file will create a basic user interface for file upload and asking questions.

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDF Question Answering</title>
</head>
<body>
    <h1>Upload PDF and Ask Questions</h1>

    <!-- File upload form -->
    <form id="uploadForm" action="/upload" method="POST" enctype="multipart/form-data">
        <label for="pdf">Upload a PDF file:</label>
        <input type="file" id="pdf" name="pdf" accept=".pdf" required>
        <button type="submit">Upload</button>
    </form>

    <hr>

    <!-- Question form -->
    <div>
        <label for="question">Ask a question about the PDF:</label>
        <input type="text" id="question" name="question" placeholder="Enter your question" required>
        <button onclick="askQuestion()">Ask</button>
    </div>

    <hr>

    <!-- Display answer -->
    <div id="answer"></div>

    <script>
        // Upload file using AJAX
        document.getElementById("uploadForm").onsubmit = function (event) {
            event.preventDefault();
            var formData = new FormData(this);
            fetch("/upload", {
                method: "POST",
                body: formData
            })
            .then(response => response.json())
            .then(data => alert(data.message))
            .catch(error => alert("Error uploading file."));
        };

        // Ask a question using AJAX
        function askQuestion() {
            var question = document.getElementById("question").value;
            if (!question) {
                alert("Please enter a question.");
                return;
            }

            var formData = new FormData();
            formData.append("question", question);

            fetch("/ask", {
                method: "POST",
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.answer) {
                    document.getElementById("answer").innerHTML = "<b>Answer:</b> " + data.answer;
                } else {
                    document.getElementById("answer").innerHTML = "Error: " + data.error;
                }
            })
            .catch(error => alert("Error asking question."));
        }
    </script>
</body>
</html>
```

#### 4. Create a `.env` file (Optional)

If you want to store your OpenAI API key securely, you can create a `.env` file with the following content:

```
OPENAI_API_KEY=your_openai_api_key
```

Then, you can use `python-dotenv` to load the API key into your application:

```bash
pip install python-dotenv
```

And modify the `app.py` to load the `.env` file:

```python
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
```

#### 5. Running the Application

Once everything is set up, you can run your application by executing:

```bash
python app.py
```

This will start the Flask development server, and you can access your application at `http://127.0.0.1:5000/`.

#### 6. Upload a PDF and Ask Questions

- Upload a PDF using the file input.
- Once the file is uploaded, you can ask any question related to the content of the PDF, and the model will return the relevant answer.

---

### Conclusion

You now have a fully functional web application that allows users to upload a PDF file, extracts its content, and uses a Meta Open Source model like LLaMA via Langchain to answer questions related to the content of that PDF.


