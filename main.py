import os
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain

os.environ["OPENAI_API_KEY"] = "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"    

loader = PyPDFLoader("")
loader = PyPDFLoader(os.path.abspath("./ipc.pdf"))
pages = loader.load_and_split()
chunks = pages

# Step 1: Convert PDF to text
import textract
doc = textract.process("./ipc.pdf")

# Step 2: Save to .txt and reopen (helps prevent issues)
with open('ipc.txt', 'w') as f:
    f.write(doc.decode('utf-8'))

with open('ipc.txt', 'r') as f:
    text = f.read()

# Step 3: Create function to count tokens
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Step 4: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = count_tokens,
)

chunks = text_splitter.create_documents([text])

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(chunks, embeddings)

# chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

# query = "how many exact number of section does ipc have"
# docs = db.similarity_search(query)
# chain.run(input_documents=docs, question=query)


from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the QA model and initialize it
qa_model = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

@app.route('/qa', methods=['POST'])
def qa_endpoint():
    try:
        # Get the input query from the request
        data = request.get_json()
        query = data.get('query')

        # Perform document similarity search
        docs = db.similarity_search(query)

        # Run the QA model on the found documents
        answers = []
        for doc in docs:
            answer = qa_model.run(input_documents=doc, question=query)
            answers.append({"document": doc, "answer": answer})

        return jsonify({"answers": answers})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
