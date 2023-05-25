from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DeepLake

load_dotenv()

os.environ.get("ACTIVELOOP_TOKEN")
username = "rihp" # replace with your username from app.activeloop.ai
projectname = "polywrap5" # replace with your project name from app.activeloop.ai

embeddings = OpenAIEmbeddings(disallowed_special=())

db = DeepLake(dataset_path=f"hub://{username}/{projectname}", read_only=True, embedding_function=embeddings)

retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['fetch_k'] = 100
retriever.search_kwargs['maximal_marginal_relevance'] = True
retriever.search_kwargs['k'] = 10

model = ChatOpenAI(model_name='gpt-3.5-turbo') # switch to 'gpt-4'
qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)

app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask_question():
    print('hello there')
    data = request.get_json()
    prompt = data['question']

    questions = [prompt]
    chat_history = []

    for question in questions:
        result = qa({"question": question, "chat_history": chat_history})
        chat_history.append((question, result['answer']))
        print(f"-> **Question**: {question} \n")
        print(f"**Answer**: {result['answer']} \n")
    
    answer = result['answer']
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run()
