import os
from langchain import hub
from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.prompts import format_document

from operator import itemgetter
from getpass import getpass

from flask import Flask, request, jsonify
# from flask_cors import CORS

app = Flask(__name__)
app.app_context()
# CORS(app)
counter = 0
memory = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="question")
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

def setup():
    prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    llm = ChatOpenAI()
    embeddings = HuggingFaceEmbeddings(model_kwargs={'device' : 'cpu'})
    index_name = "tamil-movies-4k"
    docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)
    retriever = docsearch.as_retriever()
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
    template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
    ANSWER_PROMPT = ChatPromptTemplate.from_template(template)
    loaded_memory = RunnablePassthrough.assign(chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),)
    standalone_question = {
    "standalone_question": {
        "question": lambda x: x["question"],
        "chat_history": lambda x: get_buffer_string(x["chat_history"]),
    }
    | CONDENSE_QUESTION_PROMPT
    | llm
    | StrOutputParser(),
    }
    # Now we retrieve the documents
    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }
    final_inputs = {
    "context": lambda x: _combine_documents(x["docs"]),
    "question": itemgetter("question"),
    }
    answer = {
        "answer": final_inputs | ANSWER_PROMPT | llm,
    #     "docs": itemgetter("docs"),
    }
    final_chain = loaded_memory | standalone_question | retrieved_documents | answer

    return final_chain

@app.route("/answer", methods=["POST"])
def generate_answer():
    data = request.get_json()
    if not data["query"]:
        error_message = "Required paremeters : 'query'"
        return error_message
    else:
        user_query = data["query"]
    print("Query: ", user_query)
    inputs = {"question": user_query}

    if "history" in memory.load_memory_variables({}):
        result = cr_chain.invoke({
        "question": user_query,
        "chat_history": memory.load_memory_variables({})["history"],
        })
    else:
        result = cr_chain.invoke(inputs)

    print("Result: ", result)
    memory.save_context(inputs, {"answer": result["answer"].content})
    return jsonify(result["answer"].content)

if __name__ == "__main__":
    cr_chain = setup()
    app.run(host="0.0.0.0", port=5000)