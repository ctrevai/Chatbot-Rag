from langchain.embeddings import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms.bedrock import Bedrock
from langchain.chains import ConversationalRetrievalChain


def get_llm():

    model_kwargs = {  # anthropic
        "max_tokens_to_sample": 512,
        "temperature": 0,
        "top_k": 250,
        "top_p": 1,
        "stop_sequences": ["\n\nHuman:"]
    }

    llm = Bedrock(
        credentials_profile_name="default",
        region_name="us-east-1",
        model_id="anthropic.claude-v2:1",  # set the foundation model
        model_kwargs=model_kwargs,)  # configure the properties for Claude

    return llm


pdf_path = "2022-Shareholder-Letter.pdf"


def select_pdf():
    pdf_path = "uploaded_file.pdf"
    return


def get_index():

    # create embeddings for the index
    embeddings = BedrockEmbeddings(
        credentials_profile_name="default",
        region_name="us-east-1",
    )  # Titan Embedding by default

    loader = PyPDFLoader(pdf_path)

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=1000,
        chunk_overlap=100,
    )

    # create the index
    index_creator = VectorstoreIndexCreator(
        vectorstore_cls=FAISS,
        embedding=embeddings,
        text_splitter=text_splitter,
    )

    index_from_loader = index_creator.from_loaders([loader])

    return index_from_loader


def get_memory():
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
    )
    return memory


def get_rag_chat_response(input_text, memory, index):  # chat client function

    llm = get_llm()

    conversation_with_retrieval = ConversationalRetrievalChain.from_llm(
        llm, index.vectorstore.as_retriever(), memory=memory)

    # pass the user message, history, and knowledge to the model
    chat_response = conversation_with_retrieval({"question": input_text})

    return chat_response['answer']
