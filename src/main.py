from stt import SpeechToTextConverter
from rag import RAG

def reanult_rag():
    stt = SpeechToTextConverter(
        whisper_model="base", 
    )
    rag = RAG(input_path="./data")
    #query = stt()
    query = "List the members of the board directors in 2022"
    docs = rag.retriever(query)
    for doc in docs:
        print(f"* {doc.page_content} [{doc.metadata}]")



if __name__ == "__main__":
    reanult_rag()