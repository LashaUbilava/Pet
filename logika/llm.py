# main.py
import yaml
from langchain_gigachat.chat_models import GigaChat
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from rag import get_llama_retriever

# === 1. Загрузка токенов ===
def get_gigachat_token():
    with open('autorization.yaml') as f:
        return yaml.safe_load(f)['token']

# === 2. Инициализация компонентов ===
llm = GigaChat(
    credentials=get_gigachat_token(),
    verify_ssl_certs=False,
)

retriever = get_llama_retriever()

# === 3. Промпт ===
prompt = ChatPromptTemplate.from_template(
    "Ответь на вопрос,опираясь на контекст:\n\n"
    "Если в контексте нет ответа, то ответь на основе своих знаний.\n\n"
    "Контекст:\n{context}\n\n"
    "Вопрос: {question}\n\n"
    "Ответ:"
)

# === 4. Сборка цепочки с LCEL ===
def format_docs(docs):
    """Преобразует список документов в одну строку."""
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# === 5. Запуск ===
if __name__ == "__main__":
    question = "Что такое рецидив?"
    answer = rag_chain.invoke(question)
    print("❓ Вопрос:", question)
    print("✅ Ответ:", answer)
