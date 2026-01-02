from langchain_gigachat.chat_models import GigaChat
from langchain.messages import HumanMessage, AIMessage, SystemMessage

model = GigaChat(
    # Для авторизации запросов используйте ключ, полученный в проекте GigaChat API
    credentials="ZTQxNGY5OTYtOGE2NS00OWQxLTk2YWQtMTI3NWRhYmQxOTc0OjUyNWFlODVkLWY1Y2MtNDEwNy05MWE0LTdhOGI1MGZhY2FlOA==",
    verify_ssl_certs=False,
)

system_msg = SystemMessage('Ты виртуальный помощник. Твоя задача отвечать на вопросы пользователя')
human_msg = HumanMessage('Привет, что такое осадки?')

messages = [system_msg, human_msg]
response = model.invoke(messages)
print(response)
