from langchain_gigachat.chat_models import GigaChat
from langchain.messages import HumanMessage, AIMessage, SystemMessage
import yaml

def get_yaml():
    with open('autorization.yaml', 'r') as f:
        config = yaml.safe_load(f)
        return config['token']

token = get_yaml()


model = GigaChat(
    # Для авторизации запросов используйте ключ, полученный в проекте GigaChat API
    credentials=token,
    verify_ssl_certs=False,
)

system_msg = SystemMessage('Ты виртуальный помощник. Твоя задача отвечать на вопросы пользователя')
human_msg = HumanMessage('Привет, что такое осадки?')

messages = [system_msg, human_msg]
response = model.invoke(messages)
print(response)
