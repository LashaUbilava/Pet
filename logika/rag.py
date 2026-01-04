import yaml
from llama_cloud_services import LlamaCloudIndex
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import PrivateAttr

def get_yaml():
    with open('autorization.yaml', 'r') as f:
        return yaml.safe_load(f)['key']

class LlamaCloudLangChainRetriever(BaseRetriever):
    """Адаптер для LlamaCloudIndex под LangChain LCEL."""

    _index: LlamaCloudIndex = PrivateAttr()

    def __init__(self, index: LlamaCloudIndex, **kwargs):
        super().__init__(**kwargs)
        self._index = index

    def _get_relevant_documents(self, query: str):
        retriever = self._index.as_retriever()
        nodes = retriever.retrieve(query)  # или как у вас

        docs = []
        for node in nodes:
            if hasattr(node, 'page_content'):
                text = node.page_content
            elif hasattr(node, 'text'):
                text = node.text
            else:
                text = str(node)
            docs.append(Document(page_content=text))
        return docs


#os.environ['LLAMA_CLOUD_API_KEY'] = key 

def get_llama_retriever():
    key = get_yaml()
    index = LlamaCloudIndex(
    name="MyIndex",
    project_name="Default",
    organization_id="b0bf9943-4825-46fe-be29-8ad95b6cff0b",
    api_key=key,
    )

    return LlamaCloudLangChainRetriever(index = index)
