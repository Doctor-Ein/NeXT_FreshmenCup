import json
from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file.markdown import MarkdownReader

from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)

# -------------------- 配置参数 --------------------
MARKDOWN_DIR = './Data/Paper/MinerU_Res/AlexNet'  # Markdown 文件所在目录
MILVUS_HOST = '0.0.0.0'
MILVUS_PORT = '19530'
COLLECTION_NAME = 'DL_KDB'  # Deep Learning Knowledge Database
LOCAL_MODEL_DIR = './local_models/bge-m3'  # 本地 BGE-M3 模型路径
VECTOR_DIM = 1024  # 嵌入向量维度

# 初始化本地嵌入模型
embedding = HuggingFaceEmbedding(model_name=LOCAL_MODEL_DIR)


def process_markdown_docs(markdown_dir: str, chunk_size: int = 256, chunk_overlap: int = 48) -> list:
    """
    1. 加载目录下所有 Markdown 文档
    2. 按 Markdown 结构分节点
    3. 基于 TokenTextSplitter 进行分块
    4. 返回包含 text 和 metadata 的列表
    """
    # 检查目录
    path = Path(markdown_dir)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"目录不存在: {markdown_dir}")

    # 加载 Markdown 文档
    reader = SimpleDirectoryReader(
        input_dir=markdown_dir,
        file_extractor={'.md': MarkdownReader()},
        required_exts=['.md'],
        exclude_hidden=True
    )
    documents = reader.load_data()
    print(f"成功加载 {len(documents)} 个文档")

    # 拆分 Markdown 节点
    md_parser = MarkdownNodeParser()
    nodes = md_parser.get_nodes_from_documents(documents)
    print(f"生成 {len(nodes)} 个节点")

    # 文本分块器
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=' ',
        backup_separators=['\n', '.', '?']
    )

    # 处理节点
    chunks = []
    for node in nodes:
        # 清理多余空行
        cleaned = "\n".join([l.strip() for l in node.text.splitlines() if l.strip()])
        # 基本元数据
        md = {
            'file_name': node.metadata.get('file_name', ''),
            'file_path': node.metadata.get('file_path', ''),
            'file_type': node.metadata.get('file_type', '')
        }
        # 分块并附加元数据
        for idx, txt in enumerate(splitter.split_text(cleaned)):
            chunks.append({
                'text': txt,
                'metadata': {
                    **md,
                    'chunk_id': f"{md['file_name']}_chunk_{idx}",
                    'chunk_index': idx
                }
            })

    return chunks


def create_milvus_collection(collection_name: str):
    """
    创建或重置 Milvus 集合，字段包括 id, text, metadata(JSON) 和 vector
    """
    # 如果已存在则删除
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    # 定义 schema
    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name='metadata', dtype=DataType.JSON, nullable=True),
        FieldSchema(name='vector', dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
    ]
    schema = CollectionSchema(fields=fields, description='Deep Learning Knowledge DB')

    # 创建集合
    collection = Collection(name=collection_name, schema=schema, using='default')

    # 创建索引
    index_params = {
        'index_type': 'IVF_FLAT',
        'metric_type': 'L2',
        'params': {'nlist': 1024}
    }
    collection.create_index(field_name='vector', index_params=index_params)
    collection.load()
    return collection


def store_in_milvus(chunks: list):
    """
    将处理后的文本块进行嵌入并批量插入到 Milvus
    """
    # 连接 Milvus
    connections.connect(alias='default', host=MILVUS_HOST, port=MILVUS_PORT)

    # 初始化集合
    collection = create_milvus_collection(COLLECTION_NAME)

    # 准备数据
    texts = []
    metas = []
    vecs = []
    for chunk in chunks:
        texts.append(chunk['text'])
        metas.append(chunk['metadata'])
        vecs.append(embedding.get_text_embedding(chunk['text']))

    # 批量插入
    batch_size = 500
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_metas = metas[i:i+batch_size]
        batch_vecs = vecs[i:i+batch_size]
        collection.insert([batch_texts, batch_metas, batch_vecs])

    # 持久化并加载
    # collection.flush()
    collection.load()
    print(f"🚀 成功存储 {len(texts)} 条记录到 Milvus 集合 '{COLLECTION_NAME}'")


if __name__ == '__main__':
    try:
        # 1. 文档解析与分块
        processed = process_markdown_docs(markdown_dir=MARKDOWN_DIR)
        # 2. 存储到 Milvus
        store_in_milvus(processed)
    except Exception as e:
        print(f"处理失败: {e}")
