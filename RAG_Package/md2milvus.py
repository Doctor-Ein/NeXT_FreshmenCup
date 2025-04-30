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
import json

# 配置参数
MARKDOWN_DIR = "/Users/xlx/Desktop/工作文档/电院新生杯/NeXT/RAG_Package/MD"
MILVUS_HOST = "0.0.0.0"
MILVUS_PORT = "19530"
COLLECTION_NAME = "markdown_docs"
LOCAL_MODEL_DIR = "./local_models/bge-m3"  # 本地模型路径
VECTOR_DIM = 1024  # BGE-M3的嵌入维度

embedding = HuggingFaceEmbedding(model_name=LOCAL_MODEL_DIR)

def process_markdown_docs(
    markdown_dir: str,
    chunk_size: int = 256,
    chunk_overlap: int = 48,
) -> list:
    """
    - 使用 TokenTextSplitter 提供基于 token 的更精准分块
    - 保留 Markdown 结构层级，优先在段落或标题处分割
    - 自定义最大 token 限制，减少无用上下文
    """
    # 1. 检查目录
    path = Path(markdown_dir)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"目录不存在: {markdown_dir}")

    # 2. 加载 Markdown 文档
    reader = SimpleDirectoryReader(
        input_dir=markdown_dir,
        file_extractor={".md": MarkdownReader()},
        required_exts=[".md"],
        exclude_hidden=True,
    )
    documents = reader.load_data()
    print(f"成功加载 {len(documents)} 个文档")

    # 3. 拆分成 Markdown 节点
    md_parser = MarkdownNodeParser()
    nodes = md_parser.get_nodes_from_documents(documents)
    print(f"生成 {len(nodes)} 个节点")

    # 4. 基于 Token 的分块配置
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=" ",
        backup_separators=["\n", ".", "?"],
    )

    # 5. 对每个节点进行分块并添加元数据（关键修改）
    chunks = []
    for node in nodes:
        raw_text = node.text
        cleaned_text = "\n".join(
            [line.strip() for line in raw_text.splitlines() if line.strip()]
        )
        
        # 获取源文件元数据
        metadata = {
            "file_name": node.metadata.get("file_name", ""),
            "file_path": node.metadata.get("file_path", ""),
            "file_type": node.metadata.get("file_type", "")
        }
        
        for idx, chunk_text in enumerate(splitter.split_text(cleaned_text)):
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_id": f"{metadata['file_name']}_chunk_{idx}",
                    "chunk_index": idx
                }
            })

    return chunks


# 创建Milvus集合（保持不变）
def create_milvus_collection():
    fields = [
        FieldSchema(
            name="id", 
            dtype=DataType.INT64, 
            is_primary=True, 
            auto_id=True
        ),
        FieldSchema(
            name="text", 
            dtype=DataType.VARCHAR, 
            max_length=4096
        ),
        FieldSchema(
            name="metadata", 
            dtype=DataType.JSON
        ),
        FieldSchema(
            name="vector", 
            dtype=DataType.FLOAT_VECTOR, 
            dim=VECTOR_DIM
        )
    ]
    
    schema = CollectionSchema(
        fields=fields,
        description="Markdown文档向量存储"
    )
    
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
    
    collection = Collection(
        name=COLLECTION_NAME,
        schema=schema,
        using="default"
    )
    
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 1024}
    }
    
    collection.create_index(
        field_name="vector",
        index_params=index_params
    )
    
    return collection

# 存储到Milvus（保持不变）
def store_in_milvus(nodes):
    connections.connect(
        host=MILVUS_HOST, 
        port=MILVUS_PORT
    )
    
    collection = create_milvus_collection()
    
    insert_data = []
    for node in nodes:
        insert_data.append({
            "text": node["text"],  # 使用字典访问方式
            "metadata": json.dumps(node["metadata"]),
            "vector": embedding.get_text_embedding(node["text"])
        })
    
    batch_size = 500
    for i in range(0, len(insert_data), batch_size):
        batch = insert_data[i:i+batch_size]
        entities = [
            [item["text"] for item in batch],
            [json.loads(item["metadata"]) for item in batch],
            [item["vector"] for item in batch]
        ]
        collection.insert(entities)
    
    collection.flush()
    collection.load()

if __name__ == "__main__":
    try:
        processed_nodes = process_markdown_docs(markdown_dir=MARKDOWN_DIR)
        store_in_milvus(processed_nodes)
        print(f"成功存储 {len(processed_nodes)} 个块到Milvus集合 {COLLECTION_NAME}")
    except Exception as e:
        print(f"处理失败: {str(e)}")