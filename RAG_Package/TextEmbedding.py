import json
from pathlib import Path

from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # 禁用Tokenizer的并行

# -------------------- 配置参数 --------------------
CONTENT_LIST_JSON = './Data/Paper/MinerU_Res/AlexNet/AlexNet_content_list.json'
MILVUS_HOST = '0.0.0.0'
MILVUS_PORT = '19530'
COLLECTION_NAME = 'DL_KDB'
LOCAL_MODEL_DIR = './local_models/bge-m3'
VECTOR_DIM = 1024

# 本地嵌入模型
embedding = HuggingFaceEmbedding(model_name=LOCAL_MODEL_DIR)

def process_content_list_docs(
    content_list_path: str,
    chunk_size: int = 300,
    chunk_overlap: int = 34,
) -> tuple[list, list, list]:
    """
    处理 content_list.json:
    - 解析 text 块并切分为 text_chunks（用于嵌入）
    - 保留 equation、table、image 等 block 为 raw_data（带 metadata）
    - 所有 blocks 都返回为 all_blocks，用于数据库保存或 UI 展示
    """
    path = Path(content_list_path)
    if not path.is_file():
        raise FileNotFoundError(f"❌ 找不到 content_list.json: {content_list_path}")

    data = json.loads(path.read_text(encoding='utf-8'))
    print(f"🔍 读取到 {len(data)} 个布局块")

    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=' ',
        backup_separators=['\n', '.', '?']
    )

    file_name = path.stem.replace('_content_list', '')
    text_chunks = []
    raw_data    = []

    for block_id, block in enumerate(data):
        btype = block.get('type', 'text')
        page  = block.get('page_idx', 0)

        # 补充元数据
        metadata = {
            'file_name': file_name,
            'page': page,
            'block_id': block_id,
            'type': btype
        }

        # ----- 处理纯文本 -----
        if btype == 'text':
            raw_text = block.get('text', '').strip()
            if raw_text:
                # 是否需要分块
                subs = (
                    splitter.split_text(raw_text)
                    if len(raw_text.split()) > chunk_size
                    else [raw_text]
                )
                for idx, sub in enumerate(subs):
                    text_chunks.append({
                        'text': sub,
                        'metadata': {
                            **metadata,
                            'chunk_id': f"{block_id}_chunk_{idx}",
                            'chunk_index': idx
                        }
                    })

        # ----- equation -----
        elif btype == 'equation':
            entry = {
                'type': btype,
                'metadata': metadata,
                'content': {
                    'text_format': block.get('text_format', ''),
                    'latex': block.get('text', '')
                }
            }
            raw_data.append(entry)

        # ----- table -----
        elif btype == 'table':
            entry = {
                'type': btype,
                'metadata': metadata,
                'content': {
                    'caption': block.get('table_caption', []),
                    'footnote': block.get('table_footnote', []),
                    'html': block.get('table_body', '')
                }
            }
            raw_data.append(entry)

        # ----- 其他类型：image、figure 等 -----
        else:
            entry = {
                'type': btype,
                'metadata': metadata,
                'content': block  # 保留原始内容字段
            }
            raw_data.append(entry)

    print(f"✅ 生成 {len(text_chunks)} 条文本 chunk，{len(raw_data)} 条 RawData 条目")
    return text_chunks, raw_data


def create_milvus_collection(collection_name: str):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name='metadata', dtype=DataType.JSON, nullable=True),
        FieldSchema(name='vector', dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
    ]
    schema = CollectionSchema(fields=fields, description='Deep Learning Knowledge DB')
    col = Collection(name=collection_name, schema=schema, using='default')
    col.create_index(
        field_name='vector',
        index_params={'index_type':'IVF_FLAT','metric_type':'L2','params':{'nlist':480}} # 480 ≈ 4x√15000
    )
    col.load()
    return col


def store_in_milvus(chunks: list):
    connections.connect(alias='default', host=MILVUS_HOST, port=MILVUS_PORT)
    collection = create_milvus_collection(COLLECTION_NAME)

    texts, metas, vecs = [], [], []
    for chunk in chunks:
        texts.append(chunk['text'])
        metas.append(chunk['metadata'])
        vecs.append(embedding.get_text_embedding(chunk['text']))

    batch_size = 500
    for i in range(0, len(texts), batch_size):
        collection.insert([texts[i:i+batch_size],
                           metas[i:i+batch_size],
                           vecs[i:i+batch_size]])
    collection.flush()
    collection.load()
    print(f"🚀 成功存储 {len(texts)} 条记录到 Milvus 集合 '{COLLECTION_NAME}'")

if __name__ == '__main__':
    try:
        # 1. 读取并分块（文本 chunks + RawData）
        text_chunks, raw_data = process_content_list_docs(content_list_path=CONTENT_LIST_JSON)

        # 2. 确保输出目录存在
        out_dir = Path('./JsonDataBase')
        out_dir.mkdir(exist_ok=True)

        # 3. 保存文本 chunks 到 JSON
        text_chunks_path = out_dir / 'text_chunks.json'
        with open(text_chunks_path, 'w', encoding='utf-8') as f:
            json.dump(text_chunks, f, ensure_ascii=False, indent=2)

        # 4. 保存 RawData 到 JSON
        raw_data_path = out_dir / 'raw_data.json'
        with open(raw_data_path, 'w', encoding='utf-8') as f:
            json.dump(raw_data, f, ensure_ascii=False, indent=2)

        print(f"✅ 已将 {len(text_chunks)} 条文本 chunks 保存到 {text_chunks_path}")
        print(f"✅ 已将 {len(raw_data)} 条 RawData 条目保存到 {raw_data_path}")

        # 5. 将文本 chunks 存入 Milvus
        store_in_milvus(text_chunks)

    except Exception as e:
        print(f"处理失败: {e}")
