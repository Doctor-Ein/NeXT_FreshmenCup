import json
from pathlib import Path
from pymilvus import MilvusClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 配置
MILVUS_URI       = "http://0.0.0.0:19530"
COLLECTION_NAME  = "DL_KDB"
MODEL_PATH       = "./local_models/bge-m3"
TOP_K            = 5
RERANK_TOP_K     = 5
JSON_PATH    = './JsonDataBase/text_chunks.json'

# 客户端与模型
client    = MilvusClient(uri=MILVUS_URI)
embedder  = HuggingFaceEmbedding(model_name=MODEL_PATH)

def load_blocks_from_jsondb(json_path: str) -> list:
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"❌ 找不到 text_chunks.json: {json_path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    # 在load_blocks_from_jsondb函数中添加
    for blk in data:
        meta = blk.get("metadata", {})
        if not all(k in meta for k in ("file_name", "block_id")):
            print(f"⚠️ 损坏的元数据块: {json.dumps(blk, indent=2)}")
    return data

blocks = load_blocks_from_jsondb(JSON_PATH)

class QueryEngine:
    def __init__(self, milvus_client, embedder, collection, reranker=None):
        self.client = milvus_client
        self.embedder = embedder
        self.collection = collection
        self.reranker = reranker

        # 构建基于(file_name, block_id)的索引
        self.index = {
            (blk["metadata"]["file_name"], int(blk["metadata"]["block_id"])): blk
            for blk in blocks
        }

    def query(self, text_query: str, top_k: int = TOP_K, use_rerank: bool = False, rerank_top_k: int = RERANK_TOP_K):
        q_vec = self.embedder.get_text_embedding(text_query)

        res = self.client.search(
            collection_name=self.collection,
            data=[q_vec],
            anns_field="vector",
            search_params={"metric_type": "L2", "params": {'nlist': 480}},
            limit=top_k,
            output_fields=["text", "metadata"]  # 确保metadata字段被请求
        )

        candidates = []
        for hits in res:
            for hit in hits:
                hit_metadata = hit["entity"]["metadata"]
                fname = hit_metadata["file_name"]
                blk_id = int(hit_metadata["block_id"])
                
                # 从预建索引中获取完整元数据
                full_block = self.index.get((fname, blk_id))
                if not full_block:
                    continue
                
                # 合并可能存在的元数据字段（优先使用原始块数据）
                full_metadata = {**full_block["metadata"], **hit_metadata}
                # 修改候选文档构建逻辑
                # 修改候选文档构建逻辑（QueryEngine.query()方法内）
                candidates.append({
                    "text": hit["entity"]["text"],
                    "id": str(full_metadata["block_id"]),
                    "partition": full_metadata["file_name"],  # 关键映射
                    "metadata": full_metadata
                })

        # 在重排前添加保护逻辑
        candidates = [c for c in candidates if c]  # 过滤空值
        if not candidates:
            return []

        # 结果重排处理
        if use_rerank and self.reranker:
            
            reranked = self.reranker(
                query=text_query,
                retrieved_documents=candidates,
                top_k=rerank_top_k
            )
        else:
            reranked = [{
                "text": c["text"],
                "metadata": c["metadata"],
                "score": None
            } for c in candidates[:rerank_top_k]]

        # 构建最终结果（包含完整元数据）
        results = []
        for doc in reranked:
            results.append({
                "text": doc["text"],
                "metadata": doc["metadata"]
            })

        return results


print("📦 向量库总量：", client.get_collection_stats(collection_name=COLLECTION_NAME))


# # 加载重排器（如果有）
# try:
#     from RAG_Package.reranker import MilvusReranker
#     reranker = MilvusReranker(model_name="./local_models/bge-reranker-large")
# except ImportError:
#     reranker = None
#     print("[WARNING] 未加载重排器，将禁用重排")

query_engine = QueryEngine(
    milvus_client=client,
    embedder=embedder,
    collection=COLLECTION_NAME,
    reranker=None  # ✅ 正确参数列表
)