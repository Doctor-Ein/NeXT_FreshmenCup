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
RAW_DATA_JSON    = './JsonDataBase/raw_data.json'

# 客户端与模型
client    = MilvusClient(uri=MILVUS_URI)
embedder  = HuggingFaceEmbedding(model_name=MODEL_PATH)

def load_blocks_from_jsondb(json_path: str) -> list:
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"❌ 找不到 raw_data.json: {json_path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data

class QueryEngine:
    def __init__(self, milvus_client, embedder, collection, blocks, reranker=None):
        self.client     = milvus_client
        self.embedder   = embedder
        self.collection = collection
        self.blocks     = blocks
        self.reranker   = reranker

        self.file_block_map = {}
        for blk in blocks:
            fname = blk['metadata']['file_name']
            self.file_block_map.setdefault(fname, []).append(blk)

        for fname, blist in self.file_block_map.items():
            blist.sort(key=lambda b: int(b['metadata']['block_id']))
            print(f"[DEBUG] 加载文件 {fname} 的块数: {len(blist)}")

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
            output_fields=["text", "metadata"]
        )

        candidates = []
        for hits in res:
            for hit in hits:
                m = hit["entity"]["metadata"]
                fname = m["file_name"]
                blk_id = int(m["block_id"])
                text = hit['entity']['text']

                if not text:
                    continue

                candidates.append({
                    "text": text,
                    "id": blk_id,
                    "partition": fname  # 复用 partition 字段
                })

        # 若启用重排，则进行重排处理
        if use_rerank and self.reranker:
            reranked = self.reranker(
                query=text_query,
                retrieved_documents=candidates,
                top_k=rerank_top_k
            )
            # 把文本拿出来再找邻接块
            reranked_texts = [doc["text"] for doc in reranked]
        else:
            reranked = [{"text": c["text"], "score": None, "metadata": {"id": c["id"], "partition": c["partition"]}} for c in candidates]
            reranked_texts = [c["text"] for c in candidates]

        results = []
        for doc in reranked:
            fname = doc["metadata"]["partition"]
            blk_id = int(doc["metadata"]["id"])
            main = doc["text"]

            print(f"\n[DEBUG] 命中块: file={fname}, block_id={blk_id}")

            if fname not in self.file_block_map:
                print(f"[WARNING] file_name '{fname}' 不在 file_block_map 中！")
                continue

            raw_blks = self.file_block_map[fname]
            adj = []

            for block in raw_blks:
                bid = block['metadata']['block_id']
                if bid == blk_id - 1 or bid == blk_id + 1:
                    adj.append(block)

            if not adj:
                print(f"[DEBUG] ❌ 找不到邻接块，使用默认")
                adj = [{"type": "text", "block_id": -1, "text": "[No Adjacent]"}]

            results.append({
                "main": main,
                "adjacent": adj
            })

        return results

if __name__ == "__main__":
    print("📦 向量库总量：", client.get_collection_stats(collection_name=COLLECTION_NAME))
    blocks = load_blocks_from_jsondb(RAW_DATA_JSON)

    # 加载重排器（如果有）
    try:
        from RAG_Package.reranker import MilvusReranker
        reranker = MilvusReranker(model_name="./local_models/bge-reranker-large")
    except ImportError:
        reranker = None
        print("[WARNING] 未加载重排器，将禁用重排")

    engine = QueryEngine(
        milvus_client=client,
        embedder=embedder,
        collection=COLLECTION_NAME,
        blocks=blocks,
        reranker=reranker
    )

    query_str = input("Enter your query: ")
    print(f'[INFO] query_str: {query_str}')
    out = engine.query(query_str, top_k=10, use_rerank=True, rerank_top_k=3)

    for item in out:
        print(item['main'], end='\n\n')
        for adj in item["adjacent"]:
            type = adj.get("type", "unknown")
            block_id = adj.get("block_id", "?")
            print(f"  ↳ Adjacent ({type}), Block={block_id}:")
            if type == "text":
                print("    Text:", adj.get("text", "[No Text]"))
            else:
                print("    Content:", adj.get("content", "[No Content]"))
