import json
from pathlib import Path

from pymilvus import MilvusClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 配置
MILVUS_URI = "http://0.0.0.0:19530"
COLLECTION_NAME = "DL_KDB"
MODEL_PATH      = "./local_models/bge-m3"
CONTENT_JSON    = "./Data/Paper/MinerU_Res/AlexNet/AlexNet_content_list.json"
TOP_K           = 5

# 客户端与模型
client    = MilvusClient(uri=MILVUS_URI)  
embedder = HuggingFaceEmbedding(model_name=MODEL_PATH)

class QueryEngine:
    def __init__(self, milvus_client, embedder, collection, blocks):
        self.client     = milvus_client
        self.embedder   = embedder
        self.collection = collection
        self.blocks     = blocks            # list of dict with metadata & content
        # 按 (file_name, block_id) 建索引
        self.index = {
            (b["metadata"]["file_name"], b["metadata"]["block_id"]): b
            for b in blocks
        }

    def query(self, text_query: str, top_k: int = TOP_K):
        # 1. 嵌入查询
        q_vec = self.embedder.get_text_embedding(text_query)  :contentReference[oaicite:3]{index=3}

        # 2. Milvus 检索
        res = self.client.search(
            collection_name=COLLECTION_NAME,
            data=[q_vec],
            anns_field="vector",
            search_params={"metric_type": "L2", "params": {}},
            limit=top_k,
            output_fields=["metadata"]  # metadata 包含 file_name, page, block_id :contentReference[oaicite:4]{index=4}
        )

        results = []
        for hits in res:
            for hit in hits:
                m = hit["entity"]["metadata"]
                key = (m["file_name"], m["block_id"])
                # 3. 主体块
                main = {
                    "metadata": m,
                    "text": self.index[key]["content"].get("text", "")
                }
                # 4. 邻近块（前一、后一）
                adj = []
                for nbr in (m["block_id"] - 1, m["block_id"] + 1):
                    nbr_key = (m["file_name"], nbr)
                    if nbr_key in self.index:
                        nb = self.index[nbr_key]
                        adj.append({
                            "metadata": nb["metadata"],
                            "type": nb["metadata"]["type"],
                            "content": nb["content"]
                        })
                results.append({"main": main, "adjacent": adj})
        return results

if __name__ == "__main__":
    # 初始化引擎
    engine = QueryEngine(
        milvus_client=client,
        embedder=embedder,
        collection=COLLECTION_NAME,
        blocks=all_blocks
    )

    # 用户输入
    query_str = input("Enter your query: ")
    out = engine.query(query_str, top_k=5)

    # 打印结果
    for i, item in enumerate(out, 1):
        m = item["main"]["metadata"]
        print(f"\nResult {i}: File={m['file_name']} Page={m['page']} Block={m['block_id']}")
        print("Text:", item["main"]["text"])
        for adj in item["adjacent"]:
            am = adj["metadata"]
            print(f"  ↳ Adjacent ({adj['type']}), Block={am['block_id']}: {adj['content']}")
