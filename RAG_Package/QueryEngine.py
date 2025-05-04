import json
from pathlib import Path
from pymilvus import MilvusClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # 禁止Tokenizer的并行化

# 配置
MILVUS_URI = "http://0.0.0.0:19530"
COLLECTION_NAME = "DL_KDB"
MODEL_PATH = "./local_models/bge-m3"
TOP_K = 5
RAW_DATA_JSON = './JsonDataBase/raw_data.json'  # 你之前保存的那个

# 客户端与模型
client    = MilvusClient(uri=MILVUS_URI)  
embedder = HuggingFaceEmbedding(model_name=MODEL_PATH)

def load_blocks_from_jsondb(json_path: str) -> list:
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"❌ 找不到 raw_data.json: {json_path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data

class QueryEngine:
    def __init__(self, milvus_client, embedder, collection, blocks):
        self.client     = milvus_client
        self.embedder   = embedder
        self.collection = collection
        self.blocks     = blocks

        self.file_block_map = {}
        for blk in blocks:
            fname = blk['metadata']['file_name']
            self.file_block_map.setdefault(fname, []).append(blk)

        for fname, blist in self.file_block_map.items():
            blist.sort(key=lambda b: int(b['metadata']['block_id']))
            print(f"[DEBUG] 加载文件 {fname} 的块数: {len(blist)}")  # [DEBUG]

        self.index = {
            (blk["metadata"]["file_name"], int(blk["metadata"]["block_id"])): blk
            for blk in blocks
        }

    def query(self, text_query: str, top_k: int = TOP_K):
        q_vec = self.embedder.get_text_embedding(text_query)

        res = self.client.search(
            collection_name=self.collection,
            data=[q_vec],
            anns_field="vector",
            search_params={"metric_type": "L2", "params": {'nlist':480}},
            limit=top_k,
            output_fields=["text", "metadata"]
        )

        results = []
        for hits in res:
            for hit in hits:
                m = hit["entity"]["metadata"]
                fname, blk_id = m["file_name"], int(m["block_id"])
                main = hit['entity']['text']
                if not main:
                    continue

                print(f"\n[DEBUG] 命中块: file={fname}, block_id={blk_id}")  # [DEBUG]

                if fname not in self.file_block_map:
                    print(f"[WARNING] file_name '{fname}' 不在 file_block_map 中！")  # [DEBUG]
                    continue

                all_blks = self.file_block_map[fname]
                adj = []

                flag1 = False
                flag2 = False
                for i, b in enumerate(all_blks):
                    bid = int(b["metadata"]["block_id"])
                    if bid == blk_id + 1:
                        print(f"[DEBUG] 匹配到块索引: i={i}, block_id={bid}")  # [DEBUG]
                        if i < len(all_blks) - 1:
                            adj.append(all_blks[i + 1])
                            flag1 = True
                    if bid == blk_id - 1:
                        print(f"[DEBUG] 匹配到块索引: i={i}, block_id={bid}")  # [DEBUG]
                        if i > 0:
                            adj.append(all_blks[i - 1])
                            flag2 = True
                    if flag1 and flag2:
                        break

                if not adj:
                    print(f"[DEBUG] ❌ 找不到邻接块，使用默认")  # [DEBUG]
                    adj = [{"type": "text", "block_id": -1, "text": "[No Adjacent]"}]

                results.append({
                    "main": main,
                    "adjacent": adj
                })
        return results


if __name__ == "__main__":
    print("📦 向量库总量：", client.get_collection_stats(collection_name=COLLECTION_NAME))
    blocks = load_blocks_from_jsondb(RAW_DATA_JSON)

    engine = QueryEngine(
        milvus_client=client,
        embedder=embedder,
        collection=COLLECTION_NAME,
        blocks=blocks
    )

    query_str = input("Enter your query: ")
    print(f'[INFO] query_str: {query_str}')
    out = engine.query(query_str, top_k=5)

    for item in out:
        print(item['main'],end='\n\n')

        for adj in item["adjacent"]:
            type = adj.get("type", "unknown")
            block_id = adj.get("block_id", "?")
            print(f"  ↳ Adjacent ({type}), Block={block_id}:")
            if type == "text":
                print("    Text:", adj.get("text", "[No Text]"))
            else:
                print("    Content:", adj.get("content", "[No Content]"))
