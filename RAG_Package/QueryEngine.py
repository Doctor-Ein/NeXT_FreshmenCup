import os
from pymilvus import MilvusClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from reranker import MilvusReranker

os.environ["TOKENIZERS_PARALLELISM"] = "false"

collection_name = "markdown_docs"

# 基于脚本目录构建模型路径
model_path = "./local_models/bge-m3"

# 使用 llama_index 加载本地模型
embedding = HuggingFaceEmbedding(model_name = model_path)

# 连接到 Milvus 服务
client = MilvusClient(uri="http://0.0.0.0:19530")

def queryContext(query:str):
    print("[query]:" + query)

    # 将查询上下文转化为向量
    QueryVector = embedding.get_text_embedding(query)

    # 检索 Milvus 中的向量
    search_params = {"metric_type": "L2", "params": {}}
    result = client.search(
        collection_name = collection_name,
        data = [QueryVector],  # 查询向量
        anns_field = "vector",  # 向量字段名
        search_params = search_params,  # 检索参数
        limit= 10 ,  # 返回的最相似向量数量
        output_fields = ["text"]  # 输出字段
    )

    for hits in result: # hits 是初步搜索结果的列表
        retrieved_documents = [{"text": hit['entity'].get('text','N/A'),"id": hit['id'],"partition":hit['entity'].get('partition',0)} for hit in hits]
    
    # context1 = []
    # for hits in result:
    #     for hit in hits:
    #         context1.append(f"{titles[str(hit['entity'].get('partition','0'))]}\n{hit['entity'].get('text','N/A')}\n")

    ## 重排模型
    reranker = MilvusReranker() 
    reranked_context = reranker.rerank_documents(query = query, retrieved_documents = retrieved_documents, top_k = 3)
    reranked_reranked_context = sorted(reranked_context,key = lambda x:x['metadata']['id'])

    ## 组合上下文内容
    context = ["<context>\n"]
    for doc in reranked_reranked_context:
        context.append(f"\n{doc['text']}\n")
    context.append("</context>")

    # json_obj = [context1,context]
    # with open('debug.json','w',encoding='utf-8') as file:
    #     json.dump(json_obj,file,ensure_ascii = False,indent = 4)

    return context


    # # 原始版本直接返回初步搜索的结果
    # context = ["<context>\n"]
    # for hits in result:
    #     for hit in hits:
    #         context.append(f"\n{hit['entity'].get('text','N/A')}\n")
    # context.append("\n</context>")
    # return context

if __name__ == "__main__":
    query = input("[Prompts]")
    print(queryContext(query))