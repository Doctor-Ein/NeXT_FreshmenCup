import json
import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pymilvus import connections, MilvusClient, CollectionSchema, FieldSchema, DataType

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 读取章节数据
def load_chapters(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

# 主函数
def main():
    chapters = load_chapters("./chapters.json")
    collection_name = "Dream_of_the_Red_Chamber_IVF"

    # 使用llama_index加载本地模型
    embedding = HuggingFaceEmbedding(model_name="../models/bge-large-zh-v1.5")

    # 创建Milvus集合
    dim = 1024  # bge-large-zh-v1.5 模型生成的嵌入向量的维度是 1024
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="partition", dtype=DataType.INT64, auto_id = False) # 表征分区的字段为INT
    ]
    schema = CollectionSchema(fields, description="Chapters collection")
    client = MilvusClient(uri="http://0.0.0.0:19530")  # 修改了 MilvusClient 的初始化方式
    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)
    client.create_collection(collection_name=collection_name,schema=schema)

    # 按照 title 分区插入数据
    id_counter = 0
    partition_counter = 0
    for title, paragraphs in chapters.items():
        # 创建分区
        partition_counter += 1
        partition_id = '_' + str(partition_counter)
        if not client.has_partition(collection_name=collection_name, partition_name=partition_id):
            client.create_partition(collection_name=collection_name, partition_name=partition_id) # 分区名为'_' + str(数字)

        # 准备插入的数据
        documents = []
        for paragraph in paragraphs:
            id_counter += 1
            documents.append({"text": paragraph, "id": id_counter, "partition": partition_counter})

        # 获取嵌入向量
        embeddings = [embedding.get_text_embedding(doc["text"]) for doc in documents]

        # 插入数据到分区
        data_to_insert = [{"id": doc["id"], "text": doc["text"], "vector": embedding,"partition":doc["partition"]} for doc, embedding in zip(documents, embeddings)]
        client.insert(collection_name=collection_name, partition_name=partition_id, data=data_to_insert)

    # 定义索引参数
    index_params = client.prepare_index_params()
    
    index_params.add_index(
                            field_name = "vector",
                            index_type = "IVF_FLAT",  # 选择合适的索引类型
                            metric_type = "L2",       # 选择合适的度量类型
                            params = {"nlist": 128}   # 索引参数
                        )
    
    # index_params.add_index(
    #                         field_name = "vector",
    #                         index_name= "vector_index", 
    #                         index_type = "FLAT",  # 选择合适的索引类型
    #                         metric_type = "L2"    # 选择合适的度量类型
    #                        )
    
    # 创建索引
    client.create_index(
        collection_name = collection_name,
        index_params=index_params
    )
    
    index_description = client.describe_index(collection_name=collection_name,index_name="vector")
    print(index_description)

    # 加载集合以确保数据持久化
    client.load_collection(collection_name=collection_name)
    
    print(f"集合 {collection_name} 创建成功，数据已持久化保存")

if __name__ == "__main__":
    main()