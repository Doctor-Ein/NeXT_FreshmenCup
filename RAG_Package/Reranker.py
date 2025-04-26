import os
import torch
from typing import List, Dict
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class MilvusReranker:
    def __init__(self, model_name: str = "./models/bge-reranker-large", device: str = None):
        """
        初始化Milvus专用的重排器（强制离线模式）
        
        参数:
            model_name: 本地模型路径（必须为已下载的路径）
            device: 指定设备(如'cuda', 'cpu')，默认为自动选择
        """
        # 强制设置为离线模式
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        
        self.device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 从本地加载分词器和模型（强制不使用网络）
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True,
            use_fast=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            local_files_only=True
        ).to(self.device)
        self.model.eval()
    
    def rerank_documents(
        self,
        query: str,
        retrieved_documents: List[Dict],
        batch_size: int = 32,
        top_k: int = None,
        verbose: bool = False
    ) -> List[Dict]:
        """
        对Milvus检索结果进行重排
        
        参数:
            query: 用户查询字符串
            retrieved_documents: Milvus检索结果列表，每个文档应包含:
                - text: 文档内容
                - id: 文档ID
                - partition: Milvus分区信息
            batch_size: 批处理大小
            top_k: 返回前k个结果，None表示返回全部
            verbose: 是否显示截断警告
            
        返回:
            重排后的文档列表，每个文档包含:
                - text: 文档内容
                - score: 重排分数
                - metadata: 包含id和partition的字典
                - is_truncated: 是否被截断（新增字段）
        """
        if not retrieved_documents:
            return []
        
        # 准备文档数据
        docs_text = [doc['text'] for doc in retrieved_documents]
        pairs = [(query, doc_text) for doc_text in docs_text]
        
        # 分批计算得分
        all_scores = []
        truncation_info = []  # 记录截断情况
        
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            
            # 第一步：获取截断信息（不传递给模型）
            trunc_check = self.tokenizer(
                batch_pairs,
                padding=True,
                truncation="only_second",
                max_length=512,
                return_overflowing_tokens=True,
                return_tensors="pt"
            )
            batch_truncated = trunc_check.get("num_truncated_tokens", [0]*len(batch_pairs))
            truncation_info.extend(batch_truncated)
            
            if verbose and sum(batch_truncated) > 0:
                print(f"Batch {i//batch_size}: {sum(t > 0 for t in batch_truncated)}/{len(batch_pairs)} 文档被截断")
            
            # 第二步：实际编码（不返回overflowing_tokens）
            inputs = self.tokenizer(
                batch_pairs,
                padding=True,
                truncation="only_second",
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # 计算得分
            with torch.no_grad():
                batch_scores = self.model(**inputs).logits.view(-1).float().cpu()
                all_scores.extend(batch_scores.tolist())
        
        # 组合结果并添加元数据
        scored_documents = []
        for idx, (score, truncated) in enumerate(zip(all_scores, truncation_info)):
            original_doc = retrieved_documents[idx]
            scored_documents.append({
                'text': original_doc['text'],
                'score': float(score),
                'is_truncated': truncated > 0,
                'metadata': {
                    'id': original_doc['id'],
                    'partition': original_doc['partition'],
                    'truncated_tokens': truncated if verbose else None
                }
            })
        
        # 按分数降序排序
        scored_documents.sort(key=lambda x: x['score'], reverse=True)
        
        return scored_documents[:top_k] if top_k is not None else scored_documents
    
    def __call__(self, query: str, retrieved_documents: List[Dict], **kwargs):
        """使实例可调用，方便集成到QueryEngine"""
        return self.rerank_documents(query, retrieved_documents, **kwargs)