# import os
# import json
# from pathlib import Path
# from llama_cloud_services import LlamaParse
# from typing import List, Dict, Optional

# class PDFToMarkdownParser:
#     def __init__(self, api_key: str, output_dir: str = "./output"):
#         """
#         初始化PDF解析器
#         :param api_key: LlamaCloud API密钥
#         :param output_dir: 输出目录路径
#         """
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(parents=True, exist_ok=True)
        
#         # 初始化解析器（优化列表解析配置）
#         self.parser = LlamaParse(
#             api_key=api_key,
#             num_workers=4,
#             result_type="markdown",
#             user_prompt="保留所有列表格式（有序/无序），保持原始缩进层级",
#             markdown_options={
#                 "list_handling": "strict",  # 严格处理列表格式
#                 "list_item_indent": 4,     # 列表项缩进空格数
#                 "preserve_list_continuation": True
#             },
#             save_images=True,
#             image_output_dir=str(self.output_dir / "images"),
#             extract_images=True,
#             verbose=True
#         )

#     def parse_pdf(self, pdf_path: str) -> Optional[Dict]:
#         """
#         解析PDF文件并返回结构化Markdown内容
#         :param pdf_path: PDF文件路径
#         :return: 包含解析结果的字典
#         """
#         try:
#             print(f"正在解析文件: {pdf_path}")
#             result = self.parser.parse(pdf_path)
            
#             for res in result:
#                 # 验证列表解析结果
#                 if not self._validate_list_parsing(res.text):
#                     print("警告：列表内容可能未正确解析")
                
#                 return {
#                     "metadata": res.metadata,
#                     "text": res.text,
#                     "images": res.images
#                 }
#         except Exception as e:
#             print(f"解析失败: {str(e)}")
#             return None

#     def save_results(self, data: Dict, filename: str) -> None:
#         """保存解析结果"""
#         output_path = self.output_dir / filename
        
#         # 保存Markdown文本
#         with open(output_path.with_suffix('.md'), 'w', encoding='utf-8') as f:
#             f.write(data['text'])

# if __name__ == "__main__":
#     # 配置参数
#     API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
#     PDF_PATH_PREFIX = "./Data/Paper/"

#     file_list = []
#     for root, dirs, files in os.walk(PDF_PATH_PREFIX):
#         for file in files:
#             file_list.append(os.path.join(root, file))

#     print(file_list)
    
#     if not API_KEY:
#         raise ValueError("请设置LLAMA_CLOUD_API_KEY环境变量")

#     # 执行解析
#     parser = PDFToMarkdownParser(api_key=API_KEY)
#     result = parser.parse_pdf(file_list)
    
#     if result:
#         # 保存结果并打印摘要
#         parser.save_results(result, "alexnet_parsed")
#         print(f"解析完成！结果已保存到 {parser.output_dir}")
#         print(f"列表项总数: {parser._count_list_items(result['text'])}")
        
#         # 打印示例内容（前200个字符）
#         print("\n示例内容：")
#         print(result['text'][:200] + "...")

from llama_cloud_services import LlamaParse
import os
import json

# 初始化解析器（优化列表解析配置）
if __name__ == "__main__":
    parser = LlamaParse(api_key=os.getenv('LLAMA_CLOUD_API_KEY'),
                        num_workers=4,
                        result_type="markdown",
                        user_prompt="保留所有列表格式（有序/无序），保持原始缩进层级",
                        markdown_options={
                            "list_handling": "strict",  # 严格处理列表格式
                            "list_item_indent": 4,     # 列表项缩进空格数
                            "preserve_list_continuation": True
                        },
                        save_images=True,
                        image_output_dir=str("./images"),
                        extract_images=True,
                        verbose=True
                        )
    
    file_list = []
    for root, dirs, files in os.walk("./Data/Paper"):
        for file in files:
            file_list.append(os.path.join(root, file))

    print(file_list)
    # results = parser.parse(file_list)
    # i = 0
    # for res in results:
    #     i = i + 1
    #     with open('output_markdown.json','w',encoding='utf-8') as f:
    #         document = res.get_markdown_documents()
    #         json.dump(document.text,fp=f,indent=4,ensure_ascii=False)
    result = parser.parse(file_list[0])
    with open('output.json','w',encoding='utf-8') as f:
        document = result.get_markdown_documents()
        for doc in document:
            print(doc.text,file=f)