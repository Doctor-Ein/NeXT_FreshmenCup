def process_content_list_docs(
    content_list_path: str,
    chunk_size: int = 300,
    chunk_overlap: int = 34,
) -> list:
    """
    1. 读取 MinerU 导出的 content_list.json
    2. 以每个 layout-block 为单元生成初始 chunk，只保留 type == 'text'
    3. 对超长文本再次用 TokenTextSplitter 进行分块
    4. 返回包含 text 和 metadata 的列表，metadata 包括：
       - file_name: 从文件名推断
       - page: 来自 page_idx
       - block_id: JSON 中块的索引
       - chunk_id: block_id + chunk 索引
       - chunk_index: 本节点中文本块的索引
    """
    from pathlib import Path
    import json
    from llama_index.core.text_splitter import TokenTextSplitter

    path = Path(content_list_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"找不到 content_list.json: {content_list_path}")

    # 读取 JSON
    data = json.loads(path.read_text(encoding='utf-8'))
    print(f"🔍 读取到 {len(data)} 个布局块")

    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=' ',
        backup_separators=['\n', '.', '?']
    )

    chunks = []
    file_name = path.stem.replace('_content_list', '')

    for block_id, block in enumerate(data):
        # 仅保留纯文本块
        if block.get('type') != 'text':
            continue

        raw_text = block.get('text', '').strip()
        if not raw_text:
            continue

        page = block.get('page_idx', 0)

        # 根据 token 数量决定是否二次分割
        subs = (
            splitter.split_text(raw_text)
            if len(raw_text.split()) > chunk_size
            else [raw_text]
        )

        for idx, sub in enumerate(subs):
            chunks.append({
                'text': sub,
                'metadata': {
                    'file_name': file_name,
                    'page': page,
                    'block_id': block_id,
                    'chunk_id': f"{block_id}_chunk_{idx}",
                    'chunk_index': idx
                }
            })

    print(f"✅ 生成 {len(chunks)} 条有效文本 chunk")
    return chunks
