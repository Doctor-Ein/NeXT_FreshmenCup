from huggingface_hub import snapshot_download
from pathlib import Path

MODEL_NAME = "BAAI/bge-m3"
LOCAL_MODEL_DIR = "./local_models/bge-m3"

# 创建目录
Path(LOCAL_MODEL_DIR).mkdir(parents=True, exist_ok=True)

# 强制下载所有文件（包括大文件）
snapshot_download(
    repo_id=MODEL_NAME,
    local_dir=LOCAL_MODEL_DIR,
    local_dir_use_symlinks=False,  # 避免符号链接
    ignore_patterns=["*.msgpack", "*.h5"],  # 排除非必要文件
    resume_download=True  # 支持断点续传
)

print(f"完整模型已下载到: {LOCAL_MODEL_DIR}")
print("目录内容:", [p.name for p in Path(LOCAL_MODEL_DIR).iterdir()])

def verify_model():
    required_files = {
        "config.json": (10, 100),  # 文件大小范围(KB)
        "model.safetensors": (500000, None),  # 至少500MB
        "tokenizer.json": (100, 1000)
    }
    
    for filename, (min_size, max_size) in required_files.items():
        filepath = Path(LOCAL_MODEL_DIR)/filename
        if not filepath.exists():
            raise FileNotFoundError(f"缺失关键文件: {filename}")
        
        file_size = filepath.stat().st_size / 1024  # KB
        if min_size and file_size < min_size:
            raise ValueError(f"{filename} 文件过小(仅{file_size:.1f}KB)")
        if max_size and file_size > max_size:
            raise ValueError(f"{filename} 文件过大({file_size:.1f}KB)")

    print("✅ 模型验证通过！")

if __name__ == "__main__":
    verify_model()
