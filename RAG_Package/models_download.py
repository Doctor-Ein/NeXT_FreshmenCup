from huggingface_hub import snapshot_download
from pathlib import Path

MODEL_NAME = "deepcs233/VisCoT-13b-336"
LOCAL_MODEL_DIR = "./local_models/VisCoT-13b-336"

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
    model_dir = Path(LOCAL_MODEL_DIR)
    if not model_dir.exists():
        print(f"❌ 模型目录不存在: {model_dir}")
        return

    files = [f for f in model_dir.iterdir() if f.is_file()]
    files_sorted = sorted(files, key=lambda x: x.name.lower())

    print(f"\n📂 模型目录: {model_dir.resolve()}")
    print("📄 文件列表（按名称排序）:")
    for f in files_sorted:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name:<30} {size_mb:.2f} MB")

    print(f"\n✅ 共计 {len(files_sorted)} 个文件。")

if __name__ == "__main__":
    verify_model()
