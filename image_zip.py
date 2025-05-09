import base64
import io
import re
from PIL import Image

MAX_SIZE = 5 * 1024 * 1024  # 5MB

def compress_base64_image(base64_str, media_type='image/jpeg'):
    # 移除前缀信息（如 data:image/png;base64,）
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    
    # 移除所有非 Base64 字符
    base64_str = re.sub(r'[^A-Za-z0-9+/=]', '', base64_str)
    
    # 添加必要的填充字符
    missing_padding = len(base64_str) % 4
    if missing_padding:
        base64_str += '=' * (4 - missing_padding)
    
    # 解码 Base64 字符串为二进制数据
    try:
        image_data = base64.b64decode(base64_str)
    except base64.binascii.Error as e:
        raise ValueError(f"Base64 解码失败：{e}")
    
    image = Image.open(io.BytesIO(image_data))

    # 将图像转换为 RGB 模式（如果不是）
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # 初始化压缩图像的二进制流
    compressed_io = io.BytesIO()
    quality = 85  # 初始质量设置

    # 压缩图像直到大小符合要求
    while True:
        compressed_io.seek(0)
        compressed_io.truncate()
        image.save(compressed_io, format='JPEG', quality=quality)
        size = compressed_io.tell()
        if size <= MAX_SIZE or quality <= 10:
            break
        quality -= 5  # 每次减少质量以进一步压缩

    # 编码压缩后的图像为 Base64 字符串
    compressed_io.seek(0)
    compressed_base64 = base64.b64encode(compressed_io.read()).decode('utf-8')

    # 返回符合 Invoke API 要求的字典
    return {"media_type": media_type, "data": compressed_base64}
