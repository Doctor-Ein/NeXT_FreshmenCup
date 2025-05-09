import time
import boto3
import pyaudio
from AWS_Service.config import config

# 初始化PyAudio和AWS服务客户端
p = pyaudio.PyAudio()
polly = boto3.client('polly', region_name=config['region'])

# 读取器类
class Reader:

    def __init__(self):
        self.polly = boto3.client('polly', region_name=config['region'])
        self.audio = p.open(format=pyaudio.paInt16, channels=1, rate=16000, output=True)
        self.chunk = 1024

    def read(self, data):
        
        response = self.polly.synthesize_speech(
            Text=data,
            Engine=config['polly']['Engine'],
            LanguageCode=config['polly']['LanguageCode'],
            VoiceId=config['polly']['VoiceId'],
            OutputFormat=config['polly']['OutputFormat'],
        )

        stream = response['AudioStream']

        while True:
            data = stream.read(self.chunk)
            self.audio.write(data)
            if not data:
                break

    def close(self):
        time.sleep(1)
        self.audio.stop_stream()
        self.audio.close()

# 使用示例代码
if __name__ == "__main__":
    reader = Reader()
    reader.read("你好，我是来自亚马逊 Polly 的语音。")
    reader.close()