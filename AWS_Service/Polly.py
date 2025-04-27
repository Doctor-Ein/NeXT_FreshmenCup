import os
import time
import boto3
import pyaudio

aws_region = os.getenv('AWS_REGION', 'us-east-1')

# 定义语音列表
voiceLanguageList = ['cmn-CN', 'en-US', 'ja-JP', 'ko-KR']
voiceNameList = ['Zhiyu', 'Ivy', 'Takumi', 'Seoyeon']
voicePromptList = ['Chinese', 'English', 'Japanese', 'Korean']

# 默认配置为中文
voiceIndex = 2

config = {
    'log_level': 'none',  # One of: info, debug, none
    #'last_speech': "If you have any other questions, please don't hesitate to ask. Have a great day!",
    'region': aws_region,
    'polly': {
        'Engine': 'neural',
        'LanguageCode': voiceLanguageList[voiceIndex],
        'VoiceId':voiceNameList[voiceIndex],
        'OutputFormat': 'pcm',
    }
}

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
    # 初始化读取器
    reader = Reader()

    # 要读取的文本
    text_to_read = "渡月橋の上で、ずっと君を思い出していた。"

    # 读取文本
    reader.read(text_to_read)

    # 关闭读取器
    reader.close()