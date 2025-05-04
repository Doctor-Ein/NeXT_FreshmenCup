import os
import sys
from AWS_Service.api_request_schema import api_request_list, get_model_ids

model_id = os.getenv('MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0') # 从环境变量中获取模型id
aws_region = os.getenv('AWS_REGION', 'us-east-1') # 从环境变量中获取AWS区域

if model_id not in get_model_ids(): # 验证模型存在于配置清单中
    print(f'Error: Models ID {model_id} in not a valid model ID. Set MODEL_ID env var to one of {get_model_ids()}.')
    sys.exit(0)

api_request = api_request_list[model_id] # 定义全局的api_request配置表

# 定义语音列表
voiceLanguageList = ['cmn-CN', 'en-US', 'ja-JP', 'ko-KR']
voiceNameList = ['Zhiyu', 'Ivy', 'Takumi', 'Seoyeon']
voicePromptList = ['Chinese', 'English', 'Japanese', 'Korean']

# 默认配置为中文
voiceIndex = 0

config = {
    'log_level': 'info',  # One of: info, debug, none
    'region': aws_region,
    'bedrock': {
        'api_request': api_request
    },
    'network': {
        'connect_timeout': 5,  # 连接超时时间（秒）
        'read_timeout': 10,    # 读取超时时间（秒）
        'max_retries': 3,      # 最大重试次数
        'retry_delay': 2       # 重试延迟时间（秒）
    },
    'polly': {
        'Engine': 'neural',
        'LanguageCode': voiceLanguageList[voiceIndex],
        'VoiceId':voiceNameList[voiceIndex],
        'OutputFormat': 'pcm',
    }
}