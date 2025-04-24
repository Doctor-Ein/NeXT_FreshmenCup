import json
import os
import time
import sys

# import pyaudio
import boto3

# from amazon_transcribe.client import TranscribeStreamingClient
# from amazon_transcribe.handlers import TranscriptResultStreamHandler
# from amazon_transcribe.model import TranscriptEvent, TranscriptResultStream
from api_request_schema import api_request_list, get_model_ids

model_id = os.getenv('MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0') # 从环境变量中获取模型id
aws_region = os.getenv('AWS_REGION', 'us-east-1') # 从环境变量中获取AWS区域

if model_id not in get_model_ids(): # 验证模型存在于配置清单中
    print(f'Error: Models ID {model_id} in not a valid model ID. Set MODEL_ID env var to one of {get_model_ids()}.')
    sys.exit(0)

api_request = api_request_list[model_id] # 定义全局的api_request配置表
config = {
    'log_level': 'info',  # One of: info, debug, none
    #'last_speech': "If you have any other questions, please don't hesitate to ask. Have a great day!",
    'region': aws_region,
    # 因为计划将polly包括transcribe都分离出去
    # 'polly': {
    #     'Engine': 'neural',
    #     'LanguageCode': 'cmn-CN',
    #     'VoiceId': 'Zhiyu',
    #     'OutputFormat': 'pcm',
    # },
    'bedrock': {
        'api_request': api_request
    }
}

# 两个辅助调试的输出日志函数
def printInfo():
    """
    输出系统信息文本
    功能描述：打印支持的模型、AWS区域、Amazon Bedrock模型、Polly配置和日志级别等信息
    """
    info_text = f'''
    *************************************************************
    [INFO] Supported FM models: {get_model_ids()}.
    [INFO] Change FM model by setting <MODEL_ID> environment variable. Example: export MODEL_ID=meta.llama2-70b-chat-v1

    [INFO] AWS Region: {config['region']}
    [INFO] Amazon Bedrock model: {config['bedrock']['api_request']['modelId']}
    [INFO] Polly config: engine {config['polly']['Engine']}, voice {config['polly']['VoiceId']}
    [INFO] Log level: {config['log_level']}

    [INFO] Hit ENTER to interrupt Amazon Bedrock. After you can continue speaking!
    [INFO] Go ahead with the voice chat with Amazon Bedrock!
    *************************************************************
    '''
    print(info_text) 
def printer(text, level):
    """
    打印日志信息
    功能描述：根据日志级别打印信息
    :param text: 要打印的文本
    :param level: 日志级别（info或debug）
    """
    if config['log_level'] == 'info' and level == 'info':
        print(text)
    elif config['log_level'] == 'debug' and level in ['info', 'debug']:
        print(text)

# 初始化音频处理和AWS服务客户端
bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name=config['region'])

class BedrockModelsWrapper:
    """
    Amazon Bedrock模型封装类
    功能描述：定义请求体、获取流块和文本
    """

    @staticmethod
    def define_body(text, dialogue_list = [], images = []):
        """
        定义请求体
        功能描述：根据不同的模型提供者定义请求体
        :param text: 输入文本
        :param dialogue_list: 从历史会话管理库中获取的列表，元素为会话块字典{role,content} -> 这里意思是指从特性上说不想要在把图片插入，但是本地还是要保存图片的🤔
        :param images: 输入图片的列表，元素为字典{media_type, data}
        :return: 请求体
        """
        model_id = config['bedrock']['api_request']['modelId']
        model_provider = model_id.split('.')[0]
        body = config['bedrock']['api_request']['body']

        # 实际上此次开发只需要anthropic
        if model_provider == 'amazon':
            body['inputText'] = text
        elif model_provider == 'meta':
            if 'llama3' in model_id:
                body['prompt'] = f"""
                    <|begin_of_text|>
                    <|start_header_id|>user<|end_header_id|>
                    {text}, please output in Chinese.
                    <|eot_id|>
                    <|start_header_id|>assistant<|end_header_id|>
                    """
            else: 
                body['prompt'] = f"<s>[INST] {text}, please output in Chinese. [/INST]"
        elif model_provider == 'anthropic':
            if "claude-3" in model_id:
                # content是单轮对话的内容,可以包含text和image
                content = [
                    {
                        "type": "text",
                        "text": text
                    }
                ]
                # 添加图片
                for image in images:
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image["media_type"],
                            "data": image["data"]
                        },
                    })
                
                body['messages'] = list(dialogue_list) # 使用浅拷，防止两个对象相互关联
                body['messages'].append({"role": "user", "content": content}) # 添加这一次对话的content
            else: # 这个看起来是处理anthropic的其他模型的，就不用管啦（笑）
                body['prompt'] = f'\n\nHuman: {text}\n\nAssistant:'
        elif model_provider == 'cohere':
            body['prompt'] = text
        elif model_provider == 'mistral':
            body['prompt'] = f"<s>[INST] {text}, please output in Chinese. [/INST]"
        else:
            raise Exception('Unknown model provider.')

        return body

    @staticmethod
    def get_stream_chunk(event):
        """
        获取流块
        功能描述：从事件中获取流块
        :param event: 事件对象
        :return: 流块
        """
        return event.get('chunk')

    @staticmethod
    def get_stream_text(chunk):
        """
        获取流文本
        功能描述：根据不同的模型提供者从流块中获取文本
        :param chunk: 流块
        :return: 文本
        """
        model_id = config['bedrock']['api_request']['modelId']
        model_provider = model_id.split('.')[0]

        chunk_obj = ''
        text = ''
        if model_provider == 'amazon':
            chunk_obj = json.loads(chunk.get('bytes').decode())
            text = chunk_obj['outputText']
        elif model_provider == 'meta':
            chunk_obj = json.loads(chunk.get('bytes').decode())
            text = chunk_obj['generation']
        elif model_provider == 'anthropic':
            if "claude-3" in model_id:
                chunk_obj = json.loads(chunk.get('bytes').decode())
                if chunk_obj['type'] == 'message_delta':
                    print(f"\nStop reason: {chunk_obj['delta']['stop_reason']}")
                    print(f"Stop sequence: {chunk_obj['delta']['stop_sequence']}")
                    print(f"Output tokens: {chunk_obj['usage']['output_tokens']}")

                if chunk_obj['type'] == 'content_block_delta':
                    if chunk_obj['delta']['type'] == 'text_delta':
                        #print(chunk_obj['delta']['text'], end="")
                        text = chunk_obj['delta']['text']
            else: # Claude2.x
                chunk_obj = json.loads(chunk.get('bytes').decode())
                text = chunk_obj['completion']
        elif model_provider == 'cohere':
            chunk_obj = json.loads(chunk.get('bytes').decode())
            text = ' '.join([c["text"] for c in chunk_obj['generations']])
        elif model_provider == 'mistral':
            chunk_obj = json.loads(chunk.get('bytes').decode())
            text = chunk_obj['outputs'][0]['text']
        else:
            raise NotImplementedError('Unknown model provider.')

        printer(f'[DEBUG] {chunk_obj}', 'debug')
        return text

# 纯字符级的Bedrock_Stream流处理生成器
from typing import Generator
def StreamHandler(bedrock_stream) -> Generator[str, None, None]:
    """
    字符级流式处理器
    功能：
    - 直接逐字符返回原始流内容
    - 保持Markdown等结构化文本完整性
    - 最低延迟输出
    
    适用于：
    - 需要实时字符级渲染的场景
    - 保留Markdown/代码等格式的场景
    """
    try:
        for event in bedrock_stream:
            if not (chunk := BedrockModelsWrapper.get_stream_chunk(event)):
                continue
                
            text = BedrockModelsWrapper.get_stream_text(chunk)
            if not text:
                continue

            yield text # 直接返回，处理的步骤极少
                
    except Exception as e:
        print(f"\n[Stream Error] {str(e)}")
        raise

class BedrockWrapper:
    """
    Amazon Bedrock封装类
    功能描述：调用Bedrock模型并处理响应
    """

    def __init__(self):
        """
        初始化Amazon Bedrock封装类
        """
        self.speaking = False

    def is_speaking(self):
        """
        检查是否正在说话
        :return: 是否正在说话
        """
        return self.speaking

    def invoke_bedrock(self, text, dialogue_list = [], images = []):
        """
        调用Bedrock模型
        功能描述：调用Bedrock模型并处理响应
        :param text: 输入文本
        :param data: 数据列表
        :param history: 历史记录列表
        :return: 输出文本
        """
        printer('[DEBUG] Bedrock generation started', 'debug')
        self.speaking = True
        body = BedrockModelsWrapper.define_body(text, dialogue_list, images)
        printer(f"[DEBUG] Request body: {body}", 'debug')

        try:
            body_json = json.dumps(body)
            response = bedrock_runtime.invoke_model_with_response_stream( # 利用boto3定义响应对象
                body=body_json,
                modelId=config['bedrock']['api_request']['modelId'],
                accept=config['bedrock']['api_request']['accept'],
                contentType=config['bedrock']['api_request']['contentType']
            )

            printer('[DEBUG] Capturing Bedrocks response/bedrock_stream', 'debug')
            bedrock_stream = response.get('body') # 实际的调用就这一句，但是其实是同步阻塞式的
            printer(f"[DEBUG] Bedrock_stream: {bedrock_stream}", 'debug')

            audio_gen = StreamHandler(bedrock_stream) # generator
            printer('[DEBUG] Created bedrock stream to audio generator', 'debug')

            response_text = '' # 记录此次回复的全部文本
            for audio in audio_gen: # 得先从generator中才能获取文本
                print(audio,end="") # 调试用，不要随意换行
                response_text += audio
            print(history) # 这个得拿来看看

        except Exception as e:
            print(e)
            time.sleep(2)
            self.speaking = False

        time.sleep(1)
        self.speaking = False
        printer('\n[DEBUG] Bedrock generation completed', 'debug')
        return response_text
    
if __name__ == '__main__':
    history = [] # 存储对话历史的列表
    bedrock_wrapper = BedrockWrapper() 
    while True:
        if not bedrock_wrapper.is_speaking():
            input_text = input("[Please Input]：")
            if len(input_text) != 0:
                request_text = input_text # 这里处理模型的预设提示词？
                printer(f'\n[INFO] request_text: {request_text}', 'info')

                return_output = bedrock_wrapper.invoke_bedrock(request_text, dialogue_list=history, images=[]) # 为了不混乱对话历史的顺序，不能异步调用~

                history.append({"role":"user","content":[{ "type": "text","text": input_text}]}) # 向对话历史中，插入用户输入（取出了提示词）
                history.append({"role":"assistant","content":[{ "type": "text","text": return_output}]}) # 向对话历史中，插入模型回复
