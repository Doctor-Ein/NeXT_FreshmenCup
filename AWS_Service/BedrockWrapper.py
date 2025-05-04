import json
import time
import sys
import boto3
import re

from botocore.config import Config
from .Polly import Reader
from .config import config

# 初始化AWS服务客户端
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name=config['region'],
    config=Config(
        connect_timeout=config['network']['connect_timeout'],
        read_timeout=config['network']['read_timeout'],
        retries={'max_attempts': config['network']['max_retries']}
    )
)

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
        :param dialogue_list: 从历史会话管理库中获取的列表，元素为会话块字典{role,content}
        :param images: 输入图片的列表，元素为字典{media_type, data}
        :return: 请求体
        """
        model_id = config['bedrock']['api_request']['modelId']
        model_provider = model_id.split('.')[0]
        body = config['bedrock']['api_request']['body']

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
                content = [
                    {
                        "type": "text",
                        "text": text
                    }
                ]
                for image in images:
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image["media_type"],
                            "data": image["data"]
                        },
                    })
                
                body['messages'] = list(dialogue_list)
                body['messages'].append({"role": "user", "content": content})
            else:
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
                        text = chunk_obj['delta']['text']
            else:
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

# 音频生成器函数（支持中英文断句）-> 保留这个名字属实是有点传承的意味了（笑）
def to_audio_generator(bedrock_stream):
    prefix = ''
    sentence_end_pattern = re.compile(r'([^。！？!?\.]+[。！？!?\.])')  # 捕获完整句子

    if bedrock_stream:
        for event in bedrock_stream:
            chunk = BedrockModelsWrapper.get_stream_chunk(event)
            if chunk:
                text = BedrockModelsWrapper.get_stream_text(chunk)
                full_text = prefix + text
                sentences = sentence_end_pattern.findall(full_text)
                if sentences:
                    # 找到完整句子后，逐句生成
                    for sent in sentences:
                        yield sent
                    # 将未结束的部分存入 prefix
                    prefix = sentence_end_pattern.sub('', full_text)
                else:
                    prefix = full_text  # 没有匹配到句子，继续累积
        
        # 流结束后，处理剩余内容
        if prefix:
            print(prefix, flush=True, end='')
            yield prefix

        print('\n')

class BedrockWrapper:
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

    def invoke_bedrock(self, text, dialogue_list=[], images=[]):
        """
        流式调用Bedrock模型，边生成边yield文本块
        """
        printer('[DEBUG] Bedrock generation started', 'debug')
        self.speaking = True

        body = BedrockModelsWrapper.define_body(text, dialogue_list, images)
        printer(f"[DEBUG] Request body: {body}", 'debug')

        try:
            body_json = json.dumps(body)
            response = bedrock_runtime.invoke_model_with_response_stream(
                body=body_json,
                modelId=config['bedrock']['api_request']['modelId'],
                accept=config['bedrock']['api_request']['accept'],
                contentType=config['bedrock']['api_request']['contentType']
            )

            bedrock_stream = response.get('body')
            printer(f"[DEBUG] Bedrock_stream: {bedrock_stream}", 'debug')

            audio_gen = to_audio_generator(bedrock_stream)
            printer('[DEBUG] Created bedrock stream to audio generator', 'debug')

            for audio in audio_gen:
                printer(f'[DEBUG] audio: {audio}','debug')
                yield audio  # ⭐ 每段话都 yield 出去，调用方可以逐段接收

        except Exception as e:
            printer(f'[ERROR] {str(e)}', 'info')
            time.sleep(config['network']['retry_delay'])
            self.speaking = False

            if "timeout" in str(e).lower():
                printer('[INFO] Timeout detected, attempting retry...', 'info')
                # ⚠️ 注意：递归生成器必须用 yield from
                yield from self.invoke_bedrock(text, dialogue_list, images)

        time.sleep(1)
        self.speaking = False
        printer('\n[DEBUG] Bedrock generation completed', 'debug')

    def invoke_voice(self, text, dialogue_list = [], images = []):
        """
        调用Bedrock模型
        功能描述：调用Bedrock模型并处理响应
        :param text: 输入文本
        :return:
        """
        printer('[DEBUG] Bedrock generation started', 'debug')
        self.speaking = True
        
        body = BedrockModelsWrapper.define_body(text, dialogue_list, images)
        printer(f"[DEBUG] Request body: {body}", 'debug')

        try:
            body_json = json.dumps(body)
            response = bedrock_runtime.invoke_model_with_response_stream(
                body=body_json,
                modelId=config['bedrock']['api_request']['modelId'],
                accept=config['bedrock']['api_request']['accept'],
                contentType=config['bedrock']['api_request']['contentType']
            )

            printer('[DEBUG] Capturing Bedrocks response/bedrock_stream', 'debug')
            bedrock_stream = response.get('body')
            printer(f"[DEBUG] Bedrock_stream: {bedrock_stream}", 'debug')

            audio_gen = to_audio_generator(bedrock_stream)
            printer('[DEBUG] Created bedrock stream to audio generator', 'debug')

            reader = Reader()
            response_text = ''
            print("[Assistant]:",end="")
            for audio in audio_gen:
                print(audio,end='',flush=False)
                reader.read(audio) # 没有读出来是为何🤔
                response_text += audio

            reader.close()

        except Exception as e:
            printer(f'[ERROR] {str(e)}', 'info')
            time.sleep(config['network']['retry_delay'])
            self.speaking = False
            # 发生异常时尝试重试
            if "timeout" in str(e).lower():
                printer('[INFO] Timeout detected, attempting retry...', 'info')
                return self.invoke_bedrock(text, dialogue_list, images)

        time.sleep(1)
        self.speaking = False
        printer('\n[DEBUG] Bedrock generation completed', 'debug')
        return response_text

def printer(text: str, level: str) -> None:
    """
    打印日志信息（要打印到日志系统啊😂
    功能描述：根据日志级别打印信息，错误信息重定向到 stderr
    :param text: 要打印的文本
    :param level: 日志级别（info或debug）
    """
    if level == 'error':
        print(text, file=sys.stderr)
    elif config['log_level'] == 'info' and level == 'info':
        print(text)
    elif config['log_level'] == 'debug' and level in ['info', 'debug']:
        print(text)