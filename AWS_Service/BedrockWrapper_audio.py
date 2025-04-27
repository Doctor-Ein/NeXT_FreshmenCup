import asyncio
import json
import os
import time
import pyaudio
import sys
import boto3
import sounddevice

from concurrent.futures import ThreadPoolExecutor
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent, TranscriptResultStream
from BedrockWrapper.api_request_schema import api_request_list, get_model_ids
from TextInputApp import app

# 设置模型ID和AWS区域
model_id = os.getenv('MODEL_ID', 'meta.llama3-70b-instruct-v1')
aws_region = os.getenv('AWS_REGION', 'us-east-1')

# 验证模型ID是否有效
if model_id not in get_model_ids():
    print(f'Error: Models ID {model_id} in not a valid model ID. Set MODEL_ID env var to one of {get_model_ids()}.')
    sys.exit(0)

# 创建新的事件循环
loop = asyncio.new_event_loop()  # 这里希望用东西代替的啊

# 定义语音列表
voiceLanguageList = ['cmn-CN', 'en-US', 'ja-JP', 'ko-KR']
voiceNameList = ['Zhiyu', 'Ivy', 'Takumi', 'Seoyeon']
voicePromptList = ['Chinese', 'English', 'Japanese', 'Korean']

# 默认配置为中文
voiceIndex = 0

# API请求配置
api_request = api_request_list[model_id]
config = {
    'log_level': 'none',  # One of: info, debug, none
    #'last_speech': "If you have any other questions, please don't hesitate to ask. Have a great day!",
    'region': aws_region,
    'polly': {
        'Engine': 'neural',
        'LanguageCode': voiceLanguageList[voiceIndex],
        'VoiceId':voiceNameList[voiceIndex],
        'OutputFormat': 'pcm',
    },
    'bedrock': {
        'api_request': api_request
    }
}

# 更新配置函数
def update_config():
    config['polly']['LanguageCode'] = voiceLanguageList[voiceIndex]
    config['polly']['VoiceId'] = voiceNameList[voiceIndex]


# 初始化PyAudio和AWS服务客户端
p = pyaudio.PyAudio()
polly = boto3.client('polly', region_name=config['region'])
transcribe_streaming = TranscribeStreamingClient(region=config['region'])
bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name=config['region'])

# 打印函数
def printer(text, level):
    if config['log_level'] == 'info' and level == 'info':
        print(text)
    elif config['log_level'] == 'debug' and level in ['info', 'debug']:
        print(text)


# 用户输入管理类
class UserInputManager:
    shutdown_executor = False
    executor = None

    @staticmethod
    def set_executor(executor):
        UserInputManager.executor = executor

    @staticmethod
    def start_shutdown_executor():
        UserInputManager.shutdown_executor = False
        raise Exception()  # Workaround to shutdown exec, as executor.shutdown() doesn't work as expected.

    @staticmethod
    def start_user_input_loop():
        while True:
            sys.stdin.readline().strip()
            printer(f'[DEBUG] User input to shut down executor...', 'debug')
            UserInputManager.shutdown_executor = True

    @staticmethod
    def is_executor_set():
        return UserInputManager.executor is not None

    @staticmethod
    def is_shutdown_scheduled():
        return UserInputManager.shutdown_executor


# Bedrock模型包装类
class BedrockModelsWrapper:

    @staticmethod
    def define_body(text):
        model_id = config['bedrock']['api_request']['modelId']
        model_provider = model_id.split('.')[0]
        body = config['bedrock']['api_request']['body']

        output_language=voicePromptList[voiceIndex]

        if model_provider == 'amazon':
            body['inputText'] = text
        elif model_provider == 'meta':
            if 'llama3' in model_id:
                body['prompt'] = f"""
                    <|begin_of_text|>
                    <|start_header_id|>user<|end_header_id|>
                    {text}, please output in {output_language}.
                    <|eot_id|>
                    <|start_header_id|>assistant<|end_header_id|>
                    """
            else: 
                body['prompt'] = f"<s>[INST] {text}, please output in {output_language}. [/INST]"
        elif model_provider == 'anthropic':
            if "claude-3" in model_id:
                body['messages'] = [
                    {
                        "role": "user",
                        "content":  f"{text}, please respond in {output_language}."
                    }
                ]
            else:
                body['prompt'] =  f'\n\nHuman: {text}, please respond in {output_language}.\n\nAssistant:'
        elif model_provider == 'cohere':
            body['prompt'] = f"{text}, please respond in {output_language}."
        elif model_provider == 'mistral':
            body['prompt'] = f"<s>[INST] {text}, please output in {output_language}. [/INST]>"
        else:
            raise Exception('Unknown model provider.')

        return body

    @staticmethod
    def get_stream_chunk(event):
        return event.get('chunk')

    @staticmethod
    def get_stream_text(chunk):
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
            else:
                #Claude2.x
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


# 音频生成器函数
def to_audio_generator(bedrock_stream):
    prefix = ''

    if bedrock_stream:
        for event in bedrock_stream:
            chunk = BedrockModelsWrapper.get_stream_chunk(event)
            if chunk:
                text = BedrockModelsWrapper.get_stream_text(chunk)

                if '.' in text:
                    a = text.split('.')[:-1]
                    to_polly = ''.join([prefix, '.'.join(a), '. '])
                    prefix = text.split('.')[-1]
                    print(to_polly, flush=True, end='')
                    yield to_polly
                else:
                    prefix = ''.join([prefix, text])

        if prefix != '':
            print(prefix, flush=True, end='')
            yield f'{prefix}.'

        print('\n')


# Bedrock包装类
class BedrockWrapper:

    def __init__(self):
        self.speaking = False

    def is_speaking(self):
        return self.speaking

    def invoke_bedrock(self, text):
        printer('[DEBUG] Bedrock generation started', 'debug')
        self.speaking = True

        body = BedrockModelsWrapper.define_body(text)
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
            for audio in audio_gen:
                reader.read(audio)
                with open('Dialogue/audio_history','a',encoding='utf-8') as file:
                    print(audio, file = file)

            reader.close()

        except Exception as e:
            print(e)
            time.sleep(2)
            self.speaking = False
            
        time.sleep(1)
        self.speaking = False
        printer('\n[DEBUG] Bedrock generation completed', 'debug')


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
            # Check if user signaled to shutdown Bedrock speech
            # UserInputManager.start_shutdown_executor() will raise Exception. If not ideas but is functional.
            if UserInputManager.is_executor_set() and UserInputManager.is_shutdown_scheduled():
                UserInputManager.start_shutdown_executor()

            data = stream.read(self.chunk)
            self.audio.write(data)
            if not data:
                break

    def close(self):
        time.sleep(1)
        self.audio.stop_stream()
        self.audio.close()


# 事件处理器类
class EventHandler(TranscriptResultStreamHandler):
    text = []
    last_time = 0
    sample_count = 0
    max_sample_counter = 4

    def __init__(self, transcript_result_stream: TranscriptResultStream, bedrock_wrapper):
        super().__init__(transcript_result_stream)
        self.bedrock_wrapper = bedrock_wrapper

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results
        if not self.bedrock_wrapper.is_speaking():

            if results:
                for result in results:
                    EventHandler.sample_count = 0
                    if not result.is_partial:
                        for alt in result.alternatives:
                            print(alt.transcript, flush=True, end=' ')
                            EventHandler.text.append(alt.transcript)

            else:
                EventHandler.sample_count += 1
                if EventHandler.sample_count == EventHandler.max_sample_counter:

                    # if len(EventHandler.text) == 0:
                    #     last_speech = config['last_speech']
                    #     print(last_speech, flush=True)
                        #aws_polly_tts(last_speech)
                        #os._exit(0)  # exit from a child process
                    #else:
                    if len(EventHandler.text) != 0:    
                        input_text = ' '.join(EventHandler.text)
                        printer(f'\n[INFO] User input: {input_text}', 'info')

                        executor = ThreadPoolExecutor(max_workers=1)
                        # Add executor so Bedrock execution can be shut down, if user input signals so.
                        UserInputManager.set_executor(executor)
                        loop.run_in_executor(
                            executor,
                            self.bedrock_wrapper.invoke_bedrock,
                            input_text
                        )

                    EventHandler.text.clear()
                    EventHandler.sample_count = 0


# 麦克风流类
class MicStream:

    async def mic_stream(self):
        loop = asyncio.get_event_loop()
        input_queue = asyncio.Queue()

        def callback(indata, frame_count, time_info, status):
            loop.call_soon_threadsafe(input_queue.put_nowait, (bytes(indata), status))

        stream = sounddevice.RawInputStream(
            channels=1, samplerate=16000, callback=callback, blocksize=2048 * 2, dtype="int16")
        with stream:
            while True:
                indata, status = await input_queue.get()
                yield indata, status

    async def write_chunks(self, stream):
        async for chunk, status in self.mic_stream():
            await stream.input_stream.send_audio_event(audio_chunk=chunk)

        await stream.input_stream.end_stream()

    async def basic_transcribe(self):
        # loop = asyncio.get_event_loop()
        loop.run_in_executor(ThreadPoolExecutor(max_workers=1), UserInputManager.start_user_input_loop)

        lc=voiceLanguageList[voiceIndex]
        if(lc=='cmn-CN'): # 为中文特判
            lc='zh-CN'

        stream = await transcribe_streaming.start_stream_transcription(
            language_code=lc,
            media_sample_rate_hz=16000,
            media_encoding="pcm",
        )
            
        handler = EventHandler(stream.output_stream, BedrockWrapper())
        await asyncio.gather(self.write_chunks(stream), handler.handle_events())


# 信息文本
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