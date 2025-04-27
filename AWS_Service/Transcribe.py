import os
import asyncio
import sounddevice

from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent, TranscriptResultStream

# 支持的语言列表：中文、英语、日语、韩语
# 注意：这里使用的是Transcribe的语言代码，与Polly的语言代码可能不同
voiceLanguageList = ['cmn-CN', 'en-US', 'ja-JP', 'ko-KR']

# 默认配置为中文（voiceLanguageList[0]）
voiceIndex = 0

# 初始化AWS Transcribe服务客户端
# 从环境变量获取AWS区域，默认为us-east-1
aws_region = os.getenv('AWS_REGION', 'us-east-1')
transcribe_streaming = TranscribeStreamingClient(region=aws_region)

class TranscribeHandler(TranscriptResultStreamHandler):
    """处理AWS Transcribe服务返回的转录结果
    
    继承自TranscriptResultStreamHandler，用于处理实时语音转录流
    使用asyncio.Queue在异步环境中传递转录结果
    """
    def __init__(self, transcript_result_stream: TranscriptResultStream):
        super().__init__(transcript_result_stream)
        # 创建异步队列，用于存储转录文本
        self.transcript_queue = asyncio.Queue()
        
    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        """处理转录事件的回调方法

        Args:
            transcript_event: AWS Transcribe服务返回的转录事件
        处理流程：
            1. 获取转录结果
            2. 过滤出非部分结果（完整的语音片段）
            3. 将转录文本放入异步队列
        """
        results = transcript_event.transcript.results
        if results:
            for result in results:
                if not result.is_partial:  # 只处理完整的转录结果，忽略中间状态
                    for alt in result.alternatives:
                        await self.transcript_queue.put(alt.transcript)

class MicStream:
    """处理麦克风输入流

    负责：
        1. 创建和管理麦克风输入流
        2. 将音频数据传输到AWS Transcribe服务
        3. 提供停止机制
    """
    def __init__(self):
        self.is_running = True

    async def mic_stream(self):
        """创建并管理麦克风输入流

        使用sounddevice创建音频输入流，通过异步队列传递音频数据
        返回值:
            异步生成器，产生(音频数据, 状态)元组
        """
        loop = asyncio.get_event_loop()
        input_queue = asyncio.Queue()

        def callback(indata, frame_count, time_info, status): # 回调函数：向输入队列中插入数据和状态的元组
            loop.call_soon_threadsafe(input_queue.put_nowait, (bytes(indata), status))

        # 配置音频输入流
        # channels=1: 单声道
        # samplerate=16000: 采样率16kHz（AWS Transcribe要求）
        # blocksize=2048 * 2: 缓冲区大小
        stream = sounddevice.RawInputStream(
            channels=1,
            samplerate=16000,
            callback=callback,
            blocksize=2048 * 2,
            dtype="int16"
        )
        
        with stream:
            while self.is_running:
                indata, status = await input_queue.get()
                yield indata, status

    async def write_chunks(self, stream):
        """将音频数据写入AWS Transcribe流

        Args:
            stream: AWS Transcribe服务的音频流对象
        流程：
            1. 从麦克风流中获取音频数据
            2. 发送到AWS Transcribe服务
            3. 当停止时关闭流
        """
        async for chunk, status in self.mic_stream():
            if self.is_running:
                await stream.input_stream.send_audio_event(audio_chunk=chunk)
        await stream.input_stream.end_stream()

    def stop(self):
        """停止麦克风输入流"""
        self.is_running = False

class TranscribeService:
    """AWS Transcribe服务的主要接口类

    提供：
        1. 语音转录服务的启动和停止
        2. 转录文本的获取
        3. 语言切换功能
        4. 连续音频流的控制
    """
    def __init__(self, language_index=0):
        """初始化转录服务

        Args:
            language_index: 语言索引，默认为0（中文）
        """
        global voiceIndex
        voiceIndex = language_index
        self.mic_stream = MicStream()
        self.handler = None
        self.stream = None
        self.continuous_task = None  # 用于存储连续转录的任务
        self.is_continuous = False   # 连续模式标志

    async def start_transcribe(self):
        """启动转录服务（单次模式）

        流程：
            1. 配置语言代码
            2. 创建AWS Transcribe流
            3. 启动音频处理和转录处理任务
        """
        lc = voiceLanguageList[voiceIndex]
        if lc == 'cmn-CN':  # 为中文特判，因为Transcribe和Polly的配置代码不一样😭
            lc = 'zh-CN'

        # 启动Transcribe的流
        self.stream = await transcribe_streaming.start_stream_transcription(
            language_code=lc,
            media_sample_rate_hz=16000,
            media_encoding="pcm",
        )
        
        # 创建处理器并启动异步任务
        self.handler = TranscribeHandler(self.stream.output_stream)

        # 创建两个异步任务：1. 处理音频输入; 2. 处理转录结果
        asyncio.create_task(self.write_chunks_task())
        asyncio.create_task(self.handler.handle_events()) # AWS自带的一个函数

    async def start_continuous_transcribe(self):
        """启动连续转录服务
        
        开启一个持续运行的转录服务，直到调用stop_continuous_transcribe为止
        音频流会持续读取，不会中断
        """
        if self.is_continuous:
            print("连续转录服务已经在运行中")
            return

        self.is_continuous = True
        lc = voiceLanguageList[voiceIndex]
        if lc == 'cmn-CN':  # 为中文特判，因为Transcribe和Polly的配置代码不一样😭
            lc = 'zh-CN'

        # 启动Transcribe的流
        self.stream = await transcribe_streaming.start_stream_transcription(
            language_code=lc,
            media_sample_rate_hz=16000,
            media_encoding="pcm",
        )
        
        # 创建处理器并启动异步任务
        self.handler = TranscribeHandler(self.stream.output_stream)
        
        # 创建连续处理的任务
        self.continuous_task = asyncio.gather(
            self.write_chunks_task(),
            self.handler.handle_events()  # AWS自带的一个函数
        )

    async def stop_continuous_transcribe(self):
        """停止连续转录服务
        
        优雅地关闭连续转录服务，确保资源正确释放
        """
        if not self.is_continuous:
            print("连续转录服务未在运行")
            return

        self.is_continuous = False
        self.mic_stream.stop()
        
        if self.continuous_task:
            try:
                await self.continuous_task
            except Exception as e:
                print(f"停止连续转录时发生错误: {e}")
            finally:
                self.continuous_task = None
                self.stream = None
                self.handler = None
                # 重新初始化MicStream以备下次使用
                self.mic_stream = MicStream()

    async def write_chunks_task(self):
        """音频块处理任务  

        Args:
            self: 当前实例
        将麦克风输入的音频数据写入AWS Transcribe流
        """
        await self.mic_stream.write_chunks(self.stream)

    async def get_transcript(self):
        """获取转录文本

        Returns:
            str: 转录的文本字符串
            None: 如果处理器未初始化
        Note:
            这是一种异步方法，将等待到新的转录文本到达
        """
        if self.handler:
            return await self.handler.transcript_queue.get()
        return None

    def stop(self):
        """停止转录服务：停止麦克风输入流，从而触发服务的正常关闭"""
        if self.mic_stream:
            self.mic_stream.stop()

# 使用示例：
async def main():
    # 创建转录服务实例（默认中文）
    service = TranscribeService()
    
    # 启动连续转录服务
    await service.start_continuous_transcribe()
    
    try:
        print("开始连续转录（按Ctrl+C停止）...")
        while True:
            # 获取转录文本
            transcript = await service.get_transcript()
            print(f"转录文本: {transcript}")
            
    except KeyboardInterrupt:
        print("\n正在停止连续转录服务...")
        # 停止连续转录服务
        await service.stop_continuous_transcribe()
        print("服务已停止")

if __name__ == "__main__":
    asyncio.run(main())

