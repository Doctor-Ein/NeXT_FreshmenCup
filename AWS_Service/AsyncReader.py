import threading
import queue
import time
import boto3
import pyaudio
from concurrent.futures import ThreadPoolExecutor
from AWS_Service.config import config

# 原始 Reader 类：用于播放一段音频
class Reader:
    def __init__(self):
        self.polly = boto3.client('polly', region_name=config['region'])
        self.audio = pyaudio.PyAudio().open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            output=True
        )
        self.chunk = 1024
        self.stop_flag = False

    def read(self, text):
        self.stop_flag = False
        response = self.polly.synthesize_speech(
            Text=text,
            Engine=config['polly']['Engine'],
            LanguageCode=config['polly']['LanguageCode'],
            VoiceId=config['polly']['VoiceId'],
            OutputFormat=config['polly']['OutputFormat'],
        )
        stream = response['AudioStream']

        while not self.stop_flag:
            data = stream.read(self.chunk)
            if not data:
                break
            self.audio.write(data)

    def stop(self):
        self.stop_flag = True

    def close(self):
        time.sleep(0.5)
        self.audio.stop_stream()
        self.audio.close()

# 异步朗读控制器类：接受句子队列，在后台线程朗读
class AsyncReader:
    def __init__(self, reader=None):
        self.reader = reader or Reader()
        self.queue = queue.Queue()
        self.running = True
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.current_task = None
        self.reading = False
        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            try:
                sentence = self.queue.get(timeout=1)
                if sentence is None:
                    break
                with self.lock:
                    self.reading = True
                    self.current_task = self.executor.submit(self.reader.read, sentence)
                    self.current_task.result()  # 阻塞直到朗读结束
                    self.reading = False
                self.queue.task_done()
            except queue.Empty:
                continue

    def submit(self, sentence):
        self.queue.put(sentence)

    def is_reading(self):
        with self.lock:
            return self.reading

    def interrupt(self):
        with self.lock:
            if self.reading:
                self.reader.stop()
                if self.current_task:
                    self.current_task.cancel()
                self.reading = False

    def stop(self):
        self.running = False
        self.queue.put(None)
        self.thread.join()
        self.executor.shutdown(wait=True)
        self.reader.close()

# 示例用法（请在主程序中使用）
if __name__ == "__main__":
    def mock_generator():
        yield "你好，世界。"
        time.sleep(0.5)
        yield "这是一个测试。"

    async_reader = AsyncReader()

    try:
        for sentence in mock_generator():
            print(sentence, end='', flush=True)
            async_reader.submit(sentence)
            time.sleep(0.1)
    finally:
        async_reader.queue.join()
        async_reader.stop()