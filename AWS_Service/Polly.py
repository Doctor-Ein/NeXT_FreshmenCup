import threading
import boto3
import pyaudio
from AWS_Service.config import config

class Reader(threading.Thread):
    def __init__(self, text):
        super().__init__()
        self.text = text
        self.polly = boto3.client('polly', region_name=config['region'])
        self.p = pyaudio.PyAudio()
        self.audio_stream = None  # AWS Polly返回的音频流
        self.output_stream = None  # PyAudio输出流
        self.chunk = 1024
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._is_playing = False

    def is_playing(self):
        """检查是否正在播放"""
        with self._lock:
            return self._is_playing

    def run(self):
        """在后台线程中执行语音合成和播放"""
        try:
            # 获取语音合成响应
            response = self.polly.synthesize_speech(
                Text=self.text,
                Engine=config['polly']['Engine'],
                LanguageCode=config['polly']['LanguageCode'],
                VoiceId=config['polly']['VoiceId'],
                OutputFormat=config['polly']['OutputFormat'],
            )
            self.audio_stream = response['AudioStream']
            
            # 初始化输出流
            with self._lock:
                self.output_stream = self.p.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    output=True,
                    start=False
                )
                self._is_playing = True

            # 流式播放音频
            while not self._stop_event.is_set():
                data = self.audio_stream.read(self.chunk)
                if not data:
                    break
                
                with self._lock:
                    if self.output_stream.is_stopped():
                        self.output_stream.start_stream()
                    self.output_stream.write(data)

        except Exception as e:
            print(f"播放出错: {str(e)}")
        finally:
            self._cleanup()

    def stop(self):
        """安全停止播放"""
        self._stop_event.set()
        with self._lock:
            if self.output_stream:
                try:
                    # 先尝试正常停止
                    if not self.output_stream.is_stopped():
                        self.output_stream.stop_stream()
                except OSError as e:
                    print(f"停止流时忽略错误: {str(e)}")
                finally:
                    self._is_playing = False

    def _cleanup(self):
        """安全清理资源"""
        with self._lock:
            try:
                # 关闭AWS音频流
                if self.audio_stream:
                    self.audio_stream.close()
                
                # 关闭PyAudio输出流
                if self.output_stream:
                    try:
                        if not self.output_stream.is_stopped():
                            self.output_stream.stop_stream()
                    except OSError:
                        pass
                    finally:
                        self.output_stream.close()
                        self.output_stream = None
                
                # 终止PyAudio实例
                if hasattr(self, 'p'):
                    try:
                        self.p.terminate()
                    except Exception as e:
                        print(f"终止PyAudio时忽略错误: {str(e)}")
                
                self._is_playing = False
            except Exception as e:
                print(f"清理过程中发生错误: {str(e)}")

    def __del__(self):
        """析构函数确保资源释放"""
        self._cleanup()


if __name__ == "__main__":
    reader = Reader("改进后的稳定版本语音")
    reader.start()
    
    try:
        input('输入回车打断:')
        reader.stop()
    finally:
        reader.join()
        del reader