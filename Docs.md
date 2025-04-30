# NeXT 项目内部文档

## AWS_Service
此Package包含AWS为本此比赛支持的诸多服务，包括：*AWS Bedrock*、*Transcribe*、*Polly*，都提供了一键式的调用方法。
本模块由以下几个核心组件组成：
- `BedrockModelsWrapper`：封装了请求体的构建和流式处理逻辑。
- `BedrockWrapper`：封装了与 Bedrock 模型的交互逻辑。
- `StreamHandler`：处理流式响应，逐字符返回生成的内容。
- `printer`：日志打印工具，支持不同日志级别。

### BedrockWrapper
##### 启动方法
```python
while True:
    if not bedrock_wrapper.is_speaking():
        input_text = input("[Please Input]：")
        if len(input_text) != 0:
            request_text = input_text
            printer(f'\n[INFO] request_text: {request_text}', 'info')

            return_output = bedrock_wrapper.invoke_bedrock(request_text, dialogue_list=history, images=[])

            history.append({"role":"user","content":[{ "type": "text","text": input_text}]})
            history.append({"role":"assistant","content":[{ "type": "text","text": return_output}]})
```
##### 辅助组件
- `BedrockModelsWrapper` 是一个工具类，用于封装与 Bedrock 模型交互时的请求体构建和流式处理逻辑。
- `define_body`：
  - **功能**：根据不同的模型提供者构建请求体。
  - **参数**：
    - `text`：输入文本。
    - `dialogue_list`：对话历史列表，每个元素是一个字典，包含 `role` 和 `content`。
    - `images`：输入图片的列表，每个元素是一个字典，包含 `media_type` 和 `data`。
  - **返回值**：构建好的请求体。
  **示例**：
  ```python
  body = BedrockModelsWrapper.define_body(
      text="Hello, how are you?",
      dialogue_list=[
          {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
          {"role": "assistant", "content": [{"type": "text", "text": "I'm good, thanks!"}]}
      ],
      images=[
          {"media_type": "image/jpeg", "data": "base64_encoded_image_data"}
      ]
  )
  ```
- **`get_stream_chunk`**：
  - **功能**：从返回的响应事件中提取流块（流式）
  - **参数**：
    - `event`：事件对象。
  - **返回值**：流块。
- **`get_stream_text`**：
  - **功能**：从流块中提取文本内容。
  - **参数**：
    - `chunk`：流块。
  - **返回值**：提取的文本内容。

##### 3.2 `BedrockWrapper`
`BedrockWrapper` 是一个封装类，用于调用 Bedrock 模型并处理响应。

- **`__init__`**：
- **`is_speaking`**：
  - **功能**：检查是否正在调用模型。
  - **返回值**：布尔值，表示是否正在调用模型。
- **`invoke_bedrock`**：
  - **功能**：调用 Bedrock 模型并处理响应。
  - **参数**：
    - `text`：输入文本。
    - `dialogue_list`：对话历史列表。
    - `images`：输入图片的列表。
  - **返回值**：模型生成的文本。

##### 3.3 `StreamHandler`
`StreamHandler` 是一个生成器函数，用于处理流式响应。
- **功能**：逐字符处理流式响应，返回生成的文本。
- **参数**：
  - `bedrock_stream`：流式响应对象（也就是响应的事件）
- **返回值**：生成的文本。

##### 3.4 `printer`
`printer` 是一个日志打印工具，用于记录调试信息和错误信息。
- **功能**：根据日志级别打印信息。
- **参数**：
  - `text`：要打印的文本。
  - `level`：日志级别，可以是 `info` 或 `debug`。
- **返回值**：无。

##### 4.1 环境变量
以下环境变量需要在运行代码前设置：
- **`MODEL_ID`**：模型 ID，默认值为 `anthropic.claude-3-sonnet-20240229-v1:0`。
- **`AWS_REGION`**：AWS 区域，默认值为 `us-east-1`。

##### 4.2 配置文件
模块使用一个配置字典 `config`，包含以下内容：
- **`log_level`**：日志级别，可以是 `info`、`debug` 或 `none`。
- **`region`**：AWS 区域。
- **`bedrock`**：Bedrock 模型的配置，包括 `api_request`。
- **`network`**：网络配置，包括连接超时时间、读取超时时间、最大重试次数和重试延迟时间。

**示例**：
```python
config = {
    'log_level': 'info',
    'region': 'us-east-1',
    'bedrock': {
        'api_request': {
            'modelId': 'anthropic.claude-3-sonnet-20240229-v1:0',
            'contentType': 'application/json',
            'accept': 'application/json'
        }
    },
    'network': {
        'connect_timeout': 5,
        'read_timeout': 10,
        'max_retries': 3,
        'retry_delay': 2
    }
}
```

### Transcribe
##### 启动方法：
```python
asyncio.run(main())
```
##### 

以下是一份关于该语音转录模块的详细文档，包括模块的功能、类和方法的说明，以及如何使用该模块进行语音转录。

---

# AWS Transcribe 实时语音转录模块文档

## 模块概述
本模块实现了一个基于 AWS Transcribe 的实时语音转录系统，支持多种语言的语音输入，并能够将语音实时转换为文本。模块设计为异步运行，支持连续语音输入和转录结果的实时获取。

## 功能特点
- **实时语音转录**：支持从麦克风实时获取音频流并转录为文本。
- **多语言支持**：支持中文、英语、日语和韩语。
- **异步处理**：基于 `asyncio` 实现，支持非阻塞的音频流处理和转录结果获取。
- **灵活的语言切换**：可以在运行时动态切换转录语言。
- **连续转录模式**：支持长时间连续语音输入和转录。

## 核心组件

### 1. `TranscribeHandler`
`TranscribeHandler` 是一个继承自 `TranscriptResultStreamHandler` 的类，用于处理从 AWS Transcribe 返回的转录结果。

#### 方法
- **`handle_transcript_event`**：
  - **功能**：处理转录事件，提取完整的转录文本并放入队列。
  - **参数**：
    - `transcript_event`：AWS Transcribe 返回的转录事件。
  - **返回值**：无。

### 2. `MicStream`
`MicStream` 是一个类，用于处理麦克风输入流。

#### 方法
- **`mic_stream`**：
  - **功能**：创建麦克风输入流，将音频数据分块发送。
  - **返回值**：生成器，返回音频数据块和状态。
- **`write_chunks`**：
  - **功能**：将音频数据写入 AWS Transcribe 流。
  - **参数**：
    - `stream`：AWS Transcribe 的音频流对象。
- **`stop`**：
  - **功能**：停止麦克风输入流。

### 3. `TranscribeService`
`TranscribeService` 是一个类，封装了整个转录服务的启动、停止和转录结果的获取。

#### 方法
- **`start_transcribe`**：
  - **功能**：启动普通模式的转录服务。
- **`start_continuous_transcribe`**：
  - **功能**：启动连续模式的转录服务。
- **`stop_continuous_transcribe`**：
  - **功能**：停止连续模式的转录服务。
- **`get_transcript`**：
  - **功能**：从队列中获取转录结果。
  - **返回值**：转录的文本字符串。
- **`change_language`**：
  - **功能**：动态切换转录语言。
  - **参数**：
    - `language`：语言代码（`zh`、`en`、`ja`、`ko`）。
- **`stop`**：
  - **功能**：停止转录服务。

## 使用示例

### 1. 初始化转录服务
```python
service = TranscribeService(language_index=0)  # 默认语言为中文
```

### 2. 启动连续转录
```python
await service.start_continuous_transcribe()
```

### 3. 实时获取转录结果
```python
try:
    while True:
        transcript = await service.get_transcript()
        print(transcript, end="")
except KeyboardInterrupt:
    pass
```

### 4. 停止转录服务
```python
await service.stop_continuous_transcribe()
```

### 5. 切换语言
```python
await service.change_language(language='en')  # 切换到英语
```

## 语言支持
支持的语言列表如下：
- 中文：`cmn-CN`
- 英语：`en-US`
- 日语：`ja-JP`
- 韩语：`ko-KR`

语言切换时，使用 `change_language` 方法指定语言代码：
- `zh`：中文
- `en`：英语
- `ja`：日语
- `ko`：韩语

## 注意事项
1. **AWS 配置**：确保已配置 AWS 账户，并在环境中设置 `AWS_REGION`。
2. **麦克风权限**：确保程序有权限访问麦克风。
3. **网络延迟**：网络延迟可能影响转录的实时性，建议在低延迟网络环境下使用。
4. **语言代码**：AWS Transcribe 和 Polly 的语言代码可能不同，模块中已进行适配。

## 示例代码
以下是一个完整的示例代码，展示如何使用该模块进行实时语音转录：

```python
import asyncio

async def main():
    # 初始化服务（默认中文）
    service = TranscribeService()
    
    # 启动连续转录
    print("开始语音转录（按Ctrl+C停止）...")
    await service.start_continuous_transcribe()
    
    # 实时打印结果
    try:
        while True:
            transcript = await service.get_transcript()
            print(transcript, end="")
    except KeyboardInterrupt:
        pass
    finally:
        # 停止服务
        await service.stop_continuous_transcribe()
        print("转录已停止")

if __name__ == "__main__":
    asyncio.run(main())
```

---

希望这份文档能够帮助你更好地理解和使用这个模块！如果有任何问题或需要进一步的说明，请随时联系。