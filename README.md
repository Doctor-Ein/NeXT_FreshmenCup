# NeXT（暂定代号～）
# 🌍 大模型虚拟教师应用


## 🛠 **安装与使用**
### 环境依赖
- *python 3.10* 版本，内置库和第三方库（参见`requirements.txt`），稳定的、可使用Claude的IP地域和网络环境
- 可选的嵌入模型和重排模型，可在[huggingface.co](https://huggingface.co) 下载
- *Docker* 本地部署或其他方式的 *Milvus* 向量数据库

### 使用方法
- 使用python的venv虚拟环境即可，`python -m venv myvenv`
- 打开终端（命令提示符）`myvenv\Scripts\activate`，随后终端前会有一个小括号:(myvenv)表示已经激活了虚拟环境
- 然后在主目录下`pip install -r requirements.txt`，即可自动安装相关第三方库
设置AWS的访问账户为环境变量，便于程序读取进行许可验证
- `AWS_ACCESS_KEY_ID`，`AWS_SECRET_ACCESS_KEY`
<!-- 安装模型到与项目主文件夹下（与`main.py`同级）在 `models/` 中下载对应模型
- `bge-large-zh-v1.5`
- `bge-reranker-large` -->
<!-- 在每次使用时
- 在终端切换到程序根目录，执行`conda activate LLM-Teacher`启用环境
- 通过 *Docker* 启动本地 *Milvus* 服务（详见[Milvus](https://github.com/milvus-io/milvus)，代码支持2.5.x 版本）
- 如果需要替换`data.txt`，替换后请执行`python3 Dataset/Embedding.py`完成数据嵌入
- 执行`python3 main.py`，启动我们的图形化输入输出，体验能力增强后的 LLM-Teacher 🎉🎉🎉 -->

---

## 🚀 **加入我们**
📢 **关注最新动态，欢迎支持！**

## 🌟 *致谢*
感谢以下个人和团队对项目的贡献：
- [Meteor728](https://github.com/Meteor728)：感谢你为项目开发了*Multilingual*以及RAEDME的编写，代码很棒，文档也超用心，还有认真地管理了比赛的事项以及火锅非常好吃和开心😋~（这里夸夸是我写的😃 Orz
- [StarDust](https://github.com/Rewind2Nowhere)：感谢你为项目开发*MultiModal*，并且在项目部分思路构建和Debug方面做出了卓越贡献（不是指1.5h修复4行代码🤣 Orz
- [bleem](https://github.com/bleem？)：感谢你为项目进行了全面的测试，为每个模块调整适配有效的提示词，确保了项目的稳定性！伟大的提示词工程师！所有思想和努力都看到啦☺️，Orz
- [Doctor-Ein](https://github.com/Doctor-Ein)：感谢自己没有哭哦~（来自队友的碎碎念：其实xlx commit了整个项目超过60%的代码，完成了从Milvus数据库构建到异步执行处理再到项目重构的大量工作，做出了巨大的贡献、巨大的牺牲和巨大的carry）