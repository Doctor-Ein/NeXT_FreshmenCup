import os
import base64
import jsonpath_ng
from pathlib import Path
from AWS_Service.BedrockWrapper import BedrockWrapper

bedrock = BedrockWrapper()

OUTPUTDIR = './JsonDataBase/image_summary.json'
DATAPATH = './Data/MinerU_Res/AlexNet/images'

# 首先定位图片？还是从raw_data中定位吧😂
# 一定需要补充上下文信息的感觉🤔算了图片还是靠着上下文抓取的规则吧😂
# lz先不干了，专心focus网页的构建和联系算噜～

import base64

def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def invoke_with_image(image_data):
    images = [{'media_type':'image/jpg','data':image_data}]
    prompt='please describe the image in detail, to make a summary/abstract.'
    return bedrock.invoke_model(prompt,images=images)

data_path = Path(DATAPATH)

for entry in data_path.iterdir():
    image_data = encode_image_to_base64(entry)
    print(invoke_with_image(image_data))
    break

