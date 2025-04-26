import json
import os
from pymilvus import MilvusClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from RAG_Package.Reranker import MilvusReranker

os.environ["TOKENIZERS_PARALLELISM"] = "false"

collection_name = "Dream_of_the_Red_Chamber"

# 基于脚本目录构建模型路径
model_path = "./models/bge-large-zh-v1.5"

# 使用 llama_index 加载本地模型
embedding = HuggingFaceEmbedding(model_name = model_path)

# 连接到 Milvus 服务
client = MilvusClient(uri="http://0.0.0.0:19530")

import re

def get_first_sentence(query: str) -> str:
    # 按中文句号或问号分割，保留分隔符（可选）
    sentences = re.split(r'([。？])', query)
    # 合并分隔符和句子（例如 ["你好", "。", "今天", "？"] -> ["你好。", "今天？"]）
    sentences = [sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '') for i in range(0, len(sentences)-1, 2)]
    if len(sentences) == 0:
        return query  # 如果没有分割符，返回原句
    return sentences[0]  # 返回第一句话

def queryContext(QueryContext:str):
    print("[QueryContext]:" + QueryContext)
    query = get_first_sentence(QueryContext)
    print("[query]:" + query)

    # 将查询上下文转化为向量
    QueryVector = embedding.get_text_embedding(query)

    # 检索 Milvus 中的向量
    search_params = {"metric_type": "L2", "params": {}}
    result = client.search(
        collection_name = collection_name,
        data = [QueryVector],  # 查询向量
        anns_field = "vector",  # 向量字段名
        search_params = search_params,  # 检索参数
        limit= 20 ,  # 返回的最相似向量数量
        output_fields = ["text", "partition"]  # 输出字段
    )

    for hits in result: # hits 是初步搜索结果的列表
        retrieved_documents = [{"text": hit['entity'].get('text','N/A'),"id": hit['id'],"partition":hit['entity'].get('partition',0)} for hit in hits]
    
    # context1 = []
    # for hits in result:
    #     for hit in hits:
    #         context1.append(f"{titles[str(hit['entity'].get('partition','0'))]}\n{hit['entity'].get('text','N/A')}\n")

    ## 重排模型
    reranker = MilvusReranker() 
    reranked_context = reranker.rerank_documents(query = query, retrieved_documents = retrieved_documents, top_k = 5)
    reranked_reranked_context = sorted(reranked_context,key = lambda x:x['metadata']['id'])

    ## 组合上下文内容
    context = ["<context>\n"]
    for doc in reranked_reranked_context:
        context.append(f"{titles[str(doc['metadata']['partition'])]}\n{doc['text']}\n")
    context.append("</context>")

    # json_obj = [context1,context]
    # with open('debug.json','w',encoding='utf-8') as file:
    #     json.dump(json_obj,file,ensure_ascii = False,indent = 4)

    return context


    # 原始版本直接返回初步搜索的结果
    # context = ["<context>\n"]
    # for hits in result:
    #     for hit in hits:
    #         context.append(f"{titles[str(hit['entity'].get('partition','0'))]}\n{hit['entity'].get('text','N/A')}\n")
    # context.append("\n</context>")
    # return context

titles = {
    "0": "不确定的章回",
    "1": "第一回　甄士隐梦幻识通灵贾雨村风尘怀闺秀",
    "2": "第二回　贾夫人仙逝扬州城冷子兴演说荣国府",
    "3": "第三回　贾雨村夤缘复旧职林黛玉抛父进京都",
    "4": "第四回　薄命女偏逢薄命郎葫芦僧乱判葫芦案",
    "5": "第五回　游幻境指迷十二钗饮仙醪曲演红楼梦",
    "6": "第六回　贾宝玉初试云雨情刘姥姥一进荣国府",
    "7": "第七回　送宫花贾琏戏熙凤宴宁府宝玉会秦钟",
    "8": "第八回　比通灵金莺微露意探宝钗黛玉半含酸",
    "9": "第九回　恋风流情友入家塾起嫌疑顽童闹学堂",
    "10": "第十回　金寡妇贪利权受辱张太医论病细穷源",
    "11": "十一回　庆寿辰宁府排家宴见熙凤贾瑞起淫心",
    "12": "十二回　王熙凤毒设相思局贾天祥正照风月鉴",
    "13": "十三回　回秦可卿死封龙禁尉王熙凤协理宁国府",
    "14": "十四回　林如海捐馆扬州城贾宝玉路谒北静王",
    "15": "十五回　王凤姐弄权铁槛寺秦鲸卿得趣馒头庵",
    "16": "十六回　贾元春才选凤藻宫秦鲸卿夭逝黄泉路",
    "17": "十七回　大观园试才题对额荣国府归省庆元宵",
    "18": "十八回　隔珠帘父女勉忠勤搦湘管姊弟裁题咏",
    "19": "十九回　情切切良宵花解语意绵绵静日玉生香",
    "20": "二十回　王熙凤正言弹妒意林黛玉俏语谑娇音",
    "21": "二十一回　贤袭人娇嗔箴宝玉俏平儿软语救贾琏",
    "22": "二十二回　听曲文宝玉悟禅机制灯迷贾政悲谶语",
    "23": "二十三回　西厢记妙词通戏语牡丹亭艳曲警芳心",
    "24": "二十四回　醉金刚轻财尚义侠痴女儿遗帕惹相思",
    "25": "二十五回　魇魔法姊弟逢五鬼红楼梦通灵遇双真",
    "26": "二十六回　蜂腰桥设言传心事潇湘馆春困发幽情",
    "27": "二十七回　滴翠亭杨妃戏彩蝶埋香冢飞燕泣残红",
    "28": "二十八回　蒋玉菡情赠茜香罗薛宝钗羞笼红麝串",
    "29": "二十九回　享福人福深还祷福痴情女情重愈斟情",
    "30": "三十回　宝钗借扇机带双敲龄官划蔷痴及局外",
    "31": "三十一回　撕扇子作千金一笑因麒麟伏白首双星",
    "32": "三十二回　诉肺腑心迷活宝玉含耻辱情烈死金钏",
    "33": "三十三回　手足耽耽小动唇舌不肖种种大承笞挞",
    "34": "三十四回　情中情因情感妹妹错里错以错劝哥哥",
    "35": "三五回　白玉钏亲尝莲叶羹黄金莺巧结梅花络",
    "36": "三十六回　绣鸳鸯梦兆绛芸轩识分定情悟梨香院",
    "37": "三十七回　秋爽斋偶结海棠社蘅芜苑夜拟菊花题",
    "38": "三十八回　林潇湘魁夺菊花诗薛蘅芜讽和螃蟹咏",
    "39": "三十九回　村姥姥是信口开合情哥哥偏寻根究底",
    "40": "四十回　史太君两宴大观园金鸳鸯三宣牙牌令",
    "41": "四十一回　栊翠庵茶品梅花雪怡红院劫遇母蝗虫",
    "42": "四十二回　蘅芜君兰言解疑癖潇湘子雅谑补余香",
    "43": "四十三回　闲取乐偶攒金庆寿不了情暂撮土为香",
    "44": "四十四回　变生不测凤姐泼醋喜出望外平儿理妆",
    "45": "四十五回　金兰契互剖金兰语风雨夕闷制风雨词",
    "46": "四十六回　尴尬人难免尴尬事鸳鸯女誓绝鸳鸯偶",
    "47": "四十七回　呆霸王调情遭苦打冷郎君惧祸走他乡",
    "48": "四十八回　滥情人情误思游艺慕雅女雅集苦吟诗",
    "49": "四十九回　琉璃世界白雪红梅脂粉香娃割腥啖膻",
    "50": "五十回　芦雪庵争联即景诗暖香坞雅制春灯谜",
    "51": "五十一回　薛小妹新编怀古诗胡庸医乱用虎狼药",
    "52": "五十二回　俏平儿情掩虾须镯勇晴雯病补雀金裘",
    "53": "五十三回　宁国府除夕祭宗祠荣国府元宵开夜宴",
    "54": "五十四回　史太君破陈腐旧套王熙凤效戏彩斑衣",
    "55": "五十五回　辱亲女愚妾争闲气欺幼主刁奴蓄险心",
    "56": "五十六回　敏探春兴利除宿弊时宝钗小惠全大体",
    "57": "五十七回　慧紫鹃情辞试忙玉慈姨妈爱语慰痴颦",
    "58": "五十八回　杏子阴假凤泣虚凰茜纱窗真情揆痴理",
    "59": "五十九回　柳叶渚边嗔莺咤燕绛云轩里召将飞符",
    "60": "六十回　茉莉粉替去蔷薇硝玫瑰露引来茯苓霜",
    "61": "六十一回　投鼠忌器宝玉瞒赃判冤决狱平儿行权",
    "62": "六十二回　憨湘云醉眠芍药茵呆香菱情解石榴裙",
    "63": "六十三回　寿怡红群芳开夜宴死金丹独艳理亲丧",
    "64": "六十四回　幽淑女悲题五美吟浪荡子情遗九龙佩",
    "65": "六五回　贾二舍偷娶尤二姨尤三姐思嫁柳二郎",
    "66": "六十六回　情小妹耻情归地府冷二郎一冷入空门",
    "67": "六十七回　见土仪颦卿思故里闻秘事凤姐讯家童",
    "68": "六十八回　苦尤娘赚入大观园酸凤姐大闹宁国府",
    "69": "六十九回　弄小巧用借剑杀人觉大限吞生金自逝",
    "70": "七十回　林黛玉重建桃花社　史湘云偶填柳絮词",
    "71": "七十一回　嫌隙人有心生嫌隙鸳鸯女无意遇鸳鸯",
    "72": "七十二回　王熙凤恃强羞说病来旺妇倚势霸成亲",
    "73": "七十三回　痴丫头误拾绣春囊懦小姐不问累金凤",
    "74": "七十四回　惑奸谗抄检大观园矢孤介杜绝宁国府",
    "75": "七十五回　开夜宴异兆发悲音赏中秋新词得佳谶",
    "76": "七十六回　凸碧堂品笛感凄清凹晶馆联诗悲寂寞",
    "77": "七十七回　俏丫鬟抱屈夭风流美优伶斩情归水月",
    "78": "七十八回　老学士闲征诡画词痴公子杜撰芙蓉诔",
    "79": "七十九回　薛文龙悔娶河东狮贾迎春误嫁中山狼",
    "80": "八十回　美香菱屈受贪夫棒王道士胡诌妒妇方",
    "81": "八十一回　占旺相四美钓游鱼奉严词两番入家塾",
    "82": "八十二回　老学究讲义警顽心病潇湘痴魂惊恶梦",
    "83": "八十三回　省宫闱贾元妃染恙闹闺阃薛宝钗吞声",
    "84": "八十四回　试文字宝玉始提亲探惊风贾环重结怨",
    "85": "八五回　贾存周报升郎中任薛文起复惹放流刑",
    "86": "八十六回　受私贿老官翻案牍寄闲情淑女解琴书",
    "87": "八十七回　感深秋抚琴悲往事坐禅寂走火入邪魔",
    "88": "八十八回　博庭欢宝玉赞孤儿正家法贾珍鞭悍仆",
    "89": "八十九回　人亡物在公子填词蛇影杯弓颦卿绝粒",
    "90": "九十回　失绵衣贫女耐嗷嘈送果品小郎惊叵测",
    "91": "九十一回　纵淫心宝蟾工设计布疑阵宝玉妄谈禅",
    "92": "九十二回　评女传巧姐慕贤良玩母珠贾政参聚散",
    "93": "九十三回　甄家仆投靠贾家门水月庵掀翻风月案",
    "94": "九十四回　宴海棠贾母赏花妖失宝玉通灵知奇祸",
    "95": "九五回　因讹成实元妃薨逝以假混真宝玉疯颠",
    "96": "九十六回　瞒消息凤姐设奇谋泄机关颦儿迷本性",
    "97": "九十七回　林黛玉焚稿断痴情薛宝钗出闺成大礼",
    "98": "九十八回　苦绛珠魂归离恨天病神瑛泪洒相思地",
    "99": "九十九回　守官箴恶奴同破例阅邸报老舅自担惊",
    "100": "一百回　破好事香菱结深恨悲远嫁宝玉感离情",
    "101": "一百零一回　大观园月夜感幽魂散花寺神签惊异兆",
    "102": "一百零二回　宁国府骨肉病灾襟大观园符水驱妖孽",
    "103": "一百零三回　施毒计金桂自焚身昧真禅雨村空遇旧",
    "104": "一百零四回　醉金刚小鳅生大浪痴公子余痛触前情",
    "105": "一百零五回　锦衣军查抄宁国府骢马使弹劾平安州",
    "106": "一百零六回　王熙凤致祸抱羞惭贾太君祷天消祸患",
    "107": "一百零七回　散余资贾母明大义复世职政老沐天恩",
    "108": "一百零八回　强欢笑蘅芜庆生辰死缠绵潇湘闻鬼哭",
    "109": "一百零九回　候芳魂五儿承错爱还孽债迎女返真元",
    "110": "一百一十回　史太君寿终归地府王凤姐力诎失人心",
    "111": "一百一十一回　鸳鸯女殉主登太虚狗彘奴欺天招伙盗",
    "112": "一百一十二回　活冤孽妙尼遭大劫死雠仇赵妾赴冥曹",
    "113": "一百一十三回　忏宿冤凤姐托村妪释旧憾情婢感痴郎",
    "114": "一百一十四回　王熙凤历幻返金陵甄应嘉蒙恩还玉阙",
    "115": "一百一十五回　惑偏私惜春矢素志证同类宝玉失相知",
    "116": "一百一十六回　得通灵幻境悟仙缘送慈柩故乡全孝道",
    "117": "一百一十七回　阻超凡佳人双护玉欣聚党恶子独承家",
    "118": "一百一十八回　记微嫌舅兄欺弱女惊谜语妻妾谏痴人",
    "119": "一百一十九回　中乡魁宝玉却尘缘沐皇恩贾家延世泽",
    "120": "一百二十回　甄士隐详说太虚情贾雨村归结红楼梦"
}

    # # 打印检索结果及其上下文
    # for hits in result:
    #     for hit in hits:
    #         print(f"ID: {hit['id']}, Distance: {hit['distance']}, Text: {hit['entity'].get('text', 'N/A')}")
    #         # print("Context:")
    #         # partition_name = '_' + str(hit['entity'].get('partition', 'default'))
    #         # # 构造 filter 字符串，确保 id 是整型
    #         # filter_str = f"id >= {int(hit['id']) - 5} and id <= {int(hit['id']) + 5}"
    #         # context_result = client.query(
    #         #     collection_name=collection_name,
    #         #     filter=filter_str,
    #         #     output_fields=["id", "text"],
    #         #     partition_names=[partition_name]
    #         # )
    #         # for context_hit in context_result:
    #         #     print(f"  ID: {context_hit['id']}, Text: {context_hit['text']}")
    #         print('\n')