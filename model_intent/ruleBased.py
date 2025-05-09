import re
from sentence_transformers import SentenceTransformer, util



# 意图匹配函数
def match_intent(user_input):
    physiological_intent_patterns = [
        r"(pain|fever|headache|cough|cold|diarrhea|rash|sore throat)",
        r"(i have.*(symptoms|pain|inflammation))"
    ]

    psychological_intent_patterns = [
        r"(anxiety|depression|stress|emotion|psychological|emotional|suicide|loneliness|insomnia|fear|tension)",
        r"(i feel.*(sad|helpless|afraid|sorrow))"
    ]

    user_input = user_input.lower()  # 将输入转换为小写

    # 统计各类意图匹配的数量
    physiological_count = sum(bool(re.search(pattern, user_input)) for pattern in physiological_intent_patterns)
    psychological_count = sum(bool(re.search(pattern, user_input)) for pattern in psychological_intent_patterns)
    print(f"physiological_count: {physiological_count}, psychological_count: {psychological_count}")
    
    # 根据匹配数量分配意图
    if physiological_count >= psychological_count:
        return "physiological_agent"
    else:
        return "psychological_agent"  # 当两类意图匹配数量相同时
    


# 加载语义模型
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 定义模板
physio_template = "我有身体不适或症状"
psycho_template = "我感到情绪低落或心理压力"

# 用户输入
user_input = "我最近总是感到焦虑，晚上睡不着觉"

# 计算语义相似度
embeddings = model.encode([user_input, physio_template, psycho_template], convert_to_tensor=True)
cosine_scores = util.pytorch_cos_sim(embeddings[0], embeddings[1:])

if cosine_scores[0] > cosine_scores[1]:
    print("分配给: physiological_agent")
else:
    print("分配给: psychological_agent")


# 测试
user_input = """
I'm going through some things with my feelings and myself. I barely sleep and I do nothing but think about how I'm worthless and how I shouldn't be here.
I've never tried or contemplated suicide. I've always wanted to fix my issues, but I never get around to it.
How can I change my feeling of being worthless to everyone?
"""
agent = match_intent(user_input)
print(f"assigned to: {agent}")
