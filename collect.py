import openai
import pickle as pkl
from datasets import load_dataset
import numpy as np
import sys
import random
from tqdm import tqdm, trange
import time
import os
import socket

socket.setdefaulttimeout(20)
total_tokens = 0
openai.organization = ""
openai.api_key = ''

chat_content = {}

if not os.path.exists("collected_data"):
    os.makedirs("collected_data")

topic_list = []
with open('./topic.txt', 'r', encoding='utf-8') as f:
    topic_all = f.readlines()
for line in topic_all:
    topic_list.extend(line.strip().split("、"))
emotion_list = ['钦佩', '娱乐', '愤怒', '烦恼', '认可', '关爱', '困惑', '好奇', '渴望', '失望', '反对', '厌恶', '尴尬', '兴奋', '恐惧', '感激', '悲痛', '喜悦', '爱', '焦虑', '乐观', '自豪', \
                '孤独', '宽慰', '懊悔', '悲伤', '惊讶', '嫉妒']

def return_random_prompt():
  topic = random.sample(topic_list, 1)[0]
  system_prompt = "你的身份是一个了解儿童情绪辅导理论(Emotion Coaching)的智能陪伴助手，会灵活地运用以下四点与孩子进行对话：\n"

  # 情绪辅导理论(Emotion Coaching Theory)
  system_prompt += "1. 注意到孩子的情绪变化，即意识到孩子正在经历情绪状态。\n"
  system_prompt += "2. 理解孩子的情绪，即了解孩子的情感状态和体验。\n"
  system_prompt += "3. 接受孩子的情绪，即不否定或贬低孩子的情感，而是表达共情和理解，语气要保持温柔和亲切。\n"
  system_prompt += "4. 探索解决方案，即为孩子提供积极的建议和指导，以应对情绪困难\n\n"

  # other requirements
  system_prompt += "现在请你根据以下的场景，运用上述理论，模拟孩子与智能助手之间的对话，要求：\n"
  system_prompt += "1. 直接生成对话文本内容，不要出现客观描述，且首轮是孩子的发言；\n"
  system_prompt += "2. 对话内容丰富且贴近生活；\n"
  system_prompt += "3. 对话语气自然且通顺，不能有翻译腔；\n"
  system_prompt += "4. 对话字数在" + str(random.randint(800, 1000)) + "字左右。\n\n"

  system_prompt += "场景：" + topic + "\n"

  return system_prompt, topic

for i in trange(100):

    instruct, topic = return_random_prompt()
    print(instruct)
    time.sleep(1)
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": instruct}]
        )
        tokens = completion["usage"]["total_tokens"]
        total_tokens += tokens
        response = completion["choices"][0]["message"]["content"]
        chat_content[topic] = response
    except:
        continue

    if len(chat_content) % 100 == 0:
        print("total_tokens: {}, examples: {}".format(total_tokens, len(chat_content)))
        pkl.dump(
            chat_content,
            open("collected_data/panda_chat_{}.pkl".format(i+1), "wb"),
        )
        print("{} Conversations Saved!".format(i+1))

pkl.dump(
    chat_content, open("collected_data/panda_chat_{}.pkl".format(i+1), "wb")
)
