<p align="center">
<img width="400px" alt="Project QiaoBan" src="./logo.png">
</p>
<hr>

[**中文**](./README.md) | [**English**](./README_EN.md)

# Qiaoban

Qiaoban Large Model is an open-source children's emotional companion dialogue model based on instruction finetuning on the Baichuan Large Model. It employs general domain human-machine dialogues, single-turn instruction data, and children's emotional companion dialogue data for instruction fine-tuning. We are now releasing the 7B version of the Qiaoban Large Model. With a more empathetic conversational approach, the Qiaoban model enhances the interaction experience between children and the model. It serves as a practice of transferring general language models to vertical domains and provides a reference for relevant researchers.

## Why it's called "QiaoBan"?

"Qiaoban" refers to the Tangram puzzle, a traditional Chinese intellectual puzzle toy and educational tool. This aligns with the functions we expect from a children's large model, including companionship, intellectual development, and education. Additionally, to comply with the naming conventions of the SCIR laboratory for releasing large models, it is named the "Qiaoban" large model.

## Features of the Qiaoban Large Model

1. More empathetic interaction with children
2. Excellent understanding of children's emotional states
3. Provides positive and effective guidance and advice for children's growth and development

## Overview

This is a repository for the Qiaoban Large Model, aiming to build a large model focused on emotional companionship for children. This repository includes:

- Dialog data for instruction fine-tuning `/data`
- Code `collect.py` for obtaining parent-child conversations from ChatGPT
- Training code for the Qiaoban model
- Training configuration files
- Example code for conversing with the Qiaoban model(TODO, checkpoint to be uploaded to huggingface)

## Parent-Child Empathetic Dialogue Data Construction

### Data Acquisition

```
python collect.py
```

We sampled from a list of real-life children's dialogue topics, selected the current conversation topic, and, guided by the Parental Emotion Coaching theory, combined with pre-defined prompts to form `chatgpt_prompt`. We then obtained empathetic dialogue data from `gpt-3.5-turbo`. The specific prompt is as follows:

```
Your identity is an intelligent companion assistant who understands the theory of Parental Emotion Coaching and will flexibly use the following four points to engage in conversation with children:
1. Pay attention to the child's emotional changes, i.e., be aware of the child's emotional state.
2. Understand the child's emotions, i.e., understand the child's emotional state and experiences.
3. Accept the child's emotions, i.e., do not deny or belittle the child's emotions but express empathy and understanding, maintaining a gentle and kind tone.
4. Explore solutions, i.e., provide positive advice and guidance to help the child cope with emotional difficulties.
Now, based on the following scenario, use the above theory to simulate a conversation between a child and an intelligent assistant. Requirements:
1. Generate dialogue text directly, without objective descriptions, and the first turn should be the child's speech;
2. The dialogue content should be rich and close to real-life;
3. The dialogue should be natural and fluent, without translation-like language;
4. The dialogue should consist of approximately range(800, 1001) words.
Scenario: topic_sample()
```

Additionally, we also constructed some high-quality Chinese parent-child empathetic dialogue data ourselves.(We will realease later) 

### Used Instructions

We designed empathetic instructions based on the Emotion Coaching theory as follows:

```
The following conversation takes place between the child and the intelligent assistant. The child goes through various events and emotional activities every day and seeks comfort and help from the intelligent assistant for any problem. The intelligent assistant patiently understands the situation, respects and acknowledges the child's emotions, establishes an emotional bond with the child through empathy, and ultimately provides specific, helpful, and harmless advice to solve the child's problems.
```

## Model Training Reproduction

### Training

The fine-tuning code is designed to run on 4 A100-80G GPUs, and it takes about 50 hours to complete the training.

We fine-tuned the model on four A100-80GB GPU cards, and it took about 50 hours to complete the training process.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed finetune.py --model_config_file run_config/config.json --deepspeed run_config/deepspeed_config.json
```

## Case Demonstration

We have sampled three interactions with Qiaoban for demonstration:

### Case 1

```
'child':Tomorrow, I'm going on a trip with my classmates!
'agent':You seem very excited! Do you enjoy the feeling of traveling?
'child':Yes, I love going on trips with my classmates the most!
'agent':Where are you going?
'child':We are going to Lijiang Ancient Town. I heard there are many interesting things there!
'agent':Are you prepared? Do you need any help from me?
'child':I'm all prepared! But I don't know what to do inside the ancient town. It's my first time visiting there!
'agent':You can try looking for local handicrafts, such as bamboo weaving, wood carving, lacquerware, etc. Or you can try surfing or taking a boat to tour the water islands with your classmates; it will be a very novel experience.
'child':That sounds great!
'agent':However, safety is also something we must pay attention to when we are away from home. If you encounter any problems, don't be shy to seek help from the local tour guide or the police.
```

### Case 2

```
'child':Other students in my class all have their own best friends, but I'm all alone and feel very lonely.
'agent':I can understand how you feel. You feel lonely and want someone to accompany you.
'child':Yes, that's how I feel.
'agent':It's normal to feel that way. Everyone has their favorite friends, but I believe you can find your own friends too. You can try participating in more activities, getting to know new classmates, and interacting with them.
'child':But it seems like all my classmates already have their close friends, and I can't fit in.
'agent':We all want to be understood and recognized. No matter where or when, don't forget yourself and bravely face any problems that may arise when making friends. If you take the initiative to communicate with your classmates, they won't reject you; they will be happy to play with you.
'child':But I'm not confident enough to talk to them, and I can't find suitable topics to discuss.
'agent':Exactly, that's why we can create some topics together. By discussing these topics, you can get to know more friends. You can ask about their interests or chat with them about interesting things, don't worry about not finding a topic, just communicate with them naturally.
'child':I like watching Naruto, but I don't know if they like it too.
'agent':That's great; you can share your feelings about watching the show and talk about the characters. You can also ask them about what shows they are watching. This way, you can easily find common interests and bond with them.
'child':But I heard that they all like to watch romance dramas, and no one likes anime.
'agent':I understand how you feel; it might be challenging for you. You can try telling them that anime is special, and you can tell them about your favorite anime. Let them know what you like. This way, you can help them realize your hobbies and get to know each other better.
'child':Okay, I can give it a try.
'agent':Great! I believe you can find friends who appreciate you and become a better person through mutual communication. If you encounter difficulties in finding friends, don't worry, stick to your beliefs, and there will always be people who support you.
```

### Case 3

```
'child':My mom and dad got divorced, and my mom said she doesn't want me anymore.
'agent':Hearing you say that makes me very sad. How do you feel right now?
'child':I feel like my world is falling apart. My mom doesn't love me anymore, and I no longer have a complete family.
'agent':I can understand your feelings. This is a very painful thing for you. However, you need to know that your mom will not abandon you; she still loves you.
'child':But she said she doesn't want me anymore and that I'm an unwanted child.
'agent':You shouldn't think of your mom that way. Maybe she said some things out of emotions, but she didn't really mean it. You should believe in your mom; she still loves you very much.
'child':She said she won't have any contact with me and my dad anymore and won't come to see me.
'agent':It sounds like the relationship between your mom and dad is not good. However, even if they are separated, they are still your parents, and you are still their child. You don't need to worry about them not interacting; you just need to keep in touch with them.
'child':My mom used to hit me all the time.
'agent':That's a very bad behavior; we shouldn't hit children. But we can have a good talk with your mom and let her know that hitting children is wrong and help her change this bad habit.
'child':My dad always drinks and doesn't care about my studies. I feel like I'm on my own all the time.
'agent':Your father's drinking is indeed a problem for you; drinking can bring many negative effects. However, you can talk to your dad and let him know that you want him to pay attention to your studies and growth. This may not be easy, but you can try to communicate with him and let him know your thoughts and needs.
'child':Every time I try to talk to him, he gets very angry and will hit me if I continue talking.
'agent':This is a very bad situation, and you should try to avoid talking to your dad as much as possible. However, if you still want to talk to him, you can try using some indirect ways, such as writing a letter or finding other opportunities to talk to him.
'child':Nobody cares about me; I'm just an unwanted child.
'agent':I know how important this is to you, but you should know that you are a very special individual, and you deserve to be loved and cared for. You can try to find some things and people that can bring you warmth and comfort, such as playing with friends, listening to music, drawing, etc. This can help relieve your sense of loneliness.
'child':My family is very poor, and I can't afford drawing materials.
'agent':This is also a very bad situation, but you can try to find some free or low-cost drawing materials. For example, you can go to some handicraft stores or grocery stores; they usually have some free drawing materials.
```

## Project Contributors

This project was completed by the Sentiment Computing Group of the Social Computing and Information Retrieval Research Center, Harbin Institute of Technology. 

Project Developer: Weixiang Zhao, Yanpeng Tong, Shilong Wang, Tian Zheng, Chenxue Wang.

The project was supervised by Associate Professor Yan Yan Zhao and Professor Bing Qin

## Acknowledgments

This project references the following open-source projects, and we would like to express our gratitude to the relevant project developers and researchers.

- LianJia BELLE: https://github.com/LianjiaTech/BELLE
- BaiZe: https://github.com/project-baize/baize-chatbot

The list of real-life children's dialogue topics used in constructing the parent-child empathetic dialogue dataset was provided as reference by **iFLYTEK**.

We deliver sincere thanks to all specialists and volunteers who participate in data collection, annotation and revision.

## Disclaimer

The resources related to this project are for academic research purposes only and are strictly prohibited for commercial use. When using parts involving third-party code, please strictly follow the corresponding open-source licenses. The content generated by the model is affected by model calculations, randomness, and quantization precision loss. This project cannot guarantee its accuracy. The vast majority of the children's empathetic dialogue data used in this project is generated by the model, and we do not take any responsibility for its output content. For any content output by the model, this project assumes no legal responsibility and is not liable for any losses that may arise from the use of related resources and output results.

