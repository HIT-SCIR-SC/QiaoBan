# -*- coding: utf-8 -*-
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, logging, sys

def load_tokenizer_and_model(base_model, load_8bit=False):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    try:
        if torch.backends.mps.is_available():
            device = "mps"
    except:  # noqa: E722
        pass
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )

    if not load_8bit and device != "cpu":
        model.half()  # seems to fix bugs for some users.

    model.eval()
    return tokenizer, model, device
    
def sample_decode(
    input_ids: torch.Tensor,
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    stop_words: list,
    max_length: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 25,
):
    generated_tokens = []
    past_key_values = None
    current_length = 1
    for i in range(max_length):
        with torch.no_grad():
            if past_key_values is None:
                outputs = model(input_ids)
            else:
                outputs = model(input_ids[:, -1:], past_key_values=past_key_values)
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

        # apply temperature
        logits /= temperature

        probs = torch.softmax(logits, dim=-1)
        # apply top_p
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0.0

        # apply top_k
        # if top_k is not None:
        #    probs_sort1, _ = torch.topk(probs_sort, top_k)
        #    min_top_probs_sort = torch.min(probs_sort1, dim=-1, keepdim=True).values
        #    probs_sort = torch.where(probs_sort < min_top_probs_sort, torch.full_like(probs_sort, float(0.0)), probs_sort)

        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)

        input_ids = torch.cat((input_ids, next_token), dim=-1)

        generated_tokens.append(next_token[0].item())
        text = tokenizer.decode(generated_tokens)
        
        if '</s>' in text:
            return text.rstrip('</s>')

base_model = "tomxyz/qiaoban_bc"
tokenizer, model, device = load_tokenizer_and_model(
    base_model, load_8bit=False
)

chat_history=""
prompt = "以下对话发生在孩子和智能助手之间。孩子每天会经历各种事情和情绪活动，无论遇到任何问题都会找智能助手寻求安慰和帮助。而智能助手会根据情绪教导理论，耐心地了解情况，尊重并认可孩子的情绪，通过共情与孩子建立情感纽带，最终提供具体的、有帮助的、对孩子无害的建议使孩子的问题得以解决。"
while True:
    usr_inp = input(">> 用户: ")
    if usr_inp == "exit":
        print('End the conversation')
        print('')
        break
    usr_inp = "孩子：" + usr_inp + '</s>' + "智能助手："
    chat_history += usr_inp 
    inputs = tokenizer(prompt + chat_history, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()

    output = sample_decode(
            input_ids,
            model,
            tokenizer,
            stop_words=['</s>'],
            max_length=512,
            temperature=1,
            top_p=0.95,
        )

    if "智能助手：" in output:
        output = output.replace("智能助手：", "")
    print(">>智能助手：{}".format(output))
    chat_history += '智能助手：' + output + '</s>'