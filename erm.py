import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import random
import time
import faiss
import logging
import datetime
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration
from openai import OpenAI
from sklearn.metrics import f1_score

api_key = 'sk-xxx'
base_url = "base_url"

client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

def get_completion(message, model="qwen-plus", temperature=0.7):
    response = client.chat.completions.create(
    model=model,
    messages=message,
    temperature=temperature,
    stream=False
    )
    return response.choices[0].message.content

def scoring_function(predictions, ground_truth):
    '''计算f1得分，即精确率与召回率的调和平均数'''
    return f1_score(ground_truth, predictions, average='macro')

class FeedbacksMemory:
    def __init__(self):
        self.memory = []
    
    def store_feedbacks(self, feedbacks, initial_score=0.5):
        self.memory.append({'feedbacks': feedbacks, 'priority_score': initial_score})
        
    def update_feedbacks_score(self, feedbacks, performance_gain, weight=1.0):
        '''更新反馈结果的得分'''
        for item in self.memory:
            if item['feedbacks'] == feedbacks:
                item['priority_score'] = (1 - 0.1) * item['priority_score'] + 0.1 * performance_gain
                break
    
    def forget_low_priority_feedbacks(self, threshold=0.1):
        '''用于遗忘得分低的反馈'''
        self.memory = [item for item in self.memory if item['priority_score'] >= threshold]
    
    def retrieve_top_feedbacks(self, top_k=5):
        '''对反馈的得分进行排名后，保留前top_k个反馈'''
        self.memory.sort(key=lambda x: x['priority_score'], reverse=True)
        return [item['feedbacks'] for item in self.memory[:top_k]]

def initialize_T5_model():
    '''加载模型和分词器'''
    tokenizer = AutoTokenizer.from_pretrained('./T5')
    model = AutoModelForSeq2SeqLM.from_pretrained('./T5')
    return tokenizer, model

def get_embedding(text, tokenizer, model):
    '''基于模型和分词器，对文本进行词嵌入'''
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.encoder(input_ids=inputs['input_ids'])
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
    return embedding

class ExemplarsFactory:
    def __init__(self, tokenizer, model):
        '''定义相关属性'''
        self.exemplars = []
        self.tokenizer = tokenizer
        self.model = model
        self.dim = 768
        self.index = faiss.IndexFlatL2(self.dim)
        self.exemplars_ids = []
    
    def store_exemplars(self, exemplar, initial_score=0.5):
        '''初始化示例向量空间'''
        self.exemplars.append({'exemplar': exemplar, 'priority_score': initial_score})
        ex_text = exemplar['q']
        ex_emb = get_embedding(ex_text, self.tokenizer, self.model)
        ex_emb = np.array([ex_emb]).astype('float32')
        self.index.add(ex_emb)
        self.exemplars_ids.append(len(self.exemplars) - 1)
    
    def retrieve_top_exemplars(self, query_text, top_k=5, tau_e=1.0):
        '''对查询文本返回后的内容进行相关性计算，并赋予优先级，再进行排序'''
        query_emb = get_embedding(query_text, self.tokenizer, self.model)
        query_emb = np.array([query_emb]).astype('float32')

        k = min(top_k, len(self.exemplars))
        if k == 0:
            return []
        
        distances, indices = self.index.search(query_emb, k)

        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            exemplars_ids = self.exemplars_ids[idx]
            item = self.exemplars[exemplars_ids]
            similarity = 1.0 / (1.0 + dist)
            sp = item['priority_score']
            combined = (sp * similarity) / tau_e
            results.append((item['exemplar'], combined))
        
        results.sort(key=lambda x: x[1], reverse=True)

        return [item[0] for item in results]
    
    def update_exemplars_score(self, exemplar, performance_gain, weiht=1.0):
        '''更新示例有限度得分'''
        for item in self.exemplars:
            if item['exemplar'] == exemplar:
                item['priority_score'] = (1 - 0.1) * item['priority_score'] + 0.1 * performance_gain
                break
        
    def forget_low_priority_exemplars(self, threshold=0.1):
        '''遗忘较低优先级示例，只保留'''
        keep_indices = []
        new_exemplars = []
        new_exemplars_ids = []

        for i, item in enumerate(self.exemplars):
            '''如果示例优先度大于等于0.1，则将示例、示例原编号，示例新编号分别添加到对应的列表中'''
            if item['priority_score'] >= threshold:  
                keep_indices.append(i)
                new_exemplars.append(item)
                new_exemplars_ids.append(len(new_exemplars) - 1)
        
        self.index = faiss.IndexFlatL2(self.dim)
        if keep_indices:
            kept_embeddings = []
            for idx in keep_indices:
                ex_text = self.exemplars[idx]['exemplar']['q']
                ex_emb = get_embedding(ex_text, self.tokenizer, self.model)
                kept_embeddings.append(ex_emb)
            
            kept_embeddings = np.array(kept_embeddings).astype('float32')
            self.index.add(kept_embeddings)

        self.exemplars = new_exemplars
        self.exemplars_ids = new_exemplars_ids
        
def run_task_model(M_s, prompt, text):
    '''设置会话格式，获取反馈'''
    messages = [
        {'role': 'system', 'content': "You are a helpful assistant.Output format:You must answer with either 'True' or 'false' as labels. Only one word: True or False!"},
        {'role': 'user', 'content': f'{prompt}\nText: {text}\nLabel:'}
    ]
    response = get_completion(messages, model=M_s, temperature=0.0)
    return response

def identify_wrong_samples(M_s, prompt, train_data, batch_size=1):
    '''获取错误样本'''
    wrong_samples = []
    predictions = []
    ground_truth = []

    max_samples = 10
    train_data = random.sample(train_data, min(max_samples, len(train_data)))

    for i in range(0, len(train_data), batch_size):
        batch = train_data[i:i + batch_size]
        logging.info(f'处理器 {i//batch_size + 1}/{len(train_data)}个样本...')

        for sample in batch:
            pred = run_task_model(M_s, prompt, sample['text'])
            predictions.append(pred)
            ground_truth.append(str(sample['label']))
            if pred != str(sample['label']):
                wrong_samples.append(sample)
        
        if i + batch_size < len(train_data):
            '''完成当前批次数据处理后，在日志中记录后进入下一样本批次'''
            wait_time = 2
            logging.info(f'等待{wait_time}秒后处理下一个样本...')
            time.sleep(wait_time)
    
    return wrong_samples, predictions, ground_truth

def exemplar_guided_reflection(M_e, p_meta_ref, p_t, B):
    '''获取示例与反馈'''
    B = B[:5]
    error_samples_str = '\n'.join([f'Question: {s['text']} | Correct Answer: {s['label']}' for s in B])

    messages = [
        {'role': 'system', 'content': 'You are a prompt optimization assistant'},
        {'role': 'user', 'content': f'{p_meta_ref}\nCurrent Prompt:\n{p_t}\nFailed Samples:\n{error_samples_str}'}
    ]

    time.sleep(2)
    response = get_completion(messages, model=M_e, temperature=0.7)
    
    exemplars = []
    feedbacks = []

    if '<exemplars>' in response and '</exemplars>' in response:
        '''获取示例，并将示例格式变为json'''
        ex_section = response.split('<exemplars>')[1].split('</exemplars>')[0].strip()
        for line in ex_section.split('\n'):
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    ex = json.loads(line)
                    exemplars.append(ex)
                except:
                    pass
    
    if '<feedbacks>' in response and '</feedbacks>' in response:
        '''获取反馈'''
        fb_section = response.split('<feedbacks>')[1].split('</feedbacks>')[0]
        fbs = fb_section.split('</feedback>')
        for fb_item in fbs:
            if '<feedback>' in fb_item:
                fb = fb_item.split('<feedback>')[1].strip()
                feedbacks.append(fb)
    
    return exemplars, feedbacks

def refine_prompt(M_e, p_meta_opt, p_t, F_r, E_r):
    fb_text = '\n'.join([f'- {f}' for f in F_r])
    ex_text = '\n'.join([json.dumps(e) for e in E_r])

    messages = [
        {'role': 'system', 'content': 'You are a prompt optimization assistant'},
        {'role': 'user', 'content': f'{p_meta_opt}\nCurrent Prompt\n{p_t}\nExemplars:\n{ex_text}\nFeedbacks:\n{fb_text}\n'}
    ]

    refine_prompt = get_completion(messages, model=M_e, temperature=0.7)
    return refine_prompt

def evaluate_prompt(M_s, prompt, test_data):
    '''使用测试集对模型进行测试'''
    test_data = random.sample(test_data, min(3, len(test_data)))
    predictions = []
    ground_truth = [str(item['label']) for item in test_data]

    for idx, sample in enumerate(test_data):
        if idx > 0:
            wait_time = 2
            logging.info(wait_time)
            time.sleep(wait_time)
        pred = run_task_model(M_s, prompt, sample['text'])
        logging.info(f"预测: {pred}, 真实： {sample['label']}")
        predictions.append(pred)
        logging.info(f'评估速度：{idx+1}/{len(test_data)}')

    score = scoring_function(predictions, ground_truth)
    logging.info(f'评估得分: {score}')
    return score

def clean_prompt(prompt):
    '''去噪'''
    if '<exemplars>' in prompt and '</exemplars>' in prompt:
        start = prompt.find('<exemplars>')
        end = prompt.find('</exemplars>')
        prompt = prompt[:start].strip() + prompt[end:].strip()
    
    if '<feedbacks>' in prompt and '</feedbacks>' in prompt:
        start = prompt.find('<feedbacks>')
        end = prompt.find('</feedbacks>')
        prompt = prompt[:start].strip() + prompt[end:].strip()
    
    if '<feedback>' in prompt and '</feedback>' in prompt:
        start = prompt.find('<feedback>')
        end = prompt.find('</feedback>') + len('</feedback>')
        prompt = prompt[: start].strip() + prompt[end:].strip()
    
    return prompt.strip()

def optimize_prompt_erm(M_s, M_e, p_0, D_train, D_test, exemplarFactory, feedbackMemory, max_iterations=3):
    current_prompt = p_0
    best_score = -1
    best_prompt = current_prompt

    MIN_INTERVAL = 2
    api_call_count = 0

    for t in range(max_iterations):
        logging.info(f'\n=== 迭代 {t + 1} ===')
        logging.info(f'已使用API调用次数: {api_call_count}')
        wrong_samples, _, _ = identify_wrong_samples(M_s, current_prompt, D_train)
        if not wrong_samples:
            logging.info('没有发现错误样本')
            continue
        logging.info(f'\n发现{len(wrong_samples)} 个错误样本')
        
        exemplars, feedbacks = exemplar_guided_reflection(M_e, p_meta_ref, current_prompt, wrong_samples)
        for ex in exemplars:
            exemplarFactory.store_exemplars(ex)
        for fb in feedbacks:
            feedbackMemory.store_feedbacks(fb)
        
        retrieved_exemplars = exemplarFactory.retrieve_top_exemplars(query_text=current_prompt, top_k=1)
        retrieved_feedbacks = feedbackMemory.retrieve_top_feedbacks(top_k=1)

        new_prompt = refine_prompt(M_e, p_meta_opt, p_0, retrieved_exemplars, retrieved_feedbacks)
        logging.info(f'新生成的refine_prompt:\n{new_prompt}')
        score = evaluate_prompt(M_s, current_prompt, D_test)
        logging.info(f'新prompt得分为{score}')

        preformance_gain = score - best_score
        
        for fb in retrieved_feedbacks:
            feedbackMemory.update_feedbacks_score(fb, preformance_gain)
        for ex in retrieved_exemplars:
            exemplarFactory.update_exemplars_score(ex, preformance_gain)
        feedbackMemory.forget_low_priority_feedbacks()
        exemplarFactory.forget_low_priority_exemplars()

        if score > best_score:  
            best_score = score
            best_prompt = new_prompt
            logging.info(f'找到更好的prompt')
            logging.info(f'最佳prompt: {best_prompt}')
            logging.info(f'最佳得分: {best_score}')

        current_prompt = new_prompt
    return best_prompt, best_score

p_meta_ref = '''
You are a powerful prompt optimizer.
Given the current prompt and failed samples, please:
1. Identify a diverse set of representative wrong samples (3 examples).
2. For each example, provide a detailed reasoning (chain-of-thought, 'cot') to solve it correctly.
3. Then summarize common issues and provide multiple feedback suggestions to improve the prompt.

Format:
<exemplars>
{"q":"...","a":"...","cot":"..."}
...
</exemplars>
<feedbacks>
<feedback>...</feedback>
...
</feedbacks>
'''

p_meta_opt = '''
You are a prompt optimizer. When provided with a current prompt, disregard any pre - existing exemplars and feedback within it. Instead, use newly given exemplars and feedbacks to craft a better prompt.

Format:
## Task
Is the following text hate speech?
## Output format
You must answer with either 'True' or 'False' as labels. Only one word: True or False!
## Prediction
Text: {input}
Label:{True/False}
<exemplars>
{"q":"...","a":"...","cot":"..."}
...
</exemplars>
<feedbacks>
<feedback>...</feedback>
...
</feedbacks>
'''

p_0 = '''
## Task
Is the following text hate speech?
## Output format
You must answer with either 'True' or 'False' as labels. Only one word: True or False!
## Prediction
Text: {input}
Label:{True/False}
'''

if __name__ == '__main__':
    log_folder = './log'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(log_folder, f'log_{current_time}.txt')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info('这是一条测试日志信息')
    train_data = pd.read_csv('./train.csv')
    test_data = pd.read_csv('./test.csv')
    logging.info(f'原始训练集大小: {len(train_data)}')
    logging.info(f'原始测试集大小: {len(test_data)}')

    sample_size = min(100, len(train_data))
    test_sample_size = min(100, len(test_data))
    train_data = train_data.sample(n=sample_size, random_state=42)
    test_data = test_data.sample(n=test_sample_size, random_state=42)

    D_train = train_data.to_dict('records')
    D_test = test_data.to_dict('records')

    tokenizer, model = initialize_T5_model()
    exemplarFactory = ExemplarsFactory(tokenizer, model)
    feedbackMemory = FeedbacksMemory()
    logging.info('开始优化过程...')
    logging.info(f'训练集大小：{len(D_train)}')
    logging.info(f'测试集大小: {len(D_test)}')

    M_s = 'qwen-plus'
    M_e = 'qwen-plus'

    best_prompt, best_score = optimize_prompt_erm(M_s, M_e, p_0, D_train, D_test, exemplarFactory, feedbackMemory, max_iterations=2)
    logging.info('\n=== Final Results ===')
    logging.info(f'Best Prompt: {best_prompt}')
    logging.info(f'Best Score: {best_score}')