from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==============================================================================
#                            数据预处理与后处理辅助函数
# ==============================================================================

def pre_processing_chat(conversations, add_system_ratio=0.2):
    """
    对话数据的前处理函数。
    1. 保持带工具调用 (Tool use / Agent) 数据的原样，不做改动。
    2. 为了增强模型的系统提示词适应力，以指定概率 (如 20%) 随机给原本没有 system 角色的对话，
       在开头塞入一条随机挑选的系统提示词 (System Prompt)。
    """
    # 如果对话中存在 tools（表示该对话涉及工具调用），则完整保留不做处理
    if any(conv.get('tools') for conv in conversations): return conversations

    # 预设的多样化系统提示词列表，用于丰富模型的自我认知和助手人格
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    
    # 概率性添加 system 提示词到对话列表的最开始
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    """
    对话数据后处理函数。
    针对推理型数据集（带有自适应思考过程的数据），以 80% 的概率强行抹除空的思考标签 `<think>\n\n</think>\n\n`。
    核心目的：避免空的思考标记占用多余的 token，防止模型学废（只吐空思考标签而不真正思考）。
    """
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

# ==============================================================================
#                            预训练数据集 (PretrainDataset)
# ==============================================================================

class PretrainDataset(Dataset):
    """
    无监督预训练数据集。
    预训练是自回归语言模型的第一阶段，模型通过阅读海量连续文本来建立语言本能。
    其输入为纯文本，预测目标是“预测下一个 Token”，因此 input_ids 和 labels 完全一致。
    """
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 使用 Hugging Face dataset 高效载入本地 jsonl 文件
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        # 1. 对文本进行 Tokenize 编码，最大长度留出两个位置给 bos 和 eos 标记
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        
        # 2. 头尾强行拼接起始符 <bos> 和结束符 <eos>
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        
        # 3. 对长度不足 max_length 的序列在末尾填充 pad_token_id
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # 4. 创建预测标签 (labels)，默认等于 input_ids
        labels = input_ids.clone()
        # 极为关键：将填充 Token 部分的 labels 设为 -100。
        # PyTorch 的交叉熵损失计算会自动忽略 -100 处的损失，避免模型去学习和预测填充占位符！
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels

# ==============================================================================
#                            指令微调数据集 (SFTDataset)
# ==============================================================================

class SFTDataset(Dataset):
    """
    有监督指令微调数据集 (SFT - Supervised Fine-Tuning)。
    这是从 nanoGPT 自回归生成到 MiniMind Chat 对话的核心升级点！
    在 SFT 阶段，数据格式为 `[User Prompt] [Assistant Response]`。
    为了让模型学会回答，而不是去背诵用户的提问：
    - 我们仅对 Assistant 产生的回答部分计算 Loss，并传递梯度。
    - 对 User 提问和 System 设定等位置的标签 (labels) 强行覆写为 -100 进行屏蔽。
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 定义 JSONL 中 conversations 列的多重嵌套特征字段，确保工具调用列的结构正确解析
        features = Features({'conversations': [{'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        
        # 获取助理回复段的起始标识（如 `<bos>assistant\n`）与结束标识（如 `<eos>\n`）
        # 用于动态在输入序列中定位助理回答区域
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """
        利用分词器的 apply_chat_template 模版，将对话列表格式化为标准对话纯文本。
        包括系统内置符号，如：<system>\n...\n<user>\n...\n<assistant>\n...
        """
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        """
        动态查找定位助理回复区域，生成用于 SFT 的标签 mask 序列。
        输入：完整格式化后的序列 ID 数组 input_ids
        输出：标签数组 labels（其中提问及 padding 部分为 -100，回答部分为原始 input_ids）
        """
        # 初始化全部为 -100
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 1. 查找是否匹配到 assistant 开始标记
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                # 记录助理回答内容的起始位置
                start = i + len(self.bos_id)
                end = start
                # 2. 向后扫，直到找到助理的结束标记 eos_id
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 3. 将 [start, end + eos_id] 这一区间的 labels 还原为真实的 input_ids
                # 从而在训练中，模型仅在此区间产生 CrossEntropyLoss，进行梯度反传
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                # 指针移动到结束标识之后
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        # 前处理：概率性掺入 System 预设
        conversations = pre_processing_chat(sample['conversations'])
        # 拼装为一整段包含模板标签的对话文本
        prompt = self.create_chat_prompt(conversations)
        # 后处理：去除多余的空思考标签
        prompt = post_processing_chat(prompt)
        
        # 编码并截取至最大长度
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        # 不足最大长度的在右侧补齐填充符 pad_token
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        
        # 动态定位助理回答区域并屏蔽掉用户提问区域的 Loss
        labels = self.generate_labels(input_ids)
        
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

# ==============================================================================
#                            DPO偏好数据集 (DPODataset)
# ==============================================================================

class DPODataset(Dataset):
    """
    DPO (Direct Preference Optimization，直接偏好优化) 数据集。
    DPO 是一种轻量且极度有效的对齐算法。它不需要训练单独的 Reward 模型，
    直接基于成对的偏好数据 (Chosen 优质回答 / Rejected 劣质回答) 计算概率比例进行策略更新。
    对于每个样本，DPO 期望最大化 Chosen 回答的对数似然 (Log-Probability)，
    同时最小化 Rejected 回答的对数似然。
    为了准确计算模型对生成回复的概率，我们需要对 chosen 句和 rejected 句的回复部分做 loss_mask 标记。
    """
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample['chosen']      # 优质的回答序列
        rejected = sample['rejected']  # 较差的回答序列
        
        # 分别对 Chosen 和 Rejected 拼接模板并进行后处理
        chosen_prompt = self.tokenizer.apply_chat_template(chosen, tokenize=False, add_generation_prompt=False)
        chosen_prompt = post_processing_chat(chosen_prompt)

        rejected_prompt = self.tokenizer.apply_chat_template(rejected, tokenize=False, add_generation_prompt=False)
        rejected_prompt = post_processing_chat(rejected_prompt)
        
        # 分词编码并统一填充至 max_length 长度
        chosen_encoding = self.tokenizer(chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length')
        rejected_encoding = self.tokenizer(rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length')

        # 生成 chosen 和 rejected 回答部分的 mask 矩阵 (1 表示回答，0 表示提问/填充)
        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        
        # 经典自回归移位偏移 (类似于 Casual LM 移位)，以对齐输入与预测目标
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        """
        生成 0/1 的 loss mask 序列。
        属于助理回答部分标为 1，其余部分标为 0。
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

# ==============================================================================
#                            强化学习数据集 (RLAIFDataset)
# ==============================================================================

class RLAIFDataset(Dataset):
    """
    用于 PPO/GRPO 在线强化学习算法的提示词数据集。
    在线强化学习不需要提前准备助理的回答，而是只输入用户提问 (Prompt)，
    让当前策略网络 (Actor) 实时采样生成回复，再交给 Reward 模型进行打分训练。
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio  # 决定当前 Prompt 是否概率性诱导模型开启思考标签
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        conversations = pre_processing_chat(conversations)
        use_thinking = random.random() < self.thinking_ratio
        # 仅将对话中倒数第二句（通常是最后一轮 User 提问）以前的部分格式化为对话模版，
        # 并添加待回答指示符 (add_generation_prompt=True)，让模型去实时续写生成回答
        return self.tokenizer.apply_chat_template(
            conversations[:-1],
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True
        )

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])
        # 返回仅包含 Prompt 的前缀，Answer 字段为空待填
        return {
            'prompt': prompt,
            'answer': ""
        }

# ==============================================================================
#                            智能体强化学习数据集 (AgentRLDataset)
# ==============================================================================

class AgentRLDataset(Dataset):
    """
    智能体 (Agent) / 工具调用场景下的强化学习数据集。
    除了常规对话信息，还专门承载了外部可调用工具集 (Tools) 说明以及基准真实调用目标 (gt)。
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def parse_conversations(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            # 提取系统提示词里注入的可调用 API 工具集描述
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            messages.append(message)
        # 返回最后一轮前的历史会话以及工具清单
        return messages[:-1], tools

    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample['conversations'])
        return {'messages': messages, 'tools': tools, 'gt': sample['gt']}


if __name__ == "__main__":
    pass