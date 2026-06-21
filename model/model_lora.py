import torch
from torch import optim, nn

# ==============================================================================
#                                LoRA (Low-Rank Adaptation) 模块说明
# ==============================================================================
# LoRA 是一种参数高效微调方法 (PEFT)。其核心思想是：在大模型微调过程中，基座模型的权重 W0 保持冻结，
# 通过在其旁路引入两个低秩矩阵 A 和 B 来模拟权重的增量更新 Delta W = B * A。
# 这样可以将需要训练的参数量降低数千倍，并且微调完成后可以通过矩阵相加将 LoRA 参数合并回原模型，不增加推理延迟。
# ==============================================================================

class LoRA(nn.Module):
    """
    LoRA 低秩自适应结构。
    输入维度为 in_features，输出维度为 out_features，秩为 rank (通常设为 8, 16, 32 等极小值)。
    Delta W 的形状为 [out_features, in_features]。
    低秩分解：
    - 矩阵 A 形状为 [in_features, rank]
    - 矩阵 B 形状为 [rank, out_features]
    """
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA的秩，控制低秩投影空间的维度大小
        
        # 矩阵 A：将输入特征从高维投影到低秩空间
        self.A = nn.Linear(in_features, rank, bias=False)  
        # 矩阵 B：将低秩空间特征投影回输出特征高维空间
        self.B = nn.Linear(rank, out_features, bias=False)  
        
        # 权重初始化策略：
        # 1. 矩阵 A 采用均值为 0、标准差为 0.02 的正态分布（高斯）进行初始化。
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 2. 矩阵 B 必须初始化为全 0。
        # 核心原因：在训练最开始的一步，Delta W = B * A = 0 * A = 0，从而保证此时 LoRA 旁路对原模型没有影响，
        # 模型输出与原始冻结基座模型的输出完全一致，确保训练起点的稳定性。
        self.B.weight.data.zero_()

    def forward(self, x):
        # 前向传播：先投影到低秩，再投影回高维。计算顺序：B(A(x))
        return self.B(self.A(x))


def apply_lora(model, rank=16):
    """
    利用 Python 动态猴子补丁 (Monkey Patching) 机制，自动向模型中符合条件的 Linear 层注入 LoRA 旁路。
    - model: MiniMindForCausalLM 实例
    - rank: 秩大小，默认 16
    """
    for name, module in model.named_modules():
        # 这里只针对输入维度与输出维度一致的线性层（如注意力层的 Q, O 映射投影层）注入 LoRA
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            # 实例化一个 LoRA 旁路层
            lora = LoRA(module.in_features, module.out_features, rank=rank).to(model.device)
            # 给该线性层动态附加 lora 属性
            setattr(module, "lora", lora)
            
            # 保存原始线性层的 forward 前向计算函数
            original_forward = module.forward

            # 显式绑定：定义带有 LoRA 旁路的新前向计算函数。
            # 输出 = 原线性层前向计算结果 + LoRA 旁路计算结果 (y = W0*x + B*A*x)
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            # 将原线性层的前向逻辑替换为带有 LoRA 的新逻辑 (Monkey Patching)
            module.forward = forward_with_lora


def load_lora(model, path):
    """
    从指定权重文件中加载 LoRA 微调权重。
    - model: 被注入过 LoRA 的大模型
    - path: lora 权重文件路径 (.pth)
    """
    state_dict = torch.load(path, map_location=model.device)
    # 清洗分布式训练 DDP 引入的 'module.' 前缀
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        # 如果当前模块有被注入 lora 属性，则过滤提取并加载它对应的参数
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    """
    仅提取大模型中属于 LoRA 旁路的参数并保存，不保存原冻结底座的庞大参数，从而极大地节省磁盘存储空间（仅需几兆字节）。
    - model: 训练中的带 LoRA 模型
    - path: 权重保存目标路径
    """
    raw_model = getattr(model, '_orig_mod', model) # 兼容经过 torch.compile 后的原始模型提取
    state_dict = {}
    
    # 遍历所有被注入过 lora 的线性层，抽取其状态字典
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
            
    # 将包含有所有 LoRA 线性层参数的 state_dict 写入硬盘
    torch.save(state_dict, path)


def merge_lora(model, lora_path, save_path):
    """
    无缝合并（Merge）LoRA 权重至基座模型。
    在推理部署时，为了避免每次前向都要多跑一遍 LoRA 旁路带来计算开销，
    直接计算 W_new = W0 + B * A，将二者融合成一个新的普通线性层权重保存。
    - model: 基础基座模型
    - lora_path: 待加载的 lora 权重文件路径
    - save_path: 合并后生成的完整基座模型保存路径
    """
    # 1. 先将 LoRA 参数加载进模型中
    load_lora(model, lora_path)
    raw_model = getattr(model, '_orig_mod', model)
    
    # 2. 复制一份不含 lora 的模型全局状态字典
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}
    
    # 3. 遍历线性层，进行权重数学相加： W0 = W0 + B * A
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            # 记录原权重 W0
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
            
            # 若该层被应用了 LoRA，计算 B.weight @ A.weight 并直接加到原权重上
            # 矩阵 A 形状是 [rank, in]，矩阵 B 形状是 [out, rank]， B @ A 的形状就是 [out, in]
            if hasattr(module, 'lora'):
                state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
                
    # 4. 保存合并后的完整通用权重，随后可用常规 load_state_dict 或 AutoModel 加载直接运行
    torch.save(state_dict, save_path)
