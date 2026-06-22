import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Config (模型参数配置)
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MiniMindConfig(PretrainedConfig):
    """
    MiniMind模型配置类，继承自Hugging Face的PretrainedConfig。
    定义了模型架构的所有超参数，包括通道维度、层数、是否启用MoE、注意力头数、RoPE外推等参数。
    相较于nanoGPT（GPT-2配置），增加了GQA（Grouped-Query Attention）和MoE（混合专家）特有的参数。
    """
    model_type = "minimind"

    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size                  # 隐藏层通道维度 (d_model)，例如 768
        self.num_hidden_layers = num_hidden_layers      # Transformer Block (层数) 的数量，例如 8
        self.use_moe = use_moe                          # 是否启用混合专家架构 (MoE)
        self.dropout = kwargs.get("dropout", 0.0)       # 训练中的随机失活率
        self.vocab_size = kwargs.get("vocab_size", 6400) # 词表大小，默认为 6400 (小词表适合小模型快速训练)
        self.bos_token_id = kwargs.get("bos_token_id", 1) # 句子起始Token ID (Beginning of Sentence)
        self.eos_token_id = kwargs.get("eos_token_id", 2) # 句子结束Token ID (End of Sentence)
        self.flash_attn = kwargs.get("flash_attn", True)  # 是否启用 Flash Attention 加速开发包（需要 PyTorch 2.0+ 支持）
        self.num_attention_heads = kwargs.get("num_attention_heads", 8) # Query（查询）的注意力头数
        
        # GQA核心参数：Key/Value（键值）的注意力头数。
        # 如果 num_key_value_heads == num_attention_heads，则退化为 MHA (Multi-Head Attention，类似 nanoGPT/GPT-2)
        # 如果 num_key_value_heads == 1，则为 MQA (Multi-Query Attention)
        # 如果 1 < num_key_value_heads < num_attention_heads，则为 GQA (Grouped-Query Attention，类似 Llama 3)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4) 
        
        # 每个注意力头的维度，默认等于 hidden_size / num_attention_heads
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu') # 激活函数，现代 LLM 常使用 SiLU (Swish)
        
        # SwiGLU MLP的中间层维度，通常比标准前向网络稍大。此处使用圆周率进行一定系数扩增，并向上取整至 64 的倍数
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768) # 支持的最大上下文序列长度
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6) # RMSNorm 归一化时分母防止除以 0 的极小常数
        self.rope_theta = kwargs.get("rope_theta", 1e6)      # 旋转位置编码 (RoPE) 的基数角度常数 (Llama 3 采用 1e6 或以上以扩展长文本)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True) # 词表嵌入层与LM Head输出层是否共享权重以节省显存
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False) # 推理时是否启用 RoPE 插值/外推
        
        # YaRN 位置外推配置参数 (用于无痛或低损耗扩展上下文长度)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        
        ### MoE 架构专用配置 (当 use_moe = True 时生效)
        self.num_experts = kwargs.get("num_experts", 4)               # MoE 总专家数量
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1) # 每个 Token 被分配激活的专家数 (Top-K)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size) # 专家网络中间层大小
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)      # 是否对路由器的 top-k 概率进行重新归一化 (Softmax)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4) # 门控路由辅助损失系数，用于避免专家负载极度不均

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Model (基础算子与模型)
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏

class RMSNorm(torch.nn.Module):
    """
    RMSNorm (Root Mean Square Layer Normalization)
    相比于 nanoGPT 中的 LayerNorm，RMSNorm 移除了减去均值 (Mean) 的步骤，只对通道做均方根归一化。
    数学公式：y = (x / RMS(x)) * weight，其中 RMS(x) = sqrt( mean(x^2) + eps )
    好处：计算速度更快，且在深层网络中表现与 LayerNorm 完全相当。
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) # 可学习的缩放参数 gamma

    def norm(self, x):
        # 核心：计算均方根倒数，并在通道维度进行缩放归一化
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 将输入强制转换为 float32 计算 norm 以保证数值稳定性，再还原回输入类型并与可学习参数相乘
        return (self.weight * self.norm(x.float())).type_as(x)


def precompute_freqs_cis(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: dict = None,
):
    """【第一阶段：预计算旋转角排班大表】

    将每一行（绝对位置 t）与每一列（各维度的角速度 \omega）相乘，
    预先拼装出整个序列全维度的 cos 和 sin 角度矩阵。
    """

    # -----------------------------------------------------------------
    # 1. 计算每个 2D 平面的基础角速度向量 freqs (1/w)
    # 算子拆解机制展示 (以 dim = 8, rope_base = 10000 为具体数值样例):
    #   (1) torch.arange(0, 8, 2)            =>  [0, 2, 4, 6]
    #   (2) 除以 dim (8)                      =>  [0.0, 0.25, 0.5, 0.75]
    #   (3) rope_base ** 指数                 =>  [10000^0, 10000^0.25, 10000^0.5, 10000^0.75] = [1, 10, 100, 1000]
    #   (4) 1.0 / 结果                        =>  [1.0, 0.1, 0.01, 0.001]
    # 最终 freqs 形状为 [dim // 2]，其内部数值空间形态为：
    #   freqs = [ \omega_0, \omega_1, \omega_2, \omega_3 ]
    # -----------------------------------------------------------------
    freqs, attn_factor = (
        1.0
        / (
            rope_base
            ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
        ),
        1.0,
    )


    # -----------------------------------------------------------------
    # 2. 动态扩容插值 (YaRN 算法核心)
    # 目的：当 end (如 32768) 超过模型训练极限 orig_max (如 2048) 时，启动分频拉伸。
    # -----------------------------------------------------------------
    if rope_scaling is not None:
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 16)  # 扩容倍数：32768 / 2048 = 16
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        attn_factor = rope_scaling.get("attention_factor", 1.0)

        # 只有当实际长文本需求(end)大于原始训练长度(orig_max)时，才需要进行插值变换
        if end / orig_max > 1.0:

            # (A) 依据波长反推维度的临界边界索引 (inv_dim)
            # 数学原理: 什么样的维度算高频？什么样的算低频？通过高低频阈值 beta_fast/slow 反推。
            # 算出来的 low 和 high 是两个在 [0, dim//2] 之间的维度坐标。
            inv_dim = lambda b: (
                dim * math.log(orig_max / (b * 2 * math.pi))
            ) / (2 * math.log(rope_base))
            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)

            # (B) 构造渐变坡度掩码向量 ramp (形状: [dim // 2])
            # 它的本质是一个滑梯函数：
            #   - 在 low 维度左边（高频区），值全为 0
            #   - 在 high 维度右边（低频区），值全为 1
            #   - 在 low 到 high 之间（过渡区），值从 0 线性增加到 1
            #
            # 形象化掩码形态展示 (假设共有 8 个平面，low=2, high=5):
            # 维度平面索引 i:    0      1      2      3      4      5      6      7
            # 对应的 ramp 值: [ 0.0,   0.0,   0.0,   0.33,  0.66,  1.0,   1.0,   1.0  ]
            #                 |<- 高频区全0 ->| |<-- 中频过渡 -->| |<- 低频区全1 ->|
            ramp = torch.clamp(
                (
                    torch.arange(dim // 2, device=freqs.device).float()
                    - low
                )
                / max(high - low, 0.001),
                0,
                1,
            )

            # (C) 核心插值公式: 根据掩码，对基础频率进行改造
            # 数学计算： freqs_new = freqs * (1 - ramp + ramp / factor)
            #
            # 让我们代入上面的高、中、低三区看看这个公式的妙处：
            #
            # 1. 高频区 (ramp = 0.0):
            #    freqs * (1 - 0 + 0) = freqs * 1.0  ==> 频率完全不变！保留近距离语法的绝对敏锐度！
            #
            # 2. 低频区 (ramp = 1.0):
            #    freqs * (1 - 1 + 1 / factor) = freqs / factor ==> 频率直接放慢 factor 倍(除以16)！
            #    物理意义：长波长的旋转周期被拉长了 16 倍，完美容纳 32K 级别的超长上下文。
            #
            # 3. 中频区 (ramp = 0.33):
            #    freqs * (1 - 0.33 + 0.33 / 16) ≈ freqs * 0.69  ==> 旋转速度温和地放慢一点点。
            # -----------------------------------------------------------------
            freqs = freqs * (1 - ramp + ramp / factor)

    # -----------------------------------------------------------------
    # 3. 构造全位置角度矩阵 (torch.outer 向量外积大表)
    # 物理意义: 让“每一个位置的行”去乘“每一个速度的列”，计算出当前词在当前维度要旋转的累计弧度。
    # 矩阵形状变化: 列向量 t [end, 1]  乘  行向量 freqs [1, dim//2]  =>  二维大矩阵 [end, dim//2]
    #
    # 形象化矩阵点乘对齐图 (假设已计算出 freqs=[1.0, 0.1, 0.01, 0.001], 句子长度 end=3):
    #
    #                 平面0(\omega=1.0)   平面1(\omega=0.1)   平面2(\omega=0.01)   平面3(\omega=0.001)
    #  t=0 (第1个词): [   0 * 1.0   ,     0 * 0.1   ,     0 * 0.01  ,     0 * 0.001   ]
    #  t=1 (第2个词): [   1 * 1.0   ,     1 * 0.1   ,     1 * 0.01  ,     1 * 0.001   ]
    #  t=2 (第3个词): [   2 * 1.0   ,     2 * 0.1   ,     2 * 0.01  ,     2 * 0.001   ]
    #
    #  执行 torch.outer 后，最终的角度矩阵 freqs 内部形态为：
    #  freqs = [
    #    [ 0.0,   0.0,   0.0,   0.0   ],   <-- 绝对位置 t=0 处，4个平面各自应旋转的角度
    #    [ 1.0,   0.1,   0.01,  0.001 ],   <-- 绝对位置 t=1 处，4个平面各自应旋转的角度
    #    [ 2.0,   0.2,   0.02,  0.002 ]    <-- 绝对位置 t=2 处，4个平面各自应旋转的角度
    #  ]
    # -----------------------------------------------------------------
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()

    # -----------------------------------------------------------------
    # 4. 维度复制平铺 (为下一阶段的 Half-Split 并行运算做对齐准备)
    # 为什么要在最后一个维度 cat 拼接两次？
    # 答：因为我们要把维度从 [end, dim//2] 扩容到 [end, dim]，让左半边特征和右半边特征(不理解啥是左半边/右半边的请先看下一个函数注释)能成对对齐相同的 角度。
    #
    # 形象化横向拼接结果展示 (以第 t=1 行词为例，特征维度从 4 扩展为 8)：
    #  原始单行: [ \theta_0,   \theta_1,   \theta_2,   \theta_3 ]
    #  torch.cat 后的 freqs_cos[1] 内部结构为：
    #            |<------- 左半边 4 维 ------>| |<------- 右半边 4 维 ------>|
    #  cos_row = [ cos(\theta_0), ..., cos(\theta_3) | cos(\theta_0), ..., cos(\theta_3) ]
    # -----------------------------------------------------------------
    freqs_cos = (
        torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
        * attn_factor
    )
    freqs_sin = (
        torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
        * attn_factor
    )

    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """【第二阶段：利用 Half-Split 伴随变换将角度拧入向量】

    为什么大模型工程落地时，不采用数学论文里的“相邻两两一组” [x0, x1] 旋转？
    
    【1. 理论 vs 工程 队形大对比】
    
    ● 论文原始理论（相邻组队法）：
      X = [ x0, x1,  x2, x3,  x4, x5,  x6, x7 ]
            └───┘    └───┘    └───┘    └───┘  
            平面0     平面1    平面2     平面3
      ❌ 致命缺点：在 GPU 里，想要把奇数位和偶数位抽出来交叉做 (x0*cos - x1*sin)，
         需要频繁地做非连续内存切片（Stride 极其恶心），导致 GPU 显存带宽直接拉跨。

    ● 实际工程落地（轴向对折法）：
      例如有一个维度为 8 的向量,从正中间“啪”地一刀切开，分成【左大组】和【右大组】，上下对齐组队：
      
      左大组 (X_left)  : [  x0,   x1,   x2,   x3  ]
                           │     │     │     │    <-- 垂直对应的两个元素，组成旋转平面！
      右大组 (X_right) : [  x4,   x5,   x6,   x7  ]
                          平面0  平面1  平面2  平面3
                          
      ➔ 对应关系：(x0, x4) 成了平面0，(x1, x5) 成了平面1 ...
      ⭕ 巨大优势：左半边是一块连续内存，右半边也是一块连续内存！
         GPU 只需要大口大口地把整块连续内存捞出来，直接进行“左矩阵 * cos + 右矩阵 * sin”的整块并行大矩阵运算
    其矩阵并行运算最终等价于：
        左半边输出 = q_left  * cos - q_right * sin
        右半边输出 = q_right * cos + q_left  * sin
    """

    def rotate_half(x):
        """【核心伴随重排：利用切片拼接，产生负号并交换位置】

        形象化重排形态展示 (假设当前处理的输入向量 x 长度 dim=8):
        原始向量 x   = [   x0,   x1,   x2,   x3  |   x4,   x5,   x6,   x7  ]
                      |<---- 左半边 x_left ---->| |<---- 右半边 x_right --->|

        1. 右半边整体切片加负号挪到最左边:  [-x4, -x5, -x6, -x7]
        2. 左半边原始切片原封不动移到右边:  [ x0,  x1,  x2,  x3]

        torch.cat 拼接返回后的伴随矩阵形态：
        返回新矩阵  = [  -x4,  -x5,  -x6,  -x7  |   x0,   x1,   x2,   x3  ]
        """
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]),
            dim=-1,
        )

    # 5. 广播对齐：将预计算的 [seq_len, dim] 矩阵扩容为 [seq_len, 1, dim] 以适配多头注意力 q/k
    cos_padded = cos.unsqueeze(unsqueeze_dim)
    sin_padded = sin.unsqueeze(unsqueeze_dim)

    # -----------------------------------------------------------------
    # 6. 核心并行矩阵加法（精妙之处：利用对位相加，完美复现旋转公式！）
    #
    # 我们把这一整行点乘加法拆开，看它在内部发生了什么（以单词、前四个维度和后四个维度的对应元素为例）：
    #
    # 矩阵 1 (q * cos_padded):
    #   [  q0 * cos(\theta_0) ,  ...  |  q4 * cos(\theta_0) ,  ...  ]
    #
    # 矩阵 2 (rotate_half(q) * sin_padded)  注意：此处带入了刚才 rotate_half 产生的负号与位置对调：
    # + [ -q4 * sin(\theta_0) ,  ...  |  q0 * sin(\theta_0) ,  ...  ]
    # -----------------------------------------------------------------
    # 对应位相加后的最终结果矩阵 q_embed 为：
    #   [  (q0*cos\theta_0 - q4*sin\theta_0)  |  (q4*cos\theta_0 + q0*sin\theta_0)  ]
    #      ^                                     ^
    #      │                                     │
    #      └── 完美对应传统 2D 旋转后的新左轴 x0'     └── 完美对应传统 2D 旋转后的新右轴 x1'
    # -----------------------------------------------------------------
    q_embed = ((q * cos_padded) + (rotate_half(q) * sin_padded)).to(q.dtype)
    k_embed = ((k * cos_padded) + (rotate_half(k) * sin_padded)).to(k.dtype)

    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    用于组查询注意力 (GQA)。
    当 Query 的头数多于 Key/Value 时，需要将 Key/Value 头重复扩充。
    - x: 输入的 Key 或 Value Tensor，形状为 [bs, seq_len, num_key_value_heads, head_dim]
    - n_rep: 重复次数 (num_attention_heads // num_key_value_heads)
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    # 在第3维之后插入新维度，通过 expand 复制，最后 reshape 回融合的头维度
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))

class Attention(nn.Module):
    """
    Multi-Query / Grouped-Query Attention 模块。
    相比于 nanoGPT 中的标准多头注意力 (MHA)，该模块：
    1. 允许 Key/Value 头数少于 Query 头数 (GQA)，以大幅缩减推理时 KV Cache 的显存开销。
    2. 支持在单 Token 推理时使用键值对缓存 (past_key_value) 以加速自回归生成。
    3. 支持 PyTorch 的高效 Flash Attention (scaled_dot_product_attention)。
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads # Query 总头数
        self.n_local_kv_heads = self.num_key_value_heads # Key/Value 总头数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads # GQA 中每个 KV 头需要重复的倍数
        self.head_dim = config.head_dim
        self.is_causal = True
        
        # 定义投影映射
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        
        # 旋转编码前对 Query/Key 进行层归一化 (防止多层残差累积引发的值溢出)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        # 判断并启用 PyTorch 内置的闪电注意力算法 (Flash Attention 2 / SDPA)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        # 1. 投影映射得到 Q, K, V 向量
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        # 2. 将其 reshape 为多头形式，形状：[Batch, SeqLen, Heads, HeadDim]
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        
        # 3. 对 Q 和 K 进行 RMSNorm 归一化
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        
        # 4. 注入旋转位置编码 (RoPE)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        
        # 5. KV Cache 核心逻辑：如果在推理且存在历史 KV，则拼接历史缓存
        # 拼接方向是 seq_len 维度 (dim=1)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
            
        # 如果模型处于生成阶段，记录当前完整的 K 和 V 矩阵作为下一轮的缓存
        past_kv = (xk, xv) if use_cache else None
        
        # 6. GQA 逻辑：将 K, V 的头数使用 repeat_kv 重复复制到与 Q 相同的头数。
        # 重复后转置为 [Batch, Heads, SeqLen, HeadDim]
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        
        # 7. 计算 Attention 权重与加权和
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            # 首选高性能的 Flash Attention（使用 C++ 融合内核，不产生庞大的中间 softmax 矩阵，极大减少内存访问）
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            # 兼容性手动计算注意力 (经典自回归生成，seq_len==1 时只能手动走这里)
            # 计算 QK^T / sqrt(d_k)
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.is_causal: 
                # 注入因果掩码 (Causal Mask)，使当前 Token 无法看后续的 Token (只对当前生成步的 seq_len 做上三角无穷小覆盖)
                scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            if attention_mask is not None: 
                # 注入填充掩码 (Padding Mask)，过滤掉无效的 Padding 填充 Token
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
                
            # Softmax 与加权 V 向量得到输出
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
            
        # 8. 重整输出多头形状并投影输出
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

class FeedForward(nn.Module):
    """
    SwiGLU 前馈网络 (Feed-Forward Network)。
    相比于 nanoGPT (GPT-2) 中传统的 MLP：
    - nanoGPT: FC1 (GELU) -> FC2
    - MiniMind: SwiGLU(x) = ( (x @ W_gate) * Sigmoid(x @ W_gate) ) * (x @ W_up)
    然后再对结果进行一次线性降维回原来的 hidden_size。
    这种设计通过包含门控控制项，能更好地保留和过滤通道特征。
    """
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False) # 门控权重线性层
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False) # 整合降维线性层
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)   # 激活特征线性层
        self.act_fn = ACT2FN[config.hidden_act] # 默认为 SiLU 激活函数

    def forward(self, x):
        # 门控分支激活与上投影分支点乘，再利用 down_proj 降维回原来的维度。
        # 数学等价于：DownProj( SiLU( GateProj(x) ) * UpProj(x) )
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MOEFeedForward(nn.Module):
    """
    混合专家门控网络 (MOE - Mixture of Experts)。
    当 use_moe = True 时替代标准的 FeedForward 模块。
    通过一个简单的门控网络 (Router) 决定当前 Token 交给哪一个专家模型 (FeedForward) 进行计算。
    从而在基本不增加每次前向计算时间的前提下，实现模型容量成倍扩展。
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 【路由器】：一个简单的线性层。
        # 它的输入是 Token 的维度 (hidden_size)，输出是每个专家的打分 (num_experts)。
        # 比如有 8 个专家，它就输出 8 个数字。
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        
        # 【专家列表】：用 ModuleList 装了好多台“一模一样”的 FeedForward 机器
        self.experts = nn.ModuleList([
            FeedForward(config, intermediate_size=config.moe_intermediate_size) 
            for _ in range(config.num_experts)
        ])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        # 把输入从三维 [批次, 句子长度, 维度] 拍扁成二维 [Token总数, 维度]
        # 因为路由器和专家都是针对单个 Token 进行处理的
        x_flat = x.view(-1, hidden_dim) 
        
        # ----------------------------------------------------
        # 步骤 1: 路由器发工单
        # ----------------------------------------------------
        # self.gate(x_flat) 算出每个 Token 对每个专家的原始得分
        # F.softmax 让得分变成概率（相加等于 1）
        # scores 的形状是: [Token总数, 专家总数]
        scores = F.softmax(self.gate(x_flat), dim=-1) 
        
        # ----------------------------------------------------
        # 步骤 2 & 3: 选出最强的专家 (Top-K)
        # ----------------------------------------------------
        # torch.topk 选出概率最大的前 K 个。这里 K=1
        # topk_weight: 这个专家的权重（概率值）
        # topk_idx: 这个专家的编号（比如是 0 号专家还是 3 号专家）
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        """
        topk_idx =
        [
        [2, 1],    token0 -> expert2, expert1
        [0, 2],    token1 -> expert0, expert2
        [1, 3],    token2 -> expert1, expert3
        ]

        topk_weight =
        [
        [0.7, 0.3],
        [0.8, 0.2],
        [0.6, 0.4],
        ]
        """
        # 这里的 topk_weight 是路由器(Router)分配给 Top-K 专家的打分(概率标量)，而非专家内部的矩阵参数。
        # 如果 K > 1，截取出的 Top-K 概率之和通常小于 1。为了防止加权求和时丢失信息（导致特征数值整体变小），
        # 需要对这 K 个得分重新归一化 (Re-normalize)，使它们的和强制等于 1。
        # (加上 1e-20 是为了防止极端情况下分母为 0 导致报错)
        # 注意：如果 K=1，这里相当于自己除以自己，数值上没有实质变化。
        if self.config.norm_topk_prob: 
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
            
        # 初始化一个全 0 的大矩阵 y，用来装所有专家计算完后的最终总输出
        y = torch.zeros_like(x_flat)

        # ----------------------------------------------------
        # 步骤 4: 遍历每个专家，把属于它的活干完，合并结果
        # ----------------------------------------------------
        for i, expert in enumerate(self.experts):
            # mask 的意思是：看看有哪些 Token 的 topk_idx 正好等于当前专家的编号 i
            mask = (topk_idx == i)
            
            if mask.any(): # 如果有 Token 被分配给了当前这个专家 i
                # 找到这些 Token 在大矩阵中的具体行索引（位置）
                token_idx = mask.any(dim=-1).nonzero().flatten()
                
                # 提取出这些 Token 对应的路由权重
                weight = topk_weight[mask].view(-1, 1)
                
                # 核心计算：
                # 1. x_flat[token_idx]：把属于这个专家的 Token 筛选出来。
                # 2. expert(...)：让这个专家只计算这批属于它的 Token（高效！没分到的不计算）。
                # 3. * weight：计算结果乘以它应得的路由权重。
                # 4. y.index_add_(...)：把计算好的结果，精准地“累加回”总输出矩阵 y 的对应位置上。
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
                
            elif self.training:
                # 这是一个防报错的 Trick（冷知识）：
                # 在多卡训练时，如果某个专家很不幸一个 Token 也没分到，它就不会参与计算。
                # PyTorch 的分布式训练（DDP）会报错说“这个专家怎么没梯度？”
                # 所以这里让它乘以 0 产生一个虚拟梯度，骗过 PyTorch，防止报错。
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
                
        # ----------------------------------------------------
        # 步骤 5: 负载均衡控制 (Auxiliary Loss)
        # ----------------------------------------------------
        if self.training and self.config.router_aux_loss_coef > 0:
            # 计算每个专家实际分到的 Token 比例 (load) 和 概率平均值 (scores)
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            # 算出一个惩罚项（越不均匀，这个 loss 就越大），强迫路由器“雨露均沾”
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
            
        # 最后，把拍扁的二维矩阵还原回原本的三维形状 [BatchSize, SeqLen, HiddenDim] 返回
        return y.view(batch_size, seq_len, hidden_dim)

class MiniMindBlock(nn.Module):
    """
    MiniMind Transformer 层 (等同于 Llama Block)。
    结构为 Pre-LN (在 Attention 和 MLP 前都使用 RMSNorm)。
    计算流：
    1. x_norm = input_layernorm(x)
    2. attn_out, past_kv = self_attn(x_norm)
    3. x = x + attn_out (残差连接)
    4. x_norm2 = post_attention_layernorm(x)
    5. mlp_out = mlp(x_norm2) (如果是 MoE 则走 MOEFeedForward)
    6. x = x + mlp_out (残差连接)
    """
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        # 1. 预归一化 -> 经过注意力计算
        attn_output, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        # 2. 注意力残差加和
        hidden_states = residual + attn_output
        
        # 3. 预归一化 -> 经过前向 MLP 计算并累加残差
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        
        return hidden_states, present_key_value

class MiniMindModel(nn.Module):
    """
    MiniMind 模型核心主体（不包括最顶层的分类 LM Head 输出投影层）。
    负责：
    1. 将输入 Token ID 转化为密集词嵌入向量 (embed_tokens)。
    2. 管理所有 Transformer Block 层。
    3. 维护并管理缓存的旋转位置编码 (freqs_cos, freqs_sin)。
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size) # 词嵌入层
        self.dropout = nn.Dropout(config.dropout)
        # 串联堆叠 N 层 Transformer Blocks
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # 最后一层 RMSNorm
        
        # 预计算整个 RoPE 位置频率表，并注册为不可更新的 buffer
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        # 处理 past_key_values 缓存结构
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        
        # 计算当前自回归生成的起点位置。若存在历史 KV 缓存，说明正处理第 t 个生成的 Token，首位置等于缓存长度
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        
        # Token 转换并进行 Dropout
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        
        # 应对元设备 (meta-device) 重启后缓冲丢失的自动检测及补全 (兼容 Transformers 5.x)
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
            
        # 截取当前推理片段对应的位置旋转编码矩阵
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        
        presents = []
        # 逐层传播计算
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
            
        hidden_states = self.norm(hidden_states) # 终极 LayerNorm 归一化
        
        # 汇总所有激活 MoE 专家的辅助门控均衡 Loss
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍
#                                     MiniMindForCausalLM (最终因果语言模型)
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """
    最顶层的大语言模型外壳，包含了 LM Head 投影层。
    实现：
    1. 自回归损失函数（CrossEntropyLoss），并利用 shift 偏移预测下一个 Token。
    2. 带 KV Cache 及各种解码策略（如 Temperature、Top-K、Top-P、重复惩罚项）的完整自回归 `generate` 对话流。
    """
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        # LM Head: 接收 hidden_size 维特征，输出 vocab_size 维的 logits 概率空间映射
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 是否绑定权重以大幅节省资源（嵌入层矩阵与LM Head分类矩阵共享相同的显存空间）
        if self.config.tie_word_embeddings: 
            self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        # 1. 经过核心底座计算，获取特征状态向量、新的 KV 缓存以及 MoE 路由损失
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        
        # 2. 为节省计算开销，只取末端需要计算 Logits 的切片（如在计算 loss 或只计算最后预测步时）
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        
        loss = None
        if labels is not None:
            # 经典因果自回归移位 loss 计算：
            # 预测下一个 Token 需要将 input 对应的 logits 向前偏移一位，labels 向后偏移一位对齐
            # 比如输入 A B C -> 预测 B C D，则 logits[A,B] 对应计算 labels[B,C] 的交叉熵
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            # 这里的 ignore_index=-100 极其关键！在 SFT 时被用来过滤掉 Prompt 处的 loss 计算
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
            
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
    
    # 讨论和优化参考：https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        """
        高度优化的增量文本生成函数。
        - inputs: 输入的初始提示词 Token IDs [Batch, SeqLen]
        - max_new_tokens: 最多生成多少个新 Token
        - temperature: 温度系数（调节采样随机度，值越小越偏向贪婪预测，越大越天马行空）
        - top_p / top_k: 采样截断阈值，避免生成荒谬词
        - repetition_penalty: 重复词惩罚系数（>1.0 可以抑制模型无限循环生成重复短句）
        - streamer: 字符流式实时输出机制句柄
        - use_cache: 启用 KV Cache 极速推理
        """
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        
        if streamer: streamer.put(input_ids.cpu())
        
        for _ in range(max_new_tokens):
            # 获取当前 KV 缓存历史长度
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            
            # 【KV Cache 性能质变点】：
            # 如果存在缓存，我们不再将之前的全部 Token 重新送入模型，而只输入最新产生的一个 Token：input_ids[:, past_len:]
            # 模型在前向中自动提取旧 KV 拼接，大大减少重复矩阵乘法
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            
            # 扩展注意力遮罩以容纳最新产生的 Token 位置
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            
            # 提取最后一个位置 of logits，除以温度进行缩放
            logits = outputs.logits[:, -1, :] / temperature
            
            # 惩罚项控制：对已经出现过的 Token 的概率强行进行除以惩罚系数的衰减
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i]) # 1. 找出这句里已经出现过哪些词
                    score = logits[i, seen]           # 2. 把这些老词当前的得分抽出来
                    # 3. 强行降低它们的得分
                    logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
                    
            # Top-K 采样过滤掉分值太低的候选词
            if top_k > 0: 
                # 1. 找到第 K 名的得分是多少
                k_th_score = torch.topk(logits, top_k)[0][..., -1, None]
                # 2. 谁的得分比第 K 名还低，直接打入十八层地狱（变成负无穷）
                logits[logits < k_th_score] = -float('inf')
                
            # Top-P (Nucleus) 核采样过滤累计概率外的多余词
            if top_p < 1.0:
                # 1. 把所有词按得分从高到低排个序
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                
                # 2. 算一下从高到低的“累计概率”
                # 看看加到哪个词的时候，总概率超过了 top_p（比如 0.9）
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                
                # 3. 极其精妙的一步：把 Mask 矩阵向右平移一位，并确保第一个词永远不被屏蔽
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                
                # 4. 把这个“淘汰名单”映射回原本未排序的词表里，把落榜者全部变成负无穷
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
                
            # 从过滤后的概率分布中采样或者直接取最大分值词 (Argmax)
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            
            # 如果对应批次已经产生 EOS 结束符，则后续强制填充 EOS
            if eos_token_id is not None: 
                next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
                
            # 拼接最新 Token 成为历史
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            
            if streamer: streamer.put(next_token.cpu())
            
            # 如果所有并发批次全都输出了结束符，提前终止循环
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
                
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids