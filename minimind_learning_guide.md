# 从 nanoGPT 到 MiniMind 的进阶学习指南

恭喜你完成了 **nanoGPT** 的学习！nanoGPT 是理解经典 GPT-2 架构的极佳起点。而 **MiniMind** 则带你跨入现代大语言模型（如 LLaMA、DeepSeek、Mistral）的技术殿堂。

为了帮助你建立清晰的知识网络，本指南将从**架构升级**、**学习路线大纲**、**核心技术重点**三个维度，帮助你快速消化 MiniMind 的工程实现。

---

## 一、 nanoGPT vs MiniMind 架构对比图

| 维度 | nanoGPT (GPT-2 经典架构) | MiniMind (LLaMA 现代架构) | 升级核心原因与优势 |
| :--- | :--- | :--- | :--- |
| **归一化层 (Norm)** | LayerNorm (均值和方差归一化) | RMSNorm (均归一化，只计算均方根) | 计算量减少约 7%~10%，且保留相同的梯度稳定效果。 |
| **位置编码** | Learned Absolute Position Embedding | RoPE (旋转位置编码) | 引入相对位置关系；允许通过插值/外推（如 YaRN）直接扩展上下文长度。 |
| **注意力机制** | MHA (Multi-Head Attention) | GQA (Grouped-Query Attention) | 共享 Key/Value 磁头，极大降低推理时 **KV Cache** 的显存消耗。 |
| **激活函数 & MLP** | GELU + 两个线性层 (FC1 -> FC2) | SwiGLU + 三个线性层 (Gate, Up, Down) | 引入门控机制，GLU 在大模型中被证实拥有更好的表达和泛化能力。 |
| **生成策略** | 纯自回归（每次全部重新计算） | KV Cache (键值缓存) + 增量生成 | 推理速度提升数倍至数十倍，避免重复计算历史 Token 的 Key/Value。 |
| **专家混合 (MoE)** | 无 (全稠密模型) | 门控稀疏 Mixture of Experts (可选) | 保持推理计算量（Active Params）较低的同时，通过多专家极大提升模型容量。 |

---

## 二、 学习路线大纲 (建议阅读顺序)

为了由浅入深地掌握 MiniMind，建议按照以下顺序阅读和调试代码：

### 1. 核心模型定义与升级 (第一周)
* **核心文件**：[model_minimind.py](file:///c:/Users/32770/Desktop/minimind/model/model_minimind.py)
* **目标**：理解 RMSNorm、RoPE 预计算与应用、Grouped-Query Attention (GQA)、SwiGLU 前向传播、KV Cache 在 `generate` 中的增量推理机制。
* **MoE (可选)**：理解门控网络（Router）是如何对 Token 进行打分并分配给不同 Expert 的。

### 2. 数据处理与输入构建 (第二周)
* **核心文件**：[lm_dataset.py](file:///c:/Users/32770/Desktop/minimind/dataset/lm_dataset.py)
* **目标**：
  * **预训练数据**：`PretrainDataset` 的自回归掩码。
  * **SFT 指令微调数据**：`SFTDataset` 中的 `generate_labels` 是如何做到**只对 Assistant 的回答计算 Loss，而对 User Prompt 和 System Prompt 进行掩码 (-100)** 的（这是从 nanoGPT 字符生成过渡到 Chat 对话模型的关键点！）。

### 3. 基座训练与指令微调 (第三周)
* **核心文件**：
  * [train_pretrain.py](file:///c:/Users/32770/Desktop/minimind/trainer/train_pretrain.py) (预训练)
  * [train_full_sft.py](file:///c:/Users/32770/Desktop/minimind/trainer/train_full_sft.py) (全量 SFT)
* **目标**：了解 PyTorch 中的分布式训练（DDP）、混合精度训练（AMP）、梯度累积（Gradient Accumulation）、余弦退火学习率调度。

### 4. 轻量化微调：LoRA (第四周)
* **核心文件**：[model_lora.py](file:///c:/Users/32770/Desktop/minimind/model/model_lora.py)、[train_lora.py](file:///c:/Users/32770/Desktop/minimind/trainer/train_lora.py)
* **目标**：
  * 理解旁路低秩矩阵 $A$ 和 $B$ 的原理。
  * 掌握如何通过 Python 动态猴子补丁（Monkey Patching）在不改动原始模型结构的前提下，将 LoRA 层注入到 `Linear` 层中。
  * 学习 LoRA 权重的保存与合并（Merge Lora）机制。

### 5. 人类偏好对齐与强化学习 (第五周)
* **核心文件**：
  * [train_dpo.py](file:///c:/Users/32770/Desktop/minimind/trainer/train_dpo.py) (偏好对齐：Direct Preference Optimization)
  * [train_ppo.py](file:///c:/Users/32770/Desktop/minimind/trainer/train_ppo.py) (在线强化学习：PPO)
  * [train_grpo.py](file:///c:/Users/32770/Desktop/minimind/trainer/train_grpo.py) (DeepSeek 提出的强化学习：GRPO，无需 Critic 模型)
* **目标**：理解偏好损失（Implicit Reward）的计算方法；重点理解 GRPO 是如何通过组内相对平均得分（Group Relative Advantage）替代 Critic 网络的。

---

## 三、 核心技术内容重点剖析

### 1. 旋转位置编码 (RoPE - Rotary Position Embedding)
* **相比 nanoGPT 的改进**：nanoGPT 用的是一维的绝对位置索引，模型很难外推到更长的文本。
* **原理**：RoPE 将 2D 的位置旋转作用在 Query 和 Key 上。假设第 $m$ 个 Token，对应的向量分量会在 2D 平面上旋转一个角度 $m\theta_i$。
* **YaRN 缩放**：当我们需要推理比训练更长的文本时，通过对 $\theta$ 参数进行动态频率插值和衰减，即 YaRN (Yet another RoPE extensioN)，可以无损或低损扩展 Context Window。

### 2. 组查询注意力 (GQA - Grouped-Query Attention)
* **机制**：
  * Multi-Head Attention (MHA): $N$ 个 Q 头，对应 $N$ 个 K 头和 $N$ 个 V 头。
  * Multi-Query Attention (MQA): $N$ 个 Q 头，所有 Q 头共享同一个 K 头和 V 头。
  * Grouped-Query Attention (GQA): 将 Q 头分组，每组共享一个 K 头和 V 头（MiniMind 默认 `num_attention_heads=8`，`num_key_value_heads=4`，即 $8 / 4 = 2$ 个 Q 头为一组共享一对 KV 头）。
* **重要性**：在 autoregressive 生成过程中，我们需要保存前面的 Token 产生的 KV，称为 **KV Cache**。在多轮对话中，KV Cache 占用大量显存。使用 GQA 可以将 KV Cache 的显存开销减少到原来的几分之一。

### 3. 门控线性单元激活 (SwiGLU)
* **结构**：
  $$\text{SwiGLU}(x) = \text{Swish}(xW_{\text{gate}}) \otimes xW_{\text{up}} \cdot W_{\text{down}}$$
* **代码实现**：
  `self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))`
* **作用**：相比于经典 Transformer 的 MLP 层（先扩大再缩小，中间加一个非线性层），SwiGLU 增加了门控投影项 `gate_proj`。实验表明，这种带有门控因子的前馈网络更有利于网络收敛和细节记忆。

### 4. KV Cache (键值缓存) 推理加速
* **为何必须有 KV Cache**：GPT 生成时，第 $t$ 步只依赖 $1 \dots t$ 步的输出。如果不缓存 KV，生成第 $t+1$ 个 Token 时，我们需要把前面的 $1 \dots t$ 个 Token 重新输入模型过一遍 Attention，计算复杂度是 $\mathcal{O}(L^2)$。
* **缓存后**：每次 `generate` 时，如果传入 `past_key_values`，我们只需将**最新生成的一个 Token** 送入模型，计算它的 Q、K、V。然后将新的 K、V 与之前缓存的 `past_key_values` 在序列长度维度（`dim=1`）进行拼接（`torch.cat`）。此时计算注意力时只算一行的 Q 与所有旧列的 K 乘积，极大地提高了生成速度。

### 5. 掩码交叉熵损失 (Label Masking in SFT)
* **为什么要 Mask**：在指令微调（SFT）中，我们的输入是 `[User Prompt] [Assistant Response]`。我们不希望模型去学习如何背诵用户的输入（即不计算 User Prompt 处的 loss），我们只希望模型学会**在用户说完话后，给出正确的 Assistant 回答**。
* **如何实现**：利用 PyTorch CrossEntropyLoss 的 `ignore_index=-100` 特性。
  * 在 `lm_dataset.py` 中，遍历 Tokenized 后的序列。
  * 寻找 Assistant 开始标记 `assistant\n` 到结束标记 `\n`。
  * 将不属于 Assistant 回答区域的 Token 的标签（labels）设为 `-100`。
  * 计算 Loss 时，-100 处的损失将被忽略，只有真实的 Assistant 回答部分会贡献梯度。

---

> [!TIP]
> 接下来，我将直接开始在代码文件中添加这些关键知识点对应的中文注释，方便您在阅读源码时随时印证！
