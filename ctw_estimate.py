import numpy as np
from collections import defaultdict
import hashlib
import math

class CTWEntropy:
    def __init__(self, max_symbol, max_depth=5, alpha=0.1, prune_threshold=5):
        self.max_symbol = max_symbol          # 符号集大小
        self.max_depth = max_depth            # 最大上下文深度
        self.alpha = alpha                    # 自适应平滑系数
        self.prune_threshold = prune_threshold# 剪枝阈值
        
        # 压缩存储的上下文树(哈希+低频剪枝)
        self.context_tree = defaultdict(lambda: defaultdict(int))
        self.symbol_counts = defaultdict(int) # 全局符号频率
        
        # 动态权重缓存(提升计算效率)
        self.weight_cache = {}
    def _hash_context(self, context):
        """哈希压缩长上下文[8,6](@ref)"""
        return hashlib.md5(str(context).encode()).hexdigest()[:8]
    def update_counts(self, sequence):
        """动态更新多阶上下文频率(带剪枝)[4,5](@ref)"""
        for i in range(len(sequence)):
            symbol = sequence[i]
            self.symbol_counts[symbol] += 1
            
            # 构建多阶上下文树(1~max_depth)
            for depth in range(1, self.max_depth+1):
                if i >= depth:
                    context = tuple(sequence[i-depth:i])
                    ctx_hash = self._hash_context(context)
                    self.context_tree[ctx_hash][symbol] += 1
                    
            # 定期剪枝低频节点
            if i % 5000 == 0:  
                self._prune_low_freq()
    def _prune_low_freq(self):
        """低频上下文剪枝[5,9](@ref)"""
        for ctx in list(self.context_tree.keys()):
            total = sum(self.context_tree[ctx].values())
            if total < self.prune_threshold:
                del self.context_tree[ctx]

    def adaptive_kt_estimator(self, counts, total):
        """自适应KT平滑[3,11](@ref)"""
        dynamic_alpha = self.alpha / (1 + math.log(self.max_symbol))  # 动态调整系数
        return (counts + dynamic_alpha) / (total + self.max_symbol * dynamic_alpha)

    def _dynamic_weighting(self, global_entropy, cond_entropy):
        """基于熵差的动态权重分配[4,2](@ref)"""
        weight = global_entropy / (global_entropy + cond_entropy + 1e-10)
        return min(max(weight, 0.2), 0.8)  # 限制权重范围防止极端值

    def calculate_entropy_rate(self, sequence):
        """改进的熵率计算流程"""
        self.update_counts(sequence)
        total_log_prob = 0.0
        n = len(sequence)
        
        for i in range(n):
            symbol = sequence[i]
            
            # 全局概率估计
            global_prob = self.adaptive_kt_estimator(
                self.symbol_counts[symbol], 
                sum(self.symbol_counts.values())
            )
            
            # 多阶条件概率加权
            cond_probs = []
            for depth in range(1, min(i, self.max_depth)+1):
                context = tuple(sequence[i-depth:i])
                ctx_hash = self._hash_context(context)
                ctx_total = sum(self.context_tree[ctx_hash].values())
                cond_prob = self.adaptive_kt_estimator(
                    self.context_tree[ctx_hash].get(symbol, 0),
                    ctx_total
                ) if ctx_total > 0 else global_prob
                cond_probs.append(cond_prob)
            
            # 动态权重融合(最高3阶)
            weighted_prob = global_prob
            if cond_probs:
                weights = [0.5**d for d in range(1, len(cond_probs)+1)]  # 指数衰减权重
                weights = [w/sum(weights) for w in weights]
                weighted_prob = sum(w*p for w,p in zip(weights, cond_probs))
            
            # 数值稳定性处理[9](@ref)
            weighted_prob = max(weighted_prob, 1e-20)
            total_log_prob += math.log2(weighted_prob) 
        return -total_log_prob / n