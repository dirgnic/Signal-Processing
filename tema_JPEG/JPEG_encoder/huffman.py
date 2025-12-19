"""
Huffman Coding
Bonus: Lossless compression of quantized DCT coefficients
"""

import heapq
import numpy as np
from collections import Counter


class HuffmanNode:
    """Node in Huffman tree"""
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(data):
    """Build Huffman tree from data"""
    freq = Counter(data.flatten().astype(int))
    
    heap = [HuffmanNode(symbol=s, freq=f) for s, f in freq.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)
    
    return heap[0] if heap else None


def get_huffman_codes(root, prefix='', codes=None):
    """Extract Huffman codes from tree"""
    if codes is None:
        codes = {}
    
    if root is None:
        return codes
    
    if root.symbol is not None:
        codes[root.symbol] = prefix if prefix else '0'
    
    get_huffman_codes(root.left, prefix + '0', codes)
    get_huffman_codes(root.right, prefix + '1', codes)
    
    return codes


def compute_entropy(data):
    """Compute entropy of data in bits per symbol"""
    freq = Counter(data.flatten().astype(int))
    total = sum(freq.values())
    entropy = -sum((f/total) * np.log2(f/total) for f in freq.values())
    return entropy


def compute_average_code_length(codes, data):
    """Compute average Huffman code length"""
    freq = Counter(data.flatten().astype(int))
    total = sum(freq.values())
    avg_len = sum(len(codes[s]) * freq[s] for s in freq) / total
    return avg_len
