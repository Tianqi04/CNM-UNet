import torch
import numpy as np
from numpy.linalg import norm


class Memory(object):
    """
        Create the empty memory buffer
    """

    def __init__(self, size, dimension=1 * 3 * 512 * 512):
        self.memory = {}
        self.size = size
        self.dimension = dimension

    def reset(self):
        self.memory = {}

    def get_size(self):
        return len(self.memory)

# 由于batch_size = 8, 下面要改
    def push(self, keys, logits):
        B = keys.shape[0]
        if B == 1:
            for i, key in enumerate(keys):
                if len(self.memory.keys()) > self.size:
                    self.memory.pop(list(self.memory)[0])
                # 打印 key 的形状（测试）
                # print("self.dimension", self.dimension)
                # print("key 的形状:", key.shape)
                # print(f"logits[{i}] 的形状:", logits[i].shape)

                self.memory.update(
                    {key.reshape(self.dimension).tobytes(): (logits[i])})
        else:
            if len(self.memory.keys()) > self.size:
                self.memory.pop(list(self.memory)[0])
            # 打印 key 的形状（测试）
            # print("self.dimension", self.dimension)
            # print("key 的形状:", key.shape)
            # print(f"logits[{i}] 的形状:", logits[i].shape)

            self.memory.update(
                {keys.reshape(self.dimension).tobytes(): (logits)})


    def _prepare_batch(self, sample, attention_weight):
        attention_weight = np.array(attention_weight / 0.2)
        attention_weight = np.exp(attention_weight) / (np.sum(np.exp(attention_weight)))
        ensemble_prediction = sample[0] * attention_weight[0]
        for i in range(1, len(sample)):
            ensemble_prediction = ensemble_prediction + sample[i] * attention_weight[i]

        return torch.FloatTensor(ensemble_prediction)

    def get_neighbours(self, keys, k):
        """
        Returns samples from buffer using nearest neighbour approach
        从缓冲区中返回最近的 k 个邻居样本
        keys 是输入的特征向量列表，k 是需要获取的邻居数量
        """
        samples = []

        keys = keys.reshape(len(keys), self.dimension) # self.dimension 是在内存初始化时定义的每个特征向量的大小
        total_keys = len(self.memory.keys())
        self.all_keys = np.frombuffer(
            np.asarray(list(self.memory.keys())), dtype=np.float32).reshape(total_keys, self.dimension)

        # 找到其最近的 k 个邻居
        for key in keys:
            similarity_scores = np.dot(self.all_keys, key.T) / (norm(self.all_keys, axis=1) * norm(key.T))

            K_neighbour_keys = self.all_keys[np.argpartition(similarity_scores, -k)[-k:]]
            neighbours = [self.memory[nkey.tobytes()] for nkey in K_neighbour_keys]

            attention_weight = np.dot(K_neighbour_keys, key.T) / (norm(K_neighbour_keys, axis=1) * norm(key.T))
            batch = self._prepare_batch(neighbours, attention_weight)
            samples.append(batch)

        return torch.stack(samples), np.mean(similarity_scores)
