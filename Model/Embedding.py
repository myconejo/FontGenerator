import torch
#import torch.nn as nn
import numpy as np

#generate embedding for each category with random function.

def embedding_tensor_generate(category_num, embedding_dim, stddev=0.01):
    embedding_tensor = torch.randn([category_num,1,1,embedding_dim]) * stddev
    return embedding_tensor

def get_batch_embedding(batch_num, font_nums, embedding, embedding_dim):
    batch_embed = []
    for num in font_nums:
        batch_embed.append(embedding[num].numpy())
    batch_embed = torch.from_numpy(np.array(batch_embed))
    batch_embed = batch_embed.reshape(batch_num, embedding_dim, 1,1)
    return batch_embed

def save_category_embedding(category_num, embedding_dim, stddev=0.01):
    embedding_tensor = embedding_tensor_generate(category_num, embedding_dim, stddev)
    #torch.save(embedding_tensor)