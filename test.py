import tiktoken
from GPTDataset import GPTDatasetV1
import torch
from torch.utils.data import Dataset, DataLoader

def create_dataloader_v1(txt, batch_size=4, max_length=256,stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")                       #A
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)      #B
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,                                        #C
        num_workers=0                                               #D
    )

    return dataloader

#A 初始化分词器
#B 创建GPTDatasetV1类
#C drop_last=True会在最后一批次小于指定的batch_size时丢弃该批次，以防止训练期间的损失峰值
#D 用于预处理的CPU进程数量


with open("the-verdict.txt", "r", encoding="utf-8") as f:
		raw_text = f.read()

dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)              #A
first_batch = next(data_iter)
print(first_batch)


#A 将数据加载器转换为 Python 迭代器，以便通过 Python 的内置 next() 函数获取下一个数据条目。
#结果 [tensor([[  40,  367, 2885, 1464]]), tensor([[ 367, 2885, 1464, 1807]])]