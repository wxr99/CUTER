import random
import torch


def count_memory(memory):
    # return the class index and corresponding number & class index of majority class
    elements, counts = torch.unique(memory, return_counts=True)
    index = torch.argmax(counts)
    class_list = {elements[i].item(): counts[i].item() for i in range(len(elements))}
    # print(class_list)
    # print(elements[index].item())
    return class_list, elements[index].item()


def refresh_memory(label, class_idx):
    # return the index of a random index of label whose class index is the given class_idx
    index = random.randint(0, len(label)-1)
    while label[index] != class_idx:
        index = random.randint(0, len(label)-1)
    return index