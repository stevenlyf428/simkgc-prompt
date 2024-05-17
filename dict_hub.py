import json
import os
import glob

from transformers import AutoTokenizer

from config import args
from triplet import TripletDict, EntityDict, LinkGraph
from logger_config import logger

train_triplet_dict: TripletDict = None
all_triplet_dict: TripletDict = None
link_graph: LinkGraph = None
entity_dict: EntityDict = None
tokenizer: AutoTokenizer = None


def _init_entity_dict():
    global entity_dict
    if not entity_dict:
        print(args.valid_path)
        entity_dict = EntityDict(entity_dict_dir=os.path.dirname(args.valid_path))


def _init_train_triplet_dict():
    global train_triplet_dict
    if not train_triplet_dict:
        train_triplet_dict = TripletDict(path_list=[args.train_path])


def _init_all_triplet_dict():
    global all_triplet_dict
    if not all_triplet_dict:
        path_pattern = '{}/*.txt.json'.format(os.path.dirname(args.train_path))
        all_triplet_dict = TripletDict(path_list=glob.glob(path_pattern))


def _init_link_graph():
    global link_graph
    if not link_graph:
        link_graph = LinkGraph(train_path=args.train_path)


def get_entity_dict():
    _init_entity_dict()
    return entity_dict


def get_train_triplet_dict():
    _init_train_triplet_dict()
    return train_triplet_dict


def get_all_triplet_dict():
    _init_all_triplet_dict()
    return all_triplet_dict


def get_link_graph():
    _init_link_graph()
    return link_graph


def build_tokenizer(args):
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        tokenizer.add_special_tokens({'additional_special_tokens': ['[PROMPT]']})
        pseudo_token_id = tokenizer.get_vocab()['[PROMPT]']
        logger.info('Build tokenizer from {}'.format(args.pretrained_model))


def get_tokenizer():
    if tokenizer is None:
        build_tokenizer(args)
        # global tokenizer
        # if tokenizer is None:
        #     tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        #     logger.info('Build tokenizer from {}'.format(args.pretrained_model))
    return tokenizer

def get_relation2id_dict(args):
    global relation2id_dict
    # 文件路径
    # file_path = '/mnt/data/data/home/liuyafei/论文2/temp/SimKGC-prompt/data/{}/relations.json'.format(args.task)
    file_path = '/mnt/data/data/home/liuyafei/实验检查/论文2/SimKGC-prompt/data/{}/relations.json'.format(args.task)

    # 读取 JSON 数据
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 提取并排序字典的值
    # values = sorted(data.values())
    values = data.values()

    # 创建一个新字典，其中键是原始字典的值，值是这些值的索引
    relation2id_dict = {value: index for index, value in enumerate(values)}

    # 为每个值添加带有 "inverse " 前缀的键，确保索引值递增
    offset = len(relation2id_dict)
    inverse_dict = {'inverse {}'.format(value): index + offset for index, value in enumerate(values)}

    # 合并两个字典
    relation2id_dict.update(inverse_dict)

    return relation2id_dict