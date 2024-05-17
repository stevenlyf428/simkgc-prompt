import os
import json
import torch
import torch.utils.data.dataset

from typing import Optional, List

from config import args
# from triplet import reverse_triplet
from triplet_mask import construct_mask, construct_self_negative_mask
from dict_hub import get_entity_dict, get_link_graph, get_relation2id_dict
from dict_hub import get_tokenizer
from logger_config import logger

entity_dict = get_entity_dict()
relation2id_dict = get_relation2id_dict(args)
if args.use_link_graph:
    # make the lazy data loading happen
    get_link_graph()


def _custom_tokenize(text: str,
                     text_pair: Optional[str] = None) -> dict:
    tokenizer = get_tokenizer()
    encoded_inputs = tokenizer(text=text,
                               text_pair=text_pair if text_pair else None,
                               add_special_tokens=True,
                               max_length=args.max_num_tokens,
                               return_token_type_ids=True,
                               truncation=True)
    return encoded_inputs


def _parse_entity_name(entity: str) -> str:
    if args.task.lower() == 'wn18rr' or 'wn18rr_ind':
        # family_alcidae_NN_1
        entity = ' '.join(entity.split('_')[:-2])
        return entity
    # a very small fraction of entities in wiki5m do not have name
    return entity or ''


def _concat_name_desc2(entity: str, entity_desc: str) -> str:
    """
        添加prompt模版占位符
    """
    # if entity_desc.startswith(entity):
    #     entity_desc = entity_desc[len(entity):].strip()
    # if entity_desc:
    #     return '{}: {}'.format(entity, entity_desc)
    # return entity
    # prompt_prefix = "[PROMPT] " * 20  # 重复十次 "[PROMPT] "
    if entity_desc.startswith(entity):
        entity_desc = entity_desc[len(entity):].strip()
    if entity_desc:
        return '{} : {}'.format(entity, entity_desc)
    return  entity

def _concat_name_desc(entity: str, entity_desc: str) -> str:
    if entity_desc.startswith(entity):
        entity_desc = entity_desc[len(entity):].strip()
    if entity_desc:
        return '{}: {}'.format(entity, entity_desc)
    return entity

def get_neighbor_desc(head_id: str, tail_id: str = None) -> str:
    neighbor_ids = get_link_graph().get_neighbor_ids(head_id)
    # avoid label leakage during training
    if not args.is_test:
        neighbor_ids = [n_id for n_id in neighbor_ids if n_id != tail_id]
    entities = [entity_dict.get_entity_by_id(n_id).entity for n_id in neighbor_ids]
    entities = [_parse_entity_name(entity) for entity in entities]
    return ' '.join(entities)

def get_neighbor_triple_prompt(head_id: str, tail_id: str = None) -> str:
    neighbor_ids = get_link_graph().get_neighbor_triple_prompt(head_id)
    # avoid label leakage during training
    if not args.is_test:
        neighbor_ids = [ex for ex in neighbor_ids if ex[2] != tail_id]
    entities = [ entity_dict.get_entity_by_id(ex[0]).entity+' '+ex[1]+' '+entity_dict.get_entity_by_id(ex[2]).entity for ex in neighbor_ids]
    entities = [_parse_entity_name(entity) for entity in entities]
    triple_prompt = ' '.join(entities)
    # print(triple_prompt)
    return triple_prompt

class Example:

    def __init__(self, head_id, relation, tail_id, **kwargs):
        self.head_id = head_id
        self.tail_id = tail_id
        self.relation = relation

    @property
    def head_desc(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity_desc

    @property
    def tail_desc(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity_desc

    @property
    def head(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity

    @property
    def tail(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity
    

    def vectorize(self) -> dict:
        head_desc, tail_desc = self.head_desc, self.tail_desc
        
        # if args.use_link_graph:
        #     if len(head_desc.split()) < 20:
        #         head_desc += ' ' + get_neighbor_desc(head_id=self.head_id, tail_id=self.tail_id)
        #     if len(tail_desc.split()) < 20:
        #         tail_desc += ' ' + get_neighbor_desc(head_id=self.tail_id, tail_id=self.head_id)
        # if args.use_link_graph:
        #     if len(head_desc.split()) < 20:
        #         head_desc =  get_neighbor_triple_prompt(head_id=self.head_id, tail_id=self.tail_id) + ' ' + head_desc
        #     if len(tail_desc.split()) < 20:
        #         tail_desc = get_neighbor_triple_prompt(head_id=self.tail_id, tail_id=self.head_id) + ' ' + tail_desc

        
        head_word = _parse_entity_name(self.head)
        head_text = _concat_name_desc(head_word, head_desc)
        
        hr_encoded_inputs = _custom_tokenize(text=head_text,
                                             text_pair=self.relation)
        ## 拼接hr头
        ## 形式如[v1][v2][...][vn] +h+r + [MASK] +  h_desc + context_triple
        ## 
        # head_prompt_text = _concat_name_desc2(head_word, head_desc)
        # hr_prompt_encoded_inputs = _custom_tokenize(text=head_prompt_text,
        #                                      text_pair=self.relation)
        prompt_prefix = "[PROMPT] " * 10  # 重复十次 "[PROMPT] "
        # context_triple = get_neighbor_triple_prompt(head_id=self.head_id, tail_id=self.tail_id)
        head_prompt_text = _concat_name_desc2(head_word, head_desc)
        # if head_desc.startswith(head_word):
        #     head_desc = head_desc[len(head_word):].strip()
        # if head_desc:
        #     return '{} : {} : {}'.format(prompt_prefix, head_word, head_desc)
        # new_hr_prompt = '{} [SEP] {} [SEP] {} [SEP] [MASK] [SEP] {}'.format(prompt_prefix, head_prompt_text, self.relation,context_triple)
        # if len(head_prompt_text.split()) >= 80:
        #     head_prompt_text = head_prompt_text.split()[:40]
        new_hr_prompt = '{} [SEP] {} [SEP] {}'.format(prompt_prefix, head_prompt_text, self.relation)
        tokenizer = get_tokenizer()
        hr_prompt_encoded_inputs = tokenizer(text=new_hr_prompt,
                                add_special_tokens=True,
                                max_length=args.max_num_tokens,
                                return_token_type_ids=True,
                                truncation=True)
        
        
        
        head_encoded_inputs = _custom_tokenize(text=head_text)

        tail_word = _parse_entity_name(self.tail)
        tail_encoded_inputs = _custom_tokenize(text=_concat_name_desc(tail_word, tail_desc))
        
        # tr_prompt_text = _concat_name_desc2(tail_word, tail_desc)
        # tr_prompt_encoded_inputs = _custom_tokenize(text=tr_prompt_text,
        #                                      text_pair=self.relation)
        
        # print(relation2id_dict)
        # relation2idx
        # if not self.relation.isdigit():
        #     relation2idx = relation2id_dict[self.relation]
        # else:
        #     relation2idx = str(self.relation)
        if self.relation == '':
            relation2idx = ''
        else:
            relation2idx = relation2id_dict[self.relation]

        return {'hr_token_ids': hr_encoded_inputs['input_ids'],
                'hr_token_type_ids': hr_encoded_inputs['token_type_ids'],
                'tail_token_ids': tail_encoded_inputs['input_ids'],
                'tail_token_type_ids': tail_encoded_inputs['token_type_ids'],
                'head_token_ids': head_encoded_inputs['input_ids'],
                'head_token_type_ids': head_encoded_inputs['token_type_ids'],
                'obj': self,
                # 'hr_prompt_encoded_inputs': hr_prompt_encoded_inputs,
                'relation2idx':relation2idx,
                'hr_prompt_token_ids': hr_prompt_encoded_inputs['input_ids'],
                'hr_prompt_token_type_ids': hr_prompt_encoded_inputs['token_type_ids'],
                # 'tr_prompt_token_ids': tr_prompt_encoded_inputs['input_ids'],
                # 'tr_prompt_token_type_ids': tr_prompt_encoded_inputs['token_type_ids'],
                }


class Dataset(torch.utils.data.dataset.Dataset):

    def __init__(self, path, task, examples=None):
        self.path_list = path.split(',')
        print(self.path_list)
        self.task = task
        assert all(os.path.exists(path) for path in self.path_list) or examples
        if examples:
            self.examples = examples
        else:
            self.examples = []
            for path in self.path_list:
                if not self.examples:
                    self.examples = load_data(path)
                else:
                    self.examples.extend(load_data(path))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index].vectorize()


def load_data(path: str,
              add_forward_triplet: bool = True,
              add_backward_triplet: bool = False) -> List[Example]:
    assert path.endswith('.json'), 'Unsupported format: {}'.format(path)
    assert add_forward_triplet or add_backward_triplet
    logger.info('In test mode: {}'.format(args.is_test))

    data = json.load(open(path, 'r', encoding='utf-8'))
    logger.info('Load {} examples from {}'.format(len(data), path))

    cnt = len(data)
    examples = []
    for i in range(cnt):
        obj = data[i]
        if add_forward_triplet:
            examples.append(Example(**obj))
        # if add_backward_triplet:
        #     examples.append(Example(**reverse_triplet(obj)))
        data[i] = None

    return examples


def collate(batch_data: List[dict]) -> dict:
    # hr_token_ids, hr_mask = to_indices_and_mask(
    #     [torch.LongTensor(ex['hr_token_ids']) for ex in batch_data],
    #     pad_token_id=get_tokenizer().pad_token_id)
    # hr_token_type_ids = to_indices_and_mask(
    #     [torch.LongTensor(ex['hr_token_type_ids']) for ex in batch_data],
    #     need_mask=False)

    tail_token_ids, tail_mask = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    tail_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_type_ids']) for ex in batch_data],
        need_mask=False)

    head_token_ids, head_mask = to_indices_and_mask(
        [torch.LongTensor(ex['head_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    head_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['head_token_type_ids']) for ex in batch_data],
        need_mask=False)
    
    hr_prompt_token_ids, hr_prompt_mask = to_indices_and_mask(
        [torch.LongTensor(ex['hr_prompt_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    hr_prompt_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['hr_prompt_token_type_ids']) for ex in batch_data],
        need_mask=False)
    
    # tr_prompt_token_ids, tr_prompt_mask = to_indices_and_mask(
    #     [torch.LongTensor(ex['tr_prompt_token_ids']) for ex in batch_data],
    #     pad_token_id=get_tokenizer().pad_token_id)
    # tr_prompt_token_type_ids = to_indices_and_mask(
    #     [torch.LongTensor(ex['tr_prompt_token_type_ids']) for ex in batch_data],
    #     need_mask=False)
    
    # 'relation2idx':relation2id_dict[self.relation]
    # relation2idx = [ex['relation2idx'] for ex in batch_data]
    # relation2idx = torch.LongTensor(relation2idx)
    relation2idx = [ex['relation2idx'] for ex in batch_data]

    # 确保列表中的所有元素都是整数
    # relation2idx = [int(r) for r in relation2idx]

    # 将列表转换为 LongTensor
    # if relation2idx is None:
    relation2idx_tensor = torch.LongTensor(relation2idx)

    
    
    batch_exs = [ex['obj'] for ex in batch_data]
    batch_dict = {
        # 'hr_token_ids': hr_token_ids,
        # 'hr_mask': hr_mask,
        # 'hr_token_type_ids': hr_token_type_ids,
        'tail_token_ids': tail_token_ids,
        'tail_mask': tail_mask,
        'tail_token_type_ids': tail_token_type_ids,
        'head_token_ids': head_token_ids,
        'head_mask': head_mask,
        'head_token_type_ids': head_token_type_ids,
        'batch_data': batch_exs,
        'triplet_mask': construct_mask(row_exs=batch_exs) if not args.is_test else None,
        'self_negative_mask': construct_self_negative_mask(batch_exs) if not args.is_test else None,
        'relation2idx': relation2idx_tensor,
        'hr_prompt_token_ids': hr_prompt_token_ids,
        'hr_prompt_mask': hr_prompt_mask,
        'hr_prompt_token_type_ids': hr_prompt_token_type_ids,
        # 'tr_prompt_token_ids': tr_prompt_token_ids,
        # 'tr_prompt_mask': tr_prompt_mask,
        # 'tr_prompt_token_type_ids': tr_prompt_token_type_ids,
    }

    return batch_dict


def to_indices_and_mask(batch_tensor, pad_token_id=0, need_mask=True):
    mx_len = max([t.size(0) for t in batch_tensor])
    batch_size = len(batch_tensor)
    indices = torch.LongTensor(batch_size, mx_len).fill_(pad_token_id)
    # For BERT, mask value of 1 corresponds to a valid position
    if need_mask:
        mask = torch.ByteTensor(batch_size, mx_len).fill_(0)
    for i, t in enumerate(batch_tensor):
        indices[i, :len(t)].copy_(t)
        if need_mask:
            mask[i, :len(t)].fill_(1)
    if need_mask:
        return indices, mask
    else:
        return indices
