import os
import json
import torch
import torch.utils.data.dataset

from typing import Optional, List

from config import args
from triplet import reverse_triplet
from triplet_mask import construct_mask, construct_self_negative_mask
from dict_hub import get_entity_dict, get_link_graph, get_relation2id_dict, get_tokenizer
from logger_config import logger

relation2id_dict = get_relation2id_dict(args)
entity_dict = get_entity_dict()
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
    if args.task.lower() == 'wn18rr':
        # family_alcidae_NN_1
        entity = ' '.join(entity.split('_')[:-2])
        return entity
    # a very small fraction of entities in wiki5m do not have name
    return entity or ''


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
        if args.use_link_graph:
            # print(head_desc)
            if len(head_desc.split()) < 20:
                head_desc += ' ' + get_neighbor_desc(head_id=self.head_id, tail_id=self.tail_id)
            if len(tail_desc.split()) < 20:
                tail_desc += ' ' + get_neighbor_desc(head_id=self.tail_id, tail_id=self.head_id)

        head_word = _parse_entity_name(self.head)
        head_text = _concat_name_desc(head_word, head_desc)
        hr_encoded_inputs = _custom_tokenize(text=head_text,
                                             text_pair=self.relation)

        head_encoded_inputs = _custom_tokenize(text=head_text)

        tail_word = _parse_entity_name(self.tail)
        tail_encoded_inputs = _custom_tokenize(text=_concat_name_desc(tail_word, tail_desc))
        
        tokenizer = get_tokenizer()
        

        # # 构建[PROMPT]前缀和后缀
        # prompt_prefix = "[PROMPT] " * 5
        # prompt_suffix = "[PROMPT] " * 5
        # if len(hr_encoded_inputs) > 40:
        #     # 如果超过限制，则截断head_prompt_text
        #     hr_encoded_inputs = hr_encoded_inputs[:40]
        # # 将截断的token转回字符串
        # hr_prompt_text = tokenizer.convert_tokens_to_string(hr_encoded_inputs)  
        # # 组合最终的文本
        # # new_hr_prompt = '{} {}'.format(prompt_prefix, hr_prompt_text)
        # new_hr_prompt = '{} [SEP] {} [SEP] {}'.format(prompt_prefix, hr_prompt_text, prompt_suffix)

        # # 使用tokenizer处理文本
        # hr_prompt_encoded_inputs = tokenizer(text=new_hr_prompt,
        #                                     add_special_tokens=True,
        #                                     max_length=args.max_num_tokens,
        #                                     return_token_type_ids=True,
        #                                     truncation=True)

        # # 构建[PROMPT]前缀和后缀
        # prompt_prefix = "[PROMPT] " * 5
        # prompt_mid = "[PROMPT] " * 5
        # prompt_suffix = "[PROMPT] " * 4
        # #处理head_prompt_text和self.relation以适应token限制
        # head_prompt_text_tokens = tokenizer.tokenize(head_text)
        # if len(head_prompt_text_tokens) + len(tokenizer.tokenize(self.relation)) > 35:
        #     # 如果超过限制，则截断head_prompt_text
        #     head_prompt_text_tokens = head_prompt_text_tokens[:35 - len(tokenizer.tokenize(self.relation))]

        # # 将截断的token转回字符串
        # head_prompt_text = tokenizer.convert_tokens_to_string(head_prompt_text_tokens)

        # # 组合最终的文本
        # new_hr_prompt = '{} {} {} {}'.format(prompt_prefix, head_prompt_text, self.relation, prompt_mid)
        # # 使用tokenizer处理文本
        # hr_prompt_encoded_inputs = tokenizer(text=new_hr_prompt,
        #                                     add_special_tokens=True,
        #                                     max_length=args.max_num_tokens,
        #                                     return_token_type_ids=True,
        #                                     truncation=True)
        
        # prompt_prefix = "[PROMPT] " * 20  # 重复十次 "[PROMPT] "
        # prompt_prefix1 = "[PROMPT] " * sum(args.template)  # 重复十次 "[PROMPT] "
        prompt_prefix1 = "[PROMPT] " * 10  # 重复十次 "[PROMPT] "
        # prompt_prefix2 = "[PROMPT] " * 5  # 重复十次 "[PROMPT] "
        # prompt_prefix = "[PROMPT] " * 30  # 重复十次 "[PROMPT] "
        head_prompt_text = _concat_name_desc(head_word, head_desc)
        # if len(head_prompt_text.split()) >= 35:
        #     head_prompt_text = head_prompt_text.split()[:35]
        # if len(self.relation.split()) >= 5:
        #     self.relation = self.relation.split()[:5]
        new_hr_prompt = '{} {} {}'.format(prompt_prefix1, head_prompt_text, self.relation)
        # new_hr_prompt = '{} [SEP] {} [SEP] {} [SEP] {}'.format(prompt_prefix1, head_prompt_text, self.relation, prompt_prefix2)
        
        hr_prompt_encoded_inputs = tokenizer(text=new_hr_prompt,
                                add_special_tokens=True,
                                max_length=args.max_num_tokens,
                                return_token_type_ids=True,
                                truncation=True)
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
                'relation2idx':relation2idx,
                'hr_prompt_token_ids': hr_prompt_encoded_inputs['input_ids'],
                'hr_prompt_token_type_ids': hr_prompt_encoded_inputs['token_type_ids'],
                'obj': self}


class Dataset(torch.utils.data.dataset.Dataset):

    def __init__(self, path, task, examples=None):
        self.path_list = path.split(',')
        self.task = task
        print(self.path_list)
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
              add_backward_triplet: bool = True) -> List[Example]:
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
        if add_backward_triplet:
            examples.append(Example(**reverse_triplet(obj)))
        data[i] = None

    return examples


def collate(batch_data: List[dict]) -> dict:

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
    relation2idx = [ex['relation2idx'] for ex in batch_data]
    relation2idx_tensor = torch.LongTensor(relation2idx)

    batch_exs = [ex['obj'] for ex in batch_data]
    batch_dict = {
        'relation2idx': relation2idx_tensor,
        'hr_prompt_token_ids': hr_prompt_token_ids,
        'hr_prompt_mask': hr_prompt_mask,
        'hr_prompt_token_type_ids': hr_prompt_token_type_ids,
        'tail_token_ids': tail_token_ids,
        'tail_mask': tail_mask,
        'tail_token_type_ids': tail_token_type_ids,
        'head_token_ids': head_token_ids,
        'head_mask': head_mask,
        'head_token_type_ids': head_token_type_ids,
        'batch_data': batch_exs,
        'triplet_mask': construct_mask(row_exs=batch_exs) if not args.is_test else None,
        'self_negative_mask': construct_self_negative_mask(batch_exs) if not args.is_test else None,
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