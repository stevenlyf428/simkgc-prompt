from abc import ABC
from copy import deepcopy

import torch
import torch.nn as nn

from dataclasses import dataclass
from transformers import AutoModel, AutoConfig
from Prompt_encoder import KEPromptEncoder
from transformers import AutoTokenizer
from dict_hub import get_relation2id_dict
from triplet_mask import construct_mask
from config import args
from dict_hub import get_tokenizer

relation2id_dict = get_relation2id_dict(args)

def build_model(args) -> nn.Module:
    return CustomBertModel(args)


@dataclass
class ModelOutput:
    logits: torch.tensor
    labels: torch.tensor
    inv_t: torch.tensor
    hr_vector: torch.tensor
    tail_vector: torch.tensor


class CustomBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.pre_batch = args.pre_batch
        # num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
        # random_vector = torch.randn(num_pre_batch_vectors, self.config.hidden_size)
        # self.register_buffer("pre_batch_vectors",
        #                      nn.functional.normalize(random_vector, dim=1),
        #                      persistent=False)
        # self.offset = 0
        # self.pre_batch_exs = [None for _ in range(num_pre_batch_vectors)]

        self.hr_bert = AutoModel.from_pretrained(args.pretrained_model)
        
        
        
        self.embeddings = self.hr_bert.get_input_embeddings()
        # 添加占位token
        # self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        self.tokenizer = get_tokenizer()
        # self.tokenizer.add_special_tokens({'additional_special_tokens': ['[PROMPT]']})
        self.pseudo_token_id = self.tokenizer.get_vocab()['[PROMPT]']
        
        self.tail_bert = deepcopy(self.hr_bert)
        
        
        self.spell_length = sum((1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1))
        self.template = (1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)
        self.hidden_size = self.embeddings.embedding_dim
        # self.device = 'cuda:0'
        self.relation_num = len(relation2id_dict)
        # init prompt模型并放入cuda
        self.prompt_encoder1 = KEPromptEncoder(self.template, self.hidden_size, self.tokenizer, args, self.relation_num)
        # self.prompt_encoder1 = torch.nn.DataParallel(self.prompt_encoder1).cuda()
        
        # self.prompt_encoder2 = KEPromptEncoder(self.template, self.hidden_size, self.tokenizer, args, self.relation_num)
        # self.prompt_encoder1 = self.prompt_encoder1.to(self.device)
        # self.prompt_encoder1 = self.prompt_encoder1.cuda()
        
        # self.prompt_encoder2 = self.prompt_encoder2.to(self.device)
        # if torch.cuda.device_count() > 1:
        #     self.prompt_encoder1 = torch.nn.DataParallel(self.prompt_encoder1)
        #     self.prompt_encoder2 = torch.nn.DataParallel(self.prompt_encoder2)
        
        # for param in self.hr_bert.parameters():
        #     param.requires_grad = False
        # for param in self.tail_bert.parameters():
        #     param.requires_grad = False

    def embed_input1(self, queries, rs_tensor):
        bz = queries.shape[0]   #torch.Size([10, 55])
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.tokenizer.unk_token_id
        raw_embeds = self.embeddings(queries_for_embedding) #过embedding层 torch.Size([10, 55, 768])

        blocked_indices = (queries == self.pseudo_token_id).nonzero().reshape((bz, self.spell_length, 2))[:, :, 1]  # bz    #torch.Size([10, 6])
        replace_embeds = self.prompt_encoder1(rs_tensor)
        for bidx in range(bz):
            for i in range(self.prompt_encoder1.spell_length):
                raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[bidx, i, :]  #torch.Size([10, 6, 768])
        return raw_embeds
    
    # def embed_input2(self, queries, rs_tensor):
    #     bz = queries.shape[0]   #torch.Size([10, 55])
    #     queries_for_embedding = queries.clone()
    #     queries_for_embedding[(queries == self.pseudo_token_id)] = self.tokenizer.unk_token_id
    #     raw_embeds = self.embeddings(queries_for_embedding) #过embedding层 torch.Size([10, 55, 768])

    #     blocked_indices = (queries == self.pseudo_token_id).nonzero().reshape((bz, self.spell_length, 2))[:, :, 1]  # bz    #torch.Size([10, 6])
    #     replace_embeds = self.prompt_encoder2(rs_tensor)
    #     for bidx in range(bz):
    #         for i in range(self.prompt_encoder2.spell_length):
    #             raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[bidx, i, :]  #torch.Size([10, 6, 768])
    #     return raw_embeds
    
    def _encode(self, encoder, token_ids, mask, token_type_ids):# 关键代码
        outputs = encoder(input_ids=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :] 
        cls_output = _pool_output('cls', cls_output, mask, last_hidden_state, token_ids)
        # cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state, token_ids)
        return cls_output
    
    def _encode2(self, encoder, inputs_embeds, mask, token_type_ids,token_ids):# 关键代码
        outputs = encoder(inputs_embeds=inputs_embeds,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :] 
        # cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)
        cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state, token_ids)
        return cls_output

    def forward(self, 
                # hr_token_ids, hr_mask, hr_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,
                relation2idx,
                hr_prompt_token_ids,hr_prompt_mask,hr_prompt_token_type_ids,
                # tr_prompt_token_ids,tr_prompt_mask,tr_prompt_token_type_ids,
                only_ent_embedding=False, **kwargs) -> dict:
        if only_ent_embedding:
            return self.predict_ent_embedding(tail_token_ids=tail_token_ids,
                                              tail_mask=tail_mask,
                                              tail_token_type_ids=tail_token_type_ids)
        # if only_ent_embedding:
        #     return self.predict_ent_tr_embedding( relation2idx = relation2idx,
        #                                          tr_prompt_token_ids=tr_prompt_token_ids,
        #                                       tr_prompt_mask=tr_prompt_mask,
        #                                       tr_prompt_token_type_ids=tr_prompt_token_type_ids)
            # return self.predict_ent_embedding(tail_token_ids=tail_token_ids,
            #                                   tail_mask=tail_mask,
            #                                   tail_token_type_ids=tail_token_type_ids)

        # # TO do
        # # 插入模版的hr_prompt_vetor
        # hr_prompt_vector = self._encode(self.hr_bert,
        #                          token_ids=hr_prompt_token_ids,
        #                          mask=hr_prompt_mask)
        # relation2idx = relation2idx.to(self.device)
        hr_inputs_embeds = self.embed_input1(hr_prompt_token_ids, relation2idx)    #torch.Size([10, 64, 768])
        
        hr_vector = self._encode2(self.hr_bert,
                                 inputs_embeds=hr_inputs_embeds,
                                 mask=hr_prompt_mask,
                                 token_type_ids=hr_prompt_token_type_ids, 
                                 token_ids = hr_prompt_token_ids
                                 )

        # tail_inputs_embeds = self.embed_input2(tr_prompt_token_ids, relation2idx)
        
        # tail_vector = self._encode2(self.tail_bert,
        #                            inputs_embeds=tail_inputs_embeds,
        #                            mask=tr_prompt_mask,
        #                            token_type_ids=tr_prompt_token_type_ids)

        tail_vector = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)

        # head_vector = self._encode(self.tail_bert,
        #                            token_ids=tr_prompt_mask,
        #                            mask=head_mask,
        #                            token_type_ids=tr_prompt_token_type_ids)
        head_vector = self._encode(self.tail_bert,
                                   token_ids=head_token_ids,
                                   mask=head_mask,
                                   token_type_ids=head_token_type_ids)

        # DataParallel only support tensor/dict
        return {'hr_vector': hr_vector,
                'tail_vector': tail_vector,
                'head_vector': head_vector
                }

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
        batch_size = hr_vector.size(0)
        labels = torch.arange(batch_size).to(hr_vector.device)

        logits = hr_vector.mm(tail_vector.t())
        if self.training:
            logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device)
        logits *= self.log_inv_t.exp()

        # triplet_mask = batch_dict.get('triplet_mask', None)
        # if triplet_mask is not None:
        #     logits.masked_fill_(~triplet_mask, -1e4)

        if self.pre_batch > 0 and self.training:
            pre_batch_logits = self._compute_pre_batch_logits(hr_vector, tail_vector, batch_dict)
            logits = torch.cat([logits, pre_batch_logits], dim=-1)

        if self.args.use_self_negative and self.training:
            head_vector = output_dict['head_vector']
            self_neg_logits = torch.sum(hr_vector * head_vector, dim=1) * self.log_inv_t.exp()
            self_negative_mask = batch_dict['self_negative_mask']
            self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
            logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)

        return {'logits': logits,
                'labels': labels,
                'inv_t': self.log_inv_t.detach().exp(),
                'hr_vector': hr_vector.detach(),
                'tail_vector': tail_vector.detach()}

    def _compute_pre_batch_logits(self, hr_vector: torch.tensor,
                                  tail_vector: torch.tensor,
                                  batch_dict: dict) -> torch.tensor:
        assert tail_vector.size(0) == self.batch_size
        batch_exs = batch_dict['batch_data']
        # batch_size x num_neg
        pre_batch_logits = hr_vector.mm(self.pre_batch_vectors.clone().t())
        pre_batch_logits *= self.log_inv_t.exp() * self.args.pre_batch_weight
        if self.pre_batch_exs[-1] is not None:
            pre_triplet_mask = construct_mask(batch_exs, self.pre_batch_exs).to(hr_vector.device)
            pre_batch_logits.masked_fill_(~pre_triplet_mask, -1e4)

        self.pre_batch_vectors[self.offset:(self.offset + self.batch_size)] = tail_vector.data.clone()
        self.pre_batch_exs[self.offset:(self.offset + self.batch_size)] = batch_exs
        self.offset = (self.offset + self.batch_size) % len(self.pre_batch_exs)

        return pre_batch_logits

    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids, **kwargs) -> dict:
        ent_vectors = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)
        return {'ent_vectors': ent_vectors.detach()}
    
    @torch.no_grad()
    def predict_ent_tr_embedding(self, relation2idx,tr_prompt_token_ids, tr_prompt_mask, tr_prompt_token_type_ids, **kwargs) -> dict:
        tr_inputs_embeds = self.embed_input2(tr_prompt_token_ids, relation2idx)
        
        tr_vector = self._encode2(self.tail_bert,
                                   inputs_embeds=tr_inputs_embeds,
                                   mask=tr_prompt_mask,
                                   token_type_ids=tr_prompt_token_type_ids)
        return {'ent_tr_vectors': tr_vector.detach()}


def _pool_output(pooling: str,
                cls_output: torch.tensor,
                mask: torch.tensor,
                last_hidden_state: torch.tensor,
                ) -> torch.tensor: #input_ids
    if pooling == 'cls':
        output_vector = cls_output
    elif pooling == 'max':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        last_hidden_state[input_mask_expanded == 0] = -1e4
        output_vector = torch.max(last_hidden_state, 1)[0]
    # elif pooling == 'mask':
    #     # mask_indices = -1 * torch.ones(input_ids.shape[0], dtype=torch.long)
    #     mask_token_index = torch.where(input_ids == get_tokenizer().mask_token_id)[1]
    #     # print(get_tokenizer().mask_token_id)
    #     output_vector = last_hidden_state[0, mask_token_index]
    #     # input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
    #     # last_hidden_state[input_mask_expanded == 0] = -1e4
    #     # output_vector = torch.max(last_hidden_state, 1)[0]
    elif pooling == 'mean':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        output_vector = sum_embeddings / sum_mask
    else:
        assert False, 'Unknown pooling mode: {}'.format(pooling)

    output_vector = nn.functional.normalize(output_vector, dim=1)
    return output_vector
