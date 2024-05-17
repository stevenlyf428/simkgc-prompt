import torch
import torch.nn as nn

class KEPromptEncoder(torch.nn.Module):
    def __init__(self, template, hidden_size, tokenizer, args, relation_num):
        super().__init__()
        # self.device = 'cuda:0'
        self.template = template
        self.spell_length = sum(template)
        self.hidden_size = hidden_size
        self.tokenizer = tokenizer
        self.args = args
        self.relation_num = relation_num
        # ent embedding
        self.cloze_length = template
        self.cloze_mask = [
            [1] * self.spell_length
        ]
        self.cloze_mask = torch.LongTensor(self.cloze_mask).bool().cuda()

        self.seq_indices_relation = torch.LongTensor(list(range(sum(self.template)))).cuda()
        # embedding        
        self.embedding_relation = torch.nn.Embedding(sum(self.template) * self.relation_num, self.hidden_size).cuda()
        print("init prompt encoder...")

    def forward(self, rs_tensor):#torch.Size([10])
        if sum(self.template) == 0:
            return None

         # 获取当前设备
        device = rs_tensor.device

        # 确保seq_indices_relation_spec在正确的设备上
        seq_indices_relation_spec = self.seq_indices_relation.unsqueeze(0).to(device) + rs_tensor.unsqueeze(-1).to(device) * sum(self.template)
        
        # self.embedding_relation = self.embedding_relation。
        # 直接使用self.embedding_relation
        input_embeds = self.embedding_relation(seq_indices_relation_spec)
        


        
        return input_embeds

    def get_query(self, texts, rs, prompt_tokens): # 在此插入了prompt模版
        contents = texts.split('\t\t')
        ans_list = [self.tokenizer.cls_token_id]

        # length = 5 for triple setting
        for i in range(len(contents)):
            ans_list += prompt_tokens * self.template[i]
            if len(contents[i]) != 0:
                if len(ans_list) == 1:
                    ans_list += self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(contents[i]))
                else:
                    ans_list += self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + contents[i]))
        ans_list += prompt_tokens * self.template[-1]
        ans_list += [self.tokenizer.sep_token_id]

        return [ans_list]