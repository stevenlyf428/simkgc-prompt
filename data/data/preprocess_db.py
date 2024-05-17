import os
import json
import argparse
import multiprocessing as mp

from multiprocessing import Pool
from typing import List

parser = argparse.ArgumentParser(description='preprocess')
parser.add_argument('--task', default='fb15k237', type=str, metavar='N',
                    help='dataset name')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of workers')
parser.add_argument('--train-path', default='/mnt/data/data/home/liuyafei/论文2/temp/SimKGC-prompt/data/fb20k/train.txt', type=str, metavar='N',
                    help='path to training data')
parser.add_argument('--valid-path', default='/mnt/data/data/home/liuyafei/论文2/temp/SimKGC-prompt/data/fb20k/valid.txt', type=str, metavar='N',
                    help='path to valid data')
parser.add_argument('--test-path', default='/mnt/data/data/home/liuyafei/论文2/temp/SimKGC-prompt/data/fb20k/test.txt', type=str, metavar='N',
                    help='path to valid data')

args = parser.parse_args()
mp.set_start_method('fork')

def _normalize_fb15k237_relation(relation: str) -> str:
    tokens = relation.replace('./', '/').replace('_', ' ').strip().split('/')
    dedup_tokens = []
    for token in tokens:
        if token not in dedup_tokens[-3:]:
            dedup_tokens.append(token)
    # leaf words are more important (maybe)
    relation_tokens = dedup_tokens[::-1]
    relation = ' '.join([t for idx, t in enumerate(relation_tokens)
                         if idx == 0 or relation_tokens[idx] != relation_tokens[idx - 1]])
    return relation

def _normalize_relations(examples: List[dict], normalize_fn, is_train: bool):
    relation_id_to_str = {}
    for ex in examples:
        rel_str = normalize_fn(ex['relation'])
        relation_id_to_str[ex['relation']] = rel_str
        ex['relation'] = rel_str

    # _check_sanity(relation_id_to_str)

    if is_train:
        out_path = '{}/relations.json'.format(os.path.dirname(args.train_path))
        with open(out_path, 'w', encoding='utf-8') as writer:
            json.dump(relation_id_to_str, writer, ensure_ascii=False, indent=4)
            print('Save {} relations to {}'.format(len(relation_id_to_str), out_path))

def _check_sanity(relation_id_to_str: dict):
    # We directly use normalized relation string as a key for training and evaluation,
    # make sure no two relations are normalized to the same surface form
    relation_str_to_id = {}
    for rel_id, rel_str in relation_id_to_str.items():
        if rel_str is None:
            continue
        if rel_str not in relation_str_to_id:
            relation_str_to_id[rel_str] = rel_id
        elif relation_str_to_id[rel_str] != rel_id:
            assert False, 'ERROR: {} and {} are both normalized to {}'\
                .format(relation_str_to_id[rel_str], rel_id, rel_str)
    return


# 全局字典
fb15k_id2ent = {}
fb15k_id2desc = {}

def _load_fb15k237_data_from_json(json_path: str):
    """Load data from the entity2wikidata.json file."""
    global fb15k_id2desc, fb15k_id2ent
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for entity_id, content in data.items():
        # 更新描述字典
        fb15k_id2desc[entity_id] = content.get('description', '')
        # 更新实体字典
        fb15k_id2ent[entity_id] = (
            entity_id, 
            content.get('label', '').replace('_', ' ').strip(), 
            content.get('description', '')
        )

def _process_line_fb15k237(line: str) -> dict:
    """Process a single line from the dataset."""
    fs = line.strip().split('\t')
    assert len(fs) == 3, 'Expect 3 fields for {}'.format(line)
    head_id, relation, tail_id = fs[0], fs[1], fs[2]

    _, head, _ = fb15k_id2ent.get(head_id, (head_id, '', ''))
    _, tail, _ = fb15k_id2ent.get(tail_id, (tail_id, '', ''))
    example = {
        'head_id': head_id,
        'head': head,
        'relation': relation,
        'tail_id': tail_id,
        'tail': tail
    }
    return example
   
def preprocess_fb15k237(path):
    """Preprocess FB15k-237 dataset."""
    if not fb15k_id2desc or not fb15k_id2ent:
        _load_fb15k237_data_from_json('/mnt/data/data/home/liuyafei/论文2/temp/SimKGC-prompt/data/fb20k/entity2wikidata.json')

    lines = open(path, 'r', encoding='utf-8').readlines()
    pool = Pool(processes=args.workers)
    examples = pool.map(_process_line_fb15k237, lines)
    pool.close()
    pool.join()

    _normalize_relations(examples, normalize_fn=_normalize_fb15k237_relation, is_train=(path == args.train_path))

    out_path = path + '.json'
    json.dump(examples, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    print('Save {} examples to {}'.format(len(examples), out_path))
    return examples
def dump_all_entities(examples, out_path, id2text: dict):
    id2entity = {}
    relations = set()
    for ex in examples:
        head_id = ex['head_id']
        relations.add(ex['relation'])
        if head_id not in id2entity:
            # Check if the key exists in id2text before accessing it
            if head_id in id2text:
                id2entity[head_id] = {
                    'entity_id': head_id,
                    'entity': ex['head'],
                    'entity_desc': id2text[head_id]
                }

        tail_id = ex['tail_id']
        if tail_id not in id2entity:
            # Similarly, check if the key exists for tail_id
            if tail_id in id2text:
                id2entity[tail_id] = {
                    'entity_id': tail_id,
                    'entity': ex['tail'],
                    'entity_desc': id2text[tail_id]
                }
        # if head_id not in id2entity:
        #     id2entity[head_id] = {'entity_id': head_id,
        #                           'entity': ex['head'],
        #                           'entity_desc': id2text[head_id]}
        # tail_id = ex['tail_id']
        # if tail_id not in id2entity:
        #     id2entity[tail_id] = {'entity_id': tail_id,
        #                           'entity': ex['tail'],
        #                           'entity_desc': id2text[tail_id]}
    print('Get {} entities, {} relations in total'.format(len(id2entity), len(relations)))

    json.dump(list(id2entity.values()), open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

all_examples = []
for path in [args.train_path, args.valid_path, args.test_path]:
    assert os.path.exists(path)
    print('Process {}...'.format(path))
    if args.task.lower() == 'wn18rr':
        all_examples += preprocess_wn18rr(path)
    elif args.task.lower() == 'fb15k237':
        all_examples += preprocess_fb15k237(path)
    elif args.task.lower() in ['wiki5m_trans', 'wiki5m_ind']:
        all_examples += preprocess_wiki5m(path, is_train=(path == args.train_path))
    else:
        assert False, 'Unknown task: {}'.format(args.task)

if args.task.lower() == 'wn18rr':
    id2text = {k: v[2] for k, v in wn18rr_id2ent.items()}
elif args.task.lower() == 'fb15k237':
    id2text = {k: v[2] for k, v in fb15k_id2ent.items()}
elif args.task.lower() in ['wiki5m_trans', 'wiki5m_ind']:
    id2text = wiki5m_id2text
else:
    assert False, 'Unknown task: {}'.format(args.task)

dump_all_entities(all_examples,
                    out_path='{}/entities.json'.format(os.path.dirname(args.train_path)),
                    id2text=id2text)
print('Done')