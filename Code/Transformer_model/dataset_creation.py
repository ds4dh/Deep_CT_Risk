import json, re, string

import logging
from tqdm import tqdm
working_dir = '/home/jknafou/risklick_classification'

splits_path = working_dir + '/splits_v3'
index_path = working_dir + '/index_v3'
dataset_path = working_dir + '/dataset_'
_RE_COMBINE_WHITESPACE = re.compile(r"\s+")
punctuation = string.punctuation
punctuation = punctuation.replace(':', '')
punctuation = punctuation.replace('{', '')
punctuation = punctuation.replace('}', '')
all_pattern = r"[{}]".format(punctuation)
_RE_COMBINE_PUNCTUATION = re.compile(all_pattern)

diseases_chemical_drugs_mapping = {'C01': 'Infections', 'C04' : 'Neoplasms', 'C05': 'Musculoskeletal Diseases',
        'C06': 'Digestive System Diseases', 'C07': 'Stomatognathic Diseases', 'C08': 'Respiratory Tract Diseases',
        'C09': 'Otorhinolaryngologic Diseases', 'C10': 'Nervous System Diseases','C11': 'Eye Diseases',
        'C12': 'Male Urogenital Diseases', 'C13': 'Female Urogenital Diseases and Pregnancy Complications',
        'C14': 'Cardiovascular Diseases', 'C15': 'Hemic and Lymphatic Diseases',
        'C16': 'Congenital, Hereditary, and Neonatal Diseases and Abnormalities', 'C17': 'Skin and Connective Tissue Diseases',
        'C18': 'Nutritional and Metabolic Diseases', 'C19': 'Endocrine System Diseases', 'C20': 'Immune System Diseases',
        'C21': 'Disorders of Environmental Origin','C22': 'Animal Diseases','C23': 'Pathological Conditions, Signs and Symptoms',
        'C24': 'Occupational Diseases', 'C25': 'Chemically-Induced Disorders', 'C26': 'Wounds and Injuries', 'D27': 'Chemical Actions and Uses'}

from collections import abc
def nested_dict_iter(nested):
    for key, value in nested.items():
        if isinstance(value, abc.Mapping):
            yield from nested_dict_iter(value)
        elif isinstance(value, list):
            if len(value) == sum(isinstance(v, abc.Mapping) for v in value):
                for item in value:
                    yield from nested_dict_iter(item)
            elif len(value) == 1:
                yield key, str(value[0])
            else:
                yield key, ' '.join(value)
        else:
            yield key, value

NTC_sets = {}
NTC_check = {}
for split in ['valid', 'test', 'train']:
    positive_count, negative_count = 0, 0
    NTC_sets[split] = set()
    with open(splits_path + '/' + split + '.txt', encoding='utf-8') as splits_file:
        with open(dataset_path + split + '_v3.tsv', encoding='utf-8', mode='w') as dataset_file:
            for line in tqdm(splits_file):
            # for line in splits_file:
                doc = ''
                *NCT_id, label = line.strip().split(',')
                NTC_key = ','.join(NCT_id)
                NCT_id, phase, condition = NCT_id

                if NTC_key not in NTC_check:
                    NTC_check[NTC_key] = label

                else:
                    logging.warning(splits_path + '/' + split + '.txt')
                    logging.warning(str(NTC_key) + '; ' + str(label))
                    logging.warning(str(NTC_key) + '; ' + str(NTC_check[NCT_id]))

                NTC_sets[split].add(NTC_key)
                with open(index_path + '/' + NCT_id + '.json') as NTC_file:
                    file = json.load(NTC_file)
                if 'BasicToText' in file['Features']:
                    del file['Features']['BasicToText']
                for key, item in nested_dict_iter(file['Features']):
                    if item:
                        doc += key + ' : ' +\
                           _RE_COMBINE_WHITESPACE.sub(item.replace('\n', ' ').replace('\\', ' '), ' ').rstrip() +\
                               '\t'

                dataset_file.write(NTC_key + '\t' + label + '\t' + phase + '\t' +
                                   diseases_chemical_drugs_mapping[condition] + '\t' +
                                   doc.rstrip() + '\n')