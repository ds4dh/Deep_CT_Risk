import torch, os, json, re, nltk, scipy
from scipy.special import softmax
from transformers import PreTrainedTokenizerBase
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz
from collections.abc import MutableMapping

def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str ='->') -> MutableMapping:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# working_dir = '/home/jknafou/CT_models/'
working_dir = '/data/collection/COVID_models/CT_models_v3t/'

# Sohrab code from here...
def create_summary_ct_v2(content):
    """
    """
    # TODO: At some point investigate whether it is always useful to convert fields with list of dictionaries
    #  to dictionary of lists. For sure useful for elasticsearch, but not sure about ML-based stuff.

    # Eligibility check:
    if not is_content_eligible(content):
        return None

    content = content["FullStudy"]["Study"]

    # Summary template:
    ct_summary = {'Meta': {'NCTId': None, 'OverallStatus': None, 'Split': None, 'PhaseNormalized': None},
                  'Features':
                      {
                          'Derived': {'ParticipantsFromResults': None, 'OncologyWHO': None, 'FDADrug': None},
                          'Basic': {'Phase': None, 'BriefTitle': None, 'OfficialTitle': None, 'Condition': None,
                                    'Keyword': None, 'PrimaryOutcome': None, 'Intervention': None,
                                    'EligibilityModule': None, 'Inclusion': None, 'Exclusion': None,
                                    'Organization': None, 'DescriptionModule': None,
                                    'DesignInfo': None, 'EnrollmentInfo': None, 'ContactsLocationsModule': None,
                                    },
                          'BasicToText': None
                      }
                  }
    # TODO: This derived section is not being used for the moment.
    # if get_milestone_counts_from_json(content):
    #    ct_summary['Features']['Derived']['ParticipantsFromResults'] = str(get_milestone_counts_from_json(content))

    content = content['ProtocolSection']

    ct_summary['Meta']['NCTId'] = content['IdentificationModule']['NCTId']
    ct_summary['Features']['Basic']['BriefTitle'] = content['IdentificationModule']['BriefTitle']

    if 'OfficialTitle' in content['IdentificationModule'].keys():
        ct_summary['Features']['Basic']['OfficialTitle'] = content['IdentificationModule']['OfficialTitle']

    if 'ConditionList' in content['ConditionsModule'].keys():
        ct_summary['Features']['Basic']['Condition'] = \
            content['ConditionsModule']['ConditionList']['Condition']

    if 'KeywordList' in content['ConditionsModule'].keys():
        ct_summary['Features']['Basic']['Keyword'] = content['ConditionsModule']['KeywordList']['Keyword']

    if 'OutcomesModule' in content.keys():
        if 'PrimaryOutcomeList' in content['OutcomesModule'].keys():
            # TODO: If it doesn't, then it exists in some other ways.
            ct_summary['Features']['Basic']['PrimaryOutcome'] = content['OutcomesModule']['PrimaryOutcomeList'][
                'PrimaryOutcome']

    intervention = content['ArmsInterventionsModule']['InterventionList']['Intervention']
    intervention = list_dict_to_dict_list(intervention)
    ct_summary['Features']['Basic']['Intervention'] = intervention

    eligibility = content['EligibilityModule']
    ct_summary['Features']['Basic']['EligibilityModule'] = eligibility
    if 'EligibilityCriteria' in eligibility.keys():
        # Damn ctgov for this ugly structure.
        _eligibility = eligibility['EligibilityCriteria'].lower().replace('\r', '').replace('\n', '')
        if 'inclusion criteria' in _eligibility and 'exclusion criteria' in _eligibility:
            try:
                inclusion = re.search("inclusion criteria(.*)exclusion criteria", _eligibility)
                exclusion = re.search("exclusion criteria(.*)", _eligibility)
                ct_summary['Features']['Basic']['Inclusion'] = inclusion.group(1)
                ct_summary['Features']['Basic']['Exclusion'] = exclusion.group(1)
            except:
                inclusion = re.search("inclusion criteria(.*)", _eligibility)
                exclusion = re.search("exclusion criteria(.*)inclusion criteria", _eligibility)
                ct_summary['Features']['Basic']['Inclusion'] = inclusion.group(1)
                ct_summary['Features']['Basic']['Exclusion'] = exclusion.group(1)

    ct_summary['Features']['Basic']['Organization'] = content['IdentificationModule']['Organization']

    if 'DescriptionModule' in content.keys():
        ct_summary['Features']['Basic']['DescriptionModule'] = content['DescriptionModule']
    if 'DesignInfo' in content['DesignModule'].keys():
        ct_summary['Features']['Basic']['DesignInfo'] = content['DesignModule']['DesignInfo']

    if 'EnrollmentInfo' in content['DesignModule'].keys():
        ct_summary['Features']['Basic']['EnrollmentInfo'] = content['DesignModule']['EnrollmentInfo']

    if 'ContactsLocationsModule' in content.keys():
        ct_summary['Features']['Basic']['ContactsLocationsModule'] = content['ContactsLocationsModule']

    if 'PhaseList' in content['DesignModule'].keys():
        ct_summary['Features']['Basic']['Phase'] = content['DesignModule']['PhaseList']['Phase']
        ct_summary['Meta']['PhaseNormalized'] = phase_normalize(content['DesignModule']['PhaseList']['Phase'])

    ct_summary['Features']['BasicToText'] = str(ct_summary['Features']['Basic'])

    ct_summary['Meta']['OverallStatus'] = content['StatusModule']['OverallStatus']

    # TODO:  splits.

    return ct_summary


def list_dict_to_dict_list(l_d):
    if not isinstance(l_d, list):
        return l_d
    if not isinstance(l_d[0], dict):
        return l_d

    d_l = dict()
    for l in l_d:
        for k in l.keys():
            if k not in d_l.keys():
                d_l[k] = []
            d_l[k].append(l[k])
    return d_l


def is_content_eligible(content):
    content = content["FullStudy"]["Study"]
    eligible_study = ['Interventional']
    eligible_status = ["Completed", "Terminated", "Withdrawn", "Suspended", "Unknown status"]
    if "DesignModule" not in content["ProtocolSection"].keys():
        return False

    study = content["ProtocolSection"]["DesignModule"]["StudyType"]
    if study.lower() not in ' '.join(eligible_study).lower():
        return False

    status = content["ProtocolSection"]["StatusModule"]["OverallStatus"]
    if status.lower() not in ' '.join(eligible_status).lower():
        return False

    if 'PhaseList' not in content["ProtocolSection"]['DesignModule'].keys():
        return False

    return True


def phase_normalize(phase_list, phase_path= working_dir + 'phases_normalization.json'):
    with open(phase_path, 'r') as f:
        phase_dict = json.load(f)

    phase = '; '.join(phase_list)

    return phase_dict[phase]
# ...to here.

_RE_COMBINE_WHITESPACE = re.compile(r"\s+")

def file_parser(path_to_json_list):
    parsed_files = {}
    with open(path_to_json_list, encoding='utf-8') as splits_file:
        json_list = json.load(splits_file)
        for json_dict in tqdm(json_list):
            doc = []
            json_dict = create_summary_ct_v2(json_dict)

            if 'BasicToText' in json_dict['Features']:
                del json_dict['Features']['BasicToText']

            flatten_json_dict = flatten_dict(json_dict['Features'])
            ordered_keys = list(flatten_json_dict.keys())
            ordered_keys.sort()

            for key in ordered_keys:
                if not flatten_json_dict[key]:
                    flatten_json_dict[key] = str(flatten_json_dict[key])
                if type(flatten_json_dict[key]) == list:
                    flatten_json_dict[key] = '[' + ', '.join([str(text) for text in flatten_json_dict[key]]) + ']'

                doc.append([key] + [t for t in sentence_tokenizer.tokenize(_RE_COMBINE_WHITESPACE.sub(flatten_json_dict[key].replace('\n', ' ').replace('\\', ' ').replace('->', ' '), ' ').rstrip())])

            parsed_files[json_dict['Meta']['NCTId']] = doc
    return parsed_files

sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
abbreviation = ['a', 'å', 'Ǻ', 'Å', 'b', 'c', 'd', 'e', 'ɛ', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                'o', 'Ö', 'Ø', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'µm',
                'abs', 'al', 'approx', 'bp', 'ca', 'cap', 'cf', 'co', 'd.p.c', 'dr', 'e.g', 'et', 'etc', 'er',
                'eq',
                'fig', 'figs', 'h', 'i.e', 'it', 'inc', 'min', 'ml', 'mm', 'mol', 'ms', 'no', 'nt',
                'ref', 'r.p.m', 'sci', 's.d', 'sd', 'sec', 's.e.m', 'sp', 'ssp', 'st', 'supp', 'vs', 'wt']
sentence_tokenizer._params.abbrev_types.update(abbreviation)

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from shutil import copyfile
from typing import Optional, Tuple
import sentencepiece as spm

SPIECE_UNDERLINE = "▁"

VOCAB_FILES_NAMES = {"vocab_file": working_dir + "Tokenizer/bio_english_sample.model"}
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "spm-tokenizer": working_dir + "Tokenizer/bio_english_sample.model",
    }
}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "xlm-roberta-base": 512,
}
class SPMTokenizer(PreTrainedTokenizer):

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        **kwargs
    ):
        # Mask token behave like a normal word, i.e. include the space before it
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(str(vocab_file))
        self.vocab_file = vocab_file

        self.fairseq_tokens_to_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}

        self.fairseq_offset = 1
        self.fairseq_tokens_to_ids["<mask>"] = len(self.sp_model) + self.fairseq_offset
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(self.vocab_file)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):

        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0, token_ids_1=None, already_has_special_tokens: bool = False
    ):

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0, token_ids_1=None
    ):
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    @property
    def vocab_size(self):
        return len(self.sp_model) + self.fairseq_offset + 1  # Add the <mask> token

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text):
        return self.sp_model.EncodeAsPieces(text)

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        spm_id = self.sp_model.PieceToId(token)

        # Need to return unknown token if the SP model returned 0
        return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        return self.sp_model.IdToPiece(index - self.fairseq_offset)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            # logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)

class RiskLickLineByLineClassificationDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, tokenizer: SPMTokenizer, file: str, sampling_size=None, number_inputs=8, number_text_field='all'):
        self.tokenizer = tokenizer
        self.block_size = 512
        self.file = file
        self.number_inputs = number_inputs
        self.number_text_field = number_text_field
        self.sampling_size = 1 if sampling_size==None else sampling_size
        self.examples = []
        for NCTId, text in self.file.items():
                for _ in range(self.sampling_size):
                    self.examples.append({
                        'text': text,
                        'NCTId': NCTId})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

import random
from numpy.random import choice
@dataclass
class MultiInputsRiskLickDataCollatorClassification:

    tokenizer: PreTrainedTokenizerBase
    device: torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    padding = 'max_length'
    max_length: Optional[int] = 512
    number_inputs: [int] = 8
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    split: str = 'train'

    def __call__(self, features):
        NCTIds, NCTIds_history = [], {}
        for i in range(len(features)):
            NCTIds.append([features[i]['NCTId']])
            del features[i]['NCTId']
            if NCTIds[-1][0] not in NCTIds_history:
                NCTIds_history[NCTIds[-1][0]] = {}
            input_ids = [self.tokenizer.cls_token_id]
            for input_number in range(1, self.number_inputs + 1):
                rand_float = random.uniform(0, 1)
                if rand_float < 1/2:
                    # Sampling with more probability for fields with high length
                    count = []
                    for s in features[i]['text']:
                        count.append(sum([len(t) for t in s[1:]]))
                    n_char = sum(count)
                    probability_distribution = [t/n_char for t in count]
                    random_index = choice(range(len(features[i]['text'])), len(features[i]['text']), replace=False, p=probability_distribution)

                else:
                    # Totally random field sampling
                    random_index = random.sample(range(len(features[i]['text'])), len(features[i]['text']))

                # all sentences have been taken at least once
                to_avoid_first = [s for s in NCTIds_history[NCTIds[-1][0]].keys() if NCTIds_history[NCTIds[-1][0]][s] == 'Done']
                if len(to_avoid_first) != len(random_index):
                    random.shuffle(to_avoid_first)
                    random_index = [index for index in random_index if index not in to_avoid_first] + to_avoid_first

                for j in random_index:
                    if j not in NCTIds_history[NCTIds[-1][0]]:
                        NCTIds_history[NCTIds[-1][0]][j] = []
                    key, *text = features[i]['text'][j]
                    key = key.split('->')[-1]
                    key = self.tokenizer(key, add_special_tokens=False, return_attention_mask=False, truncation=False)['input_ids']

                    space_left = input_number * self.max_length - (len(input_ids) + len(key) + 1)
                    if space_left < 1: continue
                    sentences = [self.tokenizer(t,
                                                add_special_tokens=False,
                                                return_attention_mask=False,
                                                truncation=False)['input_ids']
                                 for t in text]
                    if len(sentences) == 1:
                        random_index = [0]
                    else:
                        if sum([len(s) for s in sentences]) <= space_left:
                            random_index = list(range(len(sentences)))

                        else:
                            if random.uniform(0, 1) < 1/2:
                                count = [len(s) for s in sentences]
                                n_token = sum(count)
                                probability_distribution = [t / n_token for t in count]
                                random_index = choice(range(len(sentences)), len(sentences), replace=False,
                                                      p=probability_distribution)
                            else:
                                random_index = list(range(max([random.randint(0, len(sentences) - 1), 0]), len(sentences)))

                    tmp = []
                    sentences_number_taken, sentence_index = [], []
                    for sentence_i in random_index:
                        if len(tmp) + len(sentences[sentence_i]) <= space_left:
                            from_tmp = len(tmp)
                            tmp += sentences[sentence_i]
                            to_tmp = len(tmp)
                            sentences_number_taken.append(sentence_i)
                            sentence_index.append([from_tmp, to_tmp])
                        else:
                            break

                    if len(tmp) > 0:
                        from_input_ids = len(input_ids)
                        input_ids += key + tmp + [self.tokenizer.sep_token_id]
                        # key range:
                        key_idx = [from_input_ids, from_input_ids + len(key)]
                        # sentence range:
                        sentences_idx = [list(v) for v in from_input_ids + len(key) + np.array(sentence_index)]
                        if len(set(NCTIds_history[NCTIds[-1][0]][j])|set(sentences_number_taken)) == len(sentences) or\
                                NCTIds_history[NCTIds[-1][0]][j] == 'Done':
                            NCTIds_history[NCTIds[-1][0]][j] = 'Done'
                        else:
                            NCTIds_history[NCTIds[-1][0]][j] += sentences_number_taken

                        NCTIds[-1].append([j] + [['key'] + key_idx] + [[s] + from_to for s, from_to in zip(sentences_number_taken, sentences_idx)])
                        if len(input_ids) == input_number*self.max_length: break
                        continue

                    if len(input_ids) + 2 >= input_number*self.max_length: break


                input_ids += [self.tokenizer.pad_token_id] * (input_number*self.max_length - len(input_ids))
                if input_number != self.number_inputs:
                    input_ids += [self.tokenizer.cls_token_id]

            attention_mask = np.array([1 for _ in input_ids])
            attention_mask[np.array(input_ids)==1] = 0
            features[i] = {**features[i], **{'input_ids': input_ids[:-1],
                                             'attention_mask': attention_mask}}
            del features[i]['text']

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.number_inputs*self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt" if labels is None else None,
        )
        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        # batch = {k: torch.tensor(v, dtype=torch.int64).to(self.device) for k, v in batch.items()}
        assert len(NCTIds) == len(NCTIds)
        return NCTIds, batch

from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel, RobertaClassificationHead
from torch.nn import MSELoss, CrossEntropyLoss
from torch import mean, stack
class MultiInputsRobertaForSequenceClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, number_inputs=8):
        super().__init__(config)
        self.number_inputs = number_inputs
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        sequence_output, outputs = [], []
        for i in range(self.number_inputs):
            outputs.append(self.roberta(
                input_ids[:, i * 512:i * 512 + 512],
                attention_mask=attention_mask[:, i * 512:i * 512 + 512],
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            ))
            sequence_output.append(outputs[-1][0])

        sequence_output = stack(sequence_output, dim=3)
        sequence_output = mean(sequence_output, dim=3)

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        if outputs[0].attentions:
            attentions = ()
            for layer in range(len(outputs[0].attentions)):
                attentions += (torch.cat([outputs[i].attentions[layer] for i in range(self.number_inputs)], dim=2), )
            outputs = outputs[0]
            outputs.attentions = attentions

        else:
            outputs = outputs[0]

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def normalize_fn(v, normalizer, normalization=None):
    # in order to avoid decentralization of 0 vectors, which happens when a sentence has not been sampled
    # (only when sampling size is really small)
    if np.all(v==0):
        return v
    # Normalization
    if normalization == 'min-max':
        return (v - normalizer['min']) / (normalizer['max'] - normalizer['min'])
    elif normalization == 'standardized':
        return (v - normalizer['mean']) / normalizer['std']
    elif normalization == 'robust-mean':
        return v / normalizer
    else:
        return v

def classify(parsed_files,
             use_cuda=True,
             batch_size=16,
             sampling_size=8,
             conservative=False,
             interpretability_method=None,
             normalization=None,
             number_inputs=8,
             html_output_dir=None):

    '''


    Args:
        parsed_files: a parsed file

        use_cuda: True if you want to use GPU with cuda.

        batch_size: batch size

        sampling_size: number of time we want to sample on the CTs field

        conservative: will output only predictions where all the samples agree on the same prediction.

        interpretability_method: default is None, can be set to captum or attention_signal.

        normalization: Way to normalize interpretability scores, can be set to standardized, min-max, or robust-mean

        number_inputs: if >1 the model won't be able to use captum for interpretability, doesn't change the results much,
        however is faster when compared with proportiannaly equal sample size.
        Will sample number_inputs time and compute a latent representation for each of the input,
        the mean of those representations will then be classified.

        html_output_dir: default is None, if set, will output an html of the interpretability

    Returns:


    '''

    assert not (interpretability_method == 'captum' and number_inputs > 1), \
        'If captum interpretability method is used, please set the number of inputs to 1'

    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
    tokenizer_path = working_dir + 'Tokenizer/bio_english.model'

    tokenizer = SPMTokenizer(tokenizer_path)
    tokenizer.vocab_files_names['vocab_file'] = tokenizer_path

    if interpretability_method == 'attention_signal':
        output_attentions = True
    else:
        output_attentions = False

    model = MultiInputsRobertaForSequenceClassification.from_pretrained(working_dir, num_labels=3, output_attentions=output_attentions, number_inputs=number_inputs).eval().to(device)

    dataset = RiskLickLineByLineClassificationDataset(tokenizer, parsed_files, sampling_size, number_inputs=number_inputs)
    sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    collator = MultiInputsRiskLickDataCollatorClassification(tokenizer=tokenizer, device=device, number_inputs=number_inputs)
    data_loader = torch.utils.data.dataloader.DataLoader(dataset,
                                                         batch_size=batch_size,
                                                         sampler=sampler,
                                                         collate_fn=collator,
                                                         pin_memory=True,
                                                         num_workers=2)

    if interpretability_method == 'captum':
        def predict(inputs, attention_mask=None):
            output = model(inputs, attention_mask=attention_mask)
            return output['logits']

        def classification_fwd_func(inputs, attention_mask=None):
            pred = predict(inputs, attention_mask=attention_mask)
            return pred.max(1).values

        lig = LayerIntegratedGradients(classification_fwd_func, model.roberta.embeddings)

        def summarize_attributions(attributions):
            attributions = attributions.sum(dim=-1).squeeze(0)
            attributions = attributions / torch.norm(attributions)
            return attributions

    prob_predictions = {}
    if interpretability_method:
        scores_memory = {}

    for NCTId, batch in tqdm(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            if interpretability_method:
                for i, idx in enumerate(NCTId):
                    if idx[0] not in scores_memory:
                        scores_memory[idx[0]] = {
                            'token_scores': [],
                            'token_ids': [],
                            'mapping': []
                        }

                    if interpretability_method == 'attention_signal':
                        token_scores = []
                        for n_input in range(number_inputs):
                            tokens_score = np.sum(np.mean(outputs.attentions[0][i][:].to('cpu').numpy()[:, n_input * 512:n_input * 512 + 512], axis=0), axis=0)
                            for layer_i in range(len(outputs.attentions) - 1):
                                tokens_score = np.dot(tokens_score,
                                              np.mean(np.array(outputs.attentions[layer_i + 1][i][:].to('cpu').numpy()[:, n_input * 512:n_input * 512 + 512]), axis=0))
                            token_scores.append(tokens_score)

                        token_scores = np.array([item for sublist in token_scores for item in sublist])
                        scores_memory[idx[0]]['token_scores'].append(token_scores)
                        scores_memory[idx[0]]['token_ids'].append(batch['input_ids'][i].to('cpu').numpy())
                        scores_memory[idx[0]]['mapping'].append(idx[1:])

                    elif interpretability_method == 'captum':
                        token_scores = []
                        token_scores.append(summarize_attributions(lig.attribute(
                            batch['input_ids'][i].reshape([1, -1]),
                            additional_forward_args=batch['attention_mask'][i].reshape([1, -1])))
                        )

                        token_scores = torch.tensor([item for sublist in token_scores for item in sublist])
                        scores_memory[idx[0]]['token_scores'].append(token_scores.to('cpu').numpy())
                        scores_memory[idx[0]]['token_ids'].append(batch['input_ids'][i].to('cpu').numpy())
                        scores_memory[idx[0]]['mapping'].append(idx[1:])

            for i, idx in enumerate(NCTId):
                if idx[0] not in prob_predictions:
                    prob_predictions[idx[0]] = [softmax(outputs['logits'][i].to('cpu').numpy())]
                else:
                    prob_predictions[idx[0]].append(softmax(outputs['logits'][i].to('cpu').numpy()))

    NCTIds = set(prob_predictions.keys())
    softmax_prediction = {}
    for NCTId in NCTIds:
        pred = np.array([np.argmax(v) for v in prob_predictions[NCTId]])
        if conservative and np.all(pred[0] != pred):
            del prob_predictions[NCTId]
        else:
            prob_prediction = np.array([np.array(v) for v in prob_predictions[NCTId]])
            softmax_prediction[NCTId] = torch.tensor(np.mean(prob_prediction, axis=0))
            prob_predictions[NCTId] = int(np.argmax(softmax_prediction[NCTId]))

    assert not (len(prob_predictions) == 0 and conservative), \
        'No predictions returned with conservative setting: could be the sample size too high'

    if interpretability_method:
        sentences_interpretability, words_interpretability = {}, {}
        for key in prob_predictions.keys():
            ct_dict = {}
            normalizer = []
            for i, mapping in enumerate(scores_memory[key]['mapping']):
                for field in mapping:
                    field_number = field[0]
                    key_range = field[1]
                    if field_number not in ct_dict:
                        ct_dict[field_number] = {'key_token_ids': scores_memory[key]['token_ids'][i][key_range[1]: key_range[2]],
                                                 'key_token_scores': [scores_memory[key]['token_scores'][i][key_range[1]: key_range[2]]]}
                    else:
                        assert np.all(scores_memory[key]['token_ids'][i][key_range[1]: key_range[2]] == ct_dict[field_number]['key_token_ids'])
                        ct_dict[field_number]['key_token_scores'].append(scores_memory[key]['token_scores'][i][key_range[1]: key_range[2]])
                    normalizer.append(ct_dict[field_number]['key_token_scores'][-1])

                    for sentence_range in field[2:]:
                        sentence_number = sentence_range[0]
                        if sentence_number not in ct_dict[field_number]:
                            ct_dict[field_number][sentence_number] = {
                                'token_ids': scores_memory[key]['token_ids'][i][sentence_range[1]: sentence_range[2]],
                                'token_scores': [scores_memory[key]['token_scores'][i][sentence_range[1]: sentence_range[2]]],
                            }
                        else:
                            assert np.all(scores_memory[key]['token_ids'][i][sentence_range[1]: sentence_range[2]] == ct_dict[field_number][sentence_number]['token_ids'])
                            ct_dict[field_number][sentence_number]['token_scores'].append(scores_memory[key]['token_scores'][i][sentence_range[1]: sentence_range[2]])
                        normalizer.append(ct_dict[field_number][sentence_number]['token_scores'][-1])

            normalizer = np.array([item for sublist in normalizer for item in sublist])
            if normalization == 'min-max':
                normalizer = {'min': min(normalizer), 'max': max(normalizer)}
            elif normalization == 'standardized':
                normalizer = {'mean': np.mean(normalizer), 'std': np.std(normalizer)}
            elif normalization == 'robust-mean':
                a = 1.0 * np.array(normalizer)
                n = len(a)
                m, se = np.mean(a), scipy.stats.sem(a)
                h = se * scipy.stats.t.ppf((1 + .99) / 2., n - 1)
                normalizer = (m+h)
            else:
                normalizer = None


            sentences, tokens, tokens_score = [], [], []
            for i, field in enumerate(parsed_files[key]):
                if i not in ct_dict:
                    ct_dict[i] = {}
                    ct_dict[i]['key_token_ids'] = np.array(tokenizer.encode(field[0], add_special_tokens=False))
                    ct_dict[i]['key_token_scores'] = [np.array([0] * len(ct_dict[i]['key_token_ids']))]
                    # for j in range(len(field[1:])):
                    #     ct_dict[i][j] = {'token_ids': np.array(tokenizer.encode(field[j], add_special_tokens=False))}
                    #     ct_dict[i][j]['token_scores'] = [np.array([0] * len(ct_dict[i][j]['token_ids'] ))]

                sentences.append([field[0]])
                tokens.append([tokenizer.convert_ids_to_tokens(ct_dict[i]['key_token_ids'])])
                tokens_score.append([normalize_fn(np.mean(ct_dict[i]['key_token_scores'], axis=0), normalizer, normalization)])
                for j, s in enumerate(field[1:]):
                    sentences[-1].append(s)
                    if j not in ct_dict[i]:
                        ct_dict[i][j] = {'token_ids': np.array(tokenizer.encode(field[j], add_special_tokens=False))}
                        ct_dict[i][j]['token_scores'] = [np.array([0] * len(ct_dict[i][j]['token_ids'] ))]
                    tokens[-1].append(tokenizer.convert_ids_to_tokens(ct_dict[i][j]['token_ids']))
                    tokens_score[-1].append(normalize_fn(np.mean(ct_dict[i][j]['token_scores'], axis=0), normalizer, normalization))

            tokenized_sentences, sentence_scores, new_count, words_list, word_scores_list = [], [], 0, [], []
            for field_number in range(len(tokens)):
                words_list.append([]), word_scores_list.append([]), sentence_scores.append([])
                for sentence, token, scores in zip(sentences[field_number], tokens[field_number], tokens_score[field_number]):
                    words_list[-1].append([]), word_scores_list[-1].append([])
                    for tok, score in zip(token, scores):
                        if '▁' in tok:
                            words_list[-1][-1].append(tokenizer.convert_tokens_to_string([tok]))
                            word_scores_list[-1][-1].append(score)

                        else:
                            words_list[-1][-1][-1] = tokenizer.convert_tokens_to_string([words_list[-1][-1][-1], tok])
                            word_scores_list[-1][-1][-1] = max([score, word_scores_list[-1][-1][-1]])

                    sentence_scores[-1].append(np.mean(word_scores_list[-1][-1]))

            words_, scores_ = [], []
            for field_number, (words, scores) in enumerate(zip(words_list, word_scores_list)):
                words_.append('->'.join(sentences[field_number][0].split('->')[:-1]) + '->')
                scores_.append(0)
                words_.append(words[0][0])
                words_.append(' &nbsp -> &nbsp ')
                scores_.append(scores[0][0])
                scores_.append(0)
                for word_sentences, word_sentence_scores in zip(words[1:], scores[1:]):
                    for w, s in zip(word_sentences, word_sentence_scores):
                        words_.append(w)
                        scores_.append(s)

                words_.append(' <br>  <br> ')
                scores_.append(0)

            words_interpretability[key] = {
                'text_list': words_,
                'score_list': scores_
            }

            if html_output_dir:
                vis = viz.VisualizationDataRecord(
                    scores_,
                    torch.max(softmax_prediction[key]),
                    torch.argmax(softmax_prediction[key]),
                    torch.argmax(softmax_prediction[key]),
                    0, 0, words_, 0
                )
                html = viz.visualize_text([vis])
                html_file_path = html_output_dir + str(
                    key) + '_' + interpretability_method + '_word.html'
                with open(html_file_path, encoding='utf-8', mode='w') as f:
                    f.write(html.data)

            sentences_, sentences_scores_ = [], []
            for sentence, scores in zip(sentences, sentence_scores):
                sentences_.append(sentence[0])
                sentences_.append(' &nbsp -> &nbsp ')
                sentences_scores_.append(scores[0])
                sentences_scores_.append(0)
                for sent, score in zip(sentence[1:], scores[1:]):
                    sentences_.append(sent)
                    sentences_scores_.append(score)

                sentences_.append(' <br>  <br> ')
                sentences_scores_.append(0)

            sentences_interpretability[key] = {
                'text_list': sentences_,
                'score_list': sentences_scores_
            }

            if html_output_dir:
                vis = viz.VisualizationDataRecord(
                    sentences_scores_,
                    torch.max(softmax_prediction[key]),
                    torch.argmax(softmax_prediction[key]),
                    torch.argmax(softmax_prediction[key]),
                    0, 0, sentences_, 0
                )
                html = viz.visualize_text([vis])
                html_file_path = html_output_dir + str(
                    key) + '_' + interpretability_method + '_sentences.html'
                with open(html_file_path, encoding='utf-8', mode='w') as f:
                    f.write(html.data)

        return prob_predictions, sentences_interpretability, words_interpretability

    else:

        return prob_predictions


def main():


    # parse it with our generic function
    parsed_files = file_parser(working_dir + 'json_list.json')

    # Examples of prediction only
    # prob_predictions = classify(parsed_files, interpretability_method=None, number_inputs=1, batch_size=batch_size, sampling_size=int(sampling_size*input_multiplier)) # 58min
    # prob_predictions = classify(parsed_files, interpretability_method=None, number_inputs=input_multiplier, batch_size=int(batch_size/input_multiplier), sampling_size=sampling_size) #38min

    # Examples of interpretability
    # prob_predictions, sentence_interpretability, words_interpretability =\
    #     classify(parsed_files,
    #              use_cuda=True,
    #              batch_size=16,
    #              sampling_size=8,
    #              conservative=True,
    #              interpretability_method='attention_signal',
    #              normalization='robust-mean',
    #              number_inputs=4,
    #              html_output_dir='/home/jknafou/interpretability_risk/')
    prob_predictions, sentence_interpretability, words_interpretability =\
        classify(parsed_files,
                 use_cuda=True,
                 batch_size=16,
                 sampling_size=2,
                 conservative=True,
                 interpretability_method='captum',
                 normalization='standardized',
                 number_inputs=1,
                 html_output_dir='/home/jknafou/interpretability_risk/')



if __name__ == '__main__':

    main()

# some tests:
#                 precision    recall  f1-score   support
# with number_inputs=1
#            0     0.5517    0.6937    0.6146      6497
#            1     0.3802    0.2997    0.3352      3186
#            2     0.7951    0.7209    0.7562     10702
#
#     accuracy                         0.6464     20385
#    macro avg     0.5756    0.5714    0.5687     20385
# weighted avg     0.6527    0.6464    0.6453     20385
#
# with number_inputs=8
#            0     0.5336    0.6895    0.6016      6497
#            1     0.3536    0.2737    0.3086      3186
#            2     0.7847    0.6983    0.7390     10702
#
#     accuracy                         0.6291     20385
#    macro avg     0.5573    0.5538    0.5497     20385
# weighted avg     0.6373    0.6291    0.6279     20385