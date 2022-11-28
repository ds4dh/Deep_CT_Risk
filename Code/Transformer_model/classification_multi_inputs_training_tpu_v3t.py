import logging, os
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel, RobertaClassificationHead
from torch.nn import MSELoss, CrossEntropyLoss
from torch import mean, stack

number_inputs = 8


class MultiInputsRobertaForSequenceClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.number_inputs = number_inputs
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

        sequence_output = []
        for i in range(self.number_inputs):
            outputs = self.roberta(
                input_ids[:, i*512:i*512+512],
                attention_mask=attention_mask[:, i*512:i*512+512],
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output.append(outputs[0])

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

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def main():
    import torch_xla.core.xla_model as xm
    # process_number = 0
    process_number = xm.get_ordinal()
    # number_of_processes = 8
    number_of_processes = xm.xrt_world_size()

    # PARAMETERS
    max_position_embeddings=512
    per_device_train_batch_size=16
    total_epochs=5
    train_sampling_size=2
    dev_sampling_size=16
    test_sampling_size=64
    # total_epochs=2
    # train_sampling_size=10
    # dev_sampling_size=2
    # test_sampling_size=2
    learning_rate=2e-5
    gradient_accumulation_steps=1
    resume_training=False

    # PATH
    model_name = 'bio-roberta-small'
    # model_name = 'biobert-v1.1'
    # sub_model_name='dmis-lab'

    model_path = '/home/jknafou/models/' + model_name
    sub_model_name = ''

    dst_model_path = '/home/jknafou/models/classifiers_3_levels_v3t/' + model_name
    if process_number == 0 and not os.path.exists(dst_model_path):
        os.mkdir(dst_model_path)

    # find last checkpoint
    if resume_training:
        import glob
        ckpt_number = max([int(p.split('-')[-1]) for p in glob.glob(dst_model_path + '/*') if 'checkpoint' in p])
        ckpt_path = dst_model_path + '/checkpoint-' + str(ckpt_number)
    else:
        ckpt_number = 0
        ckpt_path = None


    # TOKENIZER
    if model_name == 'bio-roberta-small':
        tokenizer_path = model_path + '/Tokenizer/bio_english.model'
        from transformers.models.roberta.tokenization_bio_spm import SPMTokenizer
        tokenizer = SPMTokenizer(tokenizer_path)
        tokenizer.vocab_files_names['vocab_file'] = tokenizer_path

    else:
        per_device_train_batch_size = int(per_device_train_batch_size/2)
        gradient_accumulation_steps = int(gradient_accumulation_steps*2)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(sub_model_name + '/' + model_name)

    predictions_path = dst_model_path + '/predictions'
    if process_number == 0 and not os.path.exists(predictions_path):
        os.mkdir(predictions_path)

    # DATASET
    train_data_path = '/home/jknafou/risklick_classification/dataset_v3t_train.tsv'
    # train_data_path = '/home/jknafou/risklick_classification/dataset_train_v3.tsv_h100'
    dev_data_path = '/home/jknafou/risklick_classification/dataset_v3t_valid.tsv'
    test_data_path = '/home/jknafou/risklick_classification/dataset_v3t_test.tsv'
    # test_data_path = '/home/jknafou/risklick_classification/dataset_test_v3.tsv_h100'


    from transformers.data.datasets.language_modeling import RiskLickLineByLineClassificationDataset
    train_dataset = RiskLickLineByLineClassificationDataset(
        tokenizer=tokenizer,
        # file_path=test_data_path,
        file_path=train_data_path,
        block_size=max_position_embeddings,
        sampling_size=train_sampling_size,
        number_inputs=number_inputs
    )
    dev_dataset = RiskLickLineByLineClassificationDataset(
        tokenizer=tokenizer,
        # file_path=test_data_path,
        file_path=dev_data_path,
        block_size=max_position_embeddings,
        sampling_size=dev_sampling_size,
        number_inputs=number_inputs
    )
    # for batch in dev_dataset:
    #     print(batch)
    test_dataset = RiskLickLineByLineClassificationDataset(
        tokenizer=tokenizer,
        file_path=test_data_path,
        block_size=max_position_embeddings,
        sampling_size=test_sampling_size,
        number_inputs=number_inputs
    )

    # from transformers.data.data_collator import MultiInputsRiskLickDataCollatorClassification
    # data_collator = MultiInputsRiskLickDataCollatorClassification(tokenizer=tokenizer, number_inputs=number_inputs, split='eval')
    # from torch.utils.data.sampler import RandomSampler
    # sampler = RandomSampler(train_dataset)
    # from torch.utils.data.dataloader import DataLoader
    # data_loader = DataLoader(dev_dataset, batch_size=per_device_train_batch_size, sampler=sampler, collate_fn=data_collator)
    #
    # for batch in data_loader:
    #     print(sum(batch['labels'])/len(batch['labels']))
    #     print(batch['input_ids'].shape)
    #     print(batch['NCTId_sentences'])

        # output = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        # output = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        # break

    # MODEL LOADING
    from math import ceil
    total_steps = ceil(len(train_dataset) / (per_device_train_batch_size * number_of_processes * gradient_accumulation_steps)) * total_epochs
    if model_name == 'bio-roberta-small':
        tokenizer_path = model_path + '/Tokenizer/bio_english.model'
        from transformers.models.roberta.tokenization_bio_spm import SPMTokenizer
        tokenizer = SPMTokenizer(tokenizer_path)
        tokenizer.vocab_files_names['vocab_file'] = tokenizer_path
        from transformers import AutoModelForSequenceClassification
        if resume_training:
            if ckpt_number == total_steps:
                logging.warning('dst: ' + dst_model_path)
                model = MultiInputsRobertaForSequenceClassification.from_pretrained(dst_model_path, num_labels=3)
            else:
                logging.warning('ckpt: ' + ckpt_path)
                model = MultiInputsRobertaForSequenceClassification.from_pretrained(ckpt_path, num_labels=3)

        else:
            model = MultiInputsRobertaForSequenceClassification.from_pretrained(model_path, num_labels=3)

    else:
        per_device_train_batch_size = int(per_device_train_batch_size/2)
        gradient_accumulation_steps = int(gradient_accumulation_steps*2)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(sub_model_name + '/' + model_name)
        model = AutoModelForSequenceClassification.from_pretrained(sub_model_name + '/' + model_name, num_labels=3)

    from transformers.trainer_utils import IntervalStrategy
    main_metric_name = 'micro_f1-score'
    from transformers import TrainingArguments
    training_args = TrainingArguments(
        save_total_limit=10,
        output_dir=dst_model_path,
        logging_dir='/home/jknafou/models/classifiers_3_levels_v3t/run/' + model_name,
        do_train=True,
        do_eval=True,
        do_predict=True,
        num_train_epochs=total_epochs,
        evaluation_strategy=IntervalStrategy.EPOCH,
        logging_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.EPOCH,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        eval_accumulation_steps=1,
        weight_decay=0.01,
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model=main_metric_name,
        tpu_num_cores=number_of_processes,
        dataloader_num_workers=8,
        gradient_accumulation_steps=gradient_accumulation_steps
    )

    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    from numpy import array, arange, zeros, mean, argmax
    from sklearn.metrics import classification_report
    from scipy.special import softmax
    def compute_metrics(eval_pred, metric_key_prefix, best_metric, NCTIds, NCTId_sentences):
        best_metric = 0 if best_metric == None else best_metric
        predictions, labels = eval_pred
        # predictions = [[-0.31806123 -0.1698934,   0.51713085],
        #              [-0.32364443, -0.12854864,  0.5395399 ],
        #              [-0.3242733,  -0.1636772,   0.564375  ],
        #              [-0.21213527,  0.07965711,  0.2839368 ],
        #              [-0.19090039,  0.11141474,  0.25634587],
        #              [-0.19821353,  0.11115072,  0.25010714]]
        # labels = [2, 2, 2, 2, 1, 1]
        # metric_key_prefix = 'eval'
        # global_step = 149
        prob_predictions = array([softmax(p) for p in predictions])

        results_dict = {}
        results = ''
        assert len(NCTId_sentences) == len(NCTIds)
        for NCTId, l, p, NCTId_sentence in zip(NCTIds, labels, prob_predictions, NCTId_sentences):
            results += NCTId + '\t' + ','.join([str(s) for s in NCTId_sentence if s != 522]) + '\t' + str(l) + '\t' + str(p) + '\n'
            if NCTId not in results_dict.keys():
                results_dict[NCTId] = {
                    'labels': [l],
                    'prob_predictions': [p]
                }
            else:
                results_dict[NCTId]['labels'].append(l)
                results_dict[NCTId]['prob_predictions'].append(p)

        labels, predictions = [], []
        for NCTId in set(NCTIds):
            # logging.warning(results_dict[NCTId]['labels'])
            assert len(set(results_dict[NCTId]['labels'])) == 1
            labels.append(results_dict[NCTId]['labels'][0])
            predictions.append(int(argmax(mean(results_dict[NCTId]['prob_predictions'], axis=0))))

        f1_score_micro = f1_score(labels, predictions, average='micro')

        if xm.get_ordinal() == 0:
            target_names = ['No Risk', 'Medium Risk', 'High Risk']
            logging.warning(classification_report(labels, predictions, digits=4, target_names=target_names))
            if (metric_key_prefix == 'eval' and f1_score_micro > best_metric) or \
                    metric_key_prefix == 'test' :

                with open(predictions_path + '/predictions_by_attribute_' +
                          metric_key_prefix + '.txt', encoding='utf-8', mode='w') as f:
                    f.write(results.strip())

                results = ''
                for NCTId, l, p in zip(set(NCTIds), labels, predictions):
                    results += NCTId + '\t' + str(l) + '\t' + str(p) + '\n'

                with open(predictions_path + '/predictions_' +
                          metric_key_prefix + '.txt', encoding='utf-8', mode='w') as f:
                    f.write(results.strip())



        return {main_metric_name: f1_score_micro,
                'macro_f1-score': f1_score(labels, predictions, average='macro'),
                'micro_precision': precision_score(labels, predictions, average='micro'),
                'macro_precision': precision_score(labels, predictions, average='macro'),
                'micro_recall': recall_score(labels, predictions, average='micro'),
                'macro_recall': recall_score(labels, predictions, average='macro'),
                }

    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        # data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    if ckpt_number < total_steps:
        # train model
        if resume_training:
            trainer.train(ckpt_path)
        else:
            trainer.train()

        #save best model
        trainer.save_model(dst_model_path)

    #compute test set metrics on best model
    test_metrics = trainer.evaluate(test_dataset, metric_key_prefix='test')
    if process_number == 0:
        logging.warning(test_metrics)
        import json
        json_string = json.dumps(test_metrics, indent=2, sort_keys=True) + "\n"
        with open(dst_model_path + '/test_eval.txt',
                  encoding='utf-8', mode='w') as f:
            f.write(json_string)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()

#                precision    recall  f1-score   support
#
#      No Risk     0.4603    0.2976    0.3615      3232
#  Medium Risk     0.4022    0.2241    0.2878      5792
#    High Risk     0.7890    0.9137    0.8468     23451
#
#     accuracy                         0.7294     32475
#    macro avg     0.5505    0.4785    0.4987     32475
# weighted avg     0.6873    0.7294    0.6988     32475

#      No Risk     0.4581    0.2973    0.3606      3232
#  Medium Risk     0.4043    0.2248    0.2889      5792
#    High Risk     0.7888    0.9134    0.8465     23451
#
#     accuracy                         0.7293     32475
#    macro avg     0.5504    0.4785    0.4987     32475
# weighted avg     0.6873    0.7293    0.6987     32475


