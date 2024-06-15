import sys
import os

sys.path.append('.')
sys.path.append('..')


import json
import time
import argparse
from tqdm import tqdm

import torch
from transformers import IntervalStrategy, T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer

from dataloader import JsonLoader
from configs.sketch_generation_config import sketch_data_train_path, sketch_data_dev_path, sketch_data_test_path, output_log_dir, model_save_dir
from utils.file_util import pickle_load, pickle_save, HFDataset2


def flatten(test_list):
    if isinstance(test_list, list):
        temp = []
        for ele in test_list:
            temp.extend(flatten(ele))
        return temp
    else:
        return [test_list]

def lisp_to_nested_expression(lisp_string):
    '''
       Converts sketch from string to tree. 
       Raises error if sketch is not syntactically valid.
    '''
    stack = []
    current_expression = []
    tokens = lisp_string.split()

    assert lisp_string.count('(') == lisp_string.count(')'), "unbalanced sexp"

    for token in tokens:
        while token[0] == '(':
            nested_expression = []
            current_expression.append(nested_expression)
            stack.append(current_expression)
            current_expression = nested_expression
            token = token[1:]
        current_expression.append(token.replace(')', ''))
        while token[-1] == ')':
            current_expression = stack.pop()
            token = token[:-1]

    assert len(tokens) == len(flatten(current_expression[0])), "sketch_generation.py : invalid s_exp (syntax)"
    return current_expression[0]


class SketchGeneration:

    def __init__(self, params: dict, mode='qa'):
        self.params = params

        self.model_name = params.get('model_name', 't5-base')  # 'google/t5-v1_1-base'
        self.train_batch_size = params.get('train_batch_size', 4) #8
        self.eval_batch_size = params.get('eval_batch_size', 32) #32
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, cache_dir = "~/.cache/huggingface/transformers/", local_files_only=False)
        self.device = torch.device(params.get('device', 'cuda')) if torch.cuda.is_available() else 'cpu'
        print(self.device)

        self.max_target_length = 128
        self.max_source_length = 128
        self.model_eval(model_save_dir)

        # generation settings
        self.num_beams = params.get('num_beams', 10)#10

    def model_eval(self, model_dir):
        print(model_dir + '/pytorch_model.bin')
        if os.path.isfile(model_dir + '/pytorch_model.bin'):
            print("Loading trained model for eval")
            self.model = T5ForConditionalGeneration.from_pretrained(model_dir)
            self.model.eval()
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.model.to('cuda')

    def encode(self, dataloader: JsonLoader, output_path, task_prefix='translate English to Lisp: '):
        encodings_path = output_path + '/' + dataloader.get_dataset_split() + '_encodings'
        if output_path is not None and os.path.isfile(encodings_path):
            print(f"Reading train and eval data from : {output_path}")
            res = pickle_load(encodings_path)
            return res
 
        input_sequences = []
        output_sequences = []

        for idx in tqdm(range(0, dataloader.len)):  # for each question
            question_id = dataloader.get_question_id_by_idx(idx)
            question = dataloader.get_question_by_idx(idx)
            gt_sketch = dataloader.get_sketch_by_idx(idx)

            input_sequences.append(question)
            output_seq = gt_sketch
            output_sequences.append(output_seq)

        encodings = self.tokenizer([task_prefix + sequence for sequence in input_sequences], padding='max_length', max_length=self.max_source_length, truncation=True)
        input_ids, attention_mask = encodings.input_ids, encodings.attention_mask


        for i in range(0, min(5, dataloader.len)):
            print('[Input]' + input_sequences[i])
            print('[Output]' + output_sequences[i])
            print('[Input token len]' + str(len(self.tokenizer.tokenize(input_sequences[i]))))
            print('\n')

        res = {'input_ids': input_ids, 'attention_mask': attention_mask}

        # encode the targets
        if len(output_sequences):
            target_encoding = self.tokenizer(output_sequences, padding='max_length', max_length=self.max_target_length, truncation=True)
            labels = target_encoding.input_ids
            
            # replace padding token id's of the labels by -100
            labels = torch.tensor(labels)  
            labels[labels == self.tokenizer.pad_token_id] = -100
            res['labels'] = labels

        pickle_save(res, encodings_path)
        return res


    def train(self, train_dataloader, dev_dataloader, output_dir, gradient_accumulation_step=1,num_train_epochs=10):
        print("output_dir : ", output_dir)
        assert output_dir is not None and os.path.isdir(output_dir)
        if os.path.isfile(output_dir + '/pytorch_model.bin'):
            print('Model already exists in {}'.format(output_dir))
            return

        # training settings
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, cache_dir = "~/.cache/huggingface/transformers/", local_files_only=False)
        self.model.resize_token_embeddings(len(self.tokenizer))

        train_encodings = self.encode(train_dataloader, output_dir.rstrip('/'))
        dev_encodings = self.encode(dev_dataloader, output_dir.rstrip('/'))


        train_dataset = HFDataset2(train_encodings)
        dev_dataset = HFDataset2(dev_encodings)

        training_args = Seq2SeqTrainingArguments(output_dir=output_dir, do_train=True, do_eval=True, do_predict=True,
                                                 evaluation_strategy=IntervalStrategy.STEPS, save_strategy=IntervalStrategy.STEPS, save_steps=10000 , eval_steps=2500,
                                                 per_device_train_batch_size=self.train_batch_size, per_device_eval_batch_size=self.eval_batch_size, num_train_epochs=num_train_epochs,
                                                 learning_rate=3e-5, gradient_accumulation_steps=gradient_accumulation_step,
                                                 load_best_model_at_end=False)

        trainer = Seq2SeqTrainer(self.model, args=training_args, train_dataset=train_dataset, eval_dataset=dev_dataset)
        best_run = trainer.train(resume_from_checkpoint=False)

        trainer.save_model(output_dir)
   
    def solve(self, dataloader: JsonLoader, output_log_dir, file_name, start_index=-1, end_index=-1):
        logs = []
        logs_with_ans = {}
        time_str = time.strftime('_%Y_%m_%d_%H_%M_%S')
        split = dataloader.get_dataset_split()

        
        cum_str_EM = 0
        ques_counter = 0

        for idx in tqdm(range(0, dataloader.get_len())):  # for each question
            
            if start_index != -1 and end_index != -1:
                if idx < start_index or idx >= end_index:
                    continue
            
            question_id = dataloader.get_question_id_by_idx(idx)
            question = dataloader.get_question_by_idx(idx)
            
            golden_sketch = ''
            # if split != 'test':
            golden_sketch = dataloader.get_sketch_by_idx(idx)

                       
            task_prefix='translate English to Lisp: '
            input_ids = self.tokenizer(task_prefix + question, return_tensors='pt', max_length=self.max_source_length, truncation=True, padding="max_length").input_ids

            # outputs = self.model.generate(input_ids.to('cuda'), max_length=self.max_target_length, num_beams=self.num_beams, num_return_sequences=self.num_beams,
            #                             logits_processor=self.logits_processor_list,output_scores=True, return_dict_in_generate=True)
            outputs = self.model.generate(input_ids.to('cuda'), max_length=self.max_target_length, num_beams=self.num_beams, num_return_sequences=self.num_beams
                                          ,output_scores=True, return_dict_in_generate=True)


            predictions = []
            pred_score = []
            outputs_seq = outputs['sequences']
            for i in range(0, len(outputs_seq)):
                output = self.tokenizer.decode(outputs_seq[i], skip_special_tokens=True)
                if '^^' not in output:
                    output = output.replace('http://www.w3.org/2001/XMLSchema#', '^^http://www.w3.org/2001/XMLSchema#')
                predictions.append(output)
                pred_score.append(float(outputs['sequences_scores'][i]))


            valid_predictions = []
            for pred_idx, pred in enumerate(predictions):
                try:
                    form_1 = lisp_to_nested_expression(pred)
                    if (not pred in valid_predictions):
                        valid_predictions.append(pred)
                except Exception as e:
                    continue
                    
            # metrics
            valid_sketch = ''
            str_em = 0.0
            if len(valid_predictions) > 0:
                valid_sketch = valid_predictions[0]
                if valid_sketch==golden_sketch:
                    str_em = 1.0
            
            cum_str_EM += str_em
            ques_counter += 1
            avg_str_EM = round(cum_str_EM/ques_counter,2)

            log = {'serial': idx, 'qid': question_id, 'question': question, 'golden_sketch': golden_sketch, 'predicted_sketch': valid_sketch,
                   'top_predictions': valid_predictions, 'top_scores': pred_score, 'str EM': str_em,'avg_str_EM': avg_str_EM}
            logs.append(log)
            print('[' + str(idx) + ']','str EM:', avg_str_EM)

            with open(f'{output_log_dir}/{file_name}.jsonl', 'a+') as f:
                f.write(json.dumps(log) + '\n')

            logs_with_ans[question_id] = {'predicted_sketch': valid_sketch, 'golden_sketch': golden_sketch}
        # end for each question
        # save_log(logs, 'grailqa', time_str)
        with open(f'{output_log_dir}/{file_name}.json', 'w') as f:
            json.dump(logs_with_ans,f)
        # with open("")
        # save_log(logs_with_ans, f'sketch_gen', time_str)


if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=["grailqa","grailqability","webqsp"], required=True)
    parser.add_argument('--train_type', type=str, choices=["A","AU"], required=False)
    parser.add_argument('--run_valid', type=str, default='False', required=False)
    parser.add_argument('--run_test', type=str, default='False', required=False)
    parser.add_argument('--run_train_nk', type=str, default='False', required=False)
    parser.add_argument('--run_train', type=str, default='False', required=False)
    parser.add_argument('--model_name', type=str, default='t5-base', required=False)

    parser.add_argument('--device', type=str, default="cuda", required=False)
    parser.add_argument('--train_batch_size', type=int, default=8, required=False)
    parser.add_argument('--eval_batch_size', type=int, default=32, required=False)
    parser.add_argument('--gradient_accumulation_step', type=int, default=1, required=False)
    parser.add_argument('--num_train_epochs', type=int, default=10, required=False)

    parser.add_argument('--start_index', type=int, default=-1, required=False)
    parser.add_argument('--end_index', type=int, default=-1, required=False)

    args = parser.parse_args()

    params = dict()

    params['device'] = args.device
    params['train_batch_size'] = args.train_batch_size
    params['eval_batch_size'] = args.eval_batch_size
    params['gradient_accumulation_step'] = args.gradient_accumulation_step
    params['num_train_epochs'] = args.num_train_epochs
    params['model_name'] = args.model_name

    print("Hyper-params")
    print("model_name : ", args.model_name)
    print("dataset : ", args.dataset)
    print("train_type : ", args.train_type)
    print("train_batch_size : ", params['train_batch_size'])
    print("gradient_accumulation_step : ", params['gradient_accumulation_step'])
    print("num train epochs: ", params["num_train_epochs"])
    print("learning_rate : 3e-5")

    assert output_log_dir is not None and os.path.isdir(output_log_dir), f"{output_log_dir}"

    grail_train_data = JsonLoader(file_path=sketch_data_train_path, split="train",\
                                         dataset=args.dataset, train_type=args.train_type)

    if args.dataset == "grailqability":
        grail_train_nk_data = JsonLoader(file_path=sketch_data_train_path, split="train_nk",\
                                                dataset=args.dataset, train_type=args.train_type)

    grail_dev_data = JsonLoader(file_path=sketch_data_dev_path, split="dev",\
                                        dataset=args.dataset, train_type=args.train_type)

    grail_test_data = JsonLoader(file_path=sketch_data_test_path, split="test",\
                                        dataset=args.dataset, train_type=args.train_type)

    grail_qa_algorithm = SketchGeneration(params)
    grail_qa_algorithm.train(grail_train_data, grail_dev_data, model_save_dir, params['gradient_accumulation_step'], params['num_train_epochs'])
    
    file_suffix = ""
    start_index = args.start_index
    end_index = args.end_index
    if start_index != -1 and end_index != -1:
        file_suffix=f"_{start_index}_{end_index}"

    if args.run_train.lower() == 'true':
        print("Running inference on train set")
        file_name = f"sketch_preds_train{file_suffix}"
        grail_qa_algorithm.solve(grail_train_data,output_log_dir,file_name, start_index, end_index)
    if args.run_valid.lower() == 'true':
        print("Running inference on dev set")
        file_name = f"sketch_preds_dev{file_suffix}"
        grail_qa_algorithm.solve(grail_dev_data,output_log_dir,file_name, start_index, end_index)
    if args.run_test.lower() == 'true':
        print("Running inference on test set")
        file_name = f"sketch_preds_test{file_suffix}"
        grail_qa_algorithm.solve(grail_test_data, output_log_dir,file_name, start_index, end_index)
    if args.run_train_nk.lower() == 'true':
        print("Running inference on trainnk set")
        file_name = f"sketch_preds_train_nk{file_suffix}"
        grail_qa_algorithm.solve(grail_train_nk_data, output_log_dir,file_name, start_index, end_index)
