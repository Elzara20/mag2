# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import os
import time
import random
import argparse
import re
import torch

from tokenizers import Tokenizer
from models.progen.modeling_progen import ProGenForCausalLM
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, DefaultDataCollator, DataCollatorForLanguageModeling
import datasets
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
#########################################################################
# util


class print_time:
    def __init__(self, desc):
        self.desc = desc

    def __enter__(self):
        print(self.desc)
        self.t = time.time()

    def __exit__(self, type, value, traceback):
        print(f'{self.desc} took {time.time()-self.t:.02f}s')


def set_env():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def set_seed(seed, deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic



########################################################################
# model


def create_model(ckpt, fp16=True):
    if fp16:
        return ProGenForCausalLM.from_pretrained(ckpt, revision='float16', torch_dtype=torch.float16, low_cpu_mem_usage=True)
    else:
        return ProGenForCausalLM.from_pretrained(ckpt)


def create_tokenizer_custom(file):
    with open(file, 'r') as f:
        return Tokenizer.from_str(f.read())


########################################################################
# sample


def sample(device, model, tokenizer, context, max_length, num_return_sequences, top_p, temp, pad_token_id):
    print(f"context = {type(context)}")
    with torch.no_grad():
        input_ids = torch.tensor(tokenizer.encode(context).ids).view([1, -1]).to(device)
        tokens_batch = model.generate(input_ids, do_sample=True, temperature=temp, max_length=max_length, top_p=top_p, num_return_sequences=num_return_sequences, pad_token_id=pad_token_id)
        as_lists = lambda batch: [batch[i, ...].detach().cpu().numpy().tolist() for i in range(batch.shape[0])]
        return tokenizer.decode_batch(as_lists(tokens_batch))


def truncate(sample, terminals):
    pos = []
    for terminal in terminals:
        find_pos = sample.find(terminal, 1)
        if find_pos != -1:
            pos.append(find_pos)
    if len(pos) > 0:
        return sample[:(min(pos)+1)]
    else:
        return sample


def cross_entropy(logits, target, reduction='mean'):
    return torch.nn.functional.cross_entropy(input=logits, target=target, weight=None, size_average=None, reduce=None, reduction=reduction)




########################################################################
# main


def main():
    #****************************************ВЫДЕЛЕНИЕ НЕОБХОДИМЫХ ДАННЫХ ДЛЯ ДООБУЧЕНИЯ****************************************

    '''данные взяты после выравниванию по бласт
    итог из бласт сожержит: id последовательности и саму последовательность, которая может переходить на новую строку (т.е. есть в файле \n)
    что делается:
    - 'вытаскиваем последовательность'
    - убираем \n
    РЕЗУЛЬТАТ: список последовательностей представленных в виде строк
    '''

    dataset = open("/content/Ribonuclease A.txt", "r").readlines()
    # разделение
    dataset_names, dataset_seq = [], []
    seq=''
    for i in range(len(dataset)):
      res=re.findall(">.+", dataset[i])
      
      if len(res)!=0:
        dataset_seq.append(seq)
        dataset_names.append(dataset[i])
        seq=''
      else:
        seq+=dataset[i]
    dataset_seq.append(seq)
    # создание единой последовательности
    print(dataset_seq[:3])
    count_empty=0
    dataset_seq_new=[]
    for i in dataset_seq:
      if len(i)!=0:
        dataset_seq_new.append(i.replace("\n", ""))
      else:
        count_empty+=1
    print(f"new = {dataset_seq_new[:3]}")
    print(f"empty value in seq = {count_empty}")
    print(dataset_names[:3])
    print(f"all data = {len(dataset_seq_new)}")
    #***************************************************************************************************************************    

    # (0) constants

    models_151M = [ 'progen2-small' ]
    models_754M = [ 'progen2-medium', 'progen2-oas', 'progen2-base' ]
    models_2B = [ 'progen2-large', 'progen2-BFD90' ]
    models_6B = [ 'progen2-xlarge' ]
    models = models_151M + models_754M + models_2B + models_6B
    
    # (1) params

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=models, default='progen2-large')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--rng-seed', type=int, default=42)
    parser.add_argument('--rng-deterministic', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--p', type=float, default=0.95)
    parser.add_argument('--t', type=float, default=0.2)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--fp16', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--context', type=str, default='1')
    parser.add_argument('--sanity', default=True, type=lambda x: (str(x).lower() == 'true'))
    args = parser.parse_args()


    # (2) preamble

    set_env()
    set_seed(args.rng_seed, deterministic=args.rng_deterministic)

    if not torch.cuda.is_available():
        print('falling back to cpu')
        args.device = 'cpu'

    device = torch.device(args.device)
    ckpt = f'./checkpoints/{args.model}'

    if device.type == 'cpu':
        print('falling back to fp32')
        args.fp16 = False

    # (3) load

    with print_time('loading parameters'):
        model = create_model(ckpt=ckpt, fp16=args.fp16).to(device)
    # print(f"MODEL FOR SAMPLE:{model}")

    with print_time('loading tokenizer'):
        tokenizer = create_tokenizer_custom(file='tokenizer.json') 
    
    # (4) sanity

    if args.sanity:

        with print_time('sanity cross-entropy'):

            def ce(tokens):
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=args.fp16):
                        target = torch.tensor(tokenizer.encode(tokens).ids).to(device)
                        logits = model(target, labels=target).logits
                        print(f"target from ce = {target}")
                        # shift
                        logits = logits[:-1, ...]
                        target = target[1:]

                        return cross_entropy(logits=logits, target=target).item()

            x_uniref90bfd30 = '2GFLPFRGADEGLAAREAATLAARGTAARAYREDSWAVPVPRGLLGDLTARVAALGAASPPPADPLAVTLDLHHVTAEVALTTVLDAATLVHGQTRVLSAEDAAEAATAAAAATEAYLERLQDFVLFMSASVRVWRRGNAAGATGPEWDQWYTVADRDALGSAPTHLAVLGRQADALCHFVLDRVAWGTCGTPLWSGDEDLGNVVATFAGYADRLATAPRDLIM1'
            x_oas = '1EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMHWVRQAPWKGLEYVSAISSNGGSTYYANSVKGRFTISRDNSKNTLYLQMGSLRAEDMAVYYCARDESGYSYGWGYYFDYWGQGTLVTVSS2'
            x_bfd90 = '1TAPRSTRASGSEGSRPPGIPAKGRRCLPSRAGSVTPRFRHARQGTATVAKEQGRKLIASNRKARHDYHIEDTFEAGLVLTGTEVKSLRMGRASLIDGYAVFYGEELWLEGVHIPEYLNGNWTNHTPRRRRKLLLNRSELTKLAHKTSESGHTIVPLALYFKDGRAKVEIAVAKGKKAYDKRHALRERQDQREV2'

            checkpoint_x_ce = {
                'progen2-small': (x_uniref90bfd30, 2.4),
                'progen2-medium': (x_uniref90bfd30, 1.9),
                'progen2-base': (x_uniref90bfd30, 1.9),
                'progen2-large': (x_uniref90bfd30, 1.8),
                'progen2-xlarge': (x_uniref90bfd30, 1.0),
                'progen2-oas': (x_oas, 0.3),
                'progen2-BFD90': (x_bfd90, 1.3),
            }

            ce_eval = ce(checkpoint_x_ce[args.model][0])
            ce_target = checkpoint_x_ce[args.model][1]

            print(ce_target, ce_eval, abs(ce_eval - ce_target))

            assert abs(ce_eval - ce_target) < 0.1

    # (5) sample

    with print_time('sampling'):
        completions = sample(device=device, model=model, tokenizer=tokenizer, context=args.context, pad_token_id=tokenizer.encode('<|pad|>').ids[0], num_return_sequences=args.num_samples, temp=args.t, top_p=args.p, max_length=args.max_length)
        truncations = [truncate(completion, terminals=['1', '2']) for completion in completions]

        print(args.context)

        for (i, truncation) in enumerate(truncations):

            print()
            print(i)
            print(truncation)


    #***************************************************ПОДГОТОВКА ДАННЫХ + ДООБУЧЕНИЕ*****************************************************************
    '''Что нужно сделать:
    - данные нужно представить в виде словаря (почему: принцип работы Trainer, представляла в виде списка - была ошибка)
    - создать коллатор данных (почему: были проблемы с представлением данных - с <pad>, решение с гугла - создать свой коллатор)
    есть решение без создания коллатора (источник: вопрос на github связанный с дообучением progen2), но там в другом месте ошибка будет
    к тому же, если в словарь заносить torch.tensor, он в словаре становится списком - а со списками Trainer тоже ошибку выдает с представленностью данных
    - дообучить: подобрать правильные параметры и сохранить модель
    '''


    # создание словаря
    data_seq_train = dataset_seq_new[:int(len(dataset_seq_new)*0.8)]
    data_seq_val=dataset_seq_new[int(len(dataset_seq_new)*0.8):]
    data_name_train =dataset_names[:int(len(dataset_names)*0.8)]
    data_name_val=dataset_names[int(len(dataset_names)*0.8):]

    data_seq_train_new=[]
    data_seq_val_new=[]
    for i in data_seq_train:
      data_seq_train_new.append(torch.tensor(tokenizer.encode(i).ids).to(device))
    for i in data_seq_val:
      data_seq_val_new.append(torch.tensor(tokenizer.encode(i).ids).to(device))

    train_dataset1 = datasets.Dataset.from_dict({'input_ids': data_seq_train_new}) 
    val_dataset1 = datasets.Dataset.from_dict({'input_ids': data_seq_val_new})
    all_dataset = datasets.DatasetDict({'train': train_dataset1, 'val':val_dataset1})
    
    # коллатор
    class MyDataCollator(DefaultDataCollator): #DataCollatorForLanguageModeling
      def __init__(self, tokenizer):
          super(MyDataCollator, self).__init__(tokenizer)
          self.tokenizer = tokenizer
          self.tokenizer.pad_token_id = tokenizer.encode('<|pad|>').ids[0]
      def __call__(self, seq):
          seq[0]['input_ids'] = torch.tensor(seq[0]['input_ids'])
          # seq[0]['labels'] = torch.tensor(seq[0]['labels']) #&
          return seq[0] #torch.tensor(seq[0]['input_ids'])
      


    data_collator = MyDataCollator(tokenizer)
    print(f'train dataset for Trainer = {train_dataset1}')
    # дообучение
    training_args = TrainingArguments(
      output_dir="/content/small_fineTuning_model1",
      logging_dir="/content/log",
      overwrite_output_dir=True,
      num_train_epochs=3, #3,
      # per_device_train_batch_size=8, #8 - по дефолту
      # per_device_eval_batch_size=8,
      # save_steps=100, #500 - по дефолту
      learning_rate=6e-5,
      save_total_limit=2, # получим количество папок в chaeckpoint - лучшая и предлучшая модель
      # prediction_loss_only=False,
      # remove_unused_columns=False,
      # logging_steps=100,
      # fp16=False,
      load_best_model_at_end = True,
      evaluation_strategy = "steps",
      save_strategy = "steps",
      label_names=['input_ids'],
      # eval_steps = 10,
    )
    from transformers.trainer_callback import PrinterCallback, ProgressCallback, EarlyStoppingCallback, DefaultFlowCallback
    from transformers.integrations import TensorBoardCallback
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)
    


    # from torch.utils.tensorboard import SummaryWriter
    # from tensorboardX import SummaryWriter
    # writer = SummaryWriter(log_dir="/content/log")
    # training_args.tb_writer = writer

    trainer = Trainer(
      model=model,
      args=training_args,
      data_collator=data_collator,
      train_dataset=train_dataset1,
      eval_dataset=val_dataset1,
      callbacks=[PrinterCallback(), ProgressCallback(), DefaultFlowCallback(), TensorBoardCallback, early_stopping_callback],
    
    )
    
    trainer.train() 
    trainer.save_model("/content/small_fineTuning_model1")
    import matplotlib.pyplot as plt
    # представление потерь в виде графиков и запоминание данных
    with open('/content/progen/progen2/res_pic/out_trainer_state_epoch1_seq1k.txt', 'w') as f:
      print("trainer.state.log_history = ", trainer.state.log_history, file=f)

    train_loss=[]
    val_loss = []
    for i in trainer.state.log_history:
      if 'loss' in i:
        train_loss.append(i['loss'])
      if 'eval_loss' in i:
        val_loss.append(i["eval_loss"])
    import numpy as np
    x_val=np.linspace(0, len(train_loss), num=len(val_loss))
    x_train=np.linspace(0, len(val_loss), num=len(train_loss))
    plt.figure(figsize=(15, 10))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(x_val, val_loss, label='Validation Loss')
    plt.xlabel('Training steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('/content/progen/progen2/res_pic/val_trainer_epoch1_seq1k.png')
    plt.savefig('/content/progen/progen2/res_pic/val_trainer_epoch1_seq1k.pdf')
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.plot(x_train, train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Training steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('/content/progen/progen2/res_pic/train_trainer_epoch1_seq1k.png')
    plt.savefig('/content/progen/progen2/res_pic/train_trainer_epoch1_seq1k.pdf')
    plt.show()


    plt.figure(figsize=(15, 10))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Training steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('/content/progen/progen2/res_pic/trainer_epoch1_seq1k.png')
    plt.savefig('/content/progen/progen2/res_pic/trainer_epoch1_seq1k.pdf')
    plt.show()

    trainer.evaluate() 

    inp = {'input_ids': torch.tensor([ 4, 11,  9, 25, 14, 12, 15,  7, 15, 25, 14, 10, 14,  9,  9, 25, 15, 25,
         8,  8, 13, 15, 20, 11, 16, 23, 14, 15, 25, 22,  9, 16,  8, 16, 25, 14,
        22, 10,  9, 26, 11, 14,  8, 25, 15, 17, 20,  9, 16, 15, 23, 20, 11, 10,
        23, 12, 25, 10, 22, 15, 23, 10,  5, 22, 22,  9,  8, 15, 23, 23, 28, 16,
        22, 12,  9, 21, 12, 20,  9, 10,  5, 11, 23, 10, 16,  5,  5, 13,  8, 14,
         3])}
    #***************************************************************************************************************************
            


if __name__ == '__main__':
    main()
    print('done.')

