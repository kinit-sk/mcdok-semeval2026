CACHE=None
import sys
import json
from datasets import Dataset
import pandas as pd
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer, set_seed, BitsAndBytesConfig, Phi3ForSequenceClassification
import os
from sklearn.model_selection import train_test_split
from scipy.special import softmax
import argparse
import logging
import datetime
import bitsandbytes as bnb
from peft import LoraConfig, PeftConfig, PeftModel, AutoPeftModelForCausalLM, TaskType, AutoPeftModelForSequenceClassification, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import torch
import torch.nn.functional as F
from accelerate import PartialState
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, roc_auc_score
import re
from confusables import confusable_characters
import random
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

import nvidia_smi, psutil

def report_gpu():
  nvidia_smi.nvmlInit()
  handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
  info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
  print("GPU [GB]:", f'{info.used/1024/1024/1024:.2f}', "/", f'{info.total/1024/1024/1024:.1f}')
  nvidia_smi.nvmlShutdown()
  print('RAM [GB]:', f'{psutil.virtual_memory()[3]/1024/1024/1024:.2f}', "/", f'{psutil.virtual_memory()[0]/1024/1024/1024:.1f}')

def preprocess_function(examples, **fn_kwargs):
    return fn_kwargs['tokenizer'](examples["text"], truncation=True, max_length=512)

def tokenize_data(dataset, tokenizer):
    """Tokenizes the text data."""
    return dataset.map(lambda examples: tokenizer(examples["text"], truncation=True, padding=True, max_length=512, return_tensors="pt"), batched=True)

f1_metric = evaluate.load("f1")
metric = evaluate.load("bstrai/classification_report")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    results = {}
    results.update(f1_metric.compute(predictions=predictions, references = labels, average="macro"))

    return results

weights = [1.0, 1.0]
class CustomTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # compute custom loss
        #loss = F.binary_cross_entropy_with_logits(logits[:,1], labels.to(torch.float32))#, pos_weight=self.label_weights)
        
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, device=model.device, dtype=logits.dtype))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    if "base_layer" in lora_module_names:  # problem with training from peft checkpoint
        lora_module_names.remove("base_layer")
    return list(lora_module_names)

def fine_tune(train_df, valid_df, checkpoints_path, id2label, label2id, model, continue_train):
    #class weghts
    global CACHE
    global weights
    weights = 1/train_df['label'].value_counts(normalize=True).sort_index().to_numpy()
    #while len(id2label) > len(weights): weights += [0]
    print(weights)

    # pandas dataframe to huggingface Dataset
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    
    floatorbfloat = torch.bfloat16 #torch.float16
    if 'lama' in model:
        floatorbfloat = torch.bfloat16
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=floatorbfloat,
        bnb_4bit_quant_storage=floatorbfloat,
    )
    
    print('LOADING:', model)
    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=CACHE, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model_name = model
    if 'deberta' in model_name.lower():
      bnb_config=None
    
    modelclass = AutoModelForSequenceClassification
    if 'phi' in model_name.lower(): modelclass = Phi3ForSequenceClassification
    model = modelclass.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": PartialState().process_index},
        torch_dtype=floatorbfloat,
        #attn_implementation="flash_attention_2", # use sdpa, alternatively use "flash_attention_2"
        cache_dir=CACHE,
        trust_remote_code=True,
        num_labels=len(label2id), id2label=id2label, label2id=label2id
    )
    model.config.use_cache = False
    
    #DM added
    if tokenizer.pad_token is None:
      if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
      else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=32)
    try:
      model.config.pad_token_id = tokenizer.get_vocab()[tokenizer.pad_token]
    except:
      print("Warning: Exception occured while setting pad_token_id")
    
    # tokenize data for train/valid
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
    tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    #print(model)
    
    lora_alpha = 16
    lora_dropout = 0.1
    lora_r = 64
    
    target_modules=[]
    if 'falcon' in model_name.lower():
      target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    elif 'mistral' in model_name.lower():
      target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
    #elif 'llama' in model_name:
    #  pass #target_modules=['v_proj', 'q_proj', 'k_proj', 'o_proj'] #'down_proj', 'up_proj', 'gate_proj', #['v_proj', 'up_proj', 'gate_proj', 'o_proj', 'down_proj', 'k_proj', 'q_proj']
    elif 'deberta' in model_name.lower():
      target_modules=["query_proj", "key_proj", "value_proj"]
    else:
      target_modules=find_all_linear_names(model)
    print(target_modules)
    
    modules_to_save=["score"]
    if 'bert' in model_name.lower(): modules_to_save=["classifier", "pooler"]
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        #task_type="CAUSAL_LM",
        target_modules=target_modules, #"all-linear", #
        modules_to_save=modules_to_save
    )
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    
    output_dir = checkpoints_path
    per_device_train_batch_size = 1#16 #4
    gradient_accumulation_steps = 1#4
    optim = "paged_adamw_32bit" #"adamw_torch" #"paged_adamw_8bit" #
    logging_steps = 1000 #10
    save_steps = logging_steps# * 5 #10
    learning_rate = 2e-5 #2e-4
    max_grad_norm = 0.3
    max_steps = 10 #500
    num_train_epochs=3 #added
    warmup_ratio = 0.03
    lr_scheduler_type = "cosine"#"constant"
    fp16 = False
    bf16 = True
    tf32 = False
    if 'deberta' in model_name.lower():
      tf32=True
      bf16=False
    if 'lama' in model_name.lower():
        fp16 = False
        bf16 = True
        #logging_steps = 10
        #per_device_train_batch_size = 4
                
    if continue_train:
      output_dir = model_name.split('/checkpoint-')[0]
    
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size*16,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        save_total_limit=3,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        fp16=fp16,
        bf16=bf16,
        tf32=tf32,
        #max_grad_norm=max_grad_norm,
        #max_steps=max_steps, #for testing
        num_train_epochs=num_train_epochs,
        warmup_ratio=warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=lr_scheduler_type,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs = {"use_reentrant": True},
        load_best_model_at_end=True,
        #metric_for_best_model='AUC',
        eval_strategy="steps",
    )
    
    max_seq_length = 512

    trainer = CustomTrainer(
        model=model,
        #train_dataset=encoded_train_dataset,
        train_dataset=tokenized_train_dataset,
        #eval_dataset=encoded_valid_dataset,
        eval_dataset=tokenized_valid_dataset,
        #peft_config=peft_config,
        #dataset_text_field="text",
        #max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)
            
    if continue_train:
      trainer.train(resume_from_checkpoint=True)
    else:
      trainer.train()

    # save best model
    best_model_path = output_dir+'/best/'
    
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
        
    trainer.save_model(best_model_path)
    trainer.model.save_pretrained(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    for module in modules_to_save:
      try:
        torch.save(getattr(trainer.model,module).state_dict(), f'{best_model_path}/{module}-params.pt')
      except:
        print(f"Module {module} not dumped.")
    
    return #skip merging
    print('Merging model...')
    model_temp = AutoPeftModelForSequenceClassification.from_pretrained(
    #model_temp = AutoPeftModelForCausalLM.from_pretrained(
        best_model_path,
        low_cpu_mem_usage=True,
        torch_dtype=floatorbfloat,
    )
    model_temp = model_temp.merge_and_unload()        
    model_temp.save_pretrained(
       best_model_path, safe_serialization=True, max_shard_size="2GB"
    )


def test(test_df, model_path, id2label, label2id):
    print('Loading model for predictions...')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        #bnb_4bit_use_double_quant=True,
        #bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        #bnb_4bit_quant_storage=torch.float16,
    )
    
    # load best model
    model = AutoModelForSequenceClassification.from_pretrained(
       model_path, trust_remote_code=True, num_labels=len(label2id), id2label=id2label, label2id=label2id, torch_dtype="auto", device_map="auto"#, quantization_config=bnb_config
    )
    
    #DM added
    if tokenizer.pad_token is None:
      if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
      else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=32)
    try:
      model.config.pad_token_id = tokenizer.get_vocab()[tokenizer.pad_token]
    except:
      print("Warning: Exception occured while setting pad_token_id")

            
    test_dataset = Dataset.from_pandas(test_df)

    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})
    #tokenized_test_dataset = tokenize_data(test_dataset, tokenizer)
    #tokenized_test_dataset = tokenized_test_dataset.remove_columns(["unique_sample_id", "text"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    try:
      report_gpu()
    except:
      pass

    # create Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    # get logits from predictions and evaluate results using classification report
    predictions = trainer.predict(tokenized_test_dataset)
    prob_pred = softmax(predictions.predictions, axis=-1)
    preds = np.argmax(predictions.predictions, axis=-1)
    probs = [x[y] for x,y in zip(prob_pred, preds)]
    results = None
    try:
      results = metric.compute(predictions=preds, references=predictions.label_ids)
    except:
      pass
    # return dictionary of classification report
    return results, preds, probs, prob_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file_path", "-tr", required=True, help="Path to the train file.", type=str)
    parser.add_argument("--dev_file_path", "-d", required=True, help="Path to the dev file.", type=str)
    parser.add_argument("--test_file_path", "-t", required=True, help="Path to the test file.", type=str)
    parser.add_argument("--model", "-m", required=True, help="Transformer to train and test", type=str)
    parser.add_argument("--prediction_file_path", "-p", required=True, help="Path where to save the prediction file.", type=str)
    parser.add_argument('--continue_train', '-c', action='store_true')
    parser.add_argument('--random_seed', '-rs', help="Set random seed to affect dataset shuffling.", type=int, default=42)
    parser.add_argument('--test_only', '-to', action='store_true')

    args = parser.parse_args()

    train_path =  args.train_file_path
    dev_path =  args.dev_file_path
    test_path =  args.test_file_path
    model =  args.model
    prediction_path = args.prediction_file_path
    random_seed = args.random_seed
    
    set_seed(random_seed)

    label2id = {}
    id2label = {}
    for i in range(0,11): #classes of 0 to 10
      label2id[str(i)] = int(i)
      id2label[int(i)] = str(i)
    
    # Load and filter data
    train_df = pd.read_parquet(train_path)
    print(train_df.info())
    
    if dev_path != "":
      valid_df = pd.read_parquet(dev_path)
    else:
      valid_df = train_df.copy()
    print(valid_df.info())
    
    test_df = pd.read_parquet(test_path)
    #unique_ids = test_df["unique_sample_id"].values
    print(test_df.info())
    
    train_df['text'] = train_df['code']
    valid_df['text'] = valid_df['code']
    test_df['text'] = test_df['code']
    
    #subsample
    train_df = train_df.dropna(subset=['text'], ignore_index=True).drop_duplicates(subset=['text'], ignore_index=True).groupby(['generator']).apply(lambda x: x.sample(min(10000, len(x)), random_state = random_seed)).groupby(['label']).apply(lambda x: x.sample(min(2000, len(x)), random_state = random_seed)).sample(frac=1., random_state = random_seed).reset_index(drop=True)
    valid_df = valid_df.dropna(subset=['text'], ignore_index=True).drop_duplicates(subset=['text'], ignore_index=True).groupby(['generator']).apply(lambda x: x.sample(min(1000, len(x)), random_state = random_seed)).groupby(['label']).apply(lambda x: x.sample(min(500, len(x)), random_state = random_seed)).sample(frac=1., random_state = random_seed).reset_index(drop=True)
    
    # train detector model
    if args.test_only != True:
      fine_tune(train_df[['text', 'label']], valid_df[['text', 'label']], f"models_B/{model}", id2label, label2id, model, args.continue_train)

    # test detector model
    if 'ID' not in test_df.columns: test_df['ID'] = test_df.index.astype(int)
    if 'label' not in test_df.columns: test_df['label'] = 0
    results, predictions, probs, class_prob = test(test_df[['text', 'label']], f"models_B/{model}/best/", id2label, label2id)
    
    test_df['label'] = predictions
    test_df['probs'] = probs
    for i in range(0, class_prob.shape[1]):
      test_df[f'{i}_probs'] = class_prob[:,i]
    
    #logging.info(results)
    test_df[['ID', 'label']].to_csv(prediction_path, index=False)
    test_df.to_csv(prediction_path.replace('.csv', '_probs.csv'), index=False)