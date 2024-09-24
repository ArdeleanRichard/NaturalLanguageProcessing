import os

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.trainer_pt_utils import log_metrics, save_metrics, save_state
from transformers.trainer_utils import get_last_checkpoint


def load_split_datasets(dataset_name, dataset_config_name, params, tokenizer):
    wmt16 = load_dataset(dataset_name, dataset_config_name)

    train_dataset = wmt16['train']
    eval_dataset = wmt16['validation']
    column_names = train_dataset.column_names

    train_ds = train_dataset.map(
        lambda batch: tokenize(batch, tokenizer, params),
        batched=True,
        remove_columns=column_names,
    )

    val_ds = eval_dataset.map(
        lambda batch: tokenize(batch, tokenizer, params),
        batched=True,
        remove_columns=column_names,
    )

    return train_ds, val_ds


def tokenize(batch, tokenizer, params):
    # get source sentences and prepend task prefix
    sources = [x[source_lang] for x in batch["translation"]]
    sources = [task_prefix + x for x in sources]
    # tokenize source sentences
    output = tokenizer(
        sources,
        max_length=params['max_source_length'],
        truncation=True,
    )

    # get target sentences
    targets = [x[target_lang] for x in batch["translation"]]
    # tokenize target sentences
    labels = tokenizer(
        targets,
        max_length=params['max_target_length'],
        truncation=True,
    )
    # add targets to output
    output["labels"] = labels["input_ids"]

    return output

def load_model(transformer_name):
    config = AutoConfig.from_pretrained(transformer_name)
    tokenizer = AutoTokenizer.from_pretrained(transformer_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(transformer_name, config=config)

    return model, tokenizer


def train_model(model, tokenizer, params, train_ds, val_ds):
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=params['label_pad_token_id'],
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=params["output_dir"],
        per_device_train_batch_size=params["batch_size"],
        per_device_eval_batch_size=params["batch_size"],
        save_steps=params["save_steps"],
        predict_with_generate=True,
        evaluation_strategy='steps',
        eval_steps=params["save_steps"],
        learning_rate=params["learning_rate"],
        num_train_epochs=params["num_train_epochs"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    last_checkpoint = None
    if os.path.isdir(params["output_dir"]):
        last_checkpoint = get_last_checkpoint(params["output_dir"])

    if last_checkpoint is not None:
        print(f'Checkpoint detected, resuming training at {last_checkpoint}.')

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()



    metrics = train_result.metrics
    metrics['train_samples'] = len(train_ds)

    log_metrics(trainer, 'train', metrics)
    save_metrics(trainer, 'train', metrics)
    save_state(trainer)



    metrics = trainer.evaluate(
        max_length=params["max_target_length"],
        num_beams=params["num_beams"],
        metric_key_prefix='eval',
    )

    metrics['eval_samples'] = len(val_ds)

    log_metrics(trainer, 'eval', metrics)
    save_metrics(trainer, 'eval', metrics)


    return trainer


def create_model_card(transformer_name, dataset_name, dataset_config_name, trainer):
    # A model card is akin to an automatically-generated README file that includes
    # information  about the model used, the data, settings used, and performance
    # throughout the training process. This file is helpful for reproducibility as
    # it contains all of this key information in one place. These cards are often
    # uploaded to the Hugging Face Hub together with the model itself.
    kwargs = {
        'finetuned_from': transformer_name,
        'tasks': 'translation',
        'dataset_tags': dataset_name,
        'dataset_args': dataset_config_name,
        'dataset': f'{dataset_name} {dataset_config_name}',
        'language': [source_lang, target_lang],
    }
    trainer.create_model_card(**kwargs)


def compute_metrics(eval_preds, tokenizer):
    metric = evaluate.load("sacrebleu")

    preds, labels = eval_preds
    # get text for predictions
    predictions = tokenizer.batch_decode(
        preds,
        skip_special_tokens=True,
    )
    # replace -100 in labels with pad token
    labels = np.where(
        labels != -100,
        labels,
        tokenizer.pad_token_id,
    )
    # get text for gold labels
    references = tokenizer.batch_decode(
        labels,
        skip_special_tokens=True,
    )
    # metric expects list of references for each prediction
    references = [[ref] for ref in references]

    # compute bleu score
    results = metric.compute(
        predictions=predictions,
        references=references,
    )
    results = {'bleu': results['score']}

    return results


if __name__ == '__main__':
    dataset_name = 'wmt16'
    dataset_config_name = 'ro-en'
    transformer_name = 't5-small'

    source_lang = 'ro'
    target_lang = 'en'
    task_prefix = 'translate Romanian to English: '

    params = {
        "source_lang": source_lang,
        "target_lang": target_lang,
        "max_source_length": 1024,
        "max_target_length": 128,
        "task_prefix": task_prefix,
        'label_pad_token_id': -100,
        "num_beams": 1,
        'num_train_epochs': 3,
        "batch_size": 4,
        'save_steps': 25_000,
        'learning_rate': 1e-3,
        'output_dir': './models/t5-translation-example',
    }


    model, tokenizer = load_model(transformer_name)

    train_ds, val_ds = load_split_datasets(dataset_name, dataset_config_name, params, tokenizer)

    trainer = train_model(model, tokenizer, params, train_ds, val_ds)
    create_model_card(transformer_name, dataset_name, dataset_config_name, trainer)