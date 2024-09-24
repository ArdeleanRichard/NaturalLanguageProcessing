import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from tfn_hugf_translate import translate_batch, test_model, greedy_translation


def load_data(dataset_name, dataset_config_name):
    wmt16 = load_dataset(dataset_name, dataset_config_name)

    return wmt16

def load_pretrained_model(params):
    tokenizer = AutoTokenizer.from_pretrained(params["output_dir"], local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(params["output_dir"], local_files_only=True)

    return model, tokenizer




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
        "batch_size": 4,
        "num_beams": 1,
        'output_dir': './models/t5-translation-example',
    }

    test_ds = load_data(dataset_name, dataset_config_name)

    model, tokenizer = load_pretrained_model(params)


    metric_result = test_model(test_ds, params, model, tokenizer)
    print(f"SacreBLEU: {round(metric_result['score'], 1)}")

    test_text = "Acesta este un test"
    test_result = greedy_translation(params, model, tokenizer, test_text)
    print(f"Translating('{source_lang}'2'{target_lang}') '{test_text}': {test_result} ")