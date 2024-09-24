import torch
from datasets import load_dataset
import evaluate
from utils.constants import device



from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def create_model(transformer_name):
    tokenizer = AutoTokenizer.from_pretrained(transformer_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(transformer_name)

    # transformers.Trainer moves the model and data to the GPU automatically,
    # but since we won't use it in this notebook, we have to do it manually
    model = model.to(device)
    return model, tokenizer

def load_data():
    test_ds = load_dataset('wmt16', 'ro-en', split='test')
    return test_ds


def translate_batch(batch, params, model, tokenizer):
    # get source language examples and prepend task prefix
    inputs = [params['task_prefix'] + x[params['source_lang']] for x in batch["translation"]]

    # tokenize inputs
    encoded = tokenizer(
        inputs,
        max_length=params['max_source_length'],
        truncation=True,
        padding=True,
        return_tensors='pt',
    )

    # move data to gpu if needed
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)

    # generate translated sentences
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_beams=params['num_beams'],
        max_length=params['max_target_length'],
    )

    # generate predicted sentences from predicted token ids
    decoded = tokenizer.batch_decode(
        output,
        skip_special_tokens=True,
    )

    # get gold sentences in target language
    targets = [x[params["target_lang"]] for x in batch["translation"]]

    # return gold and predicted sentences
    return {
        'reference': targets,
        'prediction': decoded,
    }


def test_model(test_ds, params, model, tokenizer):
    results = test_ds.map(
        lambda batch: translate_batch(batch, params, model, tokenizer),
        batched=True,
        batch_size=params["batch_size"],
        remove_columns=test_ds.column_names,
    )
    print(results.to_pandas())

    # from datasets import load_metric
    # metric = load_metric('sacrebleu')

    metric = evaluate.load("sacrebleu")

    for r in results:
        prediction = r['prediction']
        reference = [r['reference']]
        metric.add(prediction=prediction, reference=reference)

    metric_result = metric.compute()

    return metric_result


def greedy_translation(params, model, tokenizer, text):
    # This function interacts directly with the encoder and decoder components of the T5 model,
    # so we must construct the input for both.
    #
    # The encoder’s input is constructed by prepending
    # the task prefix to the English text and tokenizing it.
    #
    # On the other hand, the decoder’s input is
    # constructed incrementally by accumulating the tokens predicted so far
    # in order to predict the next token in the sequence.
    #
    # At the beginning, before any tokens are predicted,
    # the decoder’s input is initialized with  a single token that corresponds to the beginning of the sequence.
    # We retrieve this token, called decoder_start_token_id, from the model’s configuration object.

    # prepend task prefix
    text = params['task_prefix'] + text

    # tokenize input
    encoded = tokenizer(
        text,
        max_length=params['max_source_length'],
        truncation=True,
        return_tensors='pt',
    )

    # encoder input ids
    encoder_input_ids = encoded.input_ids.to(device)

    # decoder input ids, initialized with start token id
    start = model.config.decoder_start_token_id
    decoder_input_ids = torch.LongTensor([[start]]).to(device)

    # generate tokens, one at a time
    for _ in range(params['max_target_length']):
        # get model predictions
        output = model(
            encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
        )

        # get logits for last token
        next_token_logits = output.logits[0, -1, :]

        # select most probable token
        next_token_id = torch.argmax(next_token_logits)

        # append new token to decoder_input_ids
        output_id = torch.LongTensor([[next_token_id]]).to(device)
        decoder_input_ids = torch.cat([decoder_input_ids, output_id], dim=-1)

        # if predicted token is the end of sequence, stop iterating
        if next_token_id == tokenizer.eos_token_id:
            break

    # return text corresponding to predicted token ids
    return tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)



def translate(source_lang, target_lang, task_prefix, test_text):
    transformer_name = 't5-small'
    params = {
        "source_lang": source_lang,
        "target_lang": target_lang,
        "max_source_length": 1024,
        "max_target_length": 128,
        "task_prefix": task_prefix,
        "num_beams": 1,
        "batch_size": 100,
    }

    model, tokenizer = create_model(transformer_name)

    test_ds = load_data()

    metric_result = test_model(test_ds, params, model, tokenizer)
    print(f"SacreBLEU: {round(metric_result['score'], 1)}")

    test_result = greedy_translation(params, model, tokenizer, test_text)
    print(f"Translating('{source_lang}'2'{target_lang}') '{test_text}': {test_result} ")




if __name__ == '__main__':
    translate(source_lang='en', target_lang='ro', task_prefix='translate English to Romanian: ', test_text="This is a test")
    translate(source_lang='ro', target_lang='en', task_prefix='translate Romanian to English: ', test_text="Acesta este un test")