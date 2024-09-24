import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from transformers import AutoTokenizer, AutoConfig, TrainingArguments, DataCollatorForTokenClassification, Trainer
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel

from utils.data_ancora import get_dataset_dict


def tokenize_and_align_labels(batch, tokenizer, tag_to_index):
    # https://arxiv.org/pdf/1810.04805.pdf
    # Section 5.3
    # We use the representation of the first sub-token as the input to the token-level classifier over the NER label set.

    labels = []

    # the CoNLL-U dataset is already tokenized, we use the is_split_into_words=True tokenizer argument
    # to ensure that the tokenizer respects the existing word boundaries during its sub-word tokenization.
    # tokenize batch
    tokenized_inputs = tokenizer(
        batch['words'],
        truncation=True,
        is_split_into_words=True,
    )

    # Further, while we want to predict one POS tag per word,
    # any given word may be split into smaller pieces by our tokenizer. Thus,
    # we need to align the tokenizer output to the CoNLL-U words. The original
    # BERT paper (Devlin et al., 2018) addresses this by only using the
    # embedding corresponding to the first sub-token for each word. We follow
    # the same approach for consistency. For the sub-words that do not correspond
    # to the beginning of a word, we use a special value that indicates
    # that we are not interested in their predictions. The CrossEntropyLoss
    # has a parameter called ignore_index for this purpose. The default value
    # for this parameter is −100, which we use as the label for the sub-words
    # we wish to ignore during training:

    # iterate over batch elements
    for i, tags in enumerate(batch['tags']):
        label_ids = []
        previous_word_id = None

        # get word ids for current batch element
        word_ids = tokenized_inputs.word_ids(batch_index=i)

        # iterate over tokens in batch element
        for word_id in word_ids:
            if word_id is None or word_id == previous_word_id:
                # ignore if not a word or word id has already been seen
                label_ids.append(ignore_index)
            else:
                # get tag id for corresponding word
                tag_id = tag_to_index[tags[word_id]]
                label_ids.append(tag_id)

            # remember this word id
            previous_word_id = word_id

        # save label ids for current batch element
        labels.append(label_ids)

    # store labels together with the tokenizer output
    tokenized_inputs['labels'] = labels

    return tokenized_inputs


def get_datasets_with_tokens(transformer_name):
    tokenizer = AutoTokenizer.from_pretrained(transformer_name)

    ds, tags, index_to_tag, tag_to_index = get_dataset_dict()

    # x = ds['train'][0]
    # tokenized_input = tokenizer(x['words'], is_split_into_words=True)
    # tokens = tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'])
    # word_ids = tokenized_input.word_ids()
    # print(pd.DataFrame([tokens, word_ids], index=['tokens', 'word ids']))

    train_ds = ds['train'].map(
        lambda x: tokenize_and_align_labels(x, tokenizer, tag_to_index),
        batched=True)
    eval_ds = ds['validation'].map(
        lambda x: tokenize_and_align_labels(x, tokenizer, tag_to_index),
        batched=True)
    test_ds = ds['test'].map(
        lambda x: tokenize_and_align_labels(x, tokenizer, tag_to_index),
        batched=True,
    )
    return tokenizer, train_ds, eval_ds, test_ds, tags, index_to_tag, tag_to_index


class XLMRobertaForTokenClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        # https://github.com/huggingface/transformers/blob/65659a29cf5a079842e61a63d57fa24474288998/src/transformers/models/roberta/modeling_roberta.py#L1346
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        # Here, our output is three-dimensional: (batch_size, sequence_size, num_labels).
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)
        loss = None

        # we need to reshape the logits and the labels before passing them to the CrossEntropyLoss, since it
        # expects two-dimensional input and one-dimensional labels. For this purpose,
        # we use the view() method to reshape the tensors.
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            inputs = logits.view(-1, self.num_labels)
            targets = labels.view(-1)
            loss = loss_fn(inputs, targets)
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def create_model(transformer_name):
    # The number of labels which determines the output dimension of the linear layer is equal to the number of POS tags
    config = AutoConfig.from_pretrained(
        transformer_name,
        num_labels=len(index_to_tag),
    )

    # Specifically, in our text classification model
    # the output shape was two-dimensional: (batch_size, num_labels).
    # Here, our output is three-dimensional: (batch_size, sequence_size, num_labels).

    model = (
        XLMRobertaForTokenClassification
        .from_pretrained(transformer_name, config=config)
    )

    return model


def train_model(transformer_name, model, tokenizer, train_ds, eval_ds):
    num_epochs = 2
    batch_size = 24
    weight_decay = 0.01
    model_save_dir = f'./models/{transformer_name}-finetuned-pos-es'

    training_args = TrainingArguments(
        output_dir=model_save_dir,
        log_level='error',
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy='epoch',
        weight_decay=weight_decay,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    trainer.train()

    return trainer



def compute_metrics(eval_pred):
    # compute_metrics function is adjusted to account for the fact
    # that our model uses sub-word tokens rather than complete words. Recall
    # that only the first sub-word token per word was assigned a POS tag.

    # gold labels
    label_ids = eval_pred.label_ids

    # predictions
    pred_ids = np.argmax(eval_pred.predictions, axis=-1)

    # collect gold and predicted labels, ignoring ignore_index label
    y_true, y_pred = [], []
    batch_size, seq_len = pred_ids.shape
    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != ignore_index:
                y_true.append(index_to_tag[label_ids[i][j]])
                y_pred.append(index_to_tag[pred_ids[i][j]])

    # return computed metrics
    return {'accuracy': accuracy_score(y_true, y_pred)}


def test_model(trainer, test_ds, tags):
    output = trainer.predict(test_ds)

    num_labels = model.num_labels
    label_ids = output.label_ids.reshape(-1)
    predictions = output.predictions.reshape(-1, num_labels)
    predictions = np.argmax(predictions, axis=-1)
    mask = label_ids != ignore_index

    y_true = label_ids[mask]
    y_pred = predictions[mask]
    target_names = tags[:-1]

    report = classification_report(
        y_true, y_pred,
        target_names=target_names
    )
    print(report)


if __name__ == '__main__':
    ignore_index = -100

    transformer_name = 'bert-base-cased'
    tokenizer, train_ds, eval_ds, test_ds, tags, index_to_tag, tag_to_index = get_datasets_with_tokens(transformer_name)

    model = create_model(transformer_name)
    trainer = train_model(transformer_name, model, tokenizer, train_ds, eval_ds)

    test_model(trainer, test_ds, tags)
