import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from utils.constants import agnews_labels
from utils.data_agnews import get_split_dataset_dict

from torch import nn
from transformers import AutoTokenizer, AutoConfig, TrainingArguments, Trainer
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel


def tokenize(examples, tokenizer):
    return tokenizer(examples['text'], truncation=True)


def get_datasets_from_datadict(transformer_name):
    ds = get_split_dataset_dict()

    tokenizer = AutoTokenizer.from_pretrained(transformer_name)

    train_ds = ds['train'].map(
        lambda ex: tokenize(ex, tokenizer),
        batched=True,
        remove_columns=['title', 'description', 'text'],
    )
    eval_ds = ds['validation'].map(
        lambda ex: tokenize(ex, tokenizer),
        batched=True,
        remove_columns=['title', 'description', 'text'],
    )

    test_ds = ds['test'].map(
        lambda ex: tokenize(ex, tokenizer),
        batched=True,
        remove_columns=['title', 'description', 'text'],
    )

    return tokenizer, train_ds, eval_ds, test_ds


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        # https://github.com/huggingface/transformers/blob/65659a29cf5a079842e61a63d57fa24474288998/src/transformers/models/bert/modeling_bert.py#L1486
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        cls_outputs = outputs.last_hidden_state[:, 0, :]
        cls_outputs = self.dropout(cls_outputs)
        logits = self.classifier(cls_outputs)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels.to(torch.long))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def create_model(transformer_name):
    config = AutoConfig.from_pretrained(
        transformer_name,
        num_labels=len(agnews_labels),
    )

    model = (
        BertForSequenceClassification
        .from_pretrained(transformer_name, config=config)
    )

    return model


def train_model(transformer_name, model, tokenizer, train_ds, eval_ds):
    num_epochs = 2
    batch_size = 24
    weight_decay = 0.01
    model_save_dir = f'./models/{transformer_name}-sequence-classification'

    training_args = TrainingArguments(
        output_dir=model_save_dir,
        log_level='error',
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy='epoch', # FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead
        weight_decay=weight_decay,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    trainer.train()

    return trainer




def compute_metrics(eval_pred):
    y_true = eval_pred.label_ids
    y_pred = np.argmax(eval_pred.predictions, axis=-1)
    return {'accuracy': accuracy_score(y_true, y_pred)}



def test_model(trainer, test_ds):
    output = trainer.predict(test_ds)

    y_true = output.label_ids
    y_pred = np.argmax(output.predictions, axis=-1)
    target_names = agnews_labels
    print(classification_report(y_true, y_pred, target_names=target_names))


if __name__ == '__main__':
    transformer_name = 'bert-base-cased'
    tokenizer, train_ds, eval_ds, test_ds = get_datasets_from_datadict(transformer_name)

    model = create_model(transformer_name)
    trainer = train_model(transformer_name, model, tokenizer, train_ds, eval_ds)

    test_model(trainer, test_ds)
