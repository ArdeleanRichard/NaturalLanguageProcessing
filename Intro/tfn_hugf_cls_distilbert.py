from tfn_hugf_cls_bert import get_datasets_from_datadict, train_model, test_model
from utils.constants import agnews_labels

from transformers import AutoTokenizer, AutoConfig, TrainingArguments, Trainer, DistilBertForSequenceClassification


def create_model(transformer_name):
    config = AutoConfig.from_pretrained(
        transformer_name,
        num_labels=len(agnews_labels),
    )

    model = (
        DistilBertForSequenceClassification
        .from_pretrained(transformer_name, config=config)
    )

    return model


if __name__ == '__main__':
    transformer_name = 'distilbert-base-cased'
    tokenizer, train_ds, eval_ds, test_ds = get_datasets_from_datadict(transformer_name)

    model = create_model(transformer_name)
    trainer = train_model(transformer_name, model, tokenizer, train_ds, eval_ds)

    test_model(trainer, test_ds)
