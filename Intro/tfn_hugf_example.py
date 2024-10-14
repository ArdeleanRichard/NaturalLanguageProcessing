from collections import defaultdict

from accelerate import Accelerator
from huggingface_hub import get_full_repo_name, Repository
from torch.optim import AdamW
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
from datasets import Dataset, load_dataset, DatasetDict
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig, DataCollatorForLanguageModeling, Trainer, \
    TrainingArguments, pipeline, get_scheduler

from torch.utils.data.dataloader import DataLoader





def any_keyword_in_string(string, keywords):
    for keyword in keywords:
        if keyword in string:
            return True
    return False


def filter_streaming_dataset(dataset, filters):
    filtered_dict = defaultdict(list)
    total = 0
    for sample in tqdm(iter(dataset)):
        total += 1
        if any_keyword_in_string(sample["content"], filters):
            for k, v in sample.items():
                filtered_dict[k].append(v)
    print(f"{len(filtered_dict['content'])/total:.2%} of data after filtering.")
    return Dataset.from_dict(filtered_dict)


def tokenize(element):
    outputs = tokenizer(
        element["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


def data_show():
    filters = ["pandas", "sklearn", "matplotlib", "seaborn"]
    example_1 = "import numpy as np"
    example_2 = "import pandas as pd"

    print(any_keyword_in_string(example_1, filters), any_keyword_in_string(example_2, filters))

    split = "train"  # "valid"
    filters = ["pandas", "sklearn", "matplotlib", "seaborn"]

    data = load_dataset(f"transformersbook/codeparrot-{split}", split=split, streaming=True)
    filtered_data = filter_streaming_dataset(data, filters)


def load_data():
    ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
    ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation")

    raw_datasets = DatasetDict(
        {
            "train": ds_train,  # .shuffle().select(range(50000)),
            "valid": ds_valid,  # .shuffle().select(range(500))
        }
    )

    print(raw_datasets)

    for key in raw_datasets["train"][0]:
        print(f"{key.upper()}: {raw_datasets['train'][0][key][:200]}")

    return raw_datasets



def tokenize_data(raw_datasets, context_length=128):

    tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")

    outputs = tokenizer(
        raw_datasets["train"][:2]["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )

    print(f"Input IDs length: {len(outputs['input_ids'])}")
    print(f"Input chunk lengths: {(outputs['length'])}")
    print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")

    tokenized_datasets = raw_datasets.map(
        tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
    )
    print(tokenized_datasets)

    return tokenized_datasets, tokenizer


def test_data(pipe):
    txt = """\
     # create some data
     x = np.random.randn(100)
     y = np.random.randn(100)

     # create scatter plot with x, y
     """
    print(pipe(txt, num_return_sequences=1)[0]["generated_text"])

    txt = """\
     # dataframe with profession, income and name
     df = pd.DataFrame({'profession': x, 'income':y, 'name': z})

     # calculate the mean income per profession
     """
    print(pipe(txt, num_return_sequences=1)[0]["generated_text"])

    txt = """
     # import random forest regressor from scikit-learn
     from sklearn.ensemble import RandomForestRegressor

     # fit random forest model with 300 estimators on X, y:
     """
    print(pipe(txt, num_return_sequences=1)[0]["generated_text"])







# def keytoken_weighted_loss(inputs, logits, keytoken_ids, alpha=1.0):
#     # Shift so that tokens < n predict n
#     shift_labels = inputs[..., 1:].contiguous()
#     shift_logits = logits[..., :-1, :].contiguous()
#     # Calculate per-token loss
#     loss_fct = CrossEntropyLoss(reduce=False)
#     loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
#     # Resize and average loss per sample
#     loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)
#     # Calculate and scale weighting
#     weights = torch.stack([(inputs == kt).float() for kt in keytoken_ids]).sum(axis=[0, 2])
#     weights = alpha * (1.0 + weights)
#     # Calculate weighted average
#     weighted_loss = (loss_per_sample * weights).mean()
#     return weighted_loss
#
#
# def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"], weight_decay):
#     params_with_wd, params_without_wd = [], []
#     for n, p in model.named_parameters():
#         if any(nd in n for nd in no_decay):
#             params_without_wd.append(p)
#         else:
#             params_with_wd.append(p)
#     return [
#         {"params": params_with_wd, "weight_decay": weight_decay},
#         {"params": params_without_wd, "weight_decay": 0.0},
#     ]
#
# def evaluate(accelerator, eval_dataloader):
#     model.eval()
#     losses = []
#     for step, batch in enumerate(eval_dataloader):
#         with torch.no_grad():
#             outputs = model(batch["input_ids"], labels=batch["input_ids"])
#
#         losses.append(accelerator.gather(outputs.loss))
#     loss = torch.mean(torch.cat(losses))
#     try:
#         perplexity = torch.exp(loss)
#     except OverflowError:
#         perplexity = float("inf")
#     return loss.item(), perplexity.item()



if __name__ == '__main__':
    data_show()


    # LOAD DATA
    raw_datasets = load_data()


    # TOKENIZE DATA
    context_length = 128
    tokenized_datasets, tokenizer = tokenize_data(raw_datasets, context_length)


    # TRAIN DATA
    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_ctx=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    model = GPT2LMHeadModel(config)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT-2 size: {model_size / 1000 ** 2:.1f}M parameters")

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    out = data_collator([tokenized_datasets["train"][i] for i in range(5)])
    for key in out:
        print(f"{key} shape: {out[key].shape}")

    args = TrainingArguments(
        output_dir="codeparrot-ds",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        eval_steps=5_000,
        logging_steps=5_000,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=5_000,
        fp16=True,
        push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
    )

    trainer.train()

    # TEST DATA
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pipe = pipeline("text-generation", model="huggingface-course/codeparrot-ds", device=device)
    test_data(pipe)




    # # TRAINING WITH ACCELERATE (DISTRIBUTED COMPUTATIONAL UNITS)
    # keytoken_ids = []
    # for keyword in [
    #     "plt",
    #     "pd",
    #     "sk",
    #     "fit",
    #     "predict",
    #     " plt",
    #     " pd",
    #     " sk",
    #     " fit",
    #     " predict",
    #     "testtest",
    # ]:
    #     ids = tokenizer([keyword]).input_ids[0]
    #     if len(ids) == 1:
    #         keytoken_ids.append(ids[0])
    #     else:
    #         print(f"Keyword has not single token: {keyword}")
    #
    #
    # tokenized_datasets.set_format("torch")
    # train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=32, shuffle=True)
    # eval_dataloader = DataLoader(tokenized_datasets["valid"], batch_size=32)
    #
    # weight_decay = 0.1
    # model = GPT2LMHeadModel(config)
    #
    # accelerator = Accelerator(fp16=True)
    # optimizer = AdamW(get_grouped_params(model), lr=5e-4)
    # model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)
    #
    # num_train_epochs = 1
    # num_update_steps_per_epoch = len(train_dataloader)
    # num_training_steps = num_train_epochs * num_update_steps_per_epoch
    #
    # lr_scheduler = get_scheduler(
    #     name="linear",
    #     optimizer=optimizer,
    #     num_warmup_steps=1_000,
    #     num_training_steps=num_training_steps,
    # )
    #
    # model_name = "codeparrot-ds-accelerate"
    # repo_name = get_full_repo_name(model_name)
    #
    # output_dir = "codeparrot-ds-accelerate"
    # repo = Repository(output_dir, clone_from=repo_name)
    #
    # gradient_accumulation_steps = 8
    # eval_steps = 5_000
    #
    # model.train()
    # completed_steps = 0
    # for epoch in range(num_train_epochs):
    #     for step, batch in tqdm(
    #         enumerate(train_dataloader, start=1), total=num_training_steps
    #     ):
    #         logits = model(batch["input_ids"]).logits
    #         loss = keytoken_weighted_loss(batch["input_ids"], logits, keytoken_ids)
    #         if step % 100 == 0:
    #             accelerator.print(
    #                 {
    #                     "lr": get_lr(),
    #                     "samples": step * samples_per_step,
    #                     "steps": completed_steps,
    #                     "loss/train": loss.item() * gradient_accumulation_steps,
    #                 }
    #             )
    #         loss = loss / gradient_accumulation_steps
    #         accelerator.backward(loss)
    #         if step % gradient_accumulation_steps == 0:
    #             accelerator.clip_grad_norm_(model.parameters(), 1.0)
    #             optimizer.step()
    #             lr_scheduler.step()
    #             optimizer.zero_grad()
    #             completed_steps += 1
    #         if (step % (eval_steps * gradient_accumulation_steps)) == 0:
    #             eval_loss, perplexity = evaluate()
    #             accelerator.print({"loss/eval": eval_loss, "perplexity": perplexity})
    #             model.train()
    #             accelerator.wait_for_everyone()
    #             unwrapped_model = accelerator.unwrap_model(model)
    #             unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    #             if accelerator.is_main_process:
    #                 tokenizer.save_pretrained(output_dir)
    #                 repo.push_to_hub(
    #                     commit_message=f"Training in progress step {step}", blocking=False
    #                 )
