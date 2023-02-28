def train(
        push_to_hub:bool,
        num_epoch: int,
        train_batch_size: int,
        eval_batch_size: int,
):
    import torch
    import numpy as np

    # 1. Dataset
    from datasets import load_dataset
    dataset = load_dataset("Adapting/abstract-keyphrases")

    # 2. Model
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from lrt.clustering.models import KeyBartAdapter
    tokenizer = AutoTokenizer.from_pretrained("Adapting/KeyBartAdapter")

    '''
    Or you can just use the initial model weights from Huggingface:
    model = AutoModelForSeq2SeqLM.from_pretrained("Adapting/KeyBartAdapter",
                                                  revision='9c3ed39c6ed5c7e141363e892d77cf8f589d5999')
    '''

    model = KeyBartAdapter(256)

    # 3. preprocess dataset
    dataset = dataset.shuffle()

    def preprocess_function(examples):
        inputs = examples['Abstract']
        targets = examples['Keywords']
        model_inputs = tokenizer(inputs, truncation=True)

        # Set up the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # 4. evaluation metrics
    def compute_metrics(eval_preds):
        preds = eval_preds.predictions
        labels = eval_preds.label_ids
        if isinstance(preds, tuple):
            preds = preds[0]
        print(preds.shape)
        if len(preds.shape) == 3:
            preds = preds.argmax(axis=-1)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds = [a.strip().split(';') for a in decoded_preds]
        decoded_labels = [a.strip().split(';') for a in decoded_labels]

        precs, recalls, f_scores = [], [], []
        num_match, num_pred, num_gold = [], [], []
        for pred, label in zip(decoded_preds, decoded_labels):
            pred_set = set(pred)
            label_set = set(label)
            match_set = label_set.intersection(pred_set)
            p = float(len(match_set)) / float(len(pred_set)) if len(pred_set) > 0 else 0.0
            r = float(len(match_set)) / float(len(label_set)) if len(label_set) > 0 else 0.0
            f1 = float(2 * (p * r)) / (p + r) if (p + r) > 0 else 0.0
            precs.append(p)
            recalls.append(r)
            f_scores.append(f1)
            num_match.append(len(match_set))
            num_pred.append(len(pred_set))
            num_gold.append(len(label_set))

            # print(f'raw_PRED: {raw_pred}')
            print(f'PRED: num={len(pred_set)} - {pred_set}')
            print(f'GT: num={len(label_set)} - {label_set}')
            print(f'p={p}, r={r}, f1={f1}')
            print('-' * 20)

        result = {
            'precision@M': np.mean(precs) * 100.0,
            'recall@M': np.mean(recalls) * 100.0,
            'fscore@M': np.mean(f_scores) * 100.0,
            'num_match': np.mean(num_match),
            'num_pred': np.mean(num_pred),
            'num_gold': np.mean(num_gold),
        }

        result = {k: round(v, 2) for k, v in result.items()}
        return result

    # 5. train
    from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    model_name = 'KeyBartAdapter'

    args = Seq2SeqTrainingArguments(
        model_name,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=num_epoch,
        logging_steps=4,
        load_best_model_at_end=True,
        metric_for_best_model='fscore@M',
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),  # speeds up training on modern GPUs.
        # eval_accumulation_steps=10,
    )

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # 6. push
    if push_to_hub:
        commit_msg = f'{model_name}_{num_epoch}'
        tokenizer.push_to_hub(commit_message=commit_msg, repo_id=model_name)
        model.push_to_hub(commit_message=commit_msg, repo_id=model_name)

    return model, tokenizer

if __name__ == '__main__':
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.absolute()
    sys.path.append(project_root.__str__())


    # code
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--epoch", help="number of epochs", default=30)
    parser.add_argument("--train_batch_size", help="training batch size", default=16)
    parser.add_argument("--eval_batch_size", help="evaluation batch size", default=16)
    parser.add_argument("--push", help="whether push the model to hub", action='store_true')

    args = parser.parse_args()
    print(args)

    model, tokenizer = train(
        push_to_hub= bool(args.push),
        num_epoch= int(args.epoch),
        train_batch_size= int(args.train_batch_size),
        eval_batch_size= int(args.eval_batch_size)
    )

