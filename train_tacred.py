import os
import warnings
import sys

warnings.filterwarnings("ignore", category=UserWarning, message="User provided device_type of 'cuda'")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from utils import set_seed, collate_fn
from prepro import TACREDProcessor
from evaluation import get_f1
from model import REModel
from torch.cuda.amp import GradScaler
import wandb
import csv


def train(args, model, train_features, benchmarks):
    train_dataloader = DataLoader(train_features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn,
                                  drop_last=True)
    total_steps = int(len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    use_amp = torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    print('Total steps: {}'.format(total_steps))
    print('Warmup steps: {}'.format(warmup_steps))

    num_steps = 0
    for epoch in range(int(args.num_train_epochs)):
        model.zero_grad()
        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            inputs = {'input_ids': batch[0].to(args.device),
                      'attention_mask': batch[1].to(args.device),
                      'labels': batch[2].to(args.device),
                      'ss': batch[3].to(args.device),
                      'os': batch[4].to(args.device),
                      }
            outputs = model(**inputs)
            loss = outputs[0] / args.gradient_accumulation_steps
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if step % args.gradient_accumulation_steps == 0:
                num_steps += 1
                if args.max_grad_norm > 0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                model.zero_grad()
                wandb.log({'loss': loss.item()}, step=num_steps)

            if num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0:
                for tag, features in benchmarks:
                    f1, output = evaluate(args, model, features, tag=tag)
                    wandb.log(output, step=num_steps)
                return  # early stopping

    # for tag, features in benchmarks:
    #    f1, output = evaluate(args, model, features, tag=tag)
    #    wandb.log(output, step=num_steps)


def evaluate(args, model, features, tag='dev'):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, collate_fn=collate_fn, drop_last=False)
    keys, preds = [], []
    for i_b, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {tag} set")):
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'ss': batch[3].to(args.device),
                  'os': batch[4].to(args.device),
                  }
        keys += batch[2].tolist()
        with torch.no_grad():
            logit = model(**inputs)[0]
            pred = torch.argmax(logit, dim=-1)
        preds += pred.tolist()

    keys = np.array(keys, dtype=np.int64)
    preds = np.array(preds, dtype=np.int64)
    _, _, max_f1 = get_f1(keys, preds)

    output = {
        tag + "_f1": max_f1 * 100,
    }
    print(output)
    return max_f1, output


def main():
    print("Process ID:", os.getpid(), "Args:", sys.argv)
    parser = argparse.ArgumentParser()
    training_num = None  # default=None
    epoch_num = 5.0  # default=5.0
    test_num = None  # default=None
    max_token_length = 512  # default=512
    eval_steps = 10000

    train_lists = ["train_mix_5000.json", "train_mix_6000.json", "train_mix_7000.json", "train_mix_8000.json", "train_mix_9000.json", "train_mix_10000.json", "train_mix_12000.json",
                   "train_mix_15000.json", "train_mix_20000.json", "train_mix_25000.json", "train_mix_30000.json", "train_mix_40000.json", "train_mix_50000.json", "train_mix_60000.json",
                   "train_mix_68124.json"]

    parser.add_argument("--data_dir", default="./data/tacred/skewed/mix", type=str)
    parser.add_argument("--model_name_or_path", default="roberta-large", type=str)
    parser.add_argument("--input_format", default="typed_entity_marker_punct", type=str,
                        help="in [entity_mask, entity_marker, entity_marker_punct, typed_entity_marker, typed_entity_marker_punct]")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=max_token_length, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated.")

    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=32, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int,
                        help="Number of updates steps to accumulate the gradients for, before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=epoch_num, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=42)
    parser.add_argument("--evaluation_steps", type=int, default=eval_steps,
                        help="Number of steps to evaluate the model")

    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--project_name", type=str, default="RE_baseline")
    parser.add_argument("--run_name", type=str, default="tacred")

    args = parser.parse_args()
    # Auto login to wandb (avoid interactive prompt)
    os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY", "")
    os.environ["WANDB_MODE"] = "online"  # online or offline
    os.environ["WANDB_CONSOLE"] = "wrap"  # prevent file logging, logs only to console

    # Check all training files exist before proceeding
    missing_files = []
    for file in train_lists:
        file_path = os.path.join(args.data_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    if missing_files:
        print("Error: The following training files do not exist:")
        for f in missing_files:
            print("  -", f)
        sys.exit(1)

    results_csv = open("training_results.csv", "a", newline="")
    csv_writer = csv.writer(results_csv, delimiter="\t")
    csv_writer.writerow(["dataset", "dev_f1", "dev_rev_f1", "test_f1", "test_rev_f1"])

    for train_file in train_lists:  # loop through each training file
        wandb.init(project=args.project_name, name=f"{args.run_name}_{train_file}", reinit=True)
        print(f"\n==== Training dataset: {train_file} ====\n")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
        args.device = device
        if args.seed > 0:
            set_seed(args)

        config = AutoConfig.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=args.num_class,
        )
        config.gradient_checkpointing = False
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        )

        model = REModel(args, config)
        model.to(args.device)

        train_file = os.path.join(args.data_dir, train_file)
        dev_file = os.path.join(args.data_dir, "dev.json")
        test_file = os.path.join(args.data_dir, "test.json")
        dev_rev_file = os.path.join(args.data_dir, "dev_rev.json")
        test_rev_file = os.path.join(args.data_dir, "test_rev.json")

        processor = TACREDProcessor(args, tokenizer)
        train_features = processor.read(train_file, max_examples=training_num)  # small set of examples
        dev_features = processor.read(dev_file, max_examples=test_num)
        test_features = processor.read(test_file, max_examples=test_num)
        dev_rev_features = processor.read(dev_rev_file, max_examples=test_num)
        test_rev_features = processor.read(test_rev_file, max_examples=test_num)

        if len(processor.new_tokens) > 0:
            model.encoder.resize_token_embeddings(len(tokenizer))

        benchmarks = (
            ("dev", dev_features),
            ("test", test_features),
            ("dev_rev", dev_rev_features),
            ("test_rev", test_rev_features),
        )

        train(args, model, train_features, benchmarks)

        # Predefine to hold outputs
        output_dev, output_test, output_dev_rev, output_test_rev = {}, {}, {}, {}
        # Evaluate and collect F1 scores after training
        _, output_dev = evaluate(args, model, dev_features, tag="dev")
        _, output_test = evaluate(args, model, test_features, tag="test")
        _, output_dev_rev = evaluate(args, model, dev_rev_features, tag="dev_rev")
        _, output_test_rev = evaluate(args, model, test_rev_features, tag="test_rev")

        dev_f1 = output_dev.get("dev_f1", 0.0)
        test_f1 = output_test.get("test_f1", 0.0)
        dev_rev_f1 = output_dev_rev.get("dev_rev_f1", 0.0)
        test_rev_f1 = output_test_rev.get("test_rev_f1", 0.0)
        csv_writer.writerow([
            train_file,
            f"{dev_f1:.2f}",
            f"{dev_rev_f1:.2f}",
            f"{test_f1:.2f}",
            f"{test_rev_f1:.2f}",
        ])
        results_csv.flush()
        wandb.finish()
    results_csv.close()
    print("\n==== All files processed. ====\n")


if __name__ == "__main__":
    main()
