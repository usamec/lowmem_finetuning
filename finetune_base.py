import argparse, math, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# lm‑eval‑harness bits
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator

from utils import Adafactor

def hellaswag_dataset(tokenizer, split="train"):
    """
    HellaSwag rows → single causal‑LM string:
    <ctx> <gold ending> <eos>
    """
    def add_text(row):
        gold = row["endings"][int(row["label"])]
        row["text"] = row["ctx"] + " " + gold + tokenizer.eos_token
        return row

    ds = load_dataset("Rowan/hellaswag", split=split)
    ds = ds.map(add_text, remove_columns=ds.column_names)

    def collate(batch):
        enc = tokenizer(
            [s["text"] for s in batch],
            padding="max_length",
            max_length=160,
            return_tensors="pt",
        )
        return {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask}

    return ds, collate


def evaluate(model, tokenizer, device):
    model.eval()                              # just to be safe
    lm = HFLM(pretrained=model,               # pass the *object*, not a name
              tokenizer=tokenizer,
              device=device,
              batch_size=32)


    with torch.autocast("cuda", dtype=torch.bfloat16):
        res = evaluator.simple_evaluate(
            model=lm,
            tasks=["hellaswag"],
            num_fewshot=0,
            bootstrap_iters=0,
        )
    return res["results"]["hellaswag"]["acc,none"] * 100.0


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
    ).to(device)
    model = torch.compile(model, dynamic=True)

    ds, collate = hellaswag_dataset(tok)
    g = torch.Generator()
    g.manual_seed(args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, generator=g)
    print("dl len", len(dl))

    if args.optimizer == "SGD":
        optim = torch.optim.SGD(model.model.layers.parameters(), lr=args.lr)
    elif args.optimizer == "Adafactor":
        optim = Adafactor(model.model.layers.parameters(), lr=args.lr)
    else:
        optim = torch.optim.Adam(model.model.layers.parameters(), lr=args.lr)
    print("opt", args.optimizer, optim)

    writer = SummaryWriter(f"runs_hell/{args.run_name}")
 
    acc = evaluate(model, tok, device)
    print(f"epoch 0 done — HellaSwag acc {acc:.2f}%")
    writer.add_scalar("val/acc", acc, 0)

    total_steps = 0
    running = 0.0
    for epoch in range(1, args.epochs + 1):
        torch.cuda.empty_cache()
        model.train()
        model.gradient_checkpointing_enable()
        model.config.use_cache = False


        for step, batch in enumerate(dl, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lm_logits = model(**batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = batch["input_ids"][:, 1:]
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
            loss = (loss * batch["attention_mask"][:,:-1].flatten()).mean()

            loss.backward()
            optim.step(); optim.zero_grad()

            running += loss.item()
            total_steps += 1
            if total_steps % args.log_every == 0:
                free,total=torch.cuda.mem_get_info()
                print(f"epoch {epoch} step {step}  loss {running/args.log_every:.4f} memory {(total-free)/1024/1024/1024:.2f} GB, Total: {total/1024/1024/1024:.2f} GB")
                writer.add_scalar("Train/loss", running/args.log_every, total_steps * args.batch_size)
                writer.add_scalar("Train/memoryGB", (total-free)/1024/1024/1024, total_steps *args.batch_size) 
                running = 0.0

        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        acc = evaluate(model, tok, device)
        print(f"epoch {epoch} done — HellaSwag acc {acc:.2f}%")
        writer.add_scalar("val/acc", acc, total_steps * args.batch_size)

    writer.close()

    print("Training finished.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--optimizer", default="Adam")
    p.add_argument("--lr", type=float, default=3e-6)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=100)
    args = p.parse_args()
    main(args)
