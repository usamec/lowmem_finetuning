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

def hellaswag_dataset(tokenizer, split: str = "train"):
    """
    Converts HellaSwag rows into a single causal‑LM string:

        <ctx> <gold ending> <eos>

    • `input_ids`, `attention_mask` – as usual  
    • `loss_mask` – 1 for tokens from the gold ending (incl. <eos>), 0 otherwise
    """
    # ---------- preprocess ----------
    def add_text(row):
        gold = row["endings"][int(row["label"])]
        row["text"] = row["ctx"] + " " + gold + tokenizer.eos_token
        # token count of the context only (no special tokens)
        row["ctx_len"] = len(tokenizer(row["ctx"],
                                       add_special_tokens=False)["input_ids"])
        return row

    ds = load_dataset("Rowan/hellaswag", split=split)
    ds = ds.map(add_text)
    ds = ds.remove_columns([c for c in ds.column_names
                            if c not in ("text", "ctx_len")])

    # ---------- batch collation ----------
    def collate(batch):
        texts     = [ex["text"]     for ex in batch]
        ctx_lens  = [ex["ctx_len"]  for ex in batch]

        enc = tokenizer(texts,
                        padding="max_length",
                        max_length=160,
                        return_tensors="pt")

        # build loss mask (1 = predict, 0 = ignore)
        seq_len   = enc.input_ids.size(1)
        loss_mask = torch.zeros_like(enc.input_ids, dtype=torch.float)

        for i, l in enumerate(ctx_lens):
            l = min(l, seq_len)          # safety
            loss_mask[i, l:] = 1         # gold ending + <eos>

        loss_mask *= enc.attention_mask   # drop padding positions

        return {
            "input_ids":      enc.input_ids,
            "attention_mask": enc.attention_mask,
            "loss_mask":      loss_mask,
        }

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
            loss = (loss * batch["loss_mask"][:,:-1].flatten()).mean()

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
