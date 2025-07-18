# Low memory full-parameter fine-tuning of LLMs

Simple demonstration of full-parameter fine-tuning using low-memory optimizers.
We can fine-tune Llama-3.1-8B using just 19 GB of memory!

There are two scripts: baseline and low-memory fine-tuning.

### Baseline

The baseline consists of 32-bit Adam finetuning on the Hellaswag dataset.

It can be run using `python finetune_base.py --run-name BaselineAdam`.

You can also run Adafactor optimizer using `python finetune_base.py --run-name Adafactor --lr 1e-4 --optimizer Adafactor --batch_size 8 --log_every 400`.

### Low memory optimization

If you have P parameters and use float32 precision and Adam optimizer, you will need 16P bytes of memory.
If you use Adafactor this goes down to 8P bytes of memory. Plus memory for activations.

To reduce memory consumption, we start with Adafactor optimizer and apply the following tricks:

* Gradient checkpointing, using `model.gradient_checkpointing_enable()` (do not forget to set the model into train mode). This allows us not to save every activation. But we need to recompute the missing ones during the backward pass.
* Using `linear_cross_entropy` from https://github.com/apple/ml-cross-entropy. This saves memory on the output layer (softmax over 100k options is memory hungry).
* Using gradients immediately during backward pass by `register_post_accumulate_grad_hook` and then setting `p.grad = None`.
* Using a smaller batch size. This reduces memory and also improves results, as shown in https://arxiv.org/abs/2507.07101.
* Using bf16 weights. But if we use them naively, this would not work, because bf16 has very low precision, and in bf16 1 + 0.001 = 1. Thus we use stochastic rounding during optimizer update step.

This can be run using `python finetune_lowmem.py --run-name LowMem`.

### Results

We finetune Llama-3.2-1B.

| Method | Accuracy | Memory consumption |
| ----------- | ----------- | ---- |
| No finetuning | 47.8 | - |
| FP32 Adam | 54.2 | 29.2 GB |
| FP32 Adafactor | 54.5 | 14.4 GB |
| Lowmemory BF16 Adafactor | 53.6 | 3.9 GB |

We can also fine-tune Llama-3.1-8B using only 18.6 GB of memory!
