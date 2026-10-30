import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from hra.data import ASPECTS, load_split, to_matrix
from hra.indobert import ENCODER, AspectModel, class_weights, loss_fn, predict
from hra.metrics import evaluate


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one(seed, args, tokenizer, train, val, device):
    set_seed(seed)
    model = AspectModel(AutoModel.from_pretrained(args.encoder)).to(device)
    texts = [r["text"] for r in train]
    y = to_matrix(train)
    weights = class_weights(y).to(device)

    head = list(model.head.parameters())
    body = [p for n, p in model.named_parameters() if not n.startswith("head.")]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr}, {"params": head, "lr": args.head_lr}], weight_decay=0.01)
    steps = args.epochs * ((len(texts) + args.batch - 1) // args.batch)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    val_texts, y_val = [r["text"] for r in val], to_matrix(val)
    best_f1, best_state, history = -1, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(texts))
        total = 0.0
        for i in range(0, len(order), args.batch):
            idx = order[i : i + args.batch]
            enc = tokenizer(
                [texts[k] for k in idx], truncation=True, max_length=args.max_length, padding=True, return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = loss_fn(model(**enc), torch.tensor(y[idx], device=device), weights)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(idx)

        f1 = evaluate(y_val, predict(model, tokenizer, val_texts, args.max_length, device=device)[0])["pair"]["f1"]
        history.append({"epoch": epoch, "loss": round(total / len(texts), 4), "val_pair_f1": f1})
        print(f"seed {seed} epoch {epoch}: loss {total / len(texts):.4f}, val pair F1 {f1}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_f1, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("models/indobert"))
    ap.add_argument("--encoder", default=ENCODER)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 2026])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, val, test = (load_split(s, args.data) for s in ["train", "val", "test"])
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    test_texts, y_test = [r["text"] for r in test], to_matrix(test)

    runs, best = [], None
    for seed in args.seeds:
        model, val_f1, history = train_one(seed, args, tokenizer, train, val, device)
        start = time.perf_counter()
        pred, _ = predict(model, tokenizer, test_texts, args.max_length, device=device)
        ms = (time.perf_counter() - start) * 1000 / len(test)
        result = evaluate(y_test, pred)
        result["per_aspect_pair_f1"] = {
            a: evaluate(y_test[:, [j]], pred[:, [j]])["pair"]["f1"] for j, a in enumerate(ASPECTS)
        }
        runs.append(
            {"seed": seed, "val_pair_f1": val_f1, "test": result, "history": history, "ms_per_review": round(ms, 2)}
        )
        print(f"seed {seed}: val {val_f1}, test pair F1 {result['pair']['f1']}")
        if best is None or val_f1 > best[0]:
            best = (val_f1, seed, model)

    test_f1 = np.array([r["test"]["pair"]["f1"] for r in runs])
    report = {
        "encoder": args.encoder,
        "device": device,
        "transformers": transformers.__version__,
        "settings": {k: v for k, v in vars(args).items() if k not in ("data", "out")},
        "test_pair_f1_mean": round(float(test_f1.mean()), 4),
        "test_pair_f1_std": round(float(test_f1.std()), 4),
        "saved_seed": best[1],
        "runs": runs,
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/indobert.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    best[2].save(args.out, tokenizer, args.max_length)
    print(f"\ntest pair F1 {report['test_pair_f1_mean']} ± {report['test_pair_f1_std']} over {len(runs)} seeds")
    print(f"saved seed {best[1]} (best on val) to {args.out}")


if __name__ == "__main__":
    main()
