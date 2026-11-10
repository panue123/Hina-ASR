"""PhoWhisper supervised fine-tune (single GPU; short, labelled command audio)."""
import argparse
from dataclasses import dataclass
from pathlib import Path

import jiwer
import numpy as np
import torch
from datasets import Audio, Dataset
from transformers import (
    AutoModelForSpeechSeq2Seq, AutoProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments,
)

from common import check_splits, normalize, read_manifest


@dataclass
class SpeechCollator:
    processor: object
    decoder_start_token_id: int

    def __call__(self, features):
        audio = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features], return_tensors="pt"
        )
        labels = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt"
        )
        label_ids = labels["input_ids"].masked_fill(labels["attention_mask"].ne(1), -100)
        if (label_ids[:, 0] == self.decoder_start_token_id).all().item():
            label_ids = label_ids[:, 1:]
        audio["labels"] = label_ids
        return audio


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--model", default="vinai/PhoWhisper-base")
    p.add_argument("--output", default="outputs/phowhisper-base")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--accumulation", type=int, default=8)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This starter expects an NVIDIA CUDA GPU for training (e.g., Colab GPU).")

    train_rows = read_manifest(args.train)
    val_rows = read_manifest(args.val)
    check_splits({"train": train_rows, "val": val_rows})

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model)
    processor.tokenizer.set_prefix_tokens(language="vietnamese", task="transcribe")
    model.generation_config.language = "vietnamese"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False  # gradient checkpointing requires this while training

    def encode(row):
        wave = row["audio"]
        return {
            "input_features": processor.feature_extractor(
                wave["array"], sampling_rate=16000
            ).input_features[0],
            "labels": processor.tokenizer(row["text"]).input_ids,
        }

    def to_dataset(rows):
        ds = Dataset.from_list([{"audio": r["audio"], "text": r["text"]} for r in rows])
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        return ds.map(encode, remove_columns=ds.column_names)

    train_ds, val_ds = to_dataset(train_rows), to_dataset(val_rows)

    def metrics(pred):
        pred_texts = processor.tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        labels = np.where(pred.label_ids == -100, processor.tokenizer.pad_token_id, pred.label_ids)
        ref_texts = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
        refs = [normalize(t) for t in ref_texts]
        hyps = [normalize(t) for t in pred_texts]
        return {"wer": jiwer.wer(refs, hyps), "cer": jiwer.cer(refs, hyps)}

    out = Path(args.output).resolve()
    training = Seq2SeqTrainingArguments(
        output_dir=str(out), per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accumulation, per_device_eval_batch_size=1,
        learning_rate=args.learning_rate, num_train_epochs=args.epochs,
        gradient_checkpointing=True, fp16=True, eval_strategy="epoch", save_strategy="epoch",
        logging_steps=20, predict_with_generate=True, generation_max_length=128,
        load_best_model_at_end=True, metric_for_best_model="wer", greater_is_better=False,
        save_total_limit=2, report_to="none", seed=42,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=training, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=SpeechCollator(processor, model.config.decoder_start_token_id),
        compute_metrics=metrics, tokenizer=processor.feature_extractor,
    )
    trainer.train()
    model.config.use_cache = True  # fast generation after finishing training
    trainer.save_model(str(out / "best-model"))
    processor.save_pretrained(str(out / "best-model"))
    print("Saved:", out / "best-model")


if __name__ == "__main__":
    main()
