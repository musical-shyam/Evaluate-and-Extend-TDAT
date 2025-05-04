import os
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from textattack.attack_recipes import TextFoolerJin2019
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import Dataset
from textattack import Attacker, AttackArgs
import nltk
import json
import pandas as pd
from pathlib import Path

# Custom utility imports
import sys
sys.path.append('../utils')
from distilbert_utils import get_sst_examples, generate_data_loader, transfer_device, count_correct

# Ensure necessary NLTK data is available
nltk.download('averaged_perceptron_tagger_eng')

# Setup
GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_dir = "./AT_checkpoints"
test_data_path = './../../data/SST-2/dev.tsv'
transformer_type = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = DistilBertTokenizer.from_pretrained(transformer_type)

# Load test data
_, test_examples = get_sst_examples(test_data_path, test=True, discard_values=0)
label_map = {'0': 0, '1': 1}
test_dataloader = generate_data_loader(test_examples, label_map, tokenizer, batch_size=64, do_shuffle=False)
test_texts = [ex[0] for ex in test_examples]
test_labels = [int(ex[1]) for ex in test_examples]
dataset_for_attack = list(zip(test_texts, test_labels))

# Results holder
taxonomy_results = {}

# Loop over epochs
for epoch in range(1, 13):
    ckpt_path = os.path.join(checkpoint_dir, f"AT_bert_checkpoint_epoch_{epoch}.pt")
    if not os.path.exists(ckpt_path):
        continue

    model = DistilBertForSequenceClassification.from_pretrained(transformer_type)
    model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
    model = transfer_device(GPU, model)
    model.eval()

    # Helper to get predictions
    def get_predictions(dataloader):
        preds, labels = [], []
        for input_ids, input_mask_array, batch_labels in dataloader:
            input_ids = transfer_device(GPU, input_ids)
            input_mask_array = transfer_device(GPU, input_mask_array)
            logits = model(input_ids=input_ids, attention_mask=input_mask_array)['logits']
            probs = F.softmax(logits, dim=1)
            pred_labels = probs.argmax(dim=1).cpu().tolist()
            preds.extend(pred_labels)
            labels.extend(batch_labels.tolist())
        return preds, labels

    clean_preds, true_labels = get_predictions(test_dataloader)
    print("clean done")
    # Attack with TextAttack for this model
    wrapped_model = HuggingFaceModelWrapper(model, tokenizer)
    attack = TextFoolerJin2019.build(wrapped_model)
    attack_args = AttackArgs(num_examples=-1, disable_stdout=True, shuffle=False)
    attacker = Attacker(attack, Dataset(dataset_for_attack), attack_args)
    adversarial_results = attacker.attack_dataset()

    adv_texts = [r.perturbed_text() if r.perturbed_result else r.original_text() for r in adversarial_results]
    adv_examples = [{"guid": str(i), "text_a": txt, "label": str(lbl)} for i, (txt, lbl) in enumerate(zip(adv_texts, test_labels))]
    # adv_dataloader = generate_data_loader(adv_examples, label_map, tokenizer, batch_size=64, do_shuffle=False)
    # Fix format to match what generate_data_loader expects
    adv_examples_tuples = [(ex["text_a"], ex["label"]) for ex in adv_examples]

    # Now safe to pass to loader
    adv_dataloader = generate_data_loader(adv_examples_tuples, label_map, tokenizer, batch_size=64, do_shuffle=False)

    adv_preds, _ = get_predictions(adv_dataloader)

    # Case computation
    case_counts = {"Case 1": 0, "Case 2": 0, "Case 3": 0, "Case 4": 0, "Case 5": 0}
    for clean, adv, true in zip(clean_preds, adv_preds, true_labels):
        if clean == true and adv == true:
            case_counts["Case 1"] += 1
        elif clean == true and adv != true:
            case_counts["Case 2"] += 1
        elif clean != true and adv == true:
            case_counts["Case 3"] += 1
        elif clean != true and adv != true and clean == adv:
            case_counts["Case 4"] += 1
        elif clean != true and adv != true and clean != adv:
            case_counts["Case 5"] += 1

    clean_acc = sum([1 for p, t in zip(clean_preds, true_labels) if p == t]) / len(true_labels)
    adv_acc = sum([1 for p, t in zip(adv_preds, true_labels) if p == t]) / len(true_labels)

    taxonomy_results[f"Epoch {epoch}"] = {
        "Clean Accuracy": round(clean_acc, 4),
        "Adversarial Accuracy": round(adv_acc, 4),
        "Cases": case_counts
    }

    Path("taxonomy_results.json").write_text(json.dumps(taxonomy_results, indent=2))


    # Optional: Save adversarial examples per epoch
    # adv_save_path = f"adv_examples_epoch_{epoch}.tsv"
    # with open(adv_save_path, "w") as f:
    #     f.write("Text\tLabel\n")
    #     for txt, lbl in zip(adv_texts, test_labels):
    #         f.write(f"{txt}\t{lbl}\n")

# Save final results
# Path("taxonomy_results.json").write_text(json.dumps(taxonomy_results, indent=2))

# Show table
import ace_tools as tools; tools.display_dataframe_to_user(name="TDAT Taxonomy Cases", dataframe=pd.DataFrame(taxonomy_results).T)
