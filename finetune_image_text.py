# finetune_siglip2_trainer.py
# Fine-tune SigLIP2 with LoRA using Hugging Face Trainer
# pip install transformers accelerate datasets peft torch torchvision timm

import argparse
import torch
import pandas as pd
#import tqdm
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
#from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModel,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback
)
#from datasets import load_dataset
import evaluate
from torchvision import transforms
from peft import LoraConfig, get_peft_model, PeftModel
import os
import sys
sys.path.append("/home/glados/unix-Documents/AstroSignals")
from feuerzeug.transforms import FakeChannels
from fixmatch.main.dataloading.datasets import MBFRConfident
import wandb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score


class ImageTextDataset(Dataset):
    def __init__(self, hf_dataset, processor, text_captions, transform=None, labels=False):
        self.ds = hf_dataset
        self.processor = processor
        self.img_transform = transform
        self.text = pd.read_csv(text_captions)["caption"]
        self.labels = labels

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        if self.img_transform:
            img = self.img_transform(img)
        text = self.text[idx]
        if self.labels:
            return {"pixel_values": img, "input_ids": text, "label":label}
        else:
            return {"pixel_values": img, "input_ids": text}


def collate_batch(batch, processor):
    images = [b["pixel_values"] for b in batch]
    texts = [b["input_ids"] for b in batch]
    encodings = processor(
        text=texts,
        images=images,
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="pt",)
    #if all("label" in b for b in batch) and getattr(batch[0], "include_labels", True):
    #    encodings["labels"] = torch.Tensor([b["label"] for b in batch])
    #print(encodings.keys())
    return encodings

def collate_batch_with_labels(batch, processor):
    images = [b["pixel_values"] for b in batch]
    texts = [b["input_ids"] for b in batch]
    encodings = processor(
        text=texts,
        images=images,
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="pt",)
    #if all("label" in b for b in batch) and getattr(batch[0], "include_labels", True):
    encodings["labels"] = torch.Tensor([b["label"] for b in batch])
    #print(encodings.keys())
    return encodings

def linear_probe_train_test(trainfeats, testfeats, trainlabels, testlabels, niter=10, eval_interval=10, lr = 1e-3):
    clf = nn.Linear(trainfeats.size(1), 2)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr)
    loss_fct = nn.CrossEntropyLoss()
    rdict = {"accuracy":[],"f1":[],"loss":[]}
    for i in range(niter):  # small number of epochs
        optimizer.zero_grad()
        logits = clf(trainfeats)
        loss = loss_fct(logits, trainlabels.long())
        loss.backward()
        optimizer.step()
        #if i % eval_interval == 0:
        if i in range(niter-10, niter):
            preds = torch.argmax(clf(testfeats), dim=1)
            accuracy = evaluate.load("accuracy").compute(predictions=preds, references=testlabels.long())["accuracy"]
            f1 = evaluate.load("f1").compute(predictions=preds, references=testlabels.long())["f1"]
            rdict["loss"].append(loss.item())
            rdict["f1"].append(f1)
            rdict["accuracy"].append(accuracy)
    #preds = torch.argmax(clf(testfeats), dim=1)
    #accuracy = evaluate.load("accuracy").compute(predictions=preds, references=testlabels.long())["accuracy"]
    #f1 = evaluate.load("f1").compute(predictions=preds, references=testlabels.long())["f1"]
    #return accuracy, f1, loss #rdict
    return np.mean(rdict['accuracy']),np.mean(rdict['f1']),np.mean(rdict['loss'])

def linear_probe(feats, labels, niter=10, lr = 1e-3):
    clf = nn.Linear(feats.size(1), 2)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr)
    loss_fct = nn.CrossEntropyLoss()

    for _ in range(niter):  # small number of epochs
        optimizer.zero_grad()
        logits = clf(feats)
        loss = loss_fct(logits, labels.long())
        loss.backward()
        optimizer.step()

    preds = torch.argmax(clf(feats), dim=1)
    accuracy = evaluate.load("accuracy").compute(predictions=preds, references=labels.long())["accuracy"]
    f1 = evaluate.load("f1").compute(predictions=preds, references=labels.long())["f1"]
    return accuracy, f1, loss

def evaluate_simscore(sim_matrix, labels, top_k=5):
    # Recall@1
    top1_preds_i2t = sim_matrix.argmax(dim=1)
    top1_preds_t2i = sim_matrix.argmax(dim=0)
    recall1_i2t = (top1_preds_i2t == labels).float().mean().item()
    recall1_t2i = (top1_preds_t2i == labels).float().mean().item()
    recall1 = (recall1_i2t + recall1_t2i) / 2

    # Top-k Recall
    topk_preds_i2t = sim_matrix.topk(k=top_k, dim=1).indices
    topk_preds_t2i = sim_matrix.topk(k=top_k, dim=0).indices
    topk_recall_i2t = (topk_preds_i2t == labels.unsqueeze(1)).any(dim=1).float().mean().item()
    topk_recall_t2i = (topk_preds_t2i == labels.unsqueeze(0)).any(dim=0).float().mean().item()
    topk_recall = (topk_recall_i2t + topk_recall_t2i) / 2
    return recall1, topk_recall

def evaluate_class_level_recall1(img_embeds, txt_embeds, img_labels, txt_labels):
    """
    Compute class-level Recall@1:
    Image retrieves text from the same class, and vice versa.
    """
    # Normalize embeddings
    img_embeds = F.normalize(img_embeds, dim=-1)
    txt_embeds = F.normalize(txt_embeds, dim=-1)

    # similarity matrix [N, N]
    sim_matrix = img_embeds @ txt_embeds.T  

    # --- Image → Text ---
    top1_indices_i2t = sim_matrix.argmax(dim=1)  # best match text index for each image
    correct_i2t = (txt_labels[top1_indices_i2t] == img_labels).float().mean().item()

    # --- Text → Image ---
    top1_indices_t2i = sim_matrix.argmax(dim=0)  # best match image index for each text
    correct_t2i = (img_labels[top1_indices_t2i] == txt_labels).float().mean().item()

    recall1_class_level = (correct_i2t + correct_t2i) / 2
    return recall1_class_level


def knn_classification_probe_train_test(train_embeddings, test_embeddings, train_labels, test_labels, k=5):
    """
    KNN probe for classification using embeddings.
    Majority-vote among k-nearest neighbors.
    """
    train_embeddings_np = train_embeddings.cpu().numpy()
    train_labels_np = train_labels.cpu().numpy()
    test_embeddings_np = test_embeddings.cpu().numpy()
    test_labels_np = test_labels.cpu().numpy()

    # Train-test split (LOO = leave-one-out)
    knn = KNeighborsClassifier(n_neighbors=k, weights="uniform", metric="cosine") #, metric="cosine", weights = "")
    knn.fit(train_embeddings_np, train_labels_np)

    preds = knn.predict(test_embeddings_np)
    acc = (preds == test_labels_np).mean()
    f1 = f1_score(test_labels_np, preds, average="macro")
    return acc, f1

def knn_classification_probe(embeddings, labels, k=5):
    """
    KNN probe for classification using embeddings.
    Majority-vote among k-nearest neighbors.
    """
    embeddings_np = embeddings.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Train-test split (LOO = leave-one-out)
    knn = KNeighborsClassifier(n_neighbors=k, weights="uniform", metric="cosine") #, metric="cosine", weights = "")
    knn.fit(embeddings_np, labels_np)

    preds = knn.predict(embeddings_np)
    acc = (preds == labels_np).mean()
    f1 = f1_score(labels_np, preds, average="macro")
    return acc, f1


def evaluate_classification_similarity(model, val_dataset, device, processor, batch_size=16, top_k = 5, save_features=False, niter=10, linear_lr = 1e-3, train_dataset=None):
    model.eval()
    
    if train_dataset is None:
        datasets = [val_dataset]
        dstypes = ["val"]
    else: 
        datasets = [train_dataset, val_dataset]
        dstypes = ["train","val"]

    ddict = {}
    for ds,dt in zip(datasets,dstypes):
        all_feats, txt_feats = [], []
        all_labels = []
        sim_scores = []
        img_embeds_list = []
        txt_embeds_list = []
        indices = list(range(len(ds)))
        for i in range(0, len(ds), batch_size):
            batch_indices = indices[i : i + batch_size]
            batch_items = [ds[j] for j in batch_indices]
            encodings = collate_batch_with_labels(batch_items, processor)
            pixel_values = encodings['pixel_values'].to(device)
            input_ids = encodings['input_ids'].to(device)
            labels = encodings['labels'].to(device)
            #pixel_values = torch.stack([b["pixel_values"] for b in batch_items]).to(device)
            #input_ids = torch.stack([b["input_ids"] for b in batch_items]).to(device)
            #labels = torch.tensor([b["label"] for b in batch_items]).to(device)

            with torch.no_grad():
                feats = model.get_image_features(pixel_values=pixel_values)
                img_embeds = F.normalize(feats, dim=-1)
                text_feats = model.get_text_features(input_ids)
                txt_embeds = F.normalize(text_feats, dim=-1)
                sim = torch.cosine_similarity(img_embeds, txt_embeds, dim=-1)
            
            sim_scores.append(sim.cpu())
            all_feats.append(feats.cpu())
            txt_feats.append(text_feats.cpu())
            all_labels.append(labels.cpu())
            img_embeds_list.append(img_embeds.cpu())
            txt_embeds_list.append(txt_embeds.cpu())

        ddict[f"{dt}_all_feats"] = torch.cat(all_feats)
        ddict[f"{dt}_all_txt_feats"] = torch.cat(txt_feats)
        ddict[f"{dt}_all_labels"] = torch.cat(all_labels)
        ddict[f"{dt}_sim_tensor"] = torch.cat(sim_scores)
        ddict[f"{dt}_img_embeds_all"] = torch.cat(img_embeds_list)
        ddict[f"{dt}_txt_embeds_all"] = torch.cat(txt_embeds_list)
        ddict[f"{dt}_mean_sim"] = ddict[f"{dt}_sim_tensor"].mean().item()
        ddict[f"{dt}_std_sim"] = ddict[f"{dt}_sim_tensor"].std().item()

    all_feats = ddict[f"{dt}_all_feats"]
    all_txt_feats = ddict[f"{dt}_all_txt_feats"]
    all_labels = ddict[f"{dt}_all_labels"]
    sim_tensor = ddict[f"{dt}_sim_tensor"]
    img_embeds_all = ddict[f"{dt}_img_embeds_all"]
    txt_embeds_all = ddict[f"{dt}_txt_embeds_all"]
    mean_sim = ddict[f"{dt}_mean_sim"]
    std_sim = ddict[f"{dt}_std_sim"]

    if len(datasets) == 2:
        accuracy, f1, loss = linear_probe_train_test(ddict["train_all_feats"],ddict["val_all_feats"], ddict["train_all_labels"],ddict["val_all_labels"], niter=niter, lr=linear_lr)
        txt_acc, txt_f1, loss_txt = linear_probe_train_test(ddict["train_all_feats"],ddict["val_all_feats"], ddict["train_all_labels"],ddict["val_all_labels"], niter=niter, lr=linear_lr)
        train_cat_feats = torch.cat((ddict["train_all_txt_feats"], ddict["train_all_feats"]),dim=1)
        val_cat_feats =  torch.cat((ddict["val_all_txt_feats"], ddict["val_all_feats"]),dim=1)
        cat_acc, cat_f1, cat_loss = linear_probe_train_test(train_cat_feats, val_cat_feats, ddict["train_all_labels"],ddict["val_all_labels"], niter=niter, lr=linear_lr)
        #print(torch.mean(torch.stack((all_txt_feats, all_feats)),dim=0).shape)
        train_avg_feats = torch.mean(torch.stack((ddict["train_all_txt_feats"], ddict["train_all_feats"])),dim=0)
        val_avg_feats =  torch.mean(torch.stack((ddict["val_all_txt_feats"], ddict["val_all_feats"])),dim=0)
        avg_acc, avg_f1, avg_loss = linear_probe_train_test(train_avg_feats, val_avg_feats, ddict["train_all_labels"],ddict["val_all_labels"], niter=niter, lr=linear_lr)

        vkacc, vkf1 = knn_classification_probe_train_test(ddict["train_all_feats"],ddict["val_all_feats"], ddict["train_all_labels"],ddict["val_all_labels"], k=5) #do this for train/test too? or is it fine with just test?
        tkacc, tkf1 = knn_classification_probe_train_test(ddict["train_all_txt_feats"],ddict["val_all_txt_feats"], ddict["train_all_labels"],ddict["val_all_labels"], k=5)
        catkacc, catkf1 = knn_classification_probe_train_test(train_cat_feats, val_cat_feats, ddict["train_all_labels"],ddict["val_all_labels"], k=5)
        avgkacc, avgkf1 = knn_classification_probe_train_test(train_avg_feats, val_avg_feats, ddict["train_all_labels"],ddict["val_all_labels"], k=5)
        vtrain_test_feats = torch.cat((ddict["train_all_feats"], ddict["val_all_feats"]),dim=0)
        ttrain_test_feats = torch.cat((ddict["train_all_txt_feats"], ddict["val_all_txt_feats"]),dim=0)
    else: 
        accuracy, f1, loss = linear_probe(all_feats, all_labels, niter=niter, lr=linear_lr)
        txt_acc, txt_f1, loss_txt = linear_probe(all_txt_feats, all_labels, niter=niter, lr=linear_lr)
        val_cat_feats =  torch.cat((ddict["val_all_txt_feats"], ddict["val_all_feats"]),dim=1)
        cat_acc, cat_f1, cat_loss = linear_probe(val_cat_feats, all_labels, niter=niter, lr=linear_lr)
        #print(torch.mean(torch.stack((all_txt_feats, all_feats)),dim=0).shape)
        val_avg_feats =  torch.mean(torch.stack((ddict["val_all_txt_feats"], ddict["val_all_feats"])),dim=0)
        avg_acc, avg_f1, avg_loss = linear_probe(val_avg_feats, all_labels, niter=niter, lr=linear_lr)
        vtrain_test_feats = torch.cat((all_feats, ddict["val_all_feats"]),dim=0)
        ttrain_test_feats = torch.cat((all_txt_feats,ddict["val_all_txt_feats"]),dim=0)

        vkacc, vkf1 = knn_classification_probe(img_embeds_all, all_labels, k=5) #do this for train/test too? or is it fine with just test?
        tkacc, tkf1 = knn_classification_probe(txt_embeds_all, all_labels, k=5)
        catkacc, catkf1 = knn_classification_probe(val_cat_feats, all_labels, k=5)
        avgkacc, avgkf1 = knn_classification_probe(val_avg_feats, all_labels, k=5)
    # --- Retrieval metrics ---
    # similarity matrix [N, N]
    sim_matrix = img_embeds_all @ txt_embeds_all.T
    labels = torch.arange(sim_matrix.size(0))
    recall1, topk_recall = evaluate_simscore(sim_matrix, labels, top_k=top_k)

    recall1_class_level = evaluate_class_level_recall1(img_embeds_all, txt_embeds_all, all_labels, all_labels)

    if save_features:
        return vtrain_test_feats.numpy(), ttrain_test_feats.numpy(), accuracy, f1, loss.item(), txt_acc, txt_f1, loss_txt.item(),cat_acc, cat_f1, cat_loss.item(), avg_acc, avg_f1, avg_loss.item(), mean_sim, std_sim, recall1, topk_recall, recall1_class_level, vkacc, vkf1, tkacc, tkf1, catkf1, avgkf1
    else:
        return accuracy, f1, loss.item(), txt_acc, txt_f1, loss_txt.item(),cat_acc, cat_f1, cat_loss.item(), avg_acc, avg_f1, avg_loss.item(), mean_sim, std_sim, recall1, topk_recall, recall1_class_level, vkacc, vkf1, tkacc, tkf1, catkf1, avgkf1

class ClassificationProbeCallback(TrainerCallback):
    def __init__(self, val_dataset, eval_interval=10, processor = None, top_k = 5, niter=50, linear_lr = 1e-3, save_features = False):
        self.val_dataset = val_dataset
        self.eval_interval = eval_interval
        self.processor = processor
        self.top_k = top_k
        self.linear_lr = linear_lr
        self.niter = niter
        self.save_features = save_features

    def _run_evaluation(self,state, model, args):
        accuracy, f1, clf_loss, txt_acc, txt_f1, txt_clf_loss,cat_acc, cat_f1, cat_loss, avg_acc, avg_f1, avg_loss, mean_sim, std_sim, recall1, topk_recall,recall1_class_level, vkacc, vkf1, tkacc, tkf1, catkf1, avgkf1 = evaluate_classification_similarity(model, self.val_dataset, device=model.device, processor = self.processor, batch_size = args.per_device_train_batch_size, top_k=self.top_k, niter = self.niter, linear_lr=self.linear_lr, save_features = self.save_features)
        wandb.log({
            "image classification_accuracy": accuracy,
            "image classification_f1": f1,
            "image classification_loss": clf_loss,
            "text classification_accuracy": txt_acc,
            "text classification_f1": txt_f1,
            "text classification_loss": txt_clf_loss,
            "image_text_mean_similarity": mean_sim,
            "image_text_std_similarity": std_sim,
            "retrieval_recall@1": recall1,
            f"retrieval_top{self.top_k}": topk_recall,
            "recall@1 class level":recall1_class_level,
            "vision knn-classifier accuracy": vkacc,
            "vision knn-classifier f1":vkf1,
            "text knn-classifier accuracy": tkacc,
            "text knn-classifier f1":tkf1,
            "eval/epoch": state.epoch
        })
        print(f"[Epoch {int(state.epoch)}] Image classification probe: acc={accuracy:.4f}, f1={f1:.4f}, loss={clf_loss:.4f},\n Text classification probe: acc={txt_acc:.4f}, f1={txt_f1:.4f}, loss={txt_clf_loss:.4f} \n Concatenated Image-Text classification probe: acc={cat_acc:.4f}, f1={cat_f1:.4f}, loss={cat_loss:.4f} \n Averaged Image-Text classification probe: acc={avg_acc:.4f}, f1={avg_f1:.4f}, loss={avg_loss:.4f} \n mean image-text cosine similarity: {mean_sim:.4f}, recall@1: {recall1:.4f}, topK: {topk_recall:.4f} \n class-level recall!@1: {recall1_class_level:.4f}, vision knn-classifier accuracy: {vkacc:.4f}, vision knn-classifier F1: {vkf1:.4f}, text knn-classifier accuracy: {tkacc:.4f}, text knn-classifier F1: {tkf1:.4f}, cat image-text knn-classifier f1: {catkf1:.4f}, avg image-text knn-classifier F1: {avgkf1:.4f}")

    # def on_train_begin(self, args, state, control, **kwargs):
    #     model = kwargs["model"]
    #     self._run_evaluation(state, model, args)

    def on_epoch_end(self, args, state, control, **kwargs):
        if int(state.epoch) % self.eval_interval == 0:
            model = kwargs["model"]
            self._run_evaluation(state, model, args)


class SigLIPTrainer(Trainer):
    """
    Custom Trainer for SigLIP2 with LoRA fine-tuning.

    Loss:
        - Sigmoid loss (global image-text alignment)
        - Optional LocCa loss (local patch-text alignment)
    """

    def __init__(self, *args, locca_weight=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.locca_weight = locca_weight
        self.sigmoid_loss_fct = nn.BCEWithLogitsLoss()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Remove 'labels' if present to avoid TypeError
        #labels = inputs.pop("labels", None)

        outputs = model(**inputs)  

        logits_per_image = outputs.logits_per_image  
        bsz = logits_per_image.size(0)
        target = torch.eye(bsz, device=logits_per_image.device)

        loss = self.sigmoid_loss_fct(logits_per_image, target)


        if self.locca_weight > 0.0:
            # Assuming outputs contains patch embeddings and text embeddings for local alignment
            # You need to implement compute_locca_loss based on your model
            locca_loss = self.compute_locca_loss(outputs)  # placeholder
            loss += self.locca_weight * locca_loss

        return (loss, outputs) if return_outputs else loss

    def compute_locca_loss(self, outputs):
        """
        Computes the LocCa (local contrastive) loss.
        This is a placeholder; implement based on your SigLIP2 outputs:
        - outputs.patch_image_embeds: [B, num_patches, D]
        - outputs.text_embeds: [B, D]
        """
        # Example: cosine similarity between patches and text
        patch_embeds = outputs.vision_model_output.last_hidden_state[:,1:,:] #.patch_image_embeds  # [B, P, D]
        text_embeds = outputs.text_embeds.unsqueeze(1)  # [B, 1, D]

        # cosine similarity along feature dim
        sim = torch.cosine_similarity(patch_embeds, text_embeds, dim=-1)  # [B, P]

        # maximize similarity for matching pairs
        locca_loss = 1.0 - sim.mean()  # simple loss for demonstration
        return locca_loss
    
    # def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
    #     # Remove labels so the model forward doesn't break
    #     _ = inputs.pop("labels", None)

    #     with torch.no_grad():
    #         outputs = model(
    #             pixel_values=inputs["pixel_values"],
    #             input_ids=inputs["input_ids"]
    #         )
    #     # Trainer expects a tuple: loss, logits, labels
    #     logits = outputs.logits_per_image  # or outputs depending on what you want
    #     return (None, logits, None)
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        _ = inputs.pop("labels", None)
        with torch.no_grad():
            outputs = model(pixel_values=inputs["pixel_values"], input_ids=inputs["input_ids"])
            logits_per_image = outputs.logits_per_image
            bsz = logits_per_image.size(0)
            target = torch.eye(bsz, device=logits_per_image.device)
            loss = self.sigmoid_loss_fct(logits_per_image, target)
            if self.locca_weight > 0.0:
                loss += self.locca_weight * self.compute_locca_loss(outputs)
        return (loss, logits_per_image, None)

# class SigLIPContrastiveTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
#         # remove labels for forward pass
#         _ = inputs.pop("labels", None)
#         pixel_values = inputs.pop("pixel_values")
#         input_ids = inputs.pop("input_ids")
#         #attention_mask = inputs.pop("attention_mask")

#         outputs = model(
#             pixel_values=pixel_values,
#             input_ids=input_ids,
#             #attention_mask=attention_mask,
#         )

#         logits_per_image = outputs.logits_per_image
#         logits_per_text = outputs.logits_per_text
#         bsz = logits_per_image.size(0)

#         target = torch.eye(bsz, device=logits_per_image.device)
#         loss_fct = nn.BCEWithLogitsLoss()
#         loss_i = loss_fct(logits_per_image, target)
#         loss_t = loss_fct(logits_per_text, target)
#         loss = (loss_i + loss_t) / 2
#         return (loss, outputs) if return_outputs else loss
    
#     def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
#         # Remove labels so the model forward doesn't break
#         _ = inputs.pop("labels", None)

#         with torch.no_grad():
#             outputs = model(
#                 pixel_values=inputs["pixel_values"],
#                 input_ids=inputs["input_ids"]
#             )
#         # Trainer expects a tuple: loss, logits, labels
#         logits = outputs.logits_per_image  # or outputs depending on what you want
#         return (None, logits, None)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def split_train_val(dataset, val_ratio=0.2, seed=42):
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    return torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))


def main(args):
    # processor + base model
    processor = AutoProcessor.from_pretrained(args.ckpt, do_normalize=False, do_rescale=False)
    base_model = AutoModel.from_pretrained(args.ckpt)
    #model = AutoModel.from_pretrained(args.ckpt)
    # freeze base params
    for p in base_model.parameters():
        p.requires_grad = False

    # LoRA config
    lora_config = LoraConfig(
        #task_type=TaskType.TASK_CUSTOM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_targets.split(","),
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    try:
        if args.vision_only:
            vision_model = get_peft_model(base_model.vision_model, lora_config)
            base_model.vision_model = vision_model #.base_model.model
            model = base_model
    except AttributeError:
        pass
    try:
        if args.language_only:
            text_model = get_peft_model(base_model.text_model, lora_config)
            base_model.text_model = text_model #.base_model.model
            model = base_model
    except AttributeError:
        pass

    try:
        print_trainable_parameters(model)
    except NameError:
        model = get_peft_model(base_model, lora_config)
        print_trainable_parameters(model)

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        FakeChannels()
    ])

    train_full_ds = ImageTextDataset(
        MBFRConfident(args.data_path, train=True), processor, args.train_captions, transform=val_transform, labels=True
    )

    train_ds, val_ds = split_train_val(train_full_ds, val_ratio=args.val_ratio)
    test_ds = ImageTextDataset(
        MBFRConfident(args.data_path, train=False), processor, args.test_captions, transform=val_transform, labels=True
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=args.patience,  # stop if no improvement for 5 evals
        early_stopping_threshold=0.0 # minimum improvement in monitored metric
    )

    # training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        logging_steps=10,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="wandb",
        metric_for_best_model="eval_loss",
        gradient_accumulation_steps=2,
        max_grad_norm=4
    )

    trainer = SigLIPTrainer(
        model=model,
        args=training_args,
        locca_weight = args.locca,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor,
        data_collator=lambda b: collate_batch(b, processor),
        #compute_metrics=compute_metrics,
        #locca_weight=0.1,
        callbacks=[ClassificationProbeCallback(val_ds, eval_interval=args.eval_interval, processor = processor, linear_lr = args.linear_lr, niter = args.niter), early_stopping]
    )

    if args.evaluate_default:
        imfeats, textfeats, acc, f1, loss, txt_acc, txt_f1, txt_clf_loss,cat_acc, cat_f1, cat_loss, avg_acc, avg_f1, avg_loss, mean_sim, std_sim, recall1, topk_recall,recall1_class_level, vkacc, vkf1, tkacc, tkf1, catkf1, avgkf1 = evaluate_classification_similarity(model, test_ds, device=model.device, processor=processor, save_features=True, niter=args.niter, linear_lr=args.linear_lr, train_dataset=train_full_ds)
        np.save(f"{os.path.join(args.output_dir, 'default_image_features.npy')}",imfeats.squeeze())
        np.save(f"{os.path.join(args.output_dir, 'default_text_features.npy')}",textfeats.squeeze())
        #np.save(f"{os.path.join(args.output_dir, 'labels.npy')}",labels.squeeze())
        print(f"[Test set evaluation] Image classification probe: acc={acc:.4f}, f1={f1:.4f}, loss={loss:.4f},\n Text classification probe: acc={txt_acc:.4f}, f1={txt_f1:.4f}, loss={txt_clf_loss:.4f} \n Concatenated Image-Text classification probe: acc={cat_acc:.4f}, f1={cat_f1:.4f}, loss={cat_loss:.4f} \n Averaged Image-Text classification probe: acc={avg_acc:.4f}, f1={avg_f1:.4f}, loss={avg_loss:.4f} \n mean image-text cosine similarity: {mean_sim:.4f}, recall@1: {recall1:.4f}, topK: {topk_recall:.4f} \n class-level recall!@1: {recall1_class_level:.4f}, vision knn-classifier accuracy: {vkacc:.4f}, vision knn-classifier F1: {vkf1:.4f}, text knn-classifier accuracy: {tkacc:.4f}, text knn-classifier F1: {tkf1:.4f}, cat image-text knn-classifier f1: {catkf1:.4f}, avg image-text knn-classifier F1: {avgkf1:.4f}")

    trainer.train()
    if args.vision_only:
        model.vision_model.save_pretrained(args.output_dir)
    elif args.language_only:
        model.text_model.save_pretrained(args.output_dir)
    else:
        model.save_pretrained(args.output_dir)
    imfeats, textfeats, acc, f1, loss, txt_acc, txt_f1, txt_clf_loss,cat_acc, cat_f1, cat_loss, avg_acc, avg_f1, avg_loss, mean_sim, std_sim, recall1, topk_recall,recall1_class_level, vkacc, vkf1, tkacc, tkf1, catkf1, avgkf1 = evaluate_classification_similarity(model, test_ds, device=model.device, processor=processor, save_features=True,niter=args.niter, linear_lr=args.linear_lr, train_dataset=train_full_ds)
    np.save(f"{os.path.join(args.output_dir, 'image_features.npy')}",imfeats.squeeze())
    np.save(f"{os.path.join(args.output_dir, 'text_features.npy')}",textfeats.squeeze())
    wandb.log({
            "test image classification_accuracy": acc,
            "test image classification_f1": f1,
            "test image classification_loss": loss,
            "test text classification_accuracy": txt_acc,
            "test text classification_f1": txt_f1,
            "test text classification_loss": txt_clf_loss,
            "test image-text cat classification_accuracy": cat_acc,
            "test image-text cat classification_f1": cat_f1,
            "test image-text cat classification_loss": cat_loss,
            "test image-text avg classification_accuracy": avg_acc,
            "test image-text avg classification_f1": avg_f1,
            "test image-text avg classification_loss": avg_loss,
            "test image_text_mean_similarity": mean_sim,
            "test image_text_std_similarity": std_sim,
            "test retrieval_recall@1": recall1,
            "test retrieval_top_k": topk_recall,
            "test recall@1 class level":recall1_class_level,
            "test vision knn-classifier accuracy": vkacc,
            "test vision knn-classifier f1":vkf1,
            "test text knn-classifier accuracy": tkacc,
            "test text knn-classifier f1":tkf1,
            "test image-text cat knn-classifier f1":catkf1,
            "test image-text avg knn-classifier f1":avgkf1,
        })
    print(f"[Test set evaluation] Image classification probe: acc={acc:.4f}, f1={f1:.4f}, loss={loss:.4f},\n Text classification probe: acc={txt_acc:.4f}, f1={txt_f1:.4f}, loss={txt_clf_loss:.4f} \n Concatenated Image-Text classification probe: acc={cat_acc:.4f}, f1={cat_f1:.4f}, loss={cat_loss:.4f} \n Averaged Image-Text classification probe: acc={avg_acc:.4f}, f1={avg_f1:.4f}, loss={avg_loss:.4f} \n mean image-text cosine similarity: {mean_sim:.4f}, recall@1: {recall1:.4f}, topK: {topk_recall:.4f} \n class-level recall!@1: {recall1_class_level:.4f}, vision knn-classifier accuracy: {vkacc:.4f}, vision knn-classifier F1: {vkf1:.4f}, text knn-classifier accuracy: {tkacc:.4f}, text knn-classifier F1: {tkf1:.4f}, cat image-text knn-classifier f1: {catkf1:.4f}, avg image-text knn-classifier F1: {avgkf1:.4f}")
    mkeys = ['test image classification_accuracy','test image classification_f1','test image classification_loss','test text classification_accuracy','test text classification_f1','test text classification_loss','test image-text cat classification_accuracy','test image-text cat classification_f1','test image-text cat classification_loss', 'test image-text avg classification_accuracy','test image-text avg classification_f1','test image-text avg classification_loss','test image_text_mean_similarity','test image_text_std_similarity','test retrieval_recall@1','test retrieval_top_k','test recall@1 class level', "test vision knn-classifier accuracy","test vision knn-classifier f1", "test text knn-classifier accuracy","test text knn-classifier f1", "test image-text cat knn-classifier f1","test image-text avg knn-classifier f1"]
    mdict = {k:v for k,v in zip(mkeys, [acc, f1, loss, txt_acc, txt_f1, txt_clf_loss, cat_acc, cat_f1, cat_loss, avg_acc, avg_f1, avg_loss, mean_sim, std_sim, recall1, topk_recall,recall1_class_level, vkacc, vkf1, tkacc, tkf1, catkf1, avgkf1])}
    df = pd.DataFrame(mdict, index=[0])
    df["niter"] = args.niter 
    df["linear lr"] = args.linear_lr
    df.to_csv(os.path.join(args.output_dir,"test_metrics.csv"))

def evaluate_model(args):
    processor = AutoProcessor.from_pretrained(args.ckpt, do_normalize=False, do_rescale=False)
    model = AutoModel.from_pretrained(args.ckpt)
    lora_config = LoraConfig(
        #task_type=TaskType.TASK_CUSTOM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_targets.split(","),
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    if not args.evaluate_default:
        if args.language_only:
            model.text_model = PeftModel.from_pretrained(model.text_model, args.output_dir)
        if args.vision_only:
            #vision_model = get_peft_model(base_model.vision_model, lora_config)
            #base_model.vision_model = vision_model #.base_model.model
            #model = base_model
            model.vision_model = PeftModel.from_pretrained(model.vision_model, args.output_dir)
    if torch.cuda.is_available():
        model.to("cuda") #.device = "cuda"
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        FakeChannels()
    ])

    train_full_ds = ImageTextDataset(
        MBFRConfident(args.data_path, train=True), processor, args.train_captions, transform=val_transform, labels=True
    )

    test_ds = ImageTextDataset(
        MBFRConfident(args.data_path, train=False), processor, args.test_captions, transform=val_transform, labels=True
    )

    imfeats, textfeats, acc, f1, loss, txt_acc, txt_f1, txt_clf_loss,cat_acc, cat_f1, cat_loss, avg_acc, avg_f1, avg_loss, mean_sim, std_sim, recall1, topk_recall,recall1_class_level, vkacc, vkf1, tkacc, tkf1, catkf1, avgkf1 = evaluate_classification_similarity(model, test_ds, device=model.device, processor=processor, niter=args.niter, linear_lr=args.linear_lr, train_dataset=train_full_ds, save_features=True)

    print(f"[Test set evaluation] Image classification probe: acc={acc:.4f}, f1={f1:.4f}, loss={loss:.4f},\n Text classification probe: acc={txt_acc:.4f}, f1={txt_f1:.4f}, loss={txt_clf_loss:.4f} \n Concatenated Image-Text classification probe: acc={cat_acc:.4f}, f1={cat_f1:.4f}, loss={cat_loss:.4f} \n Averaged Image-Text classification probe: acc={avg_acc:.4f}, f1={avg_f1:.4f}, loss={avg_loss:.4f} \n mean image-text cosine similarity: {mean_sim:.4f}, recall@1: {recall1:.4f}, topK: {topk_recall:.4f} \n class-level recall!@1: {recall1_class_level:.4f}, vision knn-classifier accuracy: {vkacc:.4f}, vision knn-classifier F1: {vkf1:.4f}, text knn-classifier accuracy: {tkacc:.4f}, text knn-classifier F1: {tkf1:.4f}, cat image-text knn-classifier f1: {catkf1:.4f}, avg image-text knn-classifier F1: {avgkf1:.4f}")

    mkeys = ['test image classification_accuracy','test image classification_f1','test image classification_loss','test text classification_accuracy','test text classification_f1','test text classification_loss','test image-text cat classification_accuracy','test image-text cat classification_f1','test image-text cat classification_loss', 'test image-text avg classification_accuracy','test image-text avg classification_f1','test image-text avg classification_loss','test image_text_mean_similarity','test image_text_std_similarity','test retrieval_recall@1','test retrieval_top_k','test recall@1 class level', "test vision knn-classifier accuracy","test vision knn-classifier f1", "test text knn-classifier accuracy","test text knn-classifier f1", "test image-text cat knn-classifier f1","test image-text avg knn-classifier f1"]
    mdict = {k:v for k,v in zip(mkeys, [acc, f1, loss, txt_acc, txt_f1, txt_clf_loss, cat_acc, cat_f1, cat_loss, avg_acc, avg_f1, avg_loss, mean_sim, std_sim, recall1, topk_recall,recall1_class_level, vkacc, vkf1, tkacc, tkf1, catkf1, avgkf1])}
    df = pd.DataFrame(mdict, index=[0])
    df["niter"] = args.niter 
    df["linear lr"] = args.linear_lr
    df.to_csv(os.path.join(args.output_dir,"test_metrics.csv"))

    np.save(f"{os.path.join(args.output_dir, 'image_features.npy')}",imfeats.squeeze())
    np.save(f"{os.path.join(args.output_dir, 'text_features.npy')}",textfeats.squeeze())

def get_args():
    parser = argparse.ArgumentParser(description="Fine-tune SigLIP2 with LoRA using Hugging Face Trainer")
    parser.add_argument("--data_path", type=str, default="/home/glados/unix-Documents/AstroSignals/llm_decoder/data/MiraBest", help="Path to training CSV file")
    #parser.add_argument("--valid_csv", type=str, required=True, help="Path to validation CSV file")
    parser.add_argument("--train_captions", type=str, default="/mnt/c/Users/elast/Documents/scratch/MiraBest/gemini_captions_train.csv", help="Column name for image paths")
    parser.add_argument("--test_captions", type=str, default="/mnt/c/Users/elast/Documents/scratch/MiraBest/gemini_captions_test.csv", help="Column name for text")
    parser.add_argument("--ckpt", type=str, default="google/siglip2-base-patch16-224", help="Base checkpoint")
    parser.add_argument("--output_dir", type=str, default="./siglip2-finetuned-lora", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument('--train', action = 'store_true',help='')
    parser.add_argument('--save_features', action = 'store_true',help='')
    parser.add_argument('--vision_only', action = 'store_true',help='')
    parser.add_argument('--language_only', action = 'store_true',help='')
    parser.add_argument('--evaluate_default', action = 'store_true',help='')
    parser.add_argument('--evaluate', action = 'store_true',help='')
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Learning rate")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--linear_lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--locca", type=float, default=0.0, help="locca rate")
    parser.add_argument("--patience", type=int, default=5, help="Learning rate")
    parser.add_argument("--niter", type=int, default=50, help="Number of epochs")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--eval_interval", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lora_r", type=int, default=4, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument(
        "--lora_targets",
        type=str,
        default="k_proj,v_proj,q_proj,fc1,fc2",
        help="Comma-separated list of target modules for LoRA",
    )
    return parser.parse_args()

def collate_text_batch(batch, processor):
    texts = [b["input_ids"] for b in batch]
    return processor(
        text=texts,
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="pt",
        #do_rescale=False,
        #do_normalize=False
    )

def collate_image_batch(batch, processor):
    images = [b["pixel_values"] for b in batch]
    return processor(
        images=images,
        return_tensors="pt",
        #do_rescale=False,
        #do_normalize=False
    )

def extract_text_features(args):
    processor = AutoProcessor.from_pretrained(args.ckpt)
    model = AutoModel.from_pretrained(args.ckpt)
    if args.language_only:
        model.text_model = PeftModel.from_pretrained(model.text_model, args.output_dir)
    model = model.text_model

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        FakeChannels()
    ])
    captions = args.train_captions if args.train else args.test_captions
    train_ds = ImageTextDataset(
        MBFRConfident(args.data_path, train=args.train), processor, captions, transform=val_transform
        )
    all_outputs = []
    for i,item in enumerate(train_ds):
        inputs = collate_text_batch([item], processor)
        output = model(**inputs).pooler_output.detach().numpy()
        all_outputs.append(output)
        #if i%300 == 0:
        #    np.save(f"siglip2_train_features_text_{i}.npy",np.array(all_outputs))
    return all_outputs

def extract_image_features(args):
    processor = AutoProcessor.from_pretrained(args.ckpt)
    model = AutoModel.from_pretrained(args.ckpt)
    if args.vision_only:
        model.vision_model = PeftModel.from_pretrained(model.vision_model, args.output_dir)
    model = model.vision_model

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        FakeChannels()
    ])
    captions = args.train_captions if args.train else args.test_captions
    train_ds = ImageTextDataset(
        MBFRConfident(args.data_path, train=args.train), processor, captions, transform=val_transform
    )
    #val_ds = ImageTextDataset(
    #    MBFRConfident(args.data_path, train=False), processor, args.test_captions, transform=val_transform
    #)
    all_outputs = []
    for i,item in enumerate(train_ds):
        inputs = collate_image_batch([item], processor)
        output = model(**inputs).pooler_output.detach().numpy()
        all_outputs.append(output)
        #if i%300 == 0:
        #    np.save(f"siglip2_train_features_image_{i}.npy",np.array(all_outputs))
    return all_outputs

if __name__ == "__main__":
    args = get_args()
    if args.evaluate:
        evaluate_model(args)
    else:
        main(args)
