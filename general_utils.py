import yaml
import os
from tqdm import tqdm
import torch
import numpy as np
import json
import uuid

from sklearn.metrics import f1_score


def read_yaml_file(file_path):
    with open(file_path, encoding="utf-8") as file_open:
        yaml_file = yaml.load(file_open, Loader=yaml.FullLoader)

    return yaml_file


def save_yaml_file(file_path, content):
    with open(file_path, encoding="utf-8", mode="w") as file_open:
        yaml_file = yaml.dump(content, file_open)


def save_model(model, tokenizer, path, used_config, step_no, current_dev_score=0):
    # final_path = os.path.join(path, f"checkpoint_w_dev_loss_{current_dev_score}_at_step_{step_no}_uuid_{uuid.uuid4().hex}")
    final_path = os.path.join(path, used_config["run_title"])
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    model.save_pretrained(final_path)
    tokenizer.save_vocabulary(final_path)
    save_yaml_file(os.path.join(final_path, "config_model.yaml"), used_config)
    return final_path


def update_dict_of_agg(agg_dict, new_dict, eval_on_train=True):
    if eval_on_train:
        agg_dict["TRAIN"]["Accuracy"].append(new_dict["TRAIN"]["Accuracy"])
        agg_dict["TRAIN"]["Loss"].append(new_dict["TRAIN"]["Loss"])

    agg_dict["DEV"]["Accuracy"].append(new_dict["DEV"]["Accuracy"])
    agg_dict["DEV"]["Loss"].append(new_dict["DEV"]["Loss"])

    agg_dict["TEST"]["Accuracy"].append(new_dict["TEST"]["Accuracy"])
    agg_dict["TEST"]["Loss"].append(new_dict["TEST"]["Loss"])
    return agg_dict


def save_json(path_to_file, content):
    with open(path_to_file, encoding="utf-8", mode="w") as ff:
        json.dump(content, ff)


def dump_predictions(sentence_index, soft_max_vals, predictions, labels, path_to_save_folder):
    with open(path_to_save_folder, encoding="utf-8", mode="w") as file_open:
        file_open.write("Sentence Index\tSoftMaxes\tPredictions\tLabels\n\n")
        for index in range(len(sentence_index)):
            file_open.write(str(sentence_index[index]) + "\t" + "["+str(" ".join(map(str,soft_max_vals[index])))+"]" + "\t" + str(predictions[index]) + "\t" + str(labels[index]) + "\n")


def random_mask_tokens(input_ids, atttention_mask, masking_percentage, mask_id, device):
    index_bool = torch.ones(input_ids.shape[0],input_ids.shape[1])*masking_percentage
    index_bool = torch.where((input_ids != mask_id) * (input_ids != 1) * (input_ids != -100) * (input_ids != 2) * (input_ids != 3) * (input_ids != 0), index_bool.float().to(device), torch.zeros(input_ids.shape[0],input_ids.shape[1]).to(device))
    index_bool = index_bool.to(device) 
    index_bool = torch.mul(index_bool, atttention_mask) # To remove the ones that are masked from consideration
    index_bool = torch.bernoulli(index_bool).bool()
    input_ids[index_bool] = mask_id
    return input_ids


def evaluate_predictions(model, evaluation_loader, model_class_name, device="cpu", return_pred_lists=False, isTest=False, split=""):
    model.eval()
    no_batches = tqdm(evaluation_loader, desc="Batch Evaluation Loop")
    final_eval_loss, correct = 0, 0
    total_no_steps, num_samples = 0, 0
    preds, g_truths, list_of_sentence_ids = [], [], []
    logits_list = []
    y_true = []
    y_pred = []
    confusion_matrix = np.zeros((21, 21))
    for batch in no_batches:
        batch = [x.to(device) for x in batch]
        label_ids_in = batch[3] if not isTest else None
        outputs = model(input_ids=batch[0], attention_mask=batch[1], token_type_ids=batch[2], class_label_ids=label_ids_in, input_ids_masked=batch[4])
        eval_loss, (logits,) = outputs[:2]
        final_eval_loss += eval_loss.mean().item() if not isTest else 0
        total_no_steps += 1

        if model_class_name == "ArabicDialectBERT":
            logits_list.extend(torch.nn.functional.softmax(logits, dim=-1).detach().cpu().numpy())
            label_ids = logits.argmax(axis=1)
            g_truths.extend(batch[3].detach().cpu().numpy())
            preds.extend(label_ids.detach().cpu().numpy())
            confusion_matrix[batch[3].item(), label_ids.item()] += 1
            list_of_sentence_ids.extend(batch[5].detach().cpu().numpy())
            correct += (label_ids == batch[3]).sum()
            num_samples += label_ids.size(0)
    
    if model_class_name == "ArabicDialectBERT":
        accuracy = correct / float(num_samples)
        accuracy = accuracy.item()
        y_true = np.array(g_truths)
        y_pred = np.array(preds)
        f1 = f1_score(y_true, y_pred, average="macro")
    else:
        f1 = 0 
        accuracy = 0
    
    print(f"Classes Confusion Matrix [{split}]=>")
    print(confusion_matrix)

    eval_loss = final_eval_loss / total_no_steps

    if return_pred_lists:
        return f1, accuracy, eval_loss, y_true, y_pred, list_of_sentence_ids, logits_list

    return f1, accuracy, eval_loss
