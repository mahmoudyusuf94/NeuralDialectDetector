import torch
import logging
import os
from torch.utils.data import TensorDataset, RandomSampler, WeightedRandomSampler
import numpy as np

logger = logging.getLogger(__name__)

class NADIDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, sentences, labels):
        'Initialization'
        self.sentences = sentences
        self.labels = labels

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'

        X = self.sentences[index]
        y = self.labels[index]

        return X, y


def parse_classes_list(path_to_folder, is_province=False):
    filename = "classes_22.txt" if is_province else "classes_12.txt"
    classes_path = os.path.join(path_to_folder, filename)
    with open(classes_path, encoding="utf-8") as file_open:
        lines = file_open.readlines()
    return [line.strip("\n") for line in lines]

def parse_classes_w_weights_list(path_to_folder):
    classes_path = os.path.join(path_to_folder, "classes_w_weights.txt")
    labels, weights = [], []
    with open(classes_path, encoding="utf-8") as file_open:
        lines = file_open.readlines()
    for line in lines:
        c, w = line.strip("\n").split("\t")
        labels += [c]
        weights += [float(w)]
    return labels, weights


def parse_mapping_list(path_to_folder):
    mapping_path = os.path.join(path_to_folder, "regional_mapping.txt")
    if not os.path.exists(mapping_path):
        return None
    with open(mapping_path, encoding="utf-8") as file_opened:
        lines = file_opened.readlines()
    lines = [line.strip("\n").split(",") for line in lines]
    return {key: value for key, value in lines}


def read_indexes_file(path_to_file, prediction_class=-1):
    with open(path_to_file, encoding="utf-8") as ff:
        lines_read = ff.readlines()

    if prediction_class==-1:
        raise Exception("Please Choose Prediction Class")

    lines_read = [int(x.split("\t")[0]) for x in lines_read[2:] if int(x.split("\t")[1]) == prediction_class]

    return lines_read


def parse_data(path_to_file, separator="\t", regional_mapping_content=None, class_to_filter=None, filter_w_indexes=None, pred_class=-1):
    group = ["Morocco", "Tunisia", "Algeria", "Libya"]
    with open(path_to_file, encoding="utf-8") as file_open:
        lines = file_open.readlines()
  
    lines_split = [line.strip().split("\t")[0:4] for line in lines[1:]]    

    if regional_mapping_content is not None:
        lines_split = [(x[0], x[1], regional_mapping_content[x[2]]) for x in lines_split]

    if filter_w_indexes is not None:
        indexes_list = read_indexes_file(filter_w_indexes, prediction_class=pred_class)

        lines_split = [x for index, x in enumerate(lines_split) if index in indexes_list]
        
    elif group is not None:
        lines_split = [x for x in lines_split if x[2] in group]

    print("Filtered lines for the following group")
    print(group)
    print("total examples for this group: ")
    print(len(lines_split))
    return lines_split


def balance_data(data_examples, max_examples):
    new_data_list = []
    classes_set = set([x[2] for x in data_examples])
    count_dict = {key: 0 for key in classes_set}
    for example in data_examples:
        if count_dict[example[2]] < max_examples:
            new_data_list.append(example)
            count_dict[example[2]] += 1

    return new_data_list


def prepare_random_sampler(classes_list):
    classes_list += [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    class_sample_count = np.array([len(np.where(classes_list == t)[0]) for t in np.unique(classes_list)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in classes_list])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler

def parse_and_generate_loader(path_to_data_folder, tokenizer, params, classes_list, split_set="train", locale="ar", random_sampler=True, masking_percentage=0.2, class_to_filter=None, regional_mapping=None, filter_w_indexes=None, pred_class=-1, max_seq_len=128, balance_data_max_examples=None, is_province=False, is_MSA=False, handle_imbalance_sampler=False):
    index_path = os.path.join(filter_w_indexes, f"predictions_{split_set}.tsv") if filter_w_indexes is not None else None
    
    arabic_type = "MSA" if is_MSA else "DA"
    data_examples = parse_data(os.path.join(path_to_data_folder, f"{arabic_type}_{split_set}_labeled.tsv"), regional_mapping_content=regional_mapping, class_to_filter=class_to_filter, filter_w_indexes=index_path, pred_class=pred_class)
    
    if balance_data_max_examples is not None:
        data_examples = balance_data(data_examples, balance_data_max_examples)

    dataset, imbalance_handling_sampler = load_and_cache_examples(data_examples, tokenizer, classes_list, is_province=is_province, max_seq_len=max_seq_len, masking_percentage=masking_percentage)
    if handle_imbalance_sampler:
        data_sampler = imbalance_handling_sampler
    else:
        data_sampler = RandomSampler(dataset) if random_sampler else None
    generator = torch.utils.data.DataLoader(dataset, sampler=data_sampler, **params)  # shuffle=not random_sampler,
    return generator


def parse_and_generate_loaders(path_to_data_folder, tokenizer, batch_size=2, masking_percentage=0.2, class_to_filter=None, regional_mapping=None, filter_w_indexes=None, pred_class=-1, use_regional_mapping=False, max_seq_len=128, balance_data_max_examples=None, is_province=False, is_MSA=False, sampler_imbalance=False):
    params = {'batch_size': batch_size}
    params_dev = {'batch_size': batch_size // 2}
    regional_mapping_content = parse_mapping_list(path_to_data_folder) if use_regional_mapping else None
    if regional_mapping_content is not None:
        classes_list = list(set(regional_mapping_content.values()))
        classes_list.sort()
    else:
        classes_list = parse_classes_list(path_to_data_folder, is_province) if class_to_filter is None else class_to_filter
        classes_list.sort()
        print("Classes that will run: ")
        print(classes_list)
    training_generator = parse_and_generate_loader(path_to_data_folder, tokenizer, params, classes_list, split_set="train", locale="ar", masking_percentage=masking_percentage, class_to_filter=class_to_filter, regional_mapping=regional_mapping_content, max_seq_len=max_seq_len, balance_data_max_examples=balance_data_max_examples, is_province=is_province, is_MSA=is_MSA, handle_imbalance_sampler=sampler_imbalance)
    dev_generator = parse_and_generate_loader(path_to_data_folder, tokenizer, params_dev, classes_list, split_set="dev", locale="ar", masking_percentage=masking_percentage, class_to_filter=class_to_filter, regional_mapping=regional_mapping_content, max_seq_len=max_seq_len, is_province=is_province, is_MSA=is_MSA, handle_imbalance_sampler=sampler_imbalance)
    test_generator = parse_and_generate_loader(path_to_data_folder, tokenizer, params_dev, classes_list, split_set="dev", locale="ar", masking_percentage=masking_percentage, class_to_filter=class_to_filter, regional_mapping=regional_mapping_content, filter_w_indexes=filter_w_indexes, pred_class=pred_class, max_seq_len=max_seq_len, is_province=is_province, is_MSA=is_MSA, handle_imbalance_sampler=sampler_imbalance)

    return training_generator, dev_generator, test_generator, len(classes_list), None

# From : https://github.com/monologg/JointBERT/blob/master/predict.py
def load_and_cache_examples(examples, tokenizer, classes_list, is_province, pad_token_ignore_index=0, max_seq_len=128, masking_percentage=0.2):

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = pad_token_ignore_index
    features = convert_examples_to_features(examples, classes_list, max_seq_len, tokenizer,
                                            is_province=is_province,pad_token_label_id=pad_token_label_id, masking_percentage=masking_percentage)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f[1] for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f[2] for f in features], dtype=torch.long)
    all_class_label_ids = torch.tensor([f[3] for f in features], dtype=torch.long)
    all_input_ids_w_masking = torch.tensor([f[4] for f in features], dtype=torch.long)
    all_sentence_indices = torch.tensor([f[5] for f in features], dtype=torch.long)
    
    imbalance_handling_sampler = prepare_random_sampler([f[3] for f in features])
    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_class_label_ids, all_input_ids_w_masking, all_sentence_indices)
    return dataset, imbalance_handling_sampler


# From : https://github.com/monologg/JointBERT/blob/master/predict.py
def convert_examples_to_features(examples, classes_list, max_seq_len, 
                                 tokenizer,
                                 is_province=False,
                                 pad_token_label_id=-100,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True,
                                 masking_percentage=0.2):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id
    mask_token = tokenizer.mask_token
    features = []
    tokens_len = []
    for (index, example) in enumerate(examples):
        ex_index = int(example[0].split("_")[-1])
        if index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # Tokenize word by word (for NER)
        tokens = []
        tokens_with_masking = []
        sentence_whitespace = example[1].split(' ')
        for word in sentence_whitespace:
            to_mask = bool(np.random.binomial(1, masking_percentage))                
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            if to_mask:
                word_tokens_masking = [mask_token]
            else:
                word_tokens_masking = word_tokens
            tokens.extend(word_tokens)
            tokens_with_masking.extend(word_tokens_masking)

        tokens_len.append(len(tokens))
        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]
        if len(tokens_with_masking) > max_seq_len - special_tokens_count:
            tokens_with_masking = tokens_with_masking[:(max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        tokens_with_masking += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        tokens_with_masking = [cls_token] + tokens_with_masking
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids_w_masking = tokenizer.convert_tokens_to_ids(tokens_with_masking)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        padding_length_masking = max_seq_len - len(input_ids_w_masking)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        input_ids_w_masking = input_ids_w_masking + ([pad_token_id] * padding_length_masking)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)
        assert len(input_ids_w_masking) == max_seq_len, "Error with input with masking length {} vs {}".format(len(input_ids_w_masking), max_seq_len)

        example_idx = 3 if is_province else 2
        if example[example_idx].strip("\n") in classes_list:
            class_label_id = classes_list.index(example[example_idx].strip("\n"))
        else:
            class_label_id = -1

        if index < 5:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_ids_w_masking: %s" % " ".join([str(x) for x in input_ids_w_masking]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("class_label: %s (id = %d)" % (example[2].strip("\n"), class_label_id))

        features.append((input_ids,
                          attention_mask,
                          token_type_ids,
                          class_label_id,
                          input_ids_w_masking,
                          ex_index
                          ))
    
    print(f"Mean: {np.mean(tokens_len)}")
    print(f"STD: {np.std(tokens_len)}")
    print(f"Median: {np.median(tokens_len)}")
    print(f"min: {np.min(tokens_len)}")
    print(f"max: {np.max(tokens_len)}")
    return features