import os, torch, random, logging, time, math
import numpy as np
import torch.nn.functional as F
from collections import deque
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, track
from model import Image_checklist_BioGPT
from transformers import AutoTokenizer, AutoModel, BioGptTokenizer, BioGptForCausalLM
from report_parser import parse_report
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from tqdm import tqdm
from Image_checklist_BioGPT import Image_checklist_selected_Dataset, build_checklist

BATCH_SIZE = 100
EPOCHS = 1000
LEARNING_RATE = 1e-4
LOSS = torch.nn.BCEWithLogitsLoss()
MODEL_PATH = './models/'
MODEL_NAME = 'Image_checklist_BioGPT_fix_text.pth'
# 配置 Logger
logging.basicConfig(level=logging.INFO)  # 配置info
LOGGER = logging.getLogger(f'{MODEL_NAME}')  #建立一個叫做(f'{MODEL_NAME}')的記錄器
LOG_FILE = f'{MODEL_PATH}/{MODEL_NAME}.log' #記錄檔案名稱
file_handler = logging.FileHandler(LOG_FILE)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
# FileHandler 添加到 Logger
LOGGER.addHandler(file_handler)
normalized_path = "F:/pathology_patches_normalized"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU state:',device)

def inference(test_DataLoader, model, BioBERT_tokenizer, BioGPT_tokenizer):
    test_rouge_1_p, test_rouge_1_r = 0, 0
    test_bleu = 0
    smoothie = SmoothingFunction().method1
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    model.eval()
    with torch.no_grad():
        for test_batch_img_path, test_in_seq, test_out_seq in track(test_DataLoader):
            test_in_seq = test_in_seq.to(device)
            test_out_seq = test_out_seq.to(device)
            test_ans_text = BioGPT_tokenizer.decode(test_out_seq['input_ids'][0][0], skip_special_tokens=True)
            test_text_seq = BioBERT_tokenizer("[BOS]", return_tensors="pt", truncation=True, padding='max_length', max_length=256)
            test_text_seq = test_text_seq.to(device)
            test_outputs, test_att_scores = model(test_batch_img_path, test_in_seq, test_text_seq, inference=True)
            test_pred = torch.softmax(test_outputs.logits, dim=-1)
            test_pred = torch.argmax(test_pred, dim=-1)
            test_pred = test_pred.cpu().detach().numpy()
            test_pred_text = BioGPT_tokenizer.decode(test_pred[0], skip_special_tokens=True)
            
            print("Pred: ", test_pred_text)
            print("Ans: ", test_ans_text)
            # Compute rouge score
            test_rouge_score = scorer.score(test_ans_text, test_pred_text)
            print("Rouge-1: ", test_rouge_score)
            test_rouge_1_r += test_rouge_score['rouge1'].recall
            test_rouge_1_p += test_rouge_score['rouge1'].precision
            # Compute bleu score
            test_bleu_pred_text = test_pred_text.split(" ")
            test_bleu_ans_text = test_ans_text.split(" ")
            test_bleu_score = sentence_bleu([test_bleu_ans_text], test_bleu_pred_text, smoothing_function=smoothie)
            print("BLEU scores: ", test_bleu_score)
            test_bleu += test_bleu_score
            time.sleep(3)

    test_bleu /= len(test_DataLoader)
    test_rouge_1_p /= len(test_DataLoader)
    test_rouge_1_r /= len(test_DataLoader)

    print(f"BLEU: {test_bleu} | Rouge Precision: {test_rouge_1_p} | Rouge Recall: {test_rouge_1_r}")



if __name__ == '__main__':
    data = parse_report()
    train_cases, train_text = [], [] # {0 : 30, 1 : 205, 2: 15}
    val_cases, val_text = [], [] # {0 : 10, 1 : 31, 2 : 9}
    test_cases, test_text = [], [] # {0 : 30, 1 : 55, 2 : 15}
    train_count = {0 : 0, 1 : 0, 2 : 0} # 0 : well, 1 : moderately, 2 : poorly
    val_count = {0 : 0, 1 : 0, 2 : 0}
    test_count = {0 : 0, 1 : 0, 2 : 0}

    ### Symptoms in checklist
    well_cases, poorly_cases, moderately_cases = [], [], []
    for k, v in data.items():
        for ind, sen in v['Checklist'].items():
            tmp_sen = sen.lower()
            if 'well' in tmp_sen and 'differentiated' in tmp_sen:
                well_cases.append(k)
            elif 'poor' in tmp_sen and 'differentiated' in tmp_sen:
                poorly_cases.append(k)
            elif 'moderately' in tmp_sen and 'differentiated' in tmp_sen:
                moderately_cases.append(k)
    
    for ind in range(len(well_cases)):
        if ind < 30:
            train_cases.append(well_cases[ind])
            train_text.append(data[well_cases[ind]]['Findings'])
            train_count[0] += 1
        elif ind >= 30 and ind < 40:
            val_cases.append(well_cases[ind])
            val_text.append(data[well_cases[ind]]['Findings'])
            val_count[0] += 1
        else:
            test_cases.append(well_cases[ind])
            test_text.append(data[well_cases[ind]]['Findings'])
            test_count[0] += 1

    for ind in range(len(poorly_cases)):
        if ind < 15:
            train_cases.append(poorly_cases[ind])
            train_text.append(data[poorly_cases[ind]]['Findings'])
            train_count[2] += 1
        elif ind >= 15 and ind < 24:
            val_cases.append(poorly_cases[ind])
            val_text.append(data[poorly_cases[ind]]['Findings'])
            val_count[2] += 1
        else:
            test_cases.append(poorly_cases[ind])
            test_text.append(data[poorly_cases[ind]]['Findings'])
            test_count[2] += 1

    for ind in range(len(moderately_cases)):
        if ind < 205:
            train_cases.append(moderately_cases[ind])
            train_text.append(data[moderately_cases[ind]]['Findings'])
            train_count[1] += 1
        elif ind >= 205 and ind < 236:
            val_cases.append(moderately_cases[ind])
            val_text.append(data[moderately_cases[ind]]['Findings'])
            val_count[1] += 1
        elif ind >= 236 and ind < 291:
            test_cases.append(moderately_cases[ind])
            test_text.append(data[moderately_cases[ind]]['Findings'])
            test_count[1] += 1

    print("Train data distribution: ", train_count)
    print("Val data distribution: ", val_count)
    print("Test data distribution: ", test_count)

    test_data_dict = build_checklist(test_cases, data)

    BioBERT_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    BioGPT_tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    BioGPT_tokenizer.bos_token = '[BOS]'
    BioGPT_tokenizer.eos_token = '[EOS]'
    BioGPT_tokenizer.pad_token = BioGPT_tokenizer.eos_token
    BioGPT_tokenizer.sep_token = '[SEP]'

    BioBERT_tokenizer.bos_token = '[BOS]'
    BioBERT_tokenizer.eos_token = '[EOS]'
    BioBERT_tokenizer.pad_token = BioBERT_tokenizer.eos_token
    BioBERT_tokenizer.sep_token = '[SEP]'
    
    test_dataset = Image_checklist_selected_Dataset(test_cases, test_data_dict, BioBERT_tokenizer, BioGPT_tokenizer)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=1, pin_memory=True)

    model = Image_checklist_BioGPT().to(device)
    model.load_state_dict(torch.load(f'./models/Best_Image_checklist_BioGPT_merge_transformer_encode.pth'))

    inference(test_dataloader, model, BioBERT_tokenizer, BioGPT_tokenizer)