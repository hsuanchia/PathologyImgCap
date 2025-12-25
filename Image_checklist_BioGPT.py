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

BATCH_SIZE = 100
EPOCHS = 1000
LEARNING_RATE = 1e-4
LOSS = torch.nn.BCEWithLogitsLoss()
MODEL_PATH = './models/'
MODEL_NAME = 'Image_checklist_BioGPT_CLIP.pth'
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

class Image_checklist_selected_Dataset(Dataset):
    def __init__(self, datalist, data_dict, BioBERT_tokenizer, BioGPT_tokenizer) -> None:
        self.datalist = datalist
        self.data_dict = data_dict
        self.BioGPT_tokenizer = BioGPT_tokenizer
        self.BioBERT_tokenizer = BioBERT_tokenizer

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        cur_cases = self.datalist[index]
        # Image processing
        cur_data_path = os.path.join(normalized_path, self.datalist[index])
        img_file_name = [x for x in os.listdir(cur_data_path) if '.tif' in x]
        all_img_file_ind = [(int(x.split('_')[0]), int(x.split('_')[1].replace('.tif', ''))) for x in img_file_name]
        
        pos_data_files, neg_data_files = [], []
        E1_result = open(os.path.join(cur_data_path,'patch_pred.txt'), 'r')
        for l in E1_result.readlines():
            name = l.strip().split(' : ')[0]
            pred = l.strip().split(' : ')[1]
            x, y = name.split('_')[0], name.split('_')[1]
            file_name = name + '.tif'
            if file_name not in img_file_name:
                continue
            else:
                if int(pred) == 8 or int(pred) == 3: # TUM
                    pos_data_files.append(file_name)
                else:
                    neg_data_files.append(file_name)
        selected_img_file_ind = [(int(x.split('_')[0]), int(x.split('_')[1].replace('.tif', ''))) for x in pos_data_files]
        random.shuffle(selected_img_file_ind)
        used_file = []
        patch_batch = []
        print("Build graph batch")
        for ind in tqdm(selected_img_file_ind):
            selected_patches = self.flood_fill(all_img_file_ind, used_file, ind)
            if len(selected_patches) < 1:
                continue
            used_file.extend(selected_patches)
            patch_batch.append(selected_patches)

        print("Build edge index")
        batch_edge_index = []
        bag_patch = []
        for bag in tqdm(range(len(patch_batch))):
            # Get edge index
            tmp_patch_bag = []
            edge_index = []
            for i in range(len(patch_batch[bag])):
                neighbor = [(patch_batch[bag][i][0] + 224, patch_batch[bag][i][1]),
                            (patch_batch[bag][i][0], patch_batch[bag][i][1] + 224),
                            (patch_batch[bag][i][0] - 224, patch_batch[bag][i][1]),
                            (patch_batch[bag][i][0], patch_batch[bag][i][1] - 224),
                            (patch_batch[bag][i][0] + 224, patch_batch[bag][i][1] + 224),
                            (patch_batch[bag][i][0] + 224, patch_batch[bag][i][1] - 224),
                            (patch_batch[bag][i][0] - 224, patch_batch[bag][i][1] + 224),
                            (patch_batch[bag][i][0] - 224, patch_batch[bag][i][1] - 224)]
                for j in range(len(patch_batch[bag])):
                    if patch_batch[bag][j] in neighbor:
                        edge_index.append([i, j])
                patch_file_name = f"{patch_batch[bag][i][0]}_{patch_batch[bag][i][1]}.tif"
                tmp_patch_bag.append(os.path.join(cur_data_path, patch_file_name))
            bag_patch.append(tmp_patch_bag)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            batch_edge_index.append(edge_index)

        tmp_checklist = f" {self.BioBERT_tokenizer.sep_token} ".join(self.data_dict[cur_cases]['Checklist'])
        tmp_finding = f" {self.BioGPT_tokenizer.sep_token} ".join(self.data_dict[cur_cases]['Findings'])
        checklist_seq = tmp_checklist.split(" ")
        checklist_seq = [word for word in checklist_seq if word != '']
        finding_seq = tmp_finding.split(" ")
        finding_seq = [word for word in finding_seq if word != '']
        checklist_seq = " ".join(checklist_seq)
        finding_seq = " ".join(finding_seq)
        input_text = self.BioBERT_tokenizer.bos_token + " " + checklist_seq + " " + self.BioBERT_tokenizer.eos_token
        output_text = self.BioGPT_tokenizer.bos_token + " " + finding_seq + " " + self.BioGPT_tokenizer.eos_token
        print("Input text: ", input_text)
        print("Output text: ", output_text)
        in_seq = self.BioBERT_tokenizer(input_text, return_tensors="pt", truncation=True, padding='max_length', max_length=256)
        out_seq = self.BioGPT_tokenizer(output_text, return_tensors="pt", truncation=True, padding='max_length', max_length=256)

        return bag_patch, in_seq, out_seq
    
    def flood_fill(self, img_files, used_file, start_ind, batch_size=100):
        img_files = set(map(tuple, img_files))  
        used_file = set(map(tuple, used_file))  

        selected_patches = []
        selected_queue = deque([tuple(start_ind)]) 
        
        visited = set()

        while len(selected_patches) < batch_size:
            if len(selected_queue) < 1:
                return []
            cur_patch = selected_queue.popleft()  # O(1) pop 操作
            if cur_patch in img_files and cur_patch not in used_file and cur_patch not in visited:
                selected_patches.append(cur_patch)
                visited.add(cur_patch)
                
                check_direction = [(224, 0), (0, 224), (-224, 0), (0, -224)]
                for dx, dy in check_direction:
                    tmp_ind = (cur_patch[0] + dx, cur_patch[1] + dy)  
                    if tmp_ind in img_files and tmp_ind not in used_file and tmp_ind not in visited:
                        selected_queue.append(tmp_ind)

        return selected_patches

def build_checklist(data_list, data):
    data_dict = {}
    for ind in data_list:
        tmp_checklist = []
        # Get findings description
        data_dict[ind] = {'Findings' : [], 'Checklist' : []}
        data_dict[ind]['Findings'] = data[ind]['Findings']
        # Get Clinical data
        tmp_checklist.append(data[ind]['Checklist'][6]) # Differentiated
        tmp_checklist.append(data[ind]['Checklist'][8]) # Involved lymph node & TNM stage
        tmp_checklist.append(data[ind]['Checklist'][9]) # Margin section
        # Get Classification result
        metastasis, infiltrate, necrosis = False, False, False
        for s in data[ind]['Findings']:
            if 'metastasis' in s and 'no tumor' not in s:
                metastasis = True
            if 'infiltra' in s and 'no tumor' not in s:
                infiltrate = True
            if 'necrosis' in s and 'no tumor' not in s:
                necrosis = True
        if metastasis:
            tmp_checklist.append('Metastasis : Identified')
        else:
            tmp_checklist.append('Metastasis : Not identified')
        
        if infiltrate:
            tmp_checklist.append('Infiltrate : Identified')
        else:
            tmp_checklist.append('Infiltrate : Not identified')

        if necrosis:
            tmp_checklist.append('Necrosis : Identified')
        else:
            tmp_checklist.append('Necrosis : Not identified')
        
        data_dict[ind]['Checklist'] = tmp_checklist

    return data_dict

def train(train_DataLoader, val_DataLoader, model, opt, BioGPT_tokenizer):
    best_loss = 888888
    earlystop = 0
    smoothie = SmoothingFunction().method1
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    with Progress(TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TimeElapsedColumn()) as progress:
        epoch_tqdm = progress.add_task(description="Epoch progress", total=EPOCHS)
        train_batch_tqdm = progress.add_task(description="Train progress", total=len(train_DataLoader))
        val_batch_tqdm = progress.add_task(description="Val progress", total=len(val_DataLoader))
        for num_epochs in range(EPOCHS):
            train_avg_loss, val_avg_loss = 0, 0
            train_num, val_num = 0, 0
            train_bleu, val_bleu = 0, 0
            train_rouge_1_p, val_rouge_1_p = 0, 0
            train_rouge_1_r, val_rouge_1_r = 0, 0
            model.train()
            for batch_img_path, in_seq, out_seq in train_DataLoader:
                in_seq = in_seq.to(device)
                out_seq = out_seq.to(device)
                ans_text = BioGPT_tokenizer.decode(out_seq['input_ids'][0][0], skip_special_tokens=True)
                
                outputs, att_score = model(batch_img_path, in_seq, out_seq)
                # Compute loss
                opt.zero_grad()
                outputs.loss.backward()
                opt.step()
                train_avg_loss += outputs.loss.item()
                # Convert prediction sentence
                pred = torch.softmax(outputs.logits, dim=-1)
                pred = torch.argmax(pred, dim=-1)
                pred = pred.cpu().detach().numpy()

                pred_text = BioGPT_tokenizer.decode(pred[0], skip_special_tokens=True)
                print("Pred: ", pred_text)
                print("Ans: ", ans_text)
                # Comput Rouge score
                train_rouge_score = scorer.score(ans_text, pred_text)
                print("Rouge-1: ", train_rouge_score)
                train_rouge_1_r += train_rouge_score['rouge1'].recall
                train_rouge_1_p += train_rouge_score['rouge1'].precision
                # Compute BLEU score
                bleu_ans_text = ans_text.split(" ")
                bleu_pred_text = pred_text.split(" ")
                bleu_score = sentence_bleu([bleu_ans_text], bleu_pred_text, smoothing_function=smoothie)
                print("BLEU scores: ", bleu_score)
                train_bleu += bleu_score
                train_num += 1
                progress.advance(train_batch_tqdm, advance=1)
            print("Start validation...")
            model.eval()
            with torch.no_grad():
                for val_batch_img_path, val_in_seq, val_out_seq in val_DataLoader:
                    val_in_seq = val_in_seq.to(device)
                    val_out_seq = val_out_seq.to(device)
                    val_ans_text = BioGPT_tokenizer.decode(val_out_seq['input_ids'][0][0], skip_special_tokens=True)
                    val_outputs, val_att_scores = model(val_batch_img_path, val_in_seq, val_out_seq)
                    val_avg_loss += val_outputs.loss.item()
                    val_pred = torch.softmax(val_outputs.logits, dim=-1)
                    val_pred = torch.argmax(val_pred, dim=-1)
                    val_pred = val_pred.cpu().detach().numpy()
                    val_pred_text = BioGPT_tokenizer.decode(val_pred[0], skip_special_tokens=True)
                    
                    print("Pred: ", val_pred_text)
                    print("Ans: ", val_ans_text)
                    # Compute rouge score
                    val_rouge_score = scorer.score(val_ans_text, val_pred_text)
                    print("Rouge-1: ", val_rouge_score)
                    val_rouge_1_r += val_rouge_score['rouge1'].recall
                    val_rouge_1_p += val_rouge_score['rouge1'].precision
                    # Compute bleu score
                    val_bleu_pred_text = val_pred_text.split(" ")
                    val_bleu_ans_text = val_ans_text.split(" ")
                    val_bleu_score = sentence_bleu([val_bleu_ans_text], val_bleu_pred_text, smoothing_function=smoothie)
                    print("BLEU scores: ", val_bleu_score)
                    val_bleu += val_bleu_score
                    val_num += 1
                    progress.advance(val_batch_tqdm, advance=1)

            train_avg_loss /= train_num
            val_avg_loss /= val_num
            train_bleu /= train_num
            val_bleu /= val_num
            train_rouge_1_p /= train_num
            train_rouge_1_r /= train_num
            val_rouge_1_p /= val_num
            val_rouge_1_r /= val_num

            LOGGER.info('Epoch {}: train loss: {} | train bleu: {} | train rouge precision: {} | train rouge recall: {}'.format(num_epochs, train_avg_loss, train_bleu, train_rouge_1_p, train_rouge_1_r)) #紀錄訓練資訊
            LOGGER.info('Epoch {}: val loss: {} | val bleu: {} | val rouge precision: {} | val rouge recall: {}'.format(num_epochs, val_avg_loss, val_bleu, val_rouge_1_p, val_rouge_1_r)) #紀錄訓練資訊
            print('Epoch {}: train loss: {} | train bleu: {} | train rouge precision: {} | train rouge recall: {}'.format(num_epochs, train_avg_loss, train_bleu, train_rouge_1_p, train_rouge_1_r))
            print('Epoch {}: val loss: {} | val bleu: {} | val rouge precision: {} | val rouge recall: {}'.format(num_epochs, val_avg_loss, val_bleu, val_rouge_1_p, val_rouge_1_r))
            SCHEDULER.step(val_avg_loss)
            print("Model saving")
            path = MODEL_PATH+MODEL_NAME
            torch.save(model.state_dict(), path)
            if best_loss > val_avg_loss:
                print("Best model saving...")
                best_loss = val_avg_loss
                path = MODEL_PATH + 'Best_' + MODEL_NAME
                torch.save(model.state_dict(), path)
                earlystop = 0
            else:
                earlystop += 1
                print(f"EarlyStop times: {earlystop}")
                if earlystop >= 5:
                    print("Earlystop triggered!")
                    break
            progress.reset(train_batch_tqdm)
            progress.reset(val_batch_tqdm)
            progress.advance(epoch_tqdm, advance=1)

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

    train_data_dict = build_checklist(train_cases, data)
    val_data_dict = build_checklist(val_cases, data)

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
    
    train_dataset = Image_checklist_selected_Dataset(train_cases, train_data_dict, BioBERT_tokenizer, BioGPT_tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1, pin_memory=True)
    val_dataset = Image_checklist_selected_Dataset(val_cases, val_data_dict, BioBERT_tokenizer, BioGPT_tokenizer)
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=1, pin_memory=True)

    model = Image_checklist_BioGPT().to(device)
        
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    SCHEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=2, factor=0.5, verbose=True)
    train(train_dataloader, val_dataloader, model, opt, BioGPT_tokenizer)