import torch, os, time, random, logging
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, track
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BioGptTokenizer, BioGptForCausalLM
from report_parser import extract_pathology_sections

EPOCHS = 1000
LEARNING_RATE = 1e-4
LOSS = torch.nn.CrossEntropyLoss()
MODEL_PATH = './models/'
MODEL_NAME = 'pretrain_lm_biogpt_fix.pth'
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
original_path = "H:/Pathology/"
report_path = "E:/hsuanchia_e/pathology_imgcap/diagnosis_report"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU state:', device)

class Text_dataset_word_chain(Dataset):
    def __init__(self, data_text, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.in_seq, self.out_seq = self.build_word_chain(data_text)

    def __len__(self):
        return len(self.in_seq)

    def __getitem__(self, index):
        return self.in_seq[index], self.out_seq[index]
    
    def build_word_chain(self, data_text):
        in_seq, out_seq = [], []
        for i in track(range(len(data_text)), description="Build word chain"):
            seq = " <SEP>".join(data_text[i])
            seq = seq.split(" ")
            seq = [word for word in seq if seq != '']
            tmp_seq = seq.copy()
            tmp_seq.append("<EOS>")
            for i in range(1, len(tmp_seq)):
                tmp_in_seq, tmp_out_seq = tmp_seq[:i], tmp_seq[i:]
                tmp_in_seq = " ".join(tmp_in_seq)
                tmp_in_seq = self.tokenizer(tmp_in_seq, return_tensors="pt", truncation=True, padding='max_length', max_length=256)['input_ids']
                tmp_out_seq = " ".join(tmp_out_seq)
                tmp_out_seq = self.tokenizer(tmp_out_seq, return_tensors="pt", truncation=True, padding='max_length', max_length=256)['input_ids']
                in_seq.append(tmp_in_seq)
                out_seq.append(tmp_out_seq)
        return in_seq, out_seq


class Text_dataset(Dataset):
    def __init__(self, datalist, tokenizer, word_chain=False) -> None:
        self.datalist = datalist
        self.tokenizer = tokenizer
        self.word_chain = word_chain

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        seq = " <SEP>".join(self.datalist[index])
        seq = seq.split(" ")
        seq = [word for word in seq if seq != '']
        tmp_seq = seq.copy()
        tmp_seq.append("<EOS>")
        if self.word_chain == False:
            out_seq = tmp_seq.copy()
            in_seq = tmp_seq.copy()
            out_seq = " ".join(out_seq)
            out_seq = self.tokenizer(out_seq, return_tensors="pt", truncation=True, padding='max_length', max_length=256)['input_ids']
            in_seq.insert(0, "<BOS>")
            in_seq = " ".join(in_seq)
            in_seq = self.tokenizer(in_seq, return_tensors="pt", truncation=True, padding='max_length', max_length=256)['input_ids']
        else:
            in_seq, out_seq = [], []
            for i in range(1, len(tmp_seq)):
                tmp_in_seq, tmp_out_seq = tmp_seq[:i], tmp_seq[i:]
                tmp_in_seq = " ".join(tmp_in_seq)
                tmp_in_seq = self.tokenizer(tmp_in_seq, return_tensors="pt", padding='max_length', max_length=256)['input_ids']
                tmp_out_seq = " ".join(tmp_out_seq)
                tmp_out_seq = self.tokenizer(tmp_out_seq, return_tensors="pt", padding='max_length', max_length=256)['input_ids']
                in_seq.append(tmp_in_seq)
                out_seq.append(tmp_out_seq)
            in_seq = torch.stack(in_seq)
            out_seq = torch.stack(out_seq)

        return in_seq, out_seq

def train(train_DataLoader, val_DataLoader, model, opt):
    best_loss = 888888
    earlystop = 0
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
            model.train()
            for in_seq, out_seq in train_DataLoader:
                in_seq = in_seq.to(device)
                out_seq = out_seq.to(device)
                in_seq = in_seq.squeeze(0).squeeze(1)
                out_seq = out_seq.squeeze(0).squeeze(1)
                # print(in_seq.shape)
                # print(out_seq.shape)
                outputs = model(input_ids=in_seq, labels=out_seq)
                print(outputs.loss)
                opt.zero_grad()
                outputs.loss.backward()
                opt.step()
                train_avg_loss += outputs.loss.item()
                train_num += 1
                progress.advance(train_batch_tqdm, advance=1)
            print("Start validation...")
            model.eval()
            with torch.no_grad():
                for val_in_seq, val_out_seq in val_DataLoader:
                    val_in_seq = val_in_seq.to(device)
                    val_out_seq = val_out_seq.to(device)
                    val_in_seq = val_in_seq.squeeze(0).squeeze(1)
                    val_out_seq = val_out_seq.squeeze(0).squeeze(1)
                    val_outputs = model(val_in_seq, labels=val_out_seq)
                    print(val_outputs.loss)
                    val_avg_loss += val_outputs.loss.item()
                    val_num += 1
                    progress.advance(val_batch_tqdm, advance=1)

            train_avg_loss /= train_num
            val_avg_loss /= val_num

            LOGGER.info('Epoch {}: train loss: {} |  val loss : {}'.format(num_epochs ,train_avg_loss, val_avg_loss)) #紀錄訓練資訊
            print('Epoch {}: train loss: {} |  val loss : {}'.format(num_epochs ,train_avg_loss, val_avg_loss))
            scheduler.step(val_avg_loss)
            if best_loss > val_avg_loss:
                print("Model saving...")
                best_loss = val_avg_loss
                path = MODEL_PATH+MODEL_NAME
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
    used_text, data_text = [], []
    for cases in track(os.listdir(original_path)):
        tmp_p = os.path.join(original_path, cases)
        for files in os.listdir(tmp_p):
            if '.txt' in files:
                used_text.append(files)
                break
    for reports in track(os.listdir(report_path)):
        if reports not in used_text:
            tmp_p = os.path.join(report_path, reports)
            text, _ = extract_pathology_sections(tmp_p)
            data_text.append(text)
    
    train_text, val_text = data_text[:2000], data_text[2000:]
    print("Train length: ", len(train_text))
    print("Val length: ", len(val_text))
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    special_tokens_dict = {
        'bos_token': '<BOS>',
        'eos_token': '<EOS>',
        'pad_token': '<EOS>',
        'sep_token': '<SEP>',
    }
    # 加入 special tokens
    tokenizer.add_special_tokens(special_tokens_dict)
    
    model = BioGptForCausalLM.from_pretrained('microsoft/biogpt').to(device).to(torch.float32)
    model.resize_token_embeddings(len(tokenizer))
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5, verbose=True)

    train_dataset = Text_dataset(train_text, tokenizer)
    train_DataLoader = DataLoader(train_dataset, shuffle=True, batch_size=16, pin_memory=True)
    val_dataset = Text_dataset(val_text, tokenizer)
    val_DataLoader = DataLoader(val_dataset, shuffle=True, batch_size=16, pin_memory=True)

    train(train_DataLoader, val_DataLoader, model, opt)






