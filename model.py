import torch, torchvision, gc, random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool
from torchvision.models.resnet import Bottleneck, ResNet
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BioGptForCausalLM, AutoTokenizer, AutoModel, AutoModelForCausalLM, BartModel
from torchsummaryX import summary
from tqdm import tqdm

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class ResNetTrunk(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.fc  # remove FC layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

def get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "BT": "bt_rn50_ep200.torch",
        "MoCoV2": "mocov2_rn50_ep200.torch",
        "SwAV": "swav_rn50_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url

def resnet50(pretrained, progress, key, **kwargs):
    model = ResNetTrunk(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        verbose = model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        )
        print(verbose)
    return model

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device('cuda:0'):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect() 
            gc.collect()

class Patch_dataset(Dataset):
    def __init__(self, datalist) -> None:
        self.datalist = datalist
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        patches = self.transform(Image.open(self.datalist[index][0]).convert("RGB"))

        return patches

class GatedAttentionMIL(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.attention_V = nn.Linear(input_dim, hidden_dim)
        self.attention_U = nn.Linear(input_dim, hidden_dim)
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, x):  # x: [N, D]
        A_V = torch.tanh(self.attention_V(x))     # [N, H]
        A_U = torch.sigmoid(self.attention_U(x))  # [N, H]
        A = A_V * A_U                              # [N, H]
        A = self.attention_weights(A)              # [N, 1]
        A = torch.softmax(A, dim=0)                # soft attention weights
        z = torch.sum(A * x, dim=0, keepdim=True)  # [1, D]
        return z, A  # 聚合後的 WSI 特徵 + 注意力權重

class MIL2BioGPT(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, gpt_hidden_dim=1024):
        super().__init__()
        self.image_extraction = resnet50(pretrained=True, progress=False, key="SwAV").to(device).to(torch.float32)
        for name, param in self.image_extraction.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.patch_encoder = GatedAttentionMIL(input_dim, hidden_dim)
        self.mil_encoder = GatedAttentionMIL(input_dim, gpt_hidden_dim)
        self.feature_projector = nn.Linear(input_dim, gpt_hidden_dim)

        self.biogpt = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
        self.biogpt.load_state_dict(torch.load(f'./models/pretrain_lm_biogpt.pth'))

    def forward(self, img_path, input_ids, inference=False):
        """
        patch_feats: Tensor [N, 512] → CNN extracted features
        decoder_input_ids: Tensor [1, L] → tokenized sentence for teacher forcing
        """
        graph_emb = []
        patch_att_scores = []

        for i in tqdm(range(len(img_path))):
            img = [self.transform(Image.open(path[0]).convert("RGB")) for path in img_path[i]]
            img = torch.stack(img).to(device)
            x = self.image_extraction(img)
            x = self.avgpool(x).view(x.size(0), -1)
            patch_vec, attn_weights = self.patch_encoder(x)  # [1, 512], [N, 1]
            graph_emb.append(patch_vec.detach())
            patch_att_scores.append(attn_weights.detach())

            del img, x, attn_weights
            torch_gc() 

        graph_emb = torch.stack(graph_emb)
        graph_emb = graph_emb.squeeze(1)
        graph_emb, mil_attn = self.mil_encoder(graph_emb)
        projected = self.feature_projector(graph_emb)  # [1, 1024]
        projected = projected.unsqueeze(0)

        # 將 MIL 特徵接到 BioGPT 前面當作起始 context
        # BioGPT 的輸入可以是 input_ids 或 inputs_embeds
        # 為簡單實作，這裡用 input_ids + 首 token 替換技巧
        inputs_embeds = self.biogpt.get_input_embeddings()(input_ids)
        inputs_embeds = inputs_embeds.unsqueeze(0)
        multimodal_input = torch.cat([projected, inputs_embeds[:, 1:, :]], dim=1)
        input_ids = input_ids.unsqueeze(0)
        if inference:
            for t in tqdm(range(256 - 1)):  # 循環直到倒數第二個 token（因為要預測下一個）
                
                output = self.biogpt(inputs_embeds=multimodal_input, labels=input_ids)
                logits = output.logits  # (1, seq_len + 1, vocab_size)

                # 機率取 argmax
                pred_token = torch.argmax(logits, dim=-1)  # (1, 1)

                # 把 next_token 加進 multimodal_input
                next_embed = self.biogpt.get_input_embeddings()(pred_token)
                multimodal_input = torch.cat([projected, next_embed[:, :-1, :]], dim=1)  

        output = self.biogpt(inputs_embeds=multimodal_input, labels=input_ids)
        return output, patch_att_scores

class Image_GPT2(nn.Module):
    def __init__(self, device="cuda"):
        super(Image_GPT2, self).__init__()
        self.resnet50 = resnet50(pretrained=True, progress=False, key="SwAV").to(device).to(torch.float32)
        # self.resnet50 = torch.jit.script(self.resnet50) # 加速CPU模型運算
        for name, param in self.resnet50.named_parameters():
            if "layer4" not in name and "fc" not in name:  # layer4 + fc 層允許訓練
                param.requires_grad = False
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").to(device).to(torch.float32)
        self.img_emb = nn.Linear(2048, self.gpt2.config.n_embd).to(device).to(torch.float32) # 2048 -> 768

    def forward(self, img, input_ids):
        img_features = self.resnet50(img)
        
        img_features = self.avgpool(img_features)
        img_features = img_features.view(img_features.size(0), -1)
        
        img_features = img_features.to(device)
        # print(img_features.shape)
        image_embedding = self.img_emb(img_features)  # (batch_size, 768)
        image_embedding = image_embedding.unsqueeze(1)  # (batch_size, 1, 768)

        input_ids = input_ids.to(device).to(torch.long)
        input_ids = input_ids.squeeze(0)
        # print(input_ids.shape)
        text_embedding = self.gpt2.get_input_embeddings()(input_ids)  # (batch_size, seq_len, 768)
        multimodal_input = torch.cat([image_embedding, text_embedding[:, :-1 ,:]], dim=1)  # (batch_size, seq_len, 768)

        # GPT-2 Forward Pass
        output = self.gpt2(inputs_embeds=multimodal_input, labels=input_ids)

        return output.loss, output.logits
    
class GAT(torch.nn.Module):
    def __init__(self, feature_dim=2048, hidden_dim=512, heads=1):
        super(GAT, self).__init__()
        self.conv1 = GCNConv(feature_dim, hidden_dim)
        self.gat1 = GATConv(hidden_dim, hidden_dim // 2, heads=heads, concat=True)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x, gat_attn_weights = self.gat1(x, edge_index, return_attention_weights=True)
        return x, gat_attn_weights

class GNN_BioGPT2(nn.Module):
    def __init__(self, device="cuda"):
        super(GNN_BioGPT2, self).__init__()
        self.image_extraction = resnet50(pretrained=True, progress=False, key="SwAV").to(device).to(torch.float32)
        for name, param in self.image_extraction.named_parameters():
            if "layer4" not in name and "fc" not in name:  # layer4 + fc 層允許訓練
                param.requires_grad = False
        self.Biogpt = BioGptForCausalLM.from_pretrained('microsoft/biogpt').to(device).to(torch.float32)
        self.Biogpt.load_state_dict(torch.load(f'./models/pretrain_lm_biogpt.pth'))
        self.gat = GAT(hidden_dim=512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.img_proj = nn.Linear(256, 1024)
        self.img_token_len = 50
    
    def forward(self, img_path, edge_index, input_ids):
        graph_emb = []
        gat_att_scores = []
        for i in tqdm(range(len(img_path))):
            img = [self.transform(Image.open(path[0]).convert("RGB")) for path in img_path[i]]
            img = torch.stack(img).to(device)
            x = self.image_extraction(img)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            cur_edge_index = edge_index[i][0].to(device)
            gnn_out, att_scores = self.gat(x, cur_edge_index)
            graph_emb.append(gnn_out.detach())
            patch_att_score = self.get_patch_att_score(att_scores[0].detach(), att_scores[1].detach())
            gat_att_scores.append(patch_att_score)
            x = x.detach()
            del img, x, att_scores  # 刪除變數
            torch_gc()
        graph_emb = torch.stack(graph_emb)
        graph_emb = self.img_proj(graph_emb)
        graph_emb = graph_emb.view(-1, graph_emb.size(-1))
        gat_att_scores = torch.stack(gat_att_scores)
        gat_att_scores = gat_att_scores.view(-1)
        # print(graph_emb.shape)
        # print(gat_att_scores.shape)
        topk_scores, topk_indices = torch.topk(gat_att_scores, k=self.img_token_len)
        topk_patch_features = graph_emb[topk_indices]  # shape: (K, D)
        # print(topk_patch_features.shape)

        input_ids = input_ids.to(device).to(torch.long)
        text_embedding = self.Biogpt.get_input_embeddings()(input_ids)  # (batch_size, seq_len, 1024)
        topk_patch_features = topk_patch_features.unsqueeze(0)
        multimodal_input = torch.cat([topk_patch_features, text_embedding[:, :-self.img_token_len ,:]], dim=1)  # (batch_size, seq_len, 1024)
        ignore_prefix = torch.full((1, self.img_token_len), -100, dtype=torch.long)
        ignore_prefix = ignore_prefix.to(device)
        labels = torch.cat([ignore_prefix, input_ids[:, 1:-self.img_token_len+1]], dim=1)  # (1, total_len)

        # GPT-2 Forward Pass
        output = self.Biogpt(inputs_embeds=multimodal_input, labels=labels)

        return output, topk_scores, topk_indices
    
    def get_patch_att_score(self, edge_index, att_scores):
        src, dst = edge_index
        patch_attention = torch.zeros(100)
        patch_attention = patch_attention.to(device)
        add_times = torch.zeros(100)
        add_times = add_times.to(device)
        for i in range(len(src)):
            patch_attention[src[i]] += att_scores[i][0]
            patch_attention[dst[i]] += att_scores[i][0]
            add_times[src[i]] += 1
            add_times[dst[i]] += 1
        # print(patch_attention)
        # print(add_times)
        patch_attention = patch_attention / add_times
        # print(patch_attention)
        return patch_attention

class GIN_GatedAttn(torch.nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512):
        super(GIN_GatedAttn, self).__init__()
        nn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.gin = GINConv(nn1)
        self.attention_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.attention_weights = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, batch=None):
        x = self.gin(x, edge_index)
        A = self.attention_gate(x)                      # shape: (N, hidden_dim)
        alpha = self.attention_weights(A).squeeze(-1)   # shape: (N,)
        att_scores = torch.softmax(alpha, dim=0)
        return x, att_scores  # return x for later projection + att_scores for top-k
        
class GCN_GatedAttn(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super(GCN_GatedAttn, self).__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

        self.att_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index):
        """
        Args:
            x: (num_patches, input_dim)
            edge_index: (2, num_edges)
        Returns:
            graph_feature: (1, hidden_dim)
            att_weight: (num_patches,)
        """

        residual = x
        x = self.gcn1(x, edge_index)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.gcn2(x, edge_index)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Gated Attention
        att_score = self.att_mlp(x).squeeze(-1)  # (num_patches,)
        att_weight = torch.softmax(att_score, dim=0)  # across patches

        graph_feature = torch.sum(x * att_weight.unsqueeze(-1), dim=0)  # weighted sum

        return graph_feature.unsqueeze(0), att_weight

class GAT_GatedAttn(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=[4, 4, 4], dropout=0.3):
        super(GAT_GatedAttn, self).__init__()

        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads[0], concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim * num_heads[0])

        self.gat2 = GATConv(hidden_dim * num_heads[0], hidden_dim, heads=num_heads[1], concat=True)
        self.bn2 = nn.BatchNorm1d(hidden_dim * num_heads[1])

        self.gat3 = GATConv(hidden_dim * num_heads[1], hidden_dim, heads=num_heads[2], concat=True)
        self.bn3 = nn.BatchNorm1d(hidden_dim * num_heads[2])

        self.dropout = nn.Dropout(dropout)

        # Gated Attention Pooling (雙層MLP設計)
        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_dim * num_heads[2], hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.feature_nn = nn.Sequential(
            nn.Linear(hidden_dim * num_heads[2], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.final_proj = nn.Linear((hidden_dim // 2) * 10, hidden_dim)  # 結合 Gated pooled + Global pooled

    def forward(self, x, edge_index, batch=None):
        identity = x

        # 1st GAT layer
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.dropout(x)

        # 2nd GAT layer + Residual
        x_res = x
        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.dropout(x)
        x = x + x_res  # 殘差連接

        # 3rd GAT layer + Residual
        x_res = x
        x = self.gat3(x, edge_index)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2)
        x = self.dropout(x)
        x = x + x_res  # 殘差連接

        # Gated Attention Pooling
        attn_scores = self.gate_nn(x).squeeze(-1)  # (N,)
        attn_weights = torch.softmax(attn_scores, dim=0)  # Softmax over nodes
        gated_pooled = torch.sum(attn_weights.unsqueeze(-1) * self.feature_nn(x), dim=0)  # (hidden_dim,)

        # Global Mean Pooling
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        global_pooled = global_mean_pool(x, batch)  # (batch_size, hidden_dim * num_heads[2])

        # 結合
        combined = torch.cat([gated_pooled.unsqueeze(0), global_pooled], dim=-1)
        out = self.final_proj(combined)  # (batch_size, hidden_dim)

        return out.squeeze(0), attn_scores

class GIN_BioGPT2(nn.Module):
    def __init__(self, device="cuda"):
        super(GIN_BioGPT2, self).__init__()
        self.image_extraction = resnet50(pretrained=True, progress=False, key="SwAV").to(device).to(torch.float32)
        for name, param in self.image_extraction.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False
        self.Biogpt = BioGptForCausalLM.from_pretrained('microsoft/biogpt').to(device).to(torch.float32)
        self.Biogpt.load_state_dict(torch.load(f'./models/pretrain_lm_biogpt.pth'))

        # self.gnn = GIN_GatedAttn(input_dim=2048, hidden_dim=512).to(device)
        # self.gnn = GCN_GatedAttn(input_dim=2048, hidden_dim=512).to(device)
        self.gnn = GAT_GatedAttn(input_dim=2048, hidden_dim=512).to(device)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.img_proj = nn.Linear(512, 1024)
        self.img_token_len = 50

    def forward(self, img_path, edge_index, input_ids):
        graph_emb = []
        patch_att_scores = []

        for i in tqdm(range(len(img_path))):
            img = [self.transform(Image.open(path[0]).convert("RGB")) for path in img_path[i]]
            img = torch.stack(img).to(device)
            x = self.image_extraction(img)
            x = self.avgpool(x).view(x.size(0), -1)
            cur_edge_index = edge_index[i][0].to(device)
            
            x, att_scores = self.gnn(x, cur_edge_index)
            graph_emb.append(x.detach())
            patch_att_scores.append(att_scores.detach())

            del img, x, att_scores
            torch_gc()

        graph_emb = torch.cat(graph_emb, dim=0)                    # (N, 512)
        patch_att_scores = torch.cat(patch_att_scores, dim=0)      # (N,)
        
        graph_emb = self.img_proj(graph_emb)                       # (N, 1024)
        topk_scores, topk_indices = torch.topk(patch_att_scores, k=self.img_token_len)
        topk_patch_features = graph_emb[topk_indices]              # (K, 1024)

        # normed = F.normalize(topk_patch_features, dim=1)  # 單位向量化
        # cos_sim_matrix = torch.mm(normed, normed.T)  # (100, 100)

        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # # 顯示 heatmap
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(cos_sim_matrix.cpu().numpy(), cmap='viridis')
        # plt.title("Cosine Similarity of GIN Output Embeddings")
        # plt.show()

        input_ids = input_ids.to(device).to(torch.long)
        text_embedding = self.Biogpt.get_input_embeddings()(input_ids) 
        # text_embedding = text_embedding.unsqueeze(0) # (1, seq_len, 1024)
        topk_patch_features = topk_patch_features.unsqueeze(0)  # (1, K, 1024)
        multimodal_input = torch.cat([topk_patch_features, text_embedding[:, :-self.img_token_len, :]], dim=1)

        ignore_prefix = torch.full((1, self.img_token_len), -100, dtype=torch.long).to(device)
        # input_ids = input_ids.unsqueeze(0)
        labels = torch.cat([ignore_prefix, input_ids[:, 1:-self.img_token_len + 1]], dim=1)

        output = self.Biogpt(inputs_embeds=multimodal_input, labels=labels)

        return output, topk_scores, topk_indices
    
class GIN_BioGPT2_schedule(nn.Module):
    def __init__(self, device="cuda"):
        super(GIN_BioGPT2_schedule, self).__init__()
        self.image_extraction = resnet50(pretrained=True, progress=False, key="SwAV").to(device).to(torch.float32)
        for name, param in self.image_extraction.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False
        self.Biogpt = BioGptForCausalLM.from_pretrained('microsoft/biogpt').to(device).to(torch.float32)
        self.Biogpt.load_state_dict(torch.load(f'./models/pretrain_lm_biogpt.pth'))

        self.gnn = GIN_GatedAttn(input_dim=2048, hidden_dim=512).to(device)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.img_proj = nn.Linear(512, 1024)
        self.img_token_len = 50

    def forward(self, img_path, edge_index, input_ids, teacher_ratio=0.0):
        
        graph_emb = []
        patch_att_scores = []

        for i in tqdm(range(len(img_path))):
            img = [self.transform(Image.open(path[0]).convert("RGB")) for path in img_path[i]]
            img = torch.stack(img).to(device)
            x = self.image_extraction(img)
            x = self.avgpool(x).view(x.size(0), -1)
            cur_edge_index = edge_index[i][0].to(device)
            x, att_scores = self.gnn(x, cur_edge_index)
            graph_emb.append(x.detach())
            patch_att_scores.append(att_scores.detach())

            del img, x, att_scores
            torch_gc()

        graph_emb = torch.cat(graph_emb, dim=0)                    # (N, 512)
        patch_att_scores = torch.cat(patch_att_scores, dim=0)      # (N,)

        graph_emb = self.img_proj(graph_emb)                       # (N, 1024)
        topk_scores, topk_indices = torch.topk(patch_att_scores, k=self.img_token_len)
        topk_patch_features = graph_emb[topk_indices]              # (K, 1024)

        input_ids = input_ids.to(device).to(torch.long)
        # text_embedding = text_embedding.unsqueeze(0) # (1, seq_len, 1024)
        start_token = input_ids[0]  # BOS token
        gt_tokens = input_ids[1:]     # 後面所有 token
        generated = []
        next_embed, next_token = [], []
        token_embed = self.Biogpt.get_input_embeddings()(start_token)  # (1, 1, 1024)
        token_embed = token_embed.unsqueeze(0)
        multimodal_input = torch.cat([topk_patch_features, token_embed], dim=0)
        multimodal_input = multimodal_input.unsqueeze(0)

        for t in tqdm(range(256 - 1)):  # 循環直到倒數第二個 token（因為要預測下一個）
            
            output = self.Biogpt(inputs_embeds=multimodal_input)
            logits = output.logits  # (1, seq_len + 1, vocab_size)
            last_token_logits = logits[:, -1, :]  # 最後一個 token 的預測

            # 機率取 argmax
            pred_token = torch.argmax(last_token_logits, dim=-1)  # (1, 1)

            # 決定下個 token 要接什麼：使用 ground-truth 還是 pred
            # teacher ratio = 0.0 means all use pred_token
            num = random.random()
            use_gt = num < teacher_ratio
            next_token = gt_tokens[t:t+1] if use_gt else pred_token

            # 把 next_token 加進 multimodal_input
            next_embed = self.Biogpt.get_input_embeddings()(next_token)
            next_embed = next_embed.unsqueeze(0)
            multimodal_input = torch.cat([multimodal_input, next_embed], dim=1)

            generated.append(next_token)
        # generated_ids = torch.cat(generated, dim=1)  # (1, seq_len - 1)

        # 組合 label，前面 img_token_len 和 start token 的地方設為 -100
        ignore_prefix = torch.full((1, self.img_token_len + 1), -100, dtype=torch.long).to(device)
        gt_tokens = gt_tokens.unsqueeze(0)
        labels = torch.cat([ignore_prefix, gt_tokens], dim=1)

        output = self.Biogpt(inputs_embeds=multimodal_input, labels=labels)

        return output, topk_scores, topk_indices

class GNN_BioGPT2_schedule(nn.Module):
    def __init__(self, device="cuda"):
        super(GNN_BioGPT2_schedule, self).__init__()
        self.image_extraction = resnet50(pretrained=True, progress=False, key="SwAV").to(device).to(torch.float32)
        for name, param in self.image_extraction.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False
        self.Biogpt = BioGptForCausalLM.from_pretrained('microsoft/biogpt').to(device).to(torch.float32)
        self.Biogpt.load_state_dict(torch.load(f'./models/pretrain_lm_biogpt.pth'))

        # self.gnn = GIN_GatedAttn(input_dim=2048, hidden_dim=512).to(device)
        # self.gnn = GAT_GatedAttn(input_dim=2048, hidden_dim=512).to(device)
        self.gnn = GCN_GatedAttn(input_dim=2048, hidden_dim=512).to(device)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.graph_attention = torch.nn.Sequential(
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid()
        )

        self.img_proj = nn.Linear(512, 1024)
        self.img_token_len = 1

    def forward(self, img_path, edge_index, input_ids, teacher_ratio=0.0):
        
        graph_emb = []
        patch_att_scores = []

        for i in tqdm(range(len(img_path))):
            img = [self.transform(Image.open(path[0]).convert("RGB")) for path in img_path[i]]
            img = torch.stack(img).to(device)
            x = self.image_extraction(img)
            x = self.avgpool(x).view(x.size(0), -1)
            cur_edge_index = edge_index[i][0].to(device)
            x, att_scores = self.gnn(x, cur_edge_index)
            graph_emb.append(x.detach())
            patch_att_scores.append(att_scores.detach())

            del img, x, att_scores
            torch_gc()

        graph_emb = torch.stack(graph_emb)
        graph_emb = graph_emb.squeeze(1)
        patch_att_scores = torch.stack(patch_att_scores)
        
        graph_att_scores = self.graph_attention(graph_emb)  # [num_graphs, 1]
        graph_att_scores = torch.softmax(graph_att_scores, dim=0)
        wsi_emb = torch.sum(graph_att_scores * graph_emb, dim=0)

        graph_emb = self.img_proj(wsi_emb)

        input_ids = input_ids.to(device).to(torch.long)
        start_token = input_ids[0]  # BOS token
        gt_tokens = input_ids[1:]     # 後面所有 token
        generated = []
        next_embed, next_token = [], []
        token_embed = self.Biogpt.get_input_embeddings()(start_token)  # (1, 1, 1024)
        token_embed = token_embed.unsqueeze(0)
        graph_emb = graph_emb.unsqueeze(0)
        multimodal_input = torch.cat([graph_emb, token_embed], dim=0)
        multimodal_input = multimodal_input.unsqueeze(0)

        for t in tqdm(range(256 - 1)):  # 循環直到倒數第二個 token（因為要預測下一個）
            
            output = self.Biogpt(inputs_embeds=multimodal_input)
            logits = output.logits  # (1, seq_len + 1, vocab_size)
            last_token_logits = logits[:, -1, :]  # 最後一個 token 的預測

            # 機率取 argmax
            pred_token = torch.argmax(last_token_logits, dim=-1)  # (1, 1)

            # 決定下個 token 要接什麼：使用 ground-truth 還是 pred
            # teacher ratio = 0.0 means all use pred_token
            num = random.random()
            use_gt = num < teacher_ratio
            next_token = gt_tokens[t:t+1] if use_gt else pred_token

            # 把 next_token 加進 multimodal_input
            next_embed = self.Biogpt.get_input_embeddings()(next_token)
            next_embed = next_embed.unsqueeze(0)
            multimodal_input = torch.cat([multimodal_input, next_embed], dim=1)

            generated.append(next_token)

        # 組合 label，前面 img_token_len 和 start token 的地方設為 -100
        ignore_prefix = torch.full((1, self.img_token_len + 1), -100, dtype=torch.long).to(device)
        gt_tokens = gt_tokens.unsqueeze(0)
        labels = torch.cat([ignore_prefix, gt_tokens], dim=1)

        output = self.Biogpt(inputs_embeds=multimodal_input, labels=labels)

        return output, patch_att_scores, graph_att_scores

class GNN_image_encoder(nn.Module):
    def __init__(self, device="cuda"):
        super(GNN_image_encoder, self).__init__()
        self.image_extraction = resnet50(pretrained=True, progress=False, key="SwAV").to(device).to(torch.float32)
        for name, param in self.image_extraction.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False
        self.Biogpt = BioGptForCausalLM.from_pretrained('microsoft/biogpt').to(device).to(torch.float32)
        self.Biogpt.load_state_dict(torch.load(f'./models/pretrain_lm_biogpt.pth'))
        self.gnn = GCN_GatedAttn(input_dim=2048, hidden_dim=512).to(device)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.graph_attention = torch.nn.Sequential(
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid()
        )
        self.img_token_len = 1
        self.adapter = CrossModalAdapter(input_dim=512, output_dim=1024)
    
    def forward(self, img_path, edge_index, input_ids, teacher_ratio=0.0):
        
        graph_emb = []
        patch_att_scores = []

        for i in tqdm(range(len(img_path))):
            img = [self.transform(Image.open(path[0]).convert("RGB")) for path in img_path[i]]
            img = torch.stack(img).to(device)
            x = self.image_extraction(img)
            x = self.avgpool(x).view(x.size(0), -1)
            cur_edge_index = edge_index[i][0].to(device)
            x, att_scores = self.gnn(x, cur_edge_index)
            graph_emb.append(x.detach())
            patch_att_scores.append(att_scores.detach())

            del img, x, att_scores
            torch_gc()

        ### Image vector
        graph_emb = torch.stack(graph_emb)
        graph_emb = graph_emb.squeeze(1)
        patch_att_scores = torch.stack(patch_att_scores)
        graph_att_scores = self.graph_attention(graph_emb)  # [num_graphs, 1]
        graph_att_scores = torch.softmax(graph_att_scores, dim=0)
        wsi_emb = torch.sum(graph_att_scores * graph_emb, dim=0)
        wsi_emb = wsi_emb.unsqueeze(0)
        wsi_emb = self.adapter(wsi_emb)  # (B, N, 768)

        ### Text vector
        input_ids = input_ids.to(device).to(torch.long)
        start_token = input_ids[0]  # BOS token
        gt_tokens = input_ids[1:]     # 後面所有 token
        generated = []
        next_embed, next_token = [], []
        token_embed = self.Biogpt.get_input_embeddings()(start_token)
        token_embed = token_embed.unsqueeze(0)
        # graph_emb = graph_emb.unsqueeze(0)
        # print(token_embed.shape)
        # print(wsi_emb.shape)
        multimodal_input = torch.cat([wsi_emb, token_embed], dim=0)
        multimodal_input = multimodal_input.unsqueeze(0)

        for t in tqdm(range(256 - 1)):  # 循環直到倒數第二個 token（因為要預測下一個）
            
            output = self.Biogpt(inputs_embeds=multimodal_input)
            logits = output.logits  # (1, seq_len + 1, vocab_size)
            last_token_logits = logits[:, -1, :]  # 最後一個 token 的預測

            # 機率取 argmax
            pred_token = torch.argmax(last_token_logits, dim=-1)  # (1, 1)

            # 決定下個 token 要接什麼：使用 ground-truth 還是 pred
            # teacher ratio = 0.0 means all use pred_token
            num = random.random()
            use_gt = num < teacher_ratio
            next_token = gt_tokens[t:t+1] if use_gt else pred_token

            # 把 next_token 加進 multimodal_input
            next_embed = self.Biogpt.get_input_embeddings()(next_token)
            next_embed = next_embed.unsqueeze(0)
            multimodal_input = torch.cat([multimodal_input, next_embed], dim=1)

            generated.append(next_token)

        # 組合 label，前面 img_token_len 和 start token 的地方設為 -100
        ignore_prefix = torch.full((1, self.img_token_len + 1), -100, dtype=torch.long).to(device)
        gt_tokens = gt_tokens.unsqueeze(0)
        labels = torch.cat([ignore_prefix, gt_tokens], dim=1)

        output = self.Biogpt(inputs_embeds=multimodal_input, labels=labels)

        return output, patch_att_scores, graph_att_scores

class CrossModalAdapter(nn.Module):
    def __init__(self, input_dim=512, output_dim=1024, num_layers=2, num_heads=8):
        super().__init__()
        self.linear_proj = nn.Linear(input_dim, output_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x):
        """
        x: (B, N, input_dim) 影像特徵 (例如從 ResNet 或 GNN 得來)
        return: (B, N, output_dim) 映射後的語言空間特徵
        """
        x = self.linear_proj(x)
        x = self.transformer(x)
        return x

class BioGPT_checklist_encoder(nn.Module):
    def __init__(self, device="cuda"):
        super(BioGPT_checklist_encoder, self).__init__()
        self.Biogpt = BioGptForCausalLM.from_pretrained('microsoft/biogpt').to(device)
        self.Biogpt.load_state_dict(torch.load(f'./models/pretrain_lm_biogpt.pth'))
        self.Biobert_model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1").to(device)
        self.linear = nn.Linear(768, 1024)

    def forward(self, in_seq, out_seq):
        # print(in_seq['input_ids'])
        # print(in_seq['attention_mask'])
        # print(in_seq['input_ids'].shape)
        # print(in_seq['attention_mask'].shape)
        encode_output = self.Biobert_model(input_ids=in_seq['input_ids'].squeeze(1), attention_mask=in_seq['attention_mask'].squeeze(1))
        prompt_encode = encode_output.last_hidden_state[:, 0, :]
        prompt_encode = self.linear(prompt_encode)
        out_seq_embed = self.Biogpt.get_input_embeddings()(out_seq['input_ids'].squeeze(1))
        prompt_encode = prompt_encode.unsqueeze(0)
        tmp_seq_emb = out_seq_embed[:, 1:, :]
        multimodal_input = torch.cat([prompt_encode, tmp_seq_emb], dim=1)
        # multimodal_input = multimodal_input.squeeze(0)

        output = self.Biogpt(inputs_embeds=multimodal_input, labels=out_seq['input_ids'].squeeze(1))

        return output

class BioGPT_checklist_condition(nn.Module):
    def __init__(self, encoder_tokenizer, decoder_tokenizer):
        super(BioGPT_checklist_condition, self).__init__()
        self.encoder = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1").to(device)
        # self.decoder = BartModel.from_pretrained('facebook/bart-base').to(device)
        # self.decoder = AutoModelForCausalLM.from_pretrained("PharMolix/BioMedGPT-LM-7B").to(device)
        self.decoder = BioGptForCausalLM.from_pretrained('microsoft/biogpt').to(device)
        self.decoder.load_state_dict(torch.load(f'./models/pretrain_lm_biogpt.pth'))
        self.encoder.resize_token_embeddings(len(encoder_tokenizer))
        self.decoder.resize_token_embeddings(len(decoder_tokenizer))
        self.linear = nn.Linear(768, 1024)

    def forward(self, in_seq, out_seq):
        encode_output = self.encoder(input_ids=in_seq['input_ids'].squeeze(1), attention_mask=in_seq['attention_mask'].squeeze(1))
        prompt_encode = self.linear(encode_output.last_hidden_state)

        output = self.decoder(inputs_embeds=prompt_encode, labels=out_seq['input_ids'].squeeze(1))

        return output
 
class BioGPT_checklist_encoder_rag(nn.Module):
    def __init__(self, device="cuda"):
        super(BioGPT_checklist_encoder_rag, self).__init__()
        self.Biogpt = BioGptForCausalLM.from_pretrained('microsoft/biogpt').to(device)
        self.Biogpt.load_state_dict(torch.load(f'./models/pretrain_lm_biogpt.pth'))
        self.Biobert_model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1").to(device)
        self.linear = nn.Linear(768, 1024)

    def forward(self, in_seq, out_seq, template_finding):
        # print(in_seq['input_ids'])
        # print(in_seq['attention_mask'])
        # print(in_seq['input_ids'].shape)
        # print(in_seq['attention_mask'].shape)
        encode_output = self.Biobert_model(input_ids=in_seq['input_ids'].squeeze(1), attention_mask=in_seq['attention_mask'].squeeze(1))
        prompt_encode = encode_output.last_hidden_state[:, 0, :]
        tmp_seq_embed = self.Biogpt.get_input_embeddings()(template_finding['input_ids'].squeeze(1))
        prompt_encode = self.linear(prompt_encode)
        prompt_encode = prompt_encode.unsqueeze(0)
        multimodal_input = torch.cat([prompt_encode, tmp_seq_embed[:, 1:, :]], dim=1)
        # multimodal_input = multimodal_input.squeeze(0)

        output = self.Biogpt(inputs_embeds=multimodal_input, labels=out_seq['input_ids'].squeeze(1))

        return output
    
class BioGPT_checklist_encoder_all(nn.Module):
    def __init__(self, device="cuda"):
        super(BioGPT_checklist_encoder_all, self).__init__()
        self.Biogpt = BioGptForCausalLM.from_pretrained('microsoft/biogpt').to(device)
        self.Biogpt.load_state_dict(torch.load(f'./models/pretrain_lm_biogpt.pth'))
        self.Biobert_model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1").to(device)
        self.linear = nn.Linear(768, 1024)

    def forward(self, in_seq, out_seq):
        # encode_output = self.Biobert_model(input_ids=in_seq['input_ids'].squeeze(1), attention_mask=in_seq['attention_mask'].squeeze(1))
        # prompt_encode = encode_output.last_hidden_state
        # prompt_encode = self.linear(prompt_encode)
        in_seq_embed = self.Biogpt.get_input_embeddings()(in_seq['input_ids'].squeeze(1))
        # prompt_encode = prompt_encode.unsqueeze(0)
        # tmp_seq_emb = out_seq_embed[:, 1:, :]
        # multimodal_input = torch.cat([prompt_encode, tmp_seq_emb], dim=1)
        # multimodal_input = multimodal_input.squeeze(0)
        # print(prompt_encode.shape)
        output = self.Biogpt(inputs_embeds=in_seq_embed, labels=out_seq['input_ids'].squeeze(1))

        return output

class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.attention_a = nn.Linear(input_dim, hidden_dim)
        self.attention_b = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (num_patches, 512)
        a = torch.tanh(self.attention_a(x))      # (num_patches, hidden_dim)
        b = self.attention_b(a)                  # (num_patches, 1)
        weights = torch.softmax(b, dim=0)        # (num_patches, 1)
        weighted_feature = (weights * x).sum(dim=0)  # (512,)
        return weighted_feature, weights

class Image_checklist_BioGPT(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, gpt_hidden_dim=1024, device="cuda"):
        super(Image_checklist_BioGPT, self).__init__()
        self.image_extraction = resnet50(pretrained=True, progress=False, key="SwAV").to(device).to(torch.float32)
        for name, param in self.image_extraction.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.clip_encoder_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().cuda()
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.patch_encoder = AttentionPooling(hidden_dim, hidden_dim//2)
        self.mil_encoder = GatedAttentionMIL(hidden_dim, gpt_hidden_dim)
        self.feature_projector = nn.Linear(input_dim, gpt_hidden_dim // 2)

        self.Biogpt = BioGptForCausalLM.from_pretrained('microsoft/biogpt').to(device)
        self.Biogpt.load_state_dict(torch.load(f'./models/pretrain_lm_biogpt.pth'))
        self.Biobert_model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1").to(device)
        self.text_projection = nn.Linear(768, gpt_hidden_dim // 2)

    def forward(self, img_path, in_seq, out_seq, inference=False):
        graph_emb = []
        patch_att_scores = []

        for i in tqdm(range(len(img_path))):
            img = [self.transform(Image.open(path[0]).convert("RGB")) for path in img_path[i]]
            img = torch.stack(img).to(device)
            clip_input = self.clip_processor(images=img, return_tensors="pt", padding=True).to("cuda")
            image_features = self.clip_encoder_model.get_image_features(**clip_input)
            patch_vec, attn_weights = self.patch_encoder(image_features)  # [1, 512], [N, 1]
            graph_emb.append(patch_vec.detach())
            patch_att_scores.append(attn_weights.detach())

            del img, clip_input, image_features
            torch_gc() 
        # Image
        graph_emb = torch.stack(graph_emb)
        graph_emb, mil_attn = self.mil_encoder(graph_emb)
        graph_emb = graph_emb.unsqueeze(0)
        # Input prompt
        encode_output = self.Biobert_model(input_ids=in_seq['input_ids'].squeeze(1), attention_mask=in_seq['attention_mask'].squeeze(1))

        # Output text
        if inference:
            prompt_encode = encode_output.last_hidden_state[:, 0, :]
            text_projected = self.text_projection(prompt_encode)
            text_projected = text_projected.unsqueeze(0)
            out_seq_embed = self.Biogpt.get_input_embeddings()(out_seq['input_ids'].squeeze(1))
            merge_input = torch.cat([graph_emb, text_projected], dim=-1)
            multimodal_input = torch.cat([merge_input, out_seq_embed[:, :-1, :]], dim=1)
        else:
            prompt_encode = encode_output.last_hidden_state[:, 0, :]
            text_projected = self.text_projection(prompt_encode)
            text_projected = text_projected.unsqueeze(0)
            out_seq_embed = self.Biogpt.get_input_embeddings()(out_seq['input_ids'].squeeze(1))
            # Concat image & text embedding directly
            merge_input = torch.cat([graph_emb, text_projected], dim=-1)
            # merge_input = merge_input.transpose(0, 1)  # → (L, B, 1024)
            # merge_emb = self.encoder(merge_input)   # → (L, B, 1024)
            # merge_emb = merge_emb.transpose(0, 1)  # → (B, L, 1024)
            multimodal_input = torch.cat([merge_input, out_seq_embed[:, 1:, :]], dim=1)

        output = self.Biogpt(inputs_embeds=multimodal_input, labels=out_seq['input_ids'].squeeze(1))

        return output, patch_att_scores

class GatedAttention_encoding(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GatedAttention_encoding, self).__init__()
        self.attention_V = nn.Linear(input_dim, hidden_dim)
        self.attention_U = nn.Linear(input_dim, hidden_dim)
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        A_V = torch.tanh(self.attention_V(x))        # [B, H]
        A_U = torch.sigmoid(self.attention_U(x))     # [B, H]
        A = self.attention_weights(A_V * A_U)        # [B, 1]
        A = torch.softmax(A, dim=0)                  # softmax over patches
        # A = torch.sigmoid(A)
        return A

class GatedAttention_BioGPT2(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=128):
        super(GatedAttention_BioGPT2, self).__init__()
        self.Biogpt = BioGptForCausalLM.from_pretrained('microsoft/biogpt').to(device).to(torch.float32)
        self.Biogpt.load_state_dict(torch.load(f'./models/pretrain_lm_biogpt.pth'))
        self.GatedAttention = GatedAttention_encoding(input_dim=input_dim, hidden_dim=hidden_dim)
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.img_proj = nn.Linear(input_dim, 1024)
        self.img_token_len = 1
    
    def forward(self, img_emb, input_ids):
        A = self.GatedAttention(img_emb)   
        M = img_emb * A
        M = torch.sum(M, dim=0)
        image_feature = self.img_proj(M)

        input_ids = input_ids.to(device).to(torch.long)
        text_input_ids = input_ids['input_ids'].squeeze(1)
        text_embedding = self.Biogpt.get_input_embeddings()(text_input_ids)  # (batch_size, seq_len, 1024)
        image_feature = image_feature.unsqueeze(0).unsqueeze(0)
        multimodal_input = torch.cat([image_feature, text_embedding[:, :-1 ,:]], dim=1)  # (batch_size, seq_len, 1024)
        ignore_prefix = torch.full((1, self.img_token_len), -100, dtype=torch.long)
        ignore_prefix = ignore_prefix.to(device)
        labels = torch.cat([ignore_prefix, text_input_ids], dim=1)  # (1, total_len)
        labels = labels[:, :-1]

        output = self.Biogpt(inputs_embeds=multimodal_input, labels=labels)

        return output, A  

class Gated_Attention_BioGPT_checklist(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=128):
        super(Gated_Attention_BioGPT_checklist, self).__init__()
        self.Biogpt = BioGptForCausalLM.from_pretrained('microsoft/biogpt').to(device)
        self.Biogpt.load_state_dict(torch.load(f'./models/pretrain_lm_biogpt.pth'))
        self.Biobert_model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1").to(device)
        self.GatedAttention = GatedAttention_encoding(input_dim=input_dim, hidden_dim=hidden_dim)
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.img_proj = nn.Linear(input_dim, 1024)
        self.img_token_len = 1
        self.linear = nn.Linear(768, 1024)
        self.img_text_fuse = nn.Linear(1024*2, 1024)

    def forward(self, img_emb, in_seq, out_seq):
        encode_output = self.Biobert_model(input_ids=in_seq['input_ids'].squeeze(1), attention_mask=in_seq['attention_mask'].squeeze(1))
        prompt_encode = encode_output.last_hidden_state[:, 0, :]
        prompt_encode = self.linear(prompt_encode)
        out_seq_embed = self.Biogpt.get_input_embeddings()(out_seq['input_ids'].squeeze(1))
        A = self.GatedAttention(img_emb)   
        M = img_emb * A
        M = torch.sum(M, dim=0)
        image_feature = self.img_proj(M)
        image_feature = image_feature.unsqueeze(0)
        fuse_feature = torch.cat([image_feature, prompt_encode], dim=1)
        fuse_feature = self.img_text_fuse(fuse_feature)
        tmp_seq_emb = out_seq_embed[:, 1:, :]
        fuse_feature = fuse_feature.unsqueeze(0)

        multimodal_input = torch.cat([fuse_feature, tmp_seq_emb], dim=1)
        # multimodal_input = multimodal_input.squeeze(0)

        output = self.Biogpt(inputs_embeds=multimodal_input, labels=out_seq['input_ids'].squeeze(1))

        return output, A
    
class checklist_RAG(nn.Module):
    def __init__(self):
        super(checklist_RAG, self).__init__()  
        self.Biobert_model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1").to(device)
        # self.Biogpt_model = BioGptForCausalLM.from_pretrained('microsoft/biogpt').to(device)
        # self.Biogpt_model.load_state_dict(torch.load(f'./models/pretrain_lm_biogpt.pth'))
        # self.finding_embedding = self.Biogpt_model.get_input_embeddings()
        self.checklist_proj = nn.Linear(768, 512)
        self.finding_proj = nn.Linear(768, 512)
        # self.finding_proj = nn.Linear(1024, 512)

    def forward(self, checklist_input_ids, finding_input_ids):
        checklist_output = self.Biobert_model(input_ids=checklist_input_ids['input_ids'].squeeze(1), 
                                           attention_mask=checklist_input_ids['attention_mask'].squeeze(1))
        checklist_vec = checklist_output.last_hidden_state[:, 0, :]
        # checklist_vec = self.checklist_proj(checklist_vec.mean(dim=1))

        finding_output = self.Biobert_model(input_ids=finding_input_ids['input_ids'].squeeze(1), 
                                           attention_mask=finding_input_ids['attention_mask'].squeeze(1))
        finding_vec = finding_output.last_hidden_state[:, 0, :]

        # tmp_seq = finding_input_ids['input_ids'].squeeze(1).to(device)
        # outputs = self.Biogpt_model(tmp_seq)
        
        # finding_vec = torch.softmax(outputs.logits, dim=-1)
        # finding_vec = torch.argmax(finding_vec, dim=-1)
        # finding_vec = self.finding_embedding(finding_vec)
        # finding_vec = self.finding_proj(finding_vec.mean(dim=1))
        # print(checklist_vec.shape)
        # print(finding_vec.shape)

        return checklist_vec, finding_vec


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('GPU state:',device)
    # model = MIL_AttnFeatureExtractor().to(device)
    # summary(model, [['F:/pathology_patches_normalized\\13-007296\\10304_13216.tif'], ['F:/pathology_patches_normalized\\13-007296\\10304_13440.tif'], ['F:/pathology_patches_normalized\\13-007296\\10304_13664.tif'], ['F:/pathology_patches_normalized\\13-007296\\10304_13888.tif'], ['F:/pathology_patches_normalized\\13-007296\\10304_14112.tif'], ['F:/pathology_patches_normalized\\13-007296\\10304_15232.tif']])
    # model = MIL_GPT2(batch_size=128)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # 设置 padding token
    seq = tokenizer("The transverse colon shows a moderately differentiated adenocarcinoma invading through muscularis propria into subserosal fat.", return_tensors="pt", padding='max_length', max_length=256)['input_ids']
    # summary(model, [['F:/pathology_patches_normalized\\13-007296\\10304_13216.tif'], ['F:/pathology_patches_normalized\\13-007296\\10304_13440.tif'], ['F:/pathology_patches_normalized\\13-007296\\10304_13664.tif'], ['F:/pathology_patches_normalized\\13-007296\\10304_13888.tif'], ['F:/pathology_patches_normalized\\13-007296\\10304_14112.tif'], ['F:/pathology_patches_normalized\\13-007296\\10304_15232.tif']], seq.to(device))