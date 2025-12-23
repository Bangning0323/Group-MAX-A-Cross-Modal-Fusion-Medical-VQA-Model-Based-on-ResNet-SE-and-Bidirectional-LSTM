import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
from config import FIXED_CONFIG

# Attention Module
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=FIXED_CONFIG['se_reduction']):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# Image encoder
class ResNetBaseline(nn.Module):
    """ResNet50"""
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # [B, 2048, 7, 7]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # [B, 2048]
        return x

class ResNetSE(nn.Module):
    """ResNet50+SE"""
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # [B, 2048, 7, 7]
        self.se_block = SEBlock(2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.se_block(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # [B, 2048]
        return x

# Text encoder
class TextEncoderUniLSTM(nn.Module):
    """UniLSTM"""
    def __init__(self, embedding_matrix, max_length=FIXED_CONFIG['max_length']):
        super().__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=True)
        self.lstm = nn.LSTM(embed_dim, 512, batch_first=True, bidirectional=False)
        self.text_proj = nn.Linear(512, FIXED_CONFIG['text_proj_dim'])

    def forward(self, questions):
        text_emb = self.embedding(questions)  # [B, L, 300]
        lstm_out, (h_n, _) = self.lstm(text_emb)  # h_n: [1, B, 512]
        text_feat = h_n.squeeze(0)  # [B, 512]
        text_feat = self.text_proj(text_feat)  # [B, 300]
        return text_feat

class TextEncoderBiLSTM(nn.Module):
    """BiLSTM"""
    def __init__(self, embedding_matrix, max_length=FIXED_CONFIG['max_length']):
        super().__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            FIXED_CONFIG['lstm_hidden_dim'],
            batch_first=True,
            bidirectional=True,
            num_layers=FIXED_CONFIG['lstm_num_layers']
        )
        self.text_proj = nn.Linear(FIXED_CONFIG['lstm_hidden_dim'] * 2, FIXED_CONFIG['text_proj_dim'])
        self.dropout = nn.Dropout(FIXED_CONFIG['dropout_rate'])

    def forward(self, questions):
        text_emb = self.embedding(questions)
        text_emb = self.layer_norm(text_emb)
        text_emb = self.dropout(text_emb)
        lstm_out, (h_n, _) = self.lstm(text_emb)
        text_feat = torch.cat([h_n[-2], h_n[-1]], dim=1)
        text_feat = self.text_proj(text_feat)
        return text_feat

class TextEncoderTextCNN(nn.Module):
    """TextCNN"""
    def __init__(self, embedding_matrix, max_length=FIXED_CONFIG['max_length']):
        super().__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=True)

        self.convs = nn.ModuleList([
            nn.Conv2d(1, FIXED_CONFIG['cnn_num_filters'], (k, embed_dim), padding=(k - 1, 0))
            for k in FIXED_CONFIG['cnn_filter_sizes']
        ])
        self.relu = nn.ReLU()
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        total_filters = FIXED_CONFIG['cnn_num_filters'] * len(FIXED_CONFIG['cnn_filter_sizes'])
        self.text_proj = nn.Linear(total_filters, FIXED_CONFIG['text_proj_dim'])

    def forward(self, questions):
        text_emb = self.embedding(questions)  # [B, L, 300]
        text_emb = text_emb.unsqueeze(1)  # [B, 1, L, 300]
        conv_outputs = [self.relu(conv(text_emb)).squeeze(-1) for conv in self.convs]  # [B, 128, L] * 3
        pool_outputs = [self.max_pool(out).squeeze(-1) for out in conv_outputs]  # [B, 128] * 3
        text_feat = torch.cat(pool_outputs, dim=1)  # [B, 384]
        text_feat = self.text_proj(text_feat)  # [B, 300]
        return text_feat


class ModelBaseline1(nn.Module):
    """ResNet50 + UniLSTM"""
    def __init__(self, embedding_matrix, num_classes):
        super().__init__()
        self.image_encoder = ResNetBaseline(pretrained=True)
        self.text_encoder = TextEncoderUniLSTM(embedding_matrix)
        self.fusion = nn.Sequential(
            nn.Linear(2048 + FIXED_CONFIG['text_proj_dim'], 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, questions):
        img_feat = self.image_encoder(images)  # [B, 2048]
        text_feat = self.text_encoder(questions)  # [B, 300]
        fused = torch.cat([img_feat, text_feat], dim=1)
        return self.fusion(fused)

class ModelBaseline2(nn.Module):
    """ResNet50 + TextCNN"""
    def __init__(self, embedding_matrix, num_classes):
        super().__init__()
        self.image_encoder = ResNetBaseline(pretrained=True)
        self.text_encoder = TextEncoderTextCNN(embedding_matrix)
        self.fusion = nn.Sequential(
            nn.Linear(2048 + FIXED_CONFIG['text_proj_dim'], 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, questions):
        img_feat = self.image_encoder(images)  # [B, 2048]
        text_feat = self.text_encoder(questions)  # [B, 300]
        fused = torch.cat([img_feat, text_feat], dim=1)
        return self.fusion(fused)

class ModelProposed(nn.Module):
    """ResNet-SE + BiLSTM"""
    def __init__(self, embedding_matrix, num_classes):
        super().__init__()
        self.image_encoder = ResNetSE(pretrained=True)
        self.text_encoder = TextEncoderBiLSTM(embedding_matrix)
        self.fusion = nn.Sequential(
            nn.Linear(2048 + FIXED_CONFIG['text_proj_dim'], 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, questions):
        img_feat = self.image_encoder(images)  # [B, 2048]
        text_feat = self.text_encoder(questions)  # [B, 300]
        fused = torch.cat([img_feat, text_feat], dim=1)
        return self.fusion(fused)