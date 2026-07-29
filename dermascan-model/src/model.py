"""
Definição do modelo EfficientNet-B3 para classificação de lesões de pele.
"""

import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class DermaScanModel(nn.Module):
    """
    EfficientNet-B3 com cabeçalho customizado para classificação.
    Usa weights pré-treinados no ImageNet e substitui o classificador.
    """

    def __init__(self, num_classes: int = 8):
        # 8 = classes diagnósticas do ISIC 2019 (AK, BCC, BKL, DF, MEL, NV,
        # SCC, VASC). Não confundir com HAM10000, que tem 7 (sem SCC).
        super().__init__()
        # Carrega EfficientNet-B3 pré-treinado
        self.backbone = EfficientNet.from_pretrained('efficientnet-b3')

        # Congela early layers (opcional — fazemos fine-tuning total)
        # Descongela tudo pra fine-tuning completo
        for param in self.backbone.parameters():
            param.requires_grad = True

        # Número de features da última camada do B3 é 1536
        n_features = self.backbone._fc.in_features

        # Substitui o classificador por um customizado
        self.backbone._fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)
