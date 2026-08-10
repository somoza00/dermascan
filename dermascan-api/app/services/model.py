"""
Arquitetura do modelo — espelho de `dermascan-model/src/model.py`.

POR QUE DUPLICADO: a API é deployada como pacote separado (Railway) e, em
runtime, não existe `dermascan-model` no filesystem para importar. O
checkpoint exportado por `export_model()` contém apenas o `state_dict` +
metadados, então a arquitetura precisa existir neste pacote para
reconstruí-la na carga. MANTENHA EM SINCRONIA com
`dermascan-model/src/model.py` (mesmo head, mesmos shapes) — qualquer
mudança lá exige a mesma mudança aqui.

Usamos `EfficientNet.from_name` (sem pesos ImageNet) porque o `state_dict`
do checkpoint treinado sobrescreve TODOS os parâmetros — `from_pretrained`
só adicionaria um download de ~40MB desnecessário no deploy.
"""

import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class DermaScanModel(nn.Module):
    """EfficientNet-B3 com cabeçalho customizado — idêntico ao do treino."""

    def __init__(self, num_classes: int = 8):
        # 8 = classes diagnósticas do ISIC 2019 (AK, BCC, BKL, DF, MEL, NV,
        # SCC, VASC). Não confundir com HAM10000, que tem 7 (sem SCC).
        super().__init__()
        self.backbone = EfficientNet.from_name('efficientnet-b3')

        # Número de features da última camada do B3 é 1536
        n_features = self.backbone._fc.in_features

        # Substitui o classificador pelo mesmo head customizado do treino
        # (Dropout -> Linear 1536->512 -> ReLU -> Dropout -> Linear 512->N)
        self.backbone._fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)
