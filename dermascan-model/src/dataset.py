"""
PyTorch Dataset para classificação de lesões de pele (ISIC).
"""

import logging
import os

import pandas as pd
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

# Resolução nativa do EfficientNet-B3 (B0 usa 224 — treinar B3 em 224
# sub-aproveita o backbone pré-treinado, que foi calibrado pra 300).
IMAGE_SIZE = 300
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(image_size: int = IMAGE_SIZE, train: bool = False) -> transforms.Compose:
    """Transform padrão do pipeline — use a mesma função pra treino/val/teste
    e pra pré-processar imagens na API real, garantindo que todo mundo
    normaliza a imagem exatamente como no treino.

    `Resize(image_size)` (int, não tupla) redimensiona mantendo a proporção
    original — o lado menor vira `image_size` — e o Crop seguinte corta um
    quadrado. Um `Resize((size, size))` direto estica a imagem, o que é
    clinicamente relevante aqui: assimetria e forma da borda são critérios
    diagnósticos do ABCDE de melanoma, e distorcer a proporção antes da
    inferência pode alterar esses sinais.
    """
    if train:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            # Só brilho/contraste: matiz e saturação são sinal diagnóstico em
            # lesões pigmentadas, então augmentation de cor agressiva iria
            # destruir informação clínica em vez de só variar o "estilo" da foto.
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class SkinLesionDataset(Dataset):
    """Dataset PyTorch para imagens de lesões de pele"""

    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        transform=None,
        classes: list[str] | None = None,
        validate_files: bool = True,
    ):
        """
        Args:
            df: DataFrame com colunas 'image_id' e 'dx' (diagnóstico)
            img_dir: diretório com as imagens
            transform: transforms da torchvision
            classes: lista fixa de classes (nome -> índice = posição na lista).
                PASSE SEMPRE O MESMO `classes` (computado a partir do dataset
                completo, antes do split) pros datasets de treino/val/teste.
                Se cada split calculasse seu próprio `sorted(df['dx'].unique())`,
                um split que por azar não contivesse alguma classe rara
                (ex.: DF ou VASC têm <1% do ISIC 2019) desalinharia os
                índices entre treino e validação — silenciosamente, sem
                nenhum erro, e sem garantia de que o índice de MEL é o mesmo
                nos dois. Se `classes` não for passado, cai para o
                comportamento antigo (calcula a partir do próprio `df`), útil
                só para uso isolado/notebooks exploratórios.
            validate_files: se True, remove do dataset (com aviso via log)
                linhas cujo arquivo de imagem não existe em `img_dir` — evita
                que o treino inteiro caia horas depois por causa de um
                arquivo ausente no meio de um epoch.
        """
        self.img_dir = img_dir
        self.transform = transform or build_transform(train=False)

        if classes is None:
            classes = sorted(df['dx'].unique())
        self.classes = list(classes)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        unknown = set(df['dx'].unique()) - set(self.classes)
        if unknown:
            raise ValueError(
                f"O DataFrame contém classes fora do conjunto esperado: {sorted(unknown)}. "
                "Passe `classes=` explicitamente, derivado do dataset completo antes do split."
            )

        df = df.reset_index(drop=True)
        if validate_files:
            exists_mask = df['image_id'].apply(
                lambda image_id: os.path.exists(os.path.join(img_dir, f"{image_id}.jpg"))
            )
            missing = int((~exists_mask).sum())
            if missing:
                logger.warning(
                    "%d de %d imagens listadas não foram encontradas em %s e foram ignoradas.",
                    missing, len(df), img_dir,
                )
            df = df[exists_mask].reset_index(drop=True)

        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['image_id']}.jpg")

        try:
            image = Image.open(img_path).convert('RGB')
        except (OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(
                f"Falha ao abrir a imagem '{img_path}' (linha {idx} do dataset). "
                "O arquivo pode estar corrompido — considere removê-lo do CSV."
            ) from exc

        label = self.class_to_idx[row['dx']]

        if self.transform:
            image = self.transform(image)

        return image, label

    @property
    def num_classes(self):
        return len(self.classes)
