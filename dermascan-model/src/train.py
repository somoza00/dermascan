"""
Loop de treinamento com Early Stopping e métricas voltadas para o
desbalanceamento do ISIC 2019 — val_loss/accuracy sozinhos escondem recall
baixo em classes raras (MEL incluída), porque NV domina o dataset.
"""

import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score
from tqdm import tqdm


def compute_class_weights(class_counts: dict, classes: list[str], device) -> torch.Tensor:
    """Pesos inversamente proporcionais à frequência de cada classe, para
    `CrossEntropyLoss(weight=...)`. Sem isso, o modelo tende a "apostar" em
    NV (majoritária e benigna) nos casos ambíguos — o viés mais perigoso
    possível aqui, porque reduz recall justamente das classes raras e
    malignas (MEL, SCC, BCC).

    Args:
        class_counts: {nome_da_classe: quantidade_de_amostras_no_treino}
        classes: mesma lista/ordem usada em `SkinLesionDataset.classes`
    """
    counts = np.array([class_counts[c] for c in classes], dtype=np.float64)
    weights = counts.sum() / (len(classes) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Um epoch de treino"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Train')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})

    return total_loss / len(loader), 100. * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Avaliação completa (validação ou teste)"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc='Val'):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (
        total_loss / len(loader),
        100. * correct / total,
        np.array(all_preds),
        np.array(all_labels),
    )


def clinical_metrics(preds, labels, classes: list[str], critical_class: str = 'MEL') -> dict:
    """Macro-F1 (robusto a desbalanceamento, ao contrário de accuracy) +
    recall isolado da classe crítica — a métrica que mais importa num app de
    triagem: quantos casos reais dessa classe o modelo está deixando passar
    como outra coisa. Use o retorno pra decidir qual checkpoint é "o
    melhor", não `val_loss`/accuracy sozinhos.
    """
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    per_class_recall = recall_score(labels, preds, average=None, zero_division=0, labels=range(len(classes)))
    metrics = {
        'macro_f1': float(macro_f1),
        'per_class_recall': dict(zip(classes, (float(r) for r in per_class_recall))),
    }
    if critical_class in classes:
        metrics['critical_recall'] = metrics['per_class_recall'][critical_class]
    return metrics


class EarlyStopping:
    """Interrompe o treino se a métrica de seleção não melhorar por
    `patience` epochs.

    `mode='min'` (padrão, compatível com o uso antigo baseado em val_loss)
    ou `mode='max'` para métricas tipo macro-F1, onde maior é melhor —
    recomendado aqui, já que val_loss sozinho não reflete recall em dataset
    desbalanceado.
    """

    def __init__(self, patience=5, min_delta=0.001, mode='min'):
        if mode not in ('min', 'max'):
            raise ValueError("mode deve ser 'min' ou 'max'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, score):
        improved = (
            score < self.best_score - self.min_delta
            if self.mode == 'min'
            else score > self.best_score + self.min_delta
        )
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
