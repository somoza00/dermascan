"""
Avaliação e exportação do modelo treinado.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png', normalize=False):
    """Matriz de confusão.

    `normalize=True` mostra proporções por linha (recall) em vez de
    contagens absolutas — com NV dominando o ISIC 2019, erros em classes
    raras (MEL incluída) ficam visualmente escondidos numa matriz de
    contagens brutas.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        with np.errstate(invalid='ignore', divide='ignore'):
            cm = np.nan_to_num(cm.astype(float) / cm.sum(axis=1, keepdims=True))
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusão' + (' (normalizada por linha)' if normalize else ''))
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Matriz salva em: {save_path}")


def print_metrics(y_true, y_pred, class_names, critical_class='MEL'):
    """Printa classification report e destaca o recall da classe crítica —
    em triagem de risco, um falso negativo nela (dizer que é outra coisa
    quando é MEL) é o pior cenário possível, bem pior que uma métrica
    agregada baixa."""
    print("\n=== Relatório de Classificação ===\n")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    if critical_class in class_names:
        report = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0,
        )
        recall = report[critical_class]['recall']
        print(f"⚠️  Recall de {critical_class}: {recall:.4f} "
              f"({'ATENÇÃO: abaixo de 0.90' if recall < 0.90 else 'ok'})")


def export_model(model, classes, image_size=IMAGE_SIZE, save_path='models/dermascan_v1.pt'):
    """Exporta o checkpoint com tudo que a API precisa pra reconstruir o
    pipeline de inferência sem adivinhar nada: pesos, ordem das classes
    (crítico — define qual índice de saída é MEL), resolução de entrada e
    estatísticas de normalização usadas no treino. Sem isso, quem
    implementar `RealInferenceService` na API teria que descobrir esse
    contrato lendo o código de treino.
    """
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    model.eval()
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'classes': list(classes),
            'num_classes': len(classes),
            'image_size': image_size,
            'normalize_mean': IMAGENET_MEAN,
            'normalize_std': IMAGENET_STD,
        },
        save_path,
    )
    print(f"Modelo exportado: {save_path}")
    print(f"  Classes (ordem = índice de saída): {list(classes)}")
    print(f"  Resolução de entrada: {image_size}x{image_size}")


def export_torchscript(model, example_input, save_path='models/dermascan_v1_ts.pt'):
    """Exporta como TorchScript (não precisa do código Python pra carregar).

    Atenção: TorchScript NÃO carrega o dicionário de metadados salvo por
    `export_model` (classes/image_size/normalização) — se optar por esse
    formato na API, transporte esses metadados separadamente (ex.: um JSON
    ao lado do .pt).
    """
    model.eval()
    traced = torch.jit.trace(model, example_input)
    traced.save(save_path)
    print(f"TorchScript exportado: {save_path}")
