import pandas as pd

from src.dataset import SkinLesionDataset


def test_class_to_idx_is_stable_across_splits_with_explicit_classes(tmp_path):
    """Reproduz o cenário do bug real: um split (val) não contém uma classe
    rara que o outro (train) contém. Sem `classes=` explícito, cada dataset
    calcularia seu próprio mapeamento e os índices divergiriam
    silenciosamente. Com `classes=` fixo, os dois batem sempre."""
    full_df = pd.DataFrame({
        "image_id": [f"img{i}" for i in range(6)],
        "dx": ["MEL", "NV", "NV", "NV", "DF", "NV"],
    })
    classes = sorted(full_df["dx"].unique())  # calculado uma vez, do df completo

    train_df = full_df.iloc[[0, 1, 2, 3]]  # contém MEL, NV, DF
    val_df = full_df.iloc[[4, 5]]          # também contém DF e NV, sem MEL — mas classes é fixo

    # validate_files=False: não há arquivos de verdade em tmp_path, só testando o mapeamento
    train_dataset = SkinLesionDataset(train_df, str(tmp_path), classes=classes, validate_files=False)
    val_dataset = SkinLesionDataset(val_df, str(tmp_path), classes=classes, validate_files=False)

    assert train_dataset.class_to_idx == val_dataset.class_to_idx
    assert train_dataset.class_to_idx["MEL"] == val_dataset.class_to_idx["MEL"]


def test_rejects_dataframe_with_class_outside_fixed_list(tmp_path):
    df = pd.DataFrame({"image_id": ["img0"], "dx": ["UNKNOWN_CLASS"]})

    try:
        SkinLesionDataset(df, str(tmp_path), classes=["MEL", "NV"], validate_files=False)
        assert False, "deveria ter levantado ValueError"
    except ValueError:
        pass


def test_missing_files_are_dropped_with_validate_files(tmp_path):
    df = pd.DataFrame({"image_id": ["does_not_exist"], "dx": ["NV"]})

    dataset = SkinLesionDataset(df, str(tmp_path), classes=["NV"], validate_files=True)

    assert len(dataset) == 0
