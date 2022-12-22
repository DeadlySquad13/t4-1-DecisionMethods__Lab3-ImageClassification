# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
# ---



# %% [markdown]
# # Лабораторная работа № 3 
# ## Задание
# Необходимо познакомиться с фреймворком машинного обучения PyTorch и выполнить три задания:
# 
# 1. Обучить полносвязную нейронную сеть классификации 3 классов изображений из набора данных CIFAR100 по варианту с точностью на тестовой выборке не менее 70%.
# Для задания нужно сформировать свою подвыборку CIFAR100 по варианту.
# 2. Преобразовать модель в ONNX и сохранить локально.
# 3. Протестировать обученную модель на своих изображениях.
#   - Скачать каталог с html-файлом и встроить в него файл модели, обученной на ЛР.
#   - Скачать картинки из интернета согласно варианту и открыть их в html по кнопке. Автоматически в скрипте масштабируется изображение.
#   - Выбрать нужные классы для готовой модели. Проверить на устойчивость полносвязную модель, двигая картинку.
# 
# ### Варианты для Задания
# Вы должны использовать следующие классы из CIFAR100:
# 
# Номер группы

# %%
GROUP = 6

# %% [markdown]
# Номер варианта
# Номер варианта + 30

# %%
VARIANT = 10

# %%
LABELS = [GROUP, VARIANT, VARIANT + 30]

# %%
from torchvision.datasets import CIFAR100

# %% [markdown]
# ## Загружаем тестовую и обучающую выборку CIFAR100

# %%
from torchvision.transforms import ToTensor

dataset_settings = {
    'root': 'data/cifar100',
    'transform': ToTensor(),
    'download': True,
}

train_data = CIFAR100(train=True, **dataset_settings)

test_data = CIFAR100(train=False, **dataset_settings)


# %% [markdown]
# Формируем подвыборку согласно варианту.

# %%
import torch
import numpy as np
from torch.utils.data import Subset, TensorDataset

def make_subset(dataset: TensorDataset, labels: list[np.integer]) -> Subset:
    """
    Создает подвыборку с задаными метками класса.

    :param dataset: Набор данных.
    :type dataset: TensorDataset (или любой другой тип датасета с совместимым форматом)
    :param labels: Метки класса.
    :type labels: list[np.integer]
    :return: Подвыборка.
    """
    mask = torch.tensor(np.isin(dataset.targets, labels), dtype=torch.bool)
    indices = mask.nonzero().reshape(-1)
    subset = Subset(dataset, indices)

    return subset

train_subset = make_subset(train_data, LABELS)

test_subset = make_subset(test_data, LABELS)

# %% [markdown]
# Отобразим полученные подвыборки

# %%
import matplotlib.pyplot as plt
ROWS = 3
COLS = 3

_, ax = plt.subplots(ROWS, COLS, figsize=(12, 12))

for i in range(9):
    img, label = train_subset[i]

    row_index = int(i / ROWS)
    col_index = i % COLS
    current_plt_cell = ax[row_index, col_index]

    current_plt_cell.imshow(img.permute(1,2,0)) # CHW -> HWC
    current_plt_cell.set_title(train_data.classes[label])
    current_plt_cell.axis('off')

# %% [markdown]
# ## Предобработка датасетов
# Трансформируем исходные названия меток классов на более удобные 0, 1, 2.

# %%
from functools import reduce 

def transform_label(label):
    """
    Map initial labels ([6, 10, 40] for example) to simple labels [0, 1, 2].
    """
    for i, initial_label in enumerate(LABELS):
        if label == initial_label:
            return i

    raise ValueError(f'No such label as {label} in ${LABELS}')

def map_labels(labels):
    return map(transform_label, labels)

def compose(f, g):
        return lambda x : f(g(x))

def create_masked_dataset(subset: Subset, dataset: TensorDataset, target_transforms=[lambda x: x]) -> TensorDataset:
    features = [dataset[i][0].numpy() for i in subset.indices]
    tensor_features = torch.tensor(features)

    targets = [dataset[i][1] for i in subset.indices]

    # Reducing array of transform functions to a compose function.
    # :example: [f, g] -> lambda x: f(g(x))
    composed_target_transforms = reduce(compose, target_transforms, lambda x: x)
    # TODO: make proper transform instead selfmade map.
    tensor_targets = torch.tensor([composed_target_transforms(target) for target in targets])

    return TensorDataset(tensor_features, tensor_targets)

train_masked_dataset = create_masked_dataset(train_subset, train_data, [transform_label])
test_masked_dataset = create_masked_dataset(test_subset, test_data, [transform_label])

# %% [markdown]

# %% Create device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


