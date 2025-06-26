## Нейросетевые методы машинного обучения

В качестве основных нейросетевых моделей были выбраны предобученные трансформерные языковые модели с платформы Hugging
Face. Для токенизации и дообучения использовались классы AutoTokenizer и AutoModelForSequenceClassification из
библиотеки transformers. Из библиотеки PyTorch для формирования тензоров и подачи их в модели были задействованы классы
TensorDataset и DataLoader.
Для корректной работы с каждой моделью были использованы функции, описанные ниже. Все нейросетевые модели обучались на
платформе Kaggle или Colab, на
предоставляемых в пользование GPU мощностях.

### Создание датасетов

Для создания обучающих и валидационных датасетов была написана функция make_dataset.

```python
def make_dataset(texts, labels):
    '''
    Токенизация текстов и сопоставление токенов с идентификаторами
    соответствующих им слов. Формирование PyTorch датасета
    '''
    input_ids = []        # Список для токенизированных текстов
    attention_masks = []  # Список для масок механизма внимания
    
    # Цикл проходится и токенизирует каждый текст
    for seq_to_token in texts:
        encoded_dict = tokenizer.encode_plus(
            seq_to_token,                # Последовательность для токенизации
            add_special_tokens=True,     # Добавить специальные токены в начало и в конец посл-ти
            max_length=338,              # Максимальная длина последовательности
            padding='max_length',        # Токен для заполнения до максимальной длины
            return_attention_mask=True,  # Маска механизма внимания для указания на паддинги
            return_tensors = 'pt',       # Возвращать pytorch-тензоры
            truncation=True              # Обрезать последовательность до максимальной длины
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    
    # Конкатенация входных данных в тензоры
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    # Преобразование таргетов в тензоры
    labels = torch.tensor(labels.values)
    # Формирование датасета
    dataset = TensorDataset(input_ids, attention_masks, labels)

    return dataset
```

### Функции для обучения и валидации

Для обучения и валидации моделей были использованы функции train и validate соответственно.

```python
def train(epoch):
    '''
    Обучение модели на одной эпохе
    '''
    print(f'Epoch {epoch+1}')
    model.train()           
    fin_targets = []        # Список для всех таргетов обучающей выборки
    fin_outputs = []        # Список для всех предиктов модели на обучающей выборки
    total_train_loss = 0    # Функция потерь на обучении
    
    # Цикл проходится по батчам из обучающей выборки
    for data in train_dataloader:
        ids = data[0].to(device, dtype=torch.long)              # Токены последовательностей из батча
        mask = data[1].to(device, dtype=torch.long)             # Маски механизма внимания последовательностей
        targets = data[2].to(device, dtype=torch.float)         # Таргеты из батча
        
        res = model(ids, attention_mask=mask, labels=targets)   # В модель подаются входные тензоры и таргеты
        loss = res['loss']                                      # Вычисляется значение функции потерь
        logits = res['logits']                                  # Логиты предсказаний модели
        total_train_loss += loss.item()                         # Считается функция потерь
        
        # Таргеты и выходы модели по батчу добавляются в списки. Логиты проходят через сигмоиду
        fin_targets.extend(targets.cpu().detach().numpy().tolist())
        fin_outputs.extend(torch.sigmoid(logits).cpu().detach().numpy().tolist())
        
        optimizer.zero_grad()                                   # Зануляются градиенты параметров модели
        loss.backward()                                         # По функции потерь рассчитываются градиенты
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Масштабируются градиенты
        optimizer.step()                                        

    
    fin_targets = np.array(fin_targets)
    fin_outputs = np.array(fin_outputs)
    predictions = np.zeros(fin_outputs.shape)
    predictions[np.where(fin_outputs >= 0.5)] = 1
    
    return total_train_loss / len(train_dataloader), fin_targets, predictions


def validate():
    '''
    Валидация обученной модели на тестовой выборке
    '''
    print(f'Validation')
    model.eval()             
    fin_targets = []         # Список для всех таргетов валидационной выборки
    fin_outputs = []         # Список для всех предиктов модели на валидационной выборки
    total_test_loss = 0.0    # Функция потерь на валидации
    
    with torch.no_grad():
        # Без подсчета градиентов цикл проходится по батчам
        for data in test_dataloader:
            ids = data[0].to(device, dtype=torch.long)            # Токены последовательностей из батча
            mask = data[1].to(device, dtype=torch.long)           # Маски механизма внимания последовательностей
            targets = data[2].to(device, dtype=torch.float)       # Таргеты из батча
                
            res = model(ids, attention_mask=mask, labels=targets) # В модель подаются входные тензоры и таргеты
            loss = res['loss']                                    # Вычисляется значение функции потерь
            logits = res['logits']                                # Логиты предсказаний модели
            total_test_loss += loss.item()                        # Считается функция потерь
            
            # Таргеты и выходы модели по батчу добавляются в списки. Логиты проходят через сигмоиду
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(logits).cpu().detach().numpy().tolist())
    
    fin_targets = np.array(fin_targets)
    fin_outputs = np.array(fin_outputs)
    predictions = np.zeros(fin_outputs.shape)
    predictions[np.where(fin_outputs >= 0.5)] = 1
    
    return total_test_loss / len(test_dataloader), fin_targets, predictions
```


### Функции логирования и отображения метрик

Для логирования метрик был использован SummaryWriter из библиотеки PyTorch. Для реализации этого была использована функция log_metrics. Функция log_metrics отвечает за наполнение истории для отображения обновляющихся графиков значений метрик в процессе обучения, что осуществлялось с помощью функции plot_learning_curves.

```python
def log_metrics(history, writer, loss, targets, outputs, postfix):
    '''
    Расчет значений метрик и добавление их в лог обучения для отрисовки графиков.
    Добавление значений метрик в историю обучения для отрисовки временных графиков
    '''
    metrics_dict = {
        'Loss': loss,
        'Accuracy': metrics.accuracy_score(targets, outputs),
        'Hamming_distance': hamming_distance(targets, outputs),
        'F1_micro': metrics.f1_score(targets, outputs, average='micro'),
        'F1_macro': metrics.f1_score(targets, outputs, average='macro'),
        'Recall_micro': metrics.recall_score(targets, outputs, average='micro'),
        'Recall_macro': metrics.recall_score(targets, outputs, average='macro'),
        'Precision_micro': metrics.precision_score(targets, outputs, average='micro', zero_division=0.0),
        'Precision_macro': metrics.precision_score(targets, outputs, average='macro', zero_division=0.0)
    }
    
    for metric, value in metrics_dict.items():
        if not 'macro' in metric:
            history[metric][postfix].append(value)
        writer.add_scalar(f'{metric}/{postfix}', value, epoch)

def plot_learning_curves(history):
    '''
    Отрисовка обновляющихся графиков значений метрик,
    для отслеживания в процессе обучения
    '''
    fig = plt.figure(figsize=(20, 10))
    
    for i, metric in enumerate(history.keys(), 1):
        plt.subplot(2,3,i)
        plt.title(metric, fontsize=15)
        plt.plot(range(1, epoch+2), history[metric]['train'], label='train')
        plt.plot(range(1, epoch+2), history[metric]['val'], label='val')
        plt.xticks(range(1, epoch+2))
        if i > 3:
            plt.xlabel('epoch', fontsize=15)
        plt.legend()

    plt.show()
```

### Инициализация

Функцией потерь для обучения всех нейросетевых моделей был выбран класс BCEWithLogitsLoss из библиотеки PyTorch. Инициализация гиперпараметров, моделей и служебных функций происходила по следующему алгоритму.

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

selected_model = 'cointegrated/rubert-tiny2'
tokenizer = AutoTokenizer.from_pretrained(selected_model)

BATCH_SIZE = 32
EPOCHS = 20
EARLY_STOP = 3
OPT = torch.optim.NAdam
LEARNING_RATE = 3e-5
EPSILON = 1e-8
SCHEDULER = False
SAMPLE = False
```

```python
# Инициализируется предобученная модель
model = AutoModelForSequenceClassification.from_pretrained(
    selected_model,
    problem_type='multi_label_classification', # Решается задача многоклассовой классификации. Функция потерь BCEWithLogitsLoss
    num_labels=y_test.shape[1],                # Число классов
    output_attentions = False,                 # Модель не выдает результаты работы механизма внимания
    output_hidden_states = False               # Модель не выдает скрытые состояния
)
model.to(device)

# Инициализируется оптимизатор
optimizer = OPT(
    model.parameters(),
    lr=LEARNING_RATE,
    eps=EPSILON
)

# Инициализируется шедулер
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_dataloader) * EPOCHS
)

# Инициализируется инструмент логирования
writer = SummaryWriter(
    comment= '-' + selected_model.replace('/', '-')
)
```

### Обучение модели
В ходе обучения была задействован метод ранней остановки. Метрикой для ранней остановки была выбрана F1-measure. Обучение моделей происходило по следующему алгоритму:

```python
# Переменная для хранения лучшей метрики для ранней остановки
best_f1_val = 0
# Переменная для отсчета количества эпох с момента лучшей метрики
epochs_since_best = 0
# Словарь для хранения истории метрик
history = defaultdict(lambda: defaultdict(list))

for epoch in range(EPOCHS):

    avg_train_loss, targets, outputs = train(epoch)                         # Обучение модели на одной эпохе
    log_metrics(history, writer, avg_train_loss, targets, outputs, 'train') # Логирование метрик

    avg_val_loss, targets, outputs = validate()                             # Предсказания модели на вал. выборке
    log_metrics(history, writer, avg_val_loss, targets, outputs, 'val')     # Логирование метрик

    clear_output()
    # Отрисовка кривых обучения
    plot_learning_curves(history)
    torch.save(model, model_path + 'models_files\\' + f"{selected_model.replace('/', '-')}_{epoch}.pt")
    # Расчет micro f1-score на валидационной выборке
    f1_val = metrics.f1_score(targets, outputs, average='micro')
    # Если метрика лучше предыдущей лучшей, то сохраняется модель
    if f1_val > best_f1_val:
        best_f1_val = f1_val
        torch.save(model, model_path + 'models_files\\' + f"{selected_model.replace('/', '-')}_{epoch}.pt")
        epochs_since_best = 0
    # В противном случае идет отсчет эпох до ранней остановки
    else:
        epochs_since_best += 1
    print('Best epoch:', epoch, '\nBest F1-score:',best_f1_val, '\n')
    if epochs_since_best == EARLY_STOP:
        break

writer.flush()
writer.close()
```

### Испытания

Для подбора гиперпараметров каждой модели испытания проводились на 25% данных от обучающей выборки. Скорость обучения
выбиралась из диапазона 2*10-5 – 4*10-5. В ходе испытаний рассматривались оптимизаторы Adam, NAdam, AdamW библиотеки
Pytorch. Размер батча выбирался по характеристикам моделей и изменялся в диапазоне от 8 до 128. Использование шедулера
не дало ожидаемого результата, отчего из дальнейшего исследования он был исключен.
