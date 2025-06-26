# Обработка данных
## Датасет для исследования 

Так как размер исходного датасета и количество уникальных записей достаточно велики, было принято решение создать датасет на
основании исходного с меньшим количеством записей для ускорения обработки и обучения моделей.

Для этого были выполнены следующие действия: 

1. Создание списка категорий (категория "69-я параллель" исключена из исследования) с количеством записей **больше 50**. 

```python
def get_tags_count(engine, num=50):
    # получение списка тегов, где количество записей больше 50
    
    if num > 0:
        query = f'''
        select tags, count(id) from news_table nt 
        group by tags
        having count(id) > {int(num)}
        order by count(id) desc
        '''
        df_tags = pd.read_sql_query(query, con = engine)
    return df_tags
```


**Количество уникальных тегов после фильтрации:** 89

2. Выбор 1000 уникальных строк по каждому тегу с сортировкой по полю "id" по возрастанию. 

```python
# проход по полученному списку тегов и выбор первых 1000 строк по каждому тегу при сортировке по id
sum_tags = 0
for tag in df_tags.tags:
    query = f'''
    select * from news_table nt 
    where tags = '{tag}'
    order by id
    limit 1000
    '''
    # сохранение данных по каждому тегу в датафрейм
    df_tag_for_save = pd.read_sql_query(query, con = engine)
    len_df = len(df_tag_for_save)
    sum_tags += len_df
    # сохранение полученного датафрейма по каждому тегу в csv файл с указанием количества строк
    df_tag_for_save.to_csv(f'D:\\netology_diplom\\final\\csv_files\\data_for_analyze_by_tags\\{tag}_{len_df}.csv', index = 
    False)

# получение списка файлов из папки data_for_analyze_by_tags
csv_list = os.listdir('D:\\netology_diplom\\final\\csv_files\\data_for_analyze_by_tags')
f = 0
for file in csv_list:
    print(f"\r{file[:-4]}:", end="")
    if f == 0:
        df_file = pd.read_csv('D:\\netology_diplom\\final\\csv_files\\data_for_analyze_by_tags\\' + file)
        df_full_data = df_file.drop(df_file.index, inplace = True)
    else:
        df_file = pd.read_csv('D:\\netology_diplom\\final\\csv_files\\data_for_analyze_by_tags\\' + file)
    df_full_data = pd.concat([df_full_data, df_file])
    print(f"\r{file[:-4]}:", 'OK', end="\n")

    f += 1
    #if f==3:
    #    break
df_full_data.to_csv('D:\\netology_diplom\\final\\csv_files\\full_data.csv', index = False)

df_full_data = pd.read_csv('D:\\netology_diplom\\final\\csv_files\\full_data.csv')
df_full_data['year'] = pd.to_datetime(df_full_data['date']).dt.year
df_fd = df_full_data.drop(columns = ['url','title','topic','date'])
df_fd.drop_duplicates()

df_text_tags = df_fd.drop(columns = ['id', 'year'])

df_text_tags_final = pd.get_dummies(df_text_tags, prefix = ['tags'], columns=['tags'])
for i in df_text_tags_final.columns[1:]:
    df_text_tags_final[i] = df_text_tags_final[i].astype(int)

list_col = df_text_tags_final.columns[:1].to_list()
for i in df_text_tags_final.columns[1:].to_list():
    list_col.append(i[5:])
list_col
df_text_tags_final.columns = list_col

df_text_tags_final.to_csv('D:\\netology_diplom\\final\\csv_files\\news_data_lemma.csv', index = False)
df_text_tags_final
#df_text_tags_final=pd.read_csv('D:\\netology_diplom\\final\\csv_files\\news_data_lemma.csv')
```

![](.\\Изображения\\dataset.png)

**Общее количество строк после обработки:** 70889
```
  <class 'pandas.core.frame.DataFrame'>
  RangeIndex: 70889 entries, 0 to 70888
  Data columns (total 7 columns):
  #   Column  Non-Null Count  Dtype 
  ---  ------  --------------  ----- 
  0   id      70889 non-null  int64 
  1   url     70889 non-null  object
  2   title   70889 non-null  object
  3   text    70889 non-null  object
  4   topic   70889 non-null  object
  5   tags    70889 non-null  object
  6   date    70889 non-null  object
  dtypes: int64(1), object(6)
  memory usage: 3.8+ MB
```

## Метрики 

### Правильность (Accuracy)

Доля объектов, для которых правильно предсказан класс. Объект считается классифицированным верно, если предсказанный вектор полностью совпадает с таргетом.

$$ Accuracy = (TP + TN) / (TP + TN + FP + FN) $$

Стоит иметь в виду, что Accuracy имеет несколько недостатков:
- Не учитывает дисбаланс классов
- Не учитывает цену ошибки на объектах разных классов (ошибочно положительное определение класса не так критично, как
  ошибочно отрицательное).

### Точность (Precision)

Точность показывает долю правильно предсказанных положительных объектов среди всех объектов, предсказанных положительным
классом. Иначе говоря, в рамках поставленной задачи, точность по каждому тегу показывает, сколько из определённых нами объектов с таким тегом действительно относятся к
этому тегу. 

$$ Precision = TP / (TP + FP) $$

Общее значение точности равно среднему арифметическому значению точности по всем тегам.

### Полнота (Recall)

Полнота показывает долю правильно найденных положительных объектов среди всех объектов положительного класса. Иначе
говоря, в рамках поставленной задачи, полнота по каждому тегу показывает, какую долю объектов с таким тегом удалось
выявить. 

$$ Recall = TP / (TP + FN) $$

Общее значение полноты равно среднему арифметическому значению полноты по всем тегам.


### F1-мера (F1-measure)

F1-мера представляет среднее гармоническое точности и полноты. F1-мера предполагает одинаковую важность Precision и Recall. 

$$ F1 = (2 * Recall * Precision) / 
(Recall + Precision) = $$

$$  = TP / 
(TP + 0.5*(FP + FN)) $$

Общее значение F1-меры равно среднему арифметическому значению F1-меры по всем тегам.

### Матрица ошибок (Confusion matrix)

Матрица, состоящая из комбинаций, которые могут получаться при сопоставлении ответов алгоритма/модели и истинных меток объекта:

- TP — истинно-положительные объекты ( True ositive) — объект представляет собой класс 1 и алгоритм его идентифицирует
  как класс 1

- FP — ложно-положительные объекты ( FalsePositive) — объект представляет собой класс0, алгоритм его идентифицирует как
  класс 1(незначительная ошибка)

- TN — истинно-отрицательные объекты ( TrueNegative) — объект представляет собой класс0 и алгоритм его идентифицирует
  как класс 0

- FN — ложно-отрицательные объекты ( FalseNegative) — объект представляет собой класс 1, алгоритм его идентифицирует как
  класс 0. (грубая ошибка)


![](.\\Изображения\\матрица_ошибок.png)



```python
def cm_show(y_test, y_pred, nrows=10, ncols=5, figsize_w=30, figsize_h=60, wspace=0.2, hspace=0.4):
    '''
    Вывод матрицы несоответствия для каждой тематической категории в формате:   
    [['TN', 'FP'],
     ['FN', 'TP']]
    y_test - тестовые целевые данные
    y_pred - предсказанные целевые данные
    nrows - количество строк в сетке матриц несоответствия (для корректного отображения сетки должно
    работать уравнение: nrows = figsize_h / 6)
    ncols - количество столбцов в сетке матриц несоответствия (для корректного отображения сетки должно
    работать уравнение: ncols = figsize_w / 6)
    figsize_w - ширина полотна для вывода сетки матриц несоответствия (для корректного отображения сетки должно
    работать уравнение: figsize_h = 6 * ncols)
    figsize_h - высота полотна для вывода сетки матриц несоответствия (для корректного отображения сетки должно
    работать уравнение: figsize_h = 6 * nrows)
    wspace - расстояние по ширине между отдельными ячейками сетки матриц несоответствия
    hspace - расстояние по высоте между отдельными ячейками сетки матриц несоответствия
    '''
    
    start = time.time()
    y_true = y_test.to_numpy()
    axis = plt.subplots(nrows, ncols, figsize=(figsize_w, figsize_h))[1]
    axis = axis.ravel()

    # многоклассовая матрица несоответствий
    mlcm = multilabel_confusion_matrix(y_true, y_pred)
        
    for tag_num in range(len(y_test.columns)):
    #   Если надо отобразить проценты. 
    #   Так как в рамках задачи наибольший интерес вызывает информация о точном предсказании (y_true = 1, y_pred = 1)
    #   или о грубой ошибке при предсказании (y_true = 1, y_pred = 0) категории, имеет смысл выводить информацию для
    #   дальнейшего анализа в виде доли верных и неверных предсказаний для целевого значения = 1. 
        mlcm_percent = mlcm[tag_num] / mlcm[tag_num].sum(axis=1).reshape(2, -1)
        
        true0_row = mlcm[tag_num].sum(axis=1)[0] # количество TN, FP предсказаний (нецелевое значение = 0)
        true1_row = mlcm[tag_num].sum(axis=1)[1] # количество TP, FN предсказаний (целевое значение = 1)
        
        mlcm_percent_display = ConfusionMatrixDisplay(mlcm_percent, display_labels=[0, 1])        
        mlcm_percent_display.plot(ax=axis[tag_num],cmap=plt.cm.GnBu, values_format="0.3f")
        mlcm_percent_display.ax_.set_title(f'{y_test.columns[tag_num]} (TL0: {true0_row}, TL1:{true1_row})') 

    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.show()
    return print(f'Время выполнения: {round(time.time() - start,3)} с')
```


### Похожесть (Identity)

Расстояние Хэмминга в своём стандартном виде ($ d_h $) показывает количество позиций, в которых соответствующие символы
двух объектов ($obj1,obj2$) одинаковой длины ($length$) различаются. В рамках поставленной задачи был рассмотрен вывод
доли похожести объектов ($Identity$) при помощи расстояния Хэмминга.

$$ Identity = 1 - d_h(obj1,obj2)  / length(obj1) $$

Стоит учитывать, что Identity имеет несколько недостатков:
- При увеличении количества классов приближается к единице, что усложняет
интерпретацию метрики
- Не даёт понимания, в чём конкретно похожи объекты. Например, для пар значений "10001" / "11111" и "10001" / "10110"
  похожесть будет одинаково равна 0,4, но в первой паре количество единиц равно 5, а во второй паре значений равно 3.


```python
def identity(y_test, y_pred):
    '''
    Расчёт похожести объектов через расстояние Хэмминга
    y_test - тестовые целевые данные
    y_pred - предсказанные целевые данные
    Вывод среднего значения похожести объектов по всем элементам 
    '''
    y_true = y_test.to_numpy()

    if (len(y_true) != len(y_pred)):

        raise Exception('Объекты должны быть одинаковой длины')
    identity_list = []
    for elem in range(len(y_true)):
        if (len(y_true[elem]) != len(y_pred[elem])):

            raise Exception('Элементы должны быть одинаковой длины')
        # Инициализация переменной расстояния Хэмминга
        dist_counter = 0
        for n in range(len(y_true[elem])):
            # Изменение расстояния Хэмминга при наличии разницы между объектами
            if y_true[elem][n] != y_pred[elem][n]:
                
                dist_counter += 1
                
        len_elem = len(y_true[elem])
        # Расчёт доли похожести элементов двух объектов через Расстояние Хэмминга(dist_counter/len_elem - доля непохожести двух объектов)
        identity = round(1 - dist_counter/len_elem, 3)
        identity_list.append(identity)

    return round(np.mean(identity_list),5)
```

### Расстояние Хэмминга (Hamming distance)

По причине того, что стандартный расчёт расстояния Хэмминга не несёт большого смысла для рассматриваемой задачи и
Identity обладает рядом перечисленных недостатков, было принято решение доработать формулу расчёта расстояния Хэмминга
для более точной интерпретации и понимания различий между объектами.

```python
def hamming_distance(y_test, y_pred):
    '''
    Расчёт доработанного расстояния Хэмминга с ориентацией на целевое значение = 1
    y_test - тестовые целевые данные
    y_pred - предсказанные целевые данные
    Вывод среднего значения расстояния Хэмминга по всем элементам 
    '''
    y_true = y_test.to_numpy()

    if (len(y_true) != len(y_pred)):
        raise Exception('Объекты должны быть одинаковой длины')
    distance_list = []
    
    for elem in range(len(y_true)):
        if (len(y_true[elem]) != len(y_pred[elem])):
            raise Exception('Элементы должны быть одинаковой длины')

        indices_1_y_true = set(np.where(y_true[elem])[0])
        indices_1_y_pred = set(np.where(y_pred[elem])[0])
        
        if len(indices_1_y_true) == 0 and len(indices_1_y_pred) == 0:
            dist_counter = 1

        else:
            dist_counter = len(indices_1_y_true.intersection(indices_1_y_pred)) / float(len(indices_1_y_true.union(indices_1_y_pred)))

        distance_list.append(dist_counter)
    return round(np.mean(distance_list),5)
```

Для отображения метрик используется модуль metrics библиотеки sklearn (методы **accuracy_score**,
**confusion_matrix**, **classification_report**) и описанные выше функции **hamming_distance**, **identity**.


