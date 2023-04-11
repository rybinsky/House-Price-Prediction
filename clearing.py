def missing_values_table(df):
    '''
    Description:
        Функция вычисляет процент пропущенных значений в каждом столбце
    Args:
        df (pd.DataFrame): матрица признаков
    Returns:
        mis_val_table_ren_columns (pd.DataFrame): матрица информации
    '''
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis = 1)
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending = False).round(1)
        
    # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
      
    return mis_val_table_ren_columns

def print_missing_percent(df, threshold):
    '''
    Description:
        Выводит все поля, где пропущен какой-то процент значений

    Args:
        df (pd.DataFrame): матрица признаков
        threshold (float): порог отсечения между автозаполненными признаками и нет

    Returns:
        missing_list (list): массив пропущенных значений меньше порога threshold
    '''
    missing_list = []
    for col in df.columns:
        missing_count = df[col].isna().sum()
        total_count = df[col].count() + missing_count
        missing_percent = missing_count / total_count * 100
        if missing_percent:
            if missing_percent < threshold:
                missing_list.append(col)
            print(f"Column {col}: {missing_percent:.2f}%   missing values,  type: {df[col].dtype}")
    return missing_list

"""Теперь везде где мало значений заполним простым методом, а где много обучим модели для заполнения. Напишем функцию для заполнения там где мало пропущенных данных """

def fill_missing_auto(df, mis_list):
    '''
    Description:
        Заполняет пропуски из df в колонках mis_list определенным методом
    Args:
        df (pd.DataFrame): матрица признаков
        mis_list (list): колонки, подлежащие заполнению 
    '''
    for col in mis_list:
        if df[col].dtype in ("int64", "float64"):
            df[col].fillna(df[col].mean(), inplace = True) # среднее для численных
        else:
            most_common_value = df[col].mode()[0] # самый частый для категориальных
            df[col].fillna(most_common_value, inplace = True)

def remove_collinear_features(df, threshold, verbose, target_var):
    '''
    Description:
        Удаляет зависимые признаки из матрицы с коэффициентом корреллции больше порога threshold 
        и у которых практически отсутствует коррелляция с таргетом
        Такая процедура улучшает обощающую способность и повышает интерпретируемость модели

    Args: 
        df (pd.DataFrame): матрица признаков
        threshold (float): порог отсечения признаков
        verbose (bool): если True, то печатаем лог
        target_var (str): таргет - имя колонки, некорреллируемые с которой признаки удаляем

    Returns: 
        df (pd.DataFrame): матрица, очищенная от зависимых признаков
    '''
    # Вычислим матрицу коррелляций
    corr_matrix = df.drop(target_var, 1).corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []
    dropped_feature = ""

    # Итерируемся по ней и сравниваем попарные коррелляции
    for i in iters:
        for j in range(i + 1): 
            item = corr_matrix.iloc[j: (j + 1), (i + 1): (i + 2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            if val >= threshold:
                if verbose:
                    print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                col_value_corr = df[col.values[0]].corr(df[target_var])
                row_value_corr = df[row.values[0]].corr(df[target_var])
                if verbose:
                    print("{}: {}".format(col.values[0], np.round(col_value_corr, 3)))
                    print("{}: {}".format(row.values[0], np.round(row_value_corr, 3)))
                if col_value_corr < row_value_corr:
                    drop_cols.append(col.values[0])
                    dropped_feature = "dropped: " + col.values[0]
                else:
                    drop_cols.append(row.values[0])
                    dropped_feature = "dropped: " + row.values[0]
                if verbose:
                    print(dropped_feature)
                    print("-----------------------------------------------------------------------------")

    # Выбросим одну из достаточно корреллируемых колонок
    drops = set(drop_cols)
    df = df.drop(columns = drops)

    print("dropped columns: ")
    print(list(drops))
    print("-----------------------------------------------------------------------------")
    print("used columns: ")
    print(df.columns.tolist())
    
    return df

def drop_similar_columns(df, threshold):
    '''
    Description:
        Удаляет столбцы из df, если в них больше, чем threshold одинаковых значений

    Args:
        df (pd.DataFrame): матрица признаков
        threshold (float): порог удаления

    Returns:
        df (pd.DataFrame): очищенная матрица признаков
    '''
    num_rows = df.shape[0]
    drop_cols = []
    
    for col in df.columns:
        if df[col].value_counts().iloc[0] >= num_rows * (threshold / 100):
            drop_cols.append(col)

    df = df.drop(columns = drop_cols)
    return df

def iqr_outliers_percent(df, columns, threshold = 10):
    '''
    Description:
        Выводит процент выбросов в столбцах columns матрицы признаков df

    Args:
        df (pd.DataFrame): матрица признаков
        columns (list): колонки, из которых удалять выбросы
        threshold (float): порог удаления выбросов из drop_cols

    Returns:
        drop_cols (list): список колонок, откуда можно удалить выбросы
    '''
    drop_cols = []
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        total_count = df[col].shape[0]
        outliers_percent = outliers_count / total_count * 100

        if outliers_percent < threshold:
            drop_cols.append(col)
        print(f'{col}: {outliers_percent:.2f}%')
    
    return drop_cols

def remove_outliers(df, columns , threshold = 1.5, drop_percent = 100):
    '''
    Description:
        Функция удаляет строки, в которых есть выбросы, определенные по методу Тьюки (межквартильное расстояние)

    Args:
        df (pd.DataFrame): матрица признаков
        columns (list): список числовых признаков
        threshold (float): порог в методе Тьюки
        drop_percent (float): доля удаляемых выбросов

    Returns:
        df (pd.DataFrame): матрица признаков, очищенные от какой-то доли выбросов
    '''
    bounds = []
    for column in columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        bounds.append((lower_bound, upper_bound))

    for (lower_bound, upper_bound), column in zip(bounds, columns):

        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
        outliers = outliers.sort_values()    
        n_to_remove = int(len(outliers) * drop_percent / 100)
    
        to_remove = outliers.head(n_to_remove).index.union(outliers.tail(n_to_remove).index)
        cleaned_col = df[column].drop(to_remove)
        df = df.loc[cleaned_col.index].copy()
        df.reset_index(drop = True, inplace = True)

    return df
