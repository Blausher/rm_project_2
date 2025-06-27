import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller
import statsmodels.graphics.tsaplots as spl
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def plot_columns(df, columns_to_plot, single_plot=True):
    """
    Функция для построения графиков временных рядов для указанных столбцов DataFrame.

    :param df: DataFrame, содержащий данные для визуализации
    :param columns_to_plot: Список столбцов для построения графиков
    :param single_plot: Если True, все графики будут расположены на одном полотне. Если False, графики будут разбиты на сетку.
    """
    sns.set(style="whitegrid")

    if single_plot:
        # Построим графики всех столбцов на одной фигуре
        plt.figure(figsize=(15, 10))

        for column in columns_to_plot:
            sns.lineplot(x=df['date'], y=df[column], label=column)

        # Настройки графика
        plt.title('Временные ряды для различных столбцов')
        plt.xlabel('Дата')
        plt.ylabel('Значение')
        plt.legend(title='Столбцы')
        plt.xticks(rotation=45)
    else:
        # Определим количество строк и столбцов для сетки графиков
        num_plots = len(columns_to_plot)
        nrows = int(num_plots / 2) + num_plots % 2
        ncols = 2

        plt.figure(figsize=(15, nrows * 5))

        for i, column in enumerate(columns_to_plot):
            plt.subplot(nrows, ncols, i + 1)
            sns.lineplot(x=df['date'], y=df[column], label=column)
            plt.title(f'График для {column}')
            plt.xlabel('Дата')
            plt.ylabel('Значение')
            plt.xticks(rotation=45)

        plt.tight_layout()

    plt.show()


def describe(data, p_level="10%"):
    """Проверка на тренд"""
    print(data.name)

    s_h0 = "{}: нельзя отвергнуть H0, ряд может содержать единичные корни и быть нестационарным (p-value {:0.3f})"
    s_ha = "{}: H0 отвергается, ряд стационарен (p-value {:0.3f})"
    print("Расширенный тест Дики — Фуллера (ADF):")

    test_name = "Тест с константой"
    adf_c = adfuller(data, regression="c")
    if adf_c[0] > adf_c[4][p_level]:
        print(s_h0.format(test_name, adf_c[1]))
    else:
        print(s_ha.format(test_name, adf_c[1]))

    test_name = "Тест с константой и линейным трендом"
    adf_ct = adfuller(data, regression="ct")
    if adf_ct[0] > adf_ct[4][p_level]:
        print(s_h0.format(test_name, adf_ct[1]))
    else:
        print(s_ha.format(test_name, adf_ct[1]))

    test_name = "Тест с константой, линейным и квадратичным трендом"
    adf_ctt = adfuller(data, regression="ctt")
    if adf_ctt[0] > adf_ctt[4][p_level]:
        print(s_h0.format(test_name, adf_ctt[1]))
    else:
        print(s_ha.format(test_name, adf_ctt[1]))

    test_name = "Тест без константы и тренда"
    adf_nc = adfuller(data, regression="n")
    if adf_nc[0] > adf_nc[4][p_level]:
        print(s_h0.format(test_name, adf_nc[1]))
    else:
        print(s_ha.format(test_name, adf_nc[1]))

    test_name = "Тест с константой для процентных изменений"
    adf_pct_nc = adfuller(data.pct_change()[1:], regression="c")
    if adf_pct_nc[0] > adf_pct_nc[4][p_level]:
        print(s_h0.format(test_name, adf_pct_nc[1]))
    else:
        print(s_ha.format(test_name, adf_pct_nc[1]))

def draw_graphs(data, title=""):
    inc = data.set_index('date').pct_change()[1:].reset_index()

    # Установим стиль для графиков
    # plt.style.use('seaborn-darkgrid')

    # Исторические данные и процентные изменения
    fig, axs = plt.subplots(2, 1, figsize=(14, 8))
    plt.subplots_adjust(hspace=0.4)
    axs[0].plot(data['date'], data.iloc[:, 1], color='blue', lw=2)
    axs[0].set_title(f"Historical Data for {title}", fontsize=14)
    axs[0].set_xlabel('Дата', fontsize=12)
    axs[0].set_ylabel('Value', fontsize=12)
    axs[0].grid(True, linestyle='--', linewidth=0.5)
    axs[0].tick_params(axis='x', rotation=45)
    
    axs[1].plot(inc['date'], inc.iloc[:, 1], color='green', lw=2)
    axs[1].set_title(f"Percentage Changes for {title}", fontsize=14)
    axs[1].set_xlabel('Дата', fontsize=12)
    axs[1].set_ylabel('Percentage Change', fontsize=12)
    axs[1].grid(True, linestyle='--', linewidth=0.5)
    axs[1].tick_params(axis='x', rotation=45)

    # ACF, PACF, Histogram, Q-Q plot и P-P plot
    fig, axs = plt.subplots(3, 2, figsize=(14, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # ACF и PACF
    spl.plot_acf(data.set_index('date').iloc[:, 0], lags=30, ax=axs[0, 0], color='blue')
    spl.plot_pacf(data.set_index('date').iloc[:, 0], lags=30, ax=axs[0, 1], color='blue')
    axs[0, 0].set_title('ACF', fontsize=14)
    axs[0, 1].set_title('PACF', fontsize=14)
    
    # Histogram
    axs[1, 0].hist(data.iloc[:, 1], bins=60, color='orange', alpha=0.7)
    axs[1, 0].set_title("Histogram (history)", fontsize=14)
    axs[1, 0].set_xlabel('Value', fontsize=12)
    axs[1, 0].set_ylabel('Frequency', fontsize=12)

    axs[1, 1].hist(inc.dropna().iloc[:, 1], bins=60, color='orange', alpha=0.7)
    axs[1, 1].set_title("Histogram (changes)", fontsize=14)
    axs[1, 1].set_xlabel('Percentage Change', fontsize=12)
    axs[1, 1].set_ylabel('Frequency', fontsize=12)

    # Q-Q plot and P-P plot
    distribution = ProbPlot(inc.dropna().iloc[:, 1], fit=True)
    distribution.qqplot(line="r", ax=axs[2, 0])
    axs[2, 0].set_title("Q-Q Plot", fontsize=14)
    
    distribution.ppplot(line="45", ax=axs[2, 1])
    axs[2, 1].set_title("P-P Plot", fontsize=14)

    for ax in axs.flat:
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.set_facecolor("#f2f2f2")

    plt.show()

def plot_decomposition(data, name, period=1):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.subplots_adjust(hspace=0.35)
    fig.suptitle(f"Разложение {name} на тренд/сезонность/ошибку (период {period})", fontsize=16)

    # Выполняем декомпозицию временного ряда
    data = data.set_index('date')
    result = seasonal_decompose(data[name], model="additive", period=period)
    
    # Строим графики и настраиваем их внешний вид
    ax1.plot(result.trend.index, result.trend, color='blue', linestyle='--', lw=2)
    ax2.plot(result.seasonal.index, result.seasonal, color='green', linestyle='-', lw=2)
    ax3.plot(result.resid.index, result.resid, color='red', linestyle='-', lw=2)
    
    # Устанавливаем метки осей и заголовки
    ax1.set_ylabel("Тренд", fontsize=14)
    ax1.set_xlabel("", fontsize=14)
    ax1.set_title("Trend", fontsize=12)

    ax2.set_ylabel("Сезонность", fontsize=14)
    ax2.set_xlabel("", fontsize=14)
    ax2.set_title("Seasonality", fontsize=12)

    ax3.set_ylabel("Ошибка", fontsize=14)
    ax3.set_xlabel("Дата", fontsize=14)
    ax3.set_title("Residuals", fontsize=12)

    # Включаем сетку и улучшенные стили
    for ax in [ax1, ax2, ax3]:
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.set_facecolor("#f2f2f2")
        ax.tick_params(axis='x', rotation=45)

    plt.show()

def get_description(df, ticker):
    data = df[['date', ticker]].dropna()
    draw_graphs(data, title=ticker)
    describe(df[ticker])
    plot_decomposition(data, ticker, 365 // 4)
    plot_decomposition(data, ticker, 365 // 2)


def plot_corr_matrix(df_risk2):
    num_cols = df_risk2.select_dtypes(include=['float64']).columns

    plt.figure(figsize=(18, 10))
    with plt.style.context({'xtick.labelsize':15,
                            'ytick.labelsize':15}):
        sns.set(font_scale=1.4)
        dataplot = sns.heatmap(df_risk2[num_cols].corr().abs(), 
                            cmap="YlGnBu", 
                            annot=True, 
                            fmt=".2f", 
                            annot_kws={"size": 10}, 
                            linecolor='white', 
                            linewidths=1)

    plt.title('Correlation Matrix', fontsize=30, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.show()



def plot_pca(df_risk3, columns_to_use):
    scaler = StandardScaler()
    df_risk_scaled = scaler.fit_transform(df_risk3[columns_to_use])

    pca = PCA(n_components=3)  # Оставим первые две главные компоненты для визуализации
    principal_components = pca.fit_transform(df_risk_scaled)

    df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])

    # Визуализация двух главных компонентов
    plt.figure(figsize=(10,6))
    sns.scatterplot(x='PC1', y='PC2', data=df_pca)
    plt.title('PCA of Risk Factors')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)')
    plt.grid(True)
    plt.show()

    # Визуализация объясненной дисперсии
    plt.figure(figsize=(10,6))
    plt.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, alpha=0.5, align='center')
    plt.step(range(1, len(pca.explained_variance_ratio_)+1), np.cumsum(pca.explained_variance_ratio_), where='mid')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Principal Components')
    plt.grid(True)
    plt.show()