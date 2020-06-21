import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import seaborn as sns
import shap

def add_random_noise_col(X_df):
    X_df_new = X_df.copy().reset_index(drop=True)
    rand_noise = pd.Series(np.random.rand(X_df.shape[0],))
    X_df_new["random"] = rand_noise
    return X_df_new


def plot_imp(imp_df_og, big=False, errorbar=True):
    imp_df = imp_df_og.copy()
    imp_df["color"] = "default"
    imp_df.loc[imp_df["Importance"] < 0, "color"] = "negative"
    imp_df.loc[imp_df["Feature"] == "random", "color"] = "random"

    palette = {"default":"dodgerblue", "negative":"lightgray", "random":"darkorange"}
    if big is True:
        plt.figure(figsize=(8, 6))
    else:
        plt.figure(figsize=(6, 4))
    sns.set(font_scale=1.1)
    sns.set_style("ticks")
    graph = sns.barplot(x="Importance", y="Feature", hue="color",
                data=imp_df, palette=palette, dodge=False)
    graph.get_legend().remove()
    if errorbar is True:
        graph.errorbar(x=imp_df["Importance"], y=imp_df["Feature"], xerr=imp_df["StdDev"],
                       ecolor="black", markersize=4, capsize=2, ls='none')
    plt.show()


def bootstrap_Xy(X, y):
    bs_ind = np.random.randint(0, X.shape[0], size=X.shape[0])
    bs_X = X.iloc[bs_ind].reset_index(drop=True)
    bs_y = y.iloc[bs_ind].reset_index(drop=True)
    return bs_X, bs_y


def simul_imp(n_simul, imp_method, X, y, model=None, metric=None, neg_metric=False):
    n_imps = []
    for i in range(n_simul):
        bs_X, bs_y = bootstrap_Xy(X, y)
        imp_df = imp_method(bs_X, bs_y, model, metric, neg_metric=neg_metric)
        n_imps.append(imp_df["Importance"])
    n_imps = np.array(n_imps).T

    min_imp = np.min(n_imps)
    max_imp = np.max(n_imps)
    n_imps_scaled = (n_imps - min_imp)/(max_imp - min_imp)
    mean_imps = np.mean(n_imps_scaled, axis=1)
    std_imps = np.std(n_imps_scaled, axis=1)

    simul_imp_df = pd.concat([pd.Series(X.columns), pd.Series(mean_imps), pd.Series(std_imps)], axis=1)
    simul_imp_df.columns = ["Feature", "Importance", "StdDev"]
    simul_imp_df = simul_imp_df.sort_values('Importance', ascending=False)
    return simul_imp_df.reset_index(drop=True)


def compare_imp_kfeat(k, ranking_dfs, X, y, model, metric, neg_metric=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    metrics = []
    k_feats = []
    for r in ranking_dfs:
        r = r.loc[r["Feature"] != "random"]
        top_k = r["Feature"].iloc[:k]
        X_train_k = X_train[top_k]
        X_test_k = X_test[top_k]
        model.fit(X_train_k, y_train)
        metrics.append(mean_squared_error(y_test, model.predict(X_test_k)))
        k_feats.append(list(top_k))
    methods = ["Spearman", "Drop-col", "Permutation"]
    out = [(md, round(mt, 3), f) for md, mt, f in zip(methods, metrics, k_feats)]
    out = pd.DataFrame(out, columns = ["Method", "Metric", "Top Features Used"])
    if neg_metric is True:
        out = out.sort_values('Metric', ascending=True)
    else:
        out = out.sort_values('Metric', ascending=False)
    return out


def spearman_imp(X, y, model=None, metric=None, sort=False, neg_metric=False):
    spearman_imp = []
    for x in X:
        spearman_imp.append((x, abs(spearmanr(X[x], y)[0])))
    if sort is True:
        spearman_imp = sorted(spearman_imp, key=lambda x: x[1], reverse=True)
    return pd.DataFrame(spearman_imp, columns = ["Feature", "Importance"])


def drop_col_imp(X, y, model, metric, neg_metric=False, sort=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    baseline_metric = metric(y_test, model.predict(X_test))
    drop_col_imp = []
    for col in X:
        X_train_drop = X_train.copy().drop(col, axis=1)
        X_test_drop = X_test.copy().drop(col, axis=1)
        model_drop = clone(model)
        model_drop.fit(X_train_drop, y_train)
        col_metric = metric(y_test, model_drop.predict(X_test_drop))
        if neg_metric is True:
            col_imp = col_metric - baseline_metric
        else:
            col_imp = baseline_metric - col_metric
        drop_col_imp.append((col, col_imp))
    if sort is True:
        drop_col_imp = sorted(drop_col_imp, key=lambda x: x[1], reverse=True)
    return pd.DataFrame(drop_col_imp, columns = ["Feature", "Importance"])


def permutation_imp(X, y, model, metric, neg_metric=False, sort=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    baseline_metric = metric(y_test, model.predict(X_test))
    permut_imp = []
    for col in X:
        X_train_permut = X_train.copy().reset_index(drop=True)
        X_test_permut = X_test.copy().reset_index(drop=True)
        X_train_permut[col] = X_train_permut[col].sample(frac=1).reset_index(drop=True)
        X_test_permut[col] = X_test_permut[col].sample(frac=1).reset_index(drop=True)
        col_metric = metric(y_test, model.predict(X_test_permut))
        if neg_metric is True:
            col_imp = col_metric - baseline_metric
        else:
            col_imp = baseline_metric - col_metric
        permut_imp.append((col, col_imp))
    if sort is True:
        permut_imp = sorted(permut_imp, key=lambda x: x[1], reverse=True)
    return pd.DataFrame(permut_imp, columns = ["Feature", "Importance"])
