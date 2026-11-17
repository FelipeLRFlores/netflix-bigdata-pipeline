import os
import json
import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.linear_model import LinearRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_PATH = os.path.join(BASE_DIR, "data", "netflix_catalog.parquet")

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "netflix_bigdata"
COLLECTION_NAME = "netflix_catalog"

MESES_PT = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril",
    5: "Maio", 6: "Junho", 7: "Julho", 8: "Agosto",
    9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"
}

def carregar_documentos():
    """Busca os documentos diretamente do MongoDB (ou do Parquet como fallback)."""
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        client.server_info()
        collection = client[DB_NAME][COLLECTION_NAME]
        docs = list(collection.find())
        if docs:
            print(f"[OK] {len(docs)} documentos carregados diretamente do MongoDB ({DB_NAME}.{COLLECTION_NAME}).")
            return docs, collection
    except Exception:
        pass

    print("[AVISO] Lendo dados a partir da camada Parquet local...")
    df_p = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
    return json.loads(df_p.to_json(orient="records")), None

def executar_analises():
    docs, collection = carregar_documentos()
    df = pd.DataFrame(docs)

    # Extrair ano e mês de 'release_info.date_added'
    datas = pd.to_datetime(df["release_info"].apply(lambda x: x.get("date_added") if isinstance(x, dict) else None), errors="coerce")
    df["year_added"] = datas.dt.year
    df["month_added"] = datas.dt.month

    print("\n" + "=" * 75)
    print("RELATÓRIO DE ANÁLISES DE BIG DATA - CATÁLOGO NETFLIX (AVALIAÇÃO N3)")
    print("=" * 75)

    # =========================================================================
    # 1. ANÁLISE DE TENDÊNCIAS DE LANÇAMENTO (Sazonalidade, Tipo e Gêneros)
    # =========================================================================
    print("\n1. ANÁLISE DE TENDÊNCIAS DE LANÇAMENTO (PERÍODOS DO ANO E TIPOS)")
    print("-" * 75)
    
    # Volume por Tipo nos últimos anos
    df_recent = df[df["year_added"] >= 2016]
    evolucao_tipo = df_recent.groupby(["year_added", "type"]).size().unstack(fill_value=0)
    print("-> Evolução de Adições por Tipo de Conteúdo (2016 - 2021):")
    for ano, row in evolucao_tipo.iterrows():
        print(f"   Ano {int(ano)}: {row.get('Movie', 0):4d} Filmes | {row.get('TV Show', 0):3d} Séries (Total: {row.sum()})")

    # Meses com maior volume de adições (Sazonalidade)
    sazonalidade_mes = df["month_added"].dropna().astype(int).value_counts().head(5)
    print("\n-> Sazonalidade: Top 5 Meses com Maior Volume de Adições no Ano:")
    for mes_num, total in sazonalidade_mes.items():
        print(f"   - {MESES_PT[mes_num]}: {total} novos títulos adicionados")

    # Gêneros mais adicionados nos meses de pico (Julho e Dezembro)
    df_cat_mes = df.explode("categories")
    top_generos = df_cat_mes["categories"].value_counts().head(5)
    print("\n-> Top 5 Gêneros com Maior Volume Acumulado de Lançamentos:")
    for genero, total in top_generos.items():
        print(f"   - {genero}: {total} títulos")

    # =========================================================================
    # 2. MAPEAMENTO DE PRODUÇÃO GLOBAL (Países x Comportamento Filmes/Séries)
    # =========================================================================
    print("\n\n2. MAPEAMENTO DE PRODUÇÃO GLOBAL (DISTRIBUIÇÃO POR PAÍS E FORMATO)")
    print("-" * 75)
    df_paises = df.explode("countries")
    df_paises = df_paises[df_paises["countries"].notna() & (df_paises["countries"] != "")]
    paises_tipo = df_paises.groupby(["countries", "type"]).size().unstack(fill_value=0)
    paises_tipo["Total"] = paises_tipo.get("Movie", 0) + paises_tipo.get("TV Show", 0)
    paises_top8 = paises_tipo.sort_values("Total", ascending=False).head(8)

    for pais, row in paises_top8.iterrows():
        movies = row.get("Movie", 0)
        tv = row.get("TV Show", 0)
        tot = row["Total"]
        pct_movie = (movies / tot) * 100
        pct_tv = (tv / tot) * 100
        foco = "Foco em Filmes" if pct_movie >= 50 else "Foco em Séries/Animes/K-Dramas"
        print(f"   - {pais:15s}: {tot:4d} obras ({movies:4d} Filmes [{pct_movie:4.1f}%] | {tv:3d} Séries [{pct_tv:4.1f}%]) -> {foco}")

    # =========================================================================
    # 3. PREVISÃO DE SUCESSO POR CATEGORIA (Machine Learning - Scikit-Learn)
    # =========================================================================
    print("\n\n3. PREVISÃO DE DEMANDA POR CATEGORIA (MACHINE LEARNING - REGRESSÃO LINEAR)")
    print("-" * 75)
    print("-> Modelo Scikit-Learn treinado sobre a janela de crescimento (2016-2020):")
    df_ml = df_cat_mes[(df_cat_mes["year_added"] >= 2016) & (df_cat_mes["year_added"] <= 2020)]
    categorias_alvo = top_generos.index.tolist()

    for cat in categorias_alvo:
        serie_anual = df_ml[df_ml["categories"] == cat].groupby("year_added").size()
        X = np.array(serie_anual.index).reshape(-1, 1)
        y = serie_anual.values

        modelo = LinearRegression()
        modelo.fit(X, y)

        taxa_crescimento = modelo.coef_[0]
        demanda_projetada = int(max(0, modelo.predict([[2022]])[0]))
        recomendacao = "Expandir Licenciamento (Alta Demanda)" if taxa_crescimento > 50 else "Manter Licenciamento Estável"

        print(f"   - {cat:24s} | Tendência: {taxa_crescimento:+6.1f} títulos/ano | "
              f"Projeção Anual: ~{demanda_projetada:3d} títulos | Ação: {recomendacao}")

    # =========================================================================
    # 4. ANÁLISE DE ELENCO E DIREÇÃO (Mapeamento de Rede e Parcerias)
    # =========================================================================
    print("\n\n4. ANÁLISE DE ELENCO E DIREÇÃO (MAPEAMENTO DE REDE E FREQUÊNCIA)")
    print("-" * 75)
    df_cast = df.explode("cast")
    top_atores = df_cast[df_cast["cast"].notna() & (df_cast["cast"] != "")]["cast"].value_counts().head(5)
    print("-> Top 5 Atores Mais Frequentes no Catálogo:")
    for ator, qtd in top_atores.items():
        print(f"   - {ator}: {qtd} títulos")

    df_dir = df.explode("directors")
    top_diretores = df_dir[df_dir["directors"].notna() & (df_dir["directors"] != "")]["directors"].value_counts().head(5)
    print("\n-> Top 5 Diretores Mais Frequentes no Catálogo:")
    for diretor, qtd in top_diretores.items():
        print(f"   - {diretor}: {qtd} títulos")

    # Mapeamento de Rede (Grafo de Colaboração Diretor <-> Ator)
    df_rede = df.explode("directors").explode("cast")
    df_rede = df_rede[
        df_rede["directors"].notna() & (df_rede["directors"] != "") &
        df_rede["cast"].notna() & (df_rede["cast"] != "")
    ]
    top_parcerias = df_rede.groupby(["directors", "cast"]).size().sort_values(ascending=False).head(5)
    print("\n-> Mapeamento de Rede: Top 5 Parcerias Recorrentes (Diretor <-> Ator):")
    for (diretor, ator), qtd in top_parcerias.items():
        print(f"   - [Diretor] {diretor} <---> [Ator] {ator}: {qtd} produções conjuntas")

    print("\n" + "=" * 75)

if __name__ == "__main__":
    executar_analises()