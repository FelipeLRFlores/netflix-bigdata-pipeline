import os
import sys
import json
import pandas as pd
from pymongo import MongoClient

# Descobre a pasta exata onde este script .py está salvo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Procura o CSV dentro da pasta 'data/' ou na mesma pasta do script
CSV_CANDIDATOS = [
    os.path.join(BASE_DIR, "data", "netflix_titles.csv"),
    os.path.join(BASE_DIR, "netflix_titles.csv")
]

CSV_PATH = next((p for p in CSV_CANDIDATOS if os.path.exists(p)), None)
DATA_DIR = os.path.join(BASE_DIR, "data")
PARQUET_PATH = os.path.join(DATA_DIR, "netflix_catalog.parquet")

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "netflix_bigdata"
COLLECTION_NAME = "netflix_catalog"

def split_and_clean(val):
    if pd.isna(val) or not str(val).strip():
        return []
    return [item.strip() for item in str(val).split(",") if item.strip()]

def parse_duration(val):
    if pd.isna(val) or not str(val).strip():
        return {"duration_value": None, "duration_unit": None}
    parts = str(val).strip().split(" ")
    try:
        return {"duration_value": int(parts[0]), "duration_unit": " ".join(parts[1:])}
    except ValueError:
        return {"duration_value": None, "duration_unit": str(val)}

def executar_pipeline():
    print("=" * 60, flush=True)
    print("INICIANDO PIPELINE DE BIG DATA - CATÁLOGO NETFLIX", flush=True)
    print("=" * 60, flush=True)

    if not CSV_PATH:
        print(f"\n[ERRO] Arquivo 'netflix_titles.csv' não encontrado!", flush=True)
        print(f"Coloque o CSV dentro de: {BASE_DIR} ou {DATA_DIR}", flush=True)
        sys.exit(1)

    # 1. CAPTURA DE DADOS
    print(f"\n[1/4] Lendo CSV em: {CSV_PATH}", flush=True)
    df_raw = pd.read_csv(CSV_PATH)
    print(f"      -> Total de registros capturados: {len(df_raw)}", flush=True)

    # 2. TRANSFORMAÇÃO
    print("\n[2/4] Executando limpeza e modelagem de agregados...", flush=True)
    dates_parsed = pd.to_datetime(df_raw["date_added"].str.strip(), errors="coerce")

    df_transformed = pd.DataFrame()
    df_transformed["_id"] = df_raw["show_id"]
    df_transformed["type"] = df_raw["type"]
    df_transformed["title"] = df_raw["title"]
    df_transformed["directors"] = df_raw["director"].apply(split_and_clean)
    df_transformed["cast"] = df_raw["cast"].apply(split_and_clean)
    df_transformed["countries"] = df_raw["country"].apply(split_and_clean)
    df_transformed["release_info"] = [
        {
            "date_added": d.strftime("%Y-%m-%dT00:00:00Z") if pd.notna(d) else None,
            "release_year": int(y)
        }
        for d, y in zip(dates_parsed, df_raw["release_year"])
    ]
    df_transformed["rating"] = df_raw["rating"].fillna("Not Rated")
    df_transformed["duration_info"] = df_raw["duration"].apply(parse_duration)
    df_transformed["categories"] = df_raw["listed_in"].apply(split_and_clean)
    df_transformed["description"] = df_raw["description"]

    # 3. ARMAZENAMENTO EM PARQUET
    print(f"\n[3/4] Gerando arquivo Parquet...", flush=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    df_transformed.to_parquet(PARQUET_PATH, engine="pyarrow", index=False)
    tamanho_mb = os.path.getsize(PARQUET_PATH) / (1024 * 1024)
    print(f"      -> SUCESSO! Parquet criado em: {PARQUET_PATH} ({tamanho_mb:.2f} MB)", flush=True)

    # 4. INGESTÃO NO MONGODB
    print(f"\n[4/4] Conectando ao MongoDB ({MONGO_URI})...", flush=True)
    try:
        df_parquet = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
        records = json.loads(df_parquet.to_json(orient="records"))

        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.server_info()  # Testa se o MongoDB está ligado
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]

        collection.delete_many({})
        resultado = collection.insert_many(records)

        collection.create_index("type")
        collection.create_index("countries")
        collection.create_index("categories")
        collection.create_index("release_info.release_year")

        print(f"      -> SUCESSO! {len(resultado.inserted_ids)} documentos inseridos em '{DB_NAME}.{COLLECTION_NAME}'.", flush=True)
    except Exception as e:
        print(f"\n[AVISO MONGODB] O arquivo Parquet foi criado, mas houve erro ao conectar no MongoDB:", flush=True)
        print(f"Detalhe do erro: {e}", flush=True)

    print("=" * 60, flush=True)

if __name__ == "__main__":
    executar_pipeline()