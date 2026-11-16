import json
from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "netflix_bigdata"
COLLECTION_NAME = "netflix_catalog"

def demonstrar_saidas():
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]

    print("=" * 65)
    print("DEMONSTRAÇÃO DAS SAÍDAS ESPERADAS E ANÁLISES NO MONGODB")
    print("=" * 65)

    # 1. Exemplo de Documento BSON/JSON (Modelo de Agregados)
    doc_exemplo = collection.find_one({"_id": "s8"})
    print("\n1. EXEMPLO DE DOCUMENTO ARMAZENADO (Agregado _id: 's8'):")
    print(json.dumps(doc_exemplo, indent=2, ensure_ascii=False))

    # 2. Distribuição por Tipo de Conteúdo (Movie vs TV Show)
    print("\n2. TOTAL DE TÍTULOS POR TIPO DE CONTEÚDO:")
    pipeline_tipo = [
        {"$group": {"_id": "$type", "total": {"$sum": 1}}},
        {"$sort": {"total": -1}}
    ]
    for item in collection.aggregate(pipeline_tipo):
        print(f"   - {item['_id']}: {item['total']} títulos")

    # 3. Mapeamento Global: Top 8 Países Produtores ($unwind no array countries)
    print("\n3. MAPEAMENTO DE PRODUÇÃO GLOBAL (Top 8 Países):")
    pipeline_paises = [
        {"$unwind": "$countries"},
        {"$group": {"_id": "$countries", "total": {"$sum": 1}}},
        {"$sort": {"total": -1}},
        {"$limit": 8}
    ]
    for idx, item in enumerate(collection.aggregate(pipeline_paises), 1):
        print(f"   {idx}. {item['_id']}: {item['total']} produções")

    # 4. Gêneros Mais Populares ($unwind no array categories)
    print("\n4. ANÁLISE DE TENDÊNCIAS POR GÊNERO (Top 8 Categorias):")
    pipeline_categorias = [
        {"$unwind": "$categories"},
        {"$group": {"_id": "$categories", "total": {"$sum": 1}}},
        {"$sort": {"total": -1}},
        {"$limit": 8}
    ]
    for idx, item in enumerate(collection.aggregate(pipeline_categorias), 1):
        print(f"   {idx}. {item['_id']}: {item['total']} títulos")

    # 5. Análise de Elenco e Direção (Top 5 Diretores)
    print("\n5. ANÁLISE DE DIREÇÃO (Top 5 Diretores Mais Frequentes):")
    pipeline_diretores = [
        {"$unwind": "$directors"},
        {"$group": {"_id": "$directors", "total": {"$sum": 1}}},
        {"$sort": {"total": -1}},
        {"$limit": 5}
    ]
    for idx, item in enumerate(collection.aggregate(pipeline_diretores), 1):
        print(f"   {idx}. {item['_id']}: {item['total']} obras dirigidas")

    print("=" * 65)

if __name__ == "__main__":
    demonstrar_saidas()