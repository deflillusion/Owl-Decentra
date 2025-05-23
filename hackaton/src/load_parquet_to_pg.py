import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

# Загружаем по частям
parquet_file = pq.ParquetFile("../data/DECENTRATHON_3.0.parquet")

batch_size = 100_000
i = 0

# Используем iter_batches для чтения по кускам
for batch in parquet_file.iter_batches(batch_size=batch_size):
    df = pa.Table.from_batches([batch]).to_pandas()
    df.to_sql("raw_transactions", engine, if_exists="append" if i >
              0 else "replace", index=False)
    i += len(df)
    print(f"✅ Загружено строк: {i}")
