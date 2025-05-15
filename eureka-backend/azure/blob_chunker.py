from azure.azure_config import container_client
from io import BytesIO
import pandas as pd

def chunk_blob_to_text(blob_path: str):
    parts = blob_path.strip("/").split("/")

    if len(parts) < 3:
        raise ValueError(f"Invalid blob path structure: '{blob_path}'")

    root = parts[0]         #CPET
    domain = parts[1]       #SALES
    filename = parts[-1]    #Sales_CPET_NEW.xlsx

    print(f"📄 Processing blob: {blob_path} → root={root}, domain={domain}, file={filename}")

    # Get blob client
    blob_client = container_client.get_blob_client(blob_path)

    try:
        blob_bytes = blob_client.download_blob().readall()
    except Exception as e:
        print(f"Failed to download blob: {blob_path} — {e}")
        return []

    # Load into Pandas
    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(BytesIO(blob_bytes), encoding="ISO-8859-1")
        elif filename.lower().endswith(".xlsx"):
            df = pd.read_excel(BytesIO(blob_bytes), engine="openpyxl")
        else:
            print(f"⚠️ Skipped unsupported file type: {filename}")
            return []
    except Exception as e:
        print(f"Failed to parse {filename}: {e}")
        return []

    # Chunking: one row = one chunk
    chunks = []
    for idx, row in df.iterrows():
        row_text = " | ".join([f"{col}: {str(val)}" for col, val in row.items()])
        chunks.append({
            "text": row_text,
            "metadata": {
                "root": root,
                "domain": domain,
                "filename": filename,
                "chunk_index": idx,
                "source_path": blob_path
            }
        })

    print(f"Chunked {len(chunks)} rows from {filename}")
    return chunks
