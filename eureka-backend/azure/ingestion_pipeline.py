from azure.blob_chunker import chunk_blob_to_text

TARGET_BLOBS = [
    "CPET/SALES/Sales_CPET_NEW.xlsx",
    "PACKAGING/SPEND/Fiori_Spend_Analysis.xlsx"
]

def ingest_selected_chunks():
    all_chunks = []

    for blob_path in TARGET_BLOBS:
        chunks = chunk_blob_to_text(blob_path)
        all_chunks.extend(chunks)

    return all_chunks
