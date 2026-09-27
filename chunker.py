def chunk_text(text, chunk_size=800, overlap=150):
    """
    Splits text into overlapping chunks.
    """

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

def chunk_pages(pages, chunk_size=800, overlap=150):
    chunks = []

    # Recreate the same full text used by load_pdf()
    full_text = ""
    page_ranges = []

    for page in pages:
        start = len(full_text)

        full_text += page["text"] + "\n"

        end = len(full_text)

        page_ranges.append({
            "page": page["page"],
            "start": start,
            "end": end
        })

    start = 0

    while start < len(full_text):

        end = start + chunk_size
        chunk = full_text[start:end]

        # Find which page contains the beginning of this chunk
        page_number = None

        for page in page_ranges:
            if page["start"] <= start < page["end"]:
                page_number = page["page"]
                break

        chunks.append({
            "text": chunk,
            "page": page_number
        })

        start += chunk_size - overlap

    return chunks

