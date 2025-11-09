def load_raw_text_data(data_path: str):
    with open(data_path, "r", encoding='utf-8') as f:
        data = f.read()
        # data = f.read().splitlines()
    return data

def load_sentences(data_path: str):
    with open(data_path, "r", encoding='utf-8') as f:
        data = f.read().splitlines()
    return data