import os
import csv
import glob
import shutil
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_localai import LocalAIEmbeddings
from utils import load_options

load_dotenv()
options = load_options("options.yaml")


def load_csvs(dir_path: str):
    '''
    Loads and parses all csv files in a directory and
    returns their content in a list of tuples
    '''

    if dir_path.endswith("/"):
        dir_path = dir_path[:-1]

    filepaths = glob.glob(dir_path + "/*.csv")
    all_data = []
    for filepath in filepaths:
        with open(filepath, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            lines = [l for l in reader]
            all_data.extend(lines)

    return all_data

def humanize_advancements(advancements: list[tuple])-> list[str]: 
    '''
    Converts the raw advancement data to a human readable format
    '''
    results = []
    for i, adv, desc, tab, parent, req, is_hidden, notes in advancements:
        res = ""
        if len(adv):
            res += f"Advancement name: {adv}\n"
        if len(desc):
            res += f"Description: {desc}\n"
        if len(tab):
            res += f"Tab: {tab}\n"
        if len(parent):
            res += f"Parent: {parent}\n"
        if len(req):
            res += f"Requirements: {req}\n"
        if len(notes):
            res += f"Notes: {notes}"

        if res.endswith("\n"):
            res = res[:-1]

        results.append(res) 

    return results

def embed_chunks(chunks: list[str], embedding_model: any, chroma_dir: str):
    '''
    Embeds the `chunks` using the given embedding model to a `chroma_dir`
    '''
    if os.path.exists(chroma_dir):
        shutil.rmtree(chroma_dir)

    db = Chroma.from_texts(chunks, embedding_model, persist_directory=chroma_dir)
    return db

if __name__ == "__main__":
    data = load_csvs("./data")
    humanized = humanize_advancements(data)

    embedding_model = LocalAIEmbeddings(
        openai_api_base=options["base_url"],
        openai_api_key=os.getenv("LLM_API_KEY"),
        model=options["embedding_model"],
    )
    db = embed_chunks(humanized, embedding_model, options["chroma_dir"])
    print(db.search("Apples", search_type="similarity"))