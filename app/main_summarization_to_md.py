import os
import tqdm
import openai
from langchain import OpenAI
from dotenv import load_dotenv
from utils.embedding_handlers import get_embedding_clusters
from utils.summarization_handlers import get_chunks_summaries, get_final_summary
from utils.text_handlers import load_document, text_splitter, normalize_text, export_as_md


def main(path, output_file_name, first_page, last_page):
    load_dotenv()
    openai.api_key = os.environ['OPENAI_API_KEY']
    vault_path = os.getenv("VAULT_PATH")
    text = load_document(path=path, first_page=first_page, last_page=last_page)
    docs = text_splitter(text=text)
    selected_indices = get_embedding_clusters(docs=docs, num_clusters=7)
    summaries = get_chunks_summaries(selected_indices=selected_indices, docs=docs)
    final_summary = get_final_summary(summaries=summaries)
    export_as_md(text=normalize_text(text=final_summary), vault_path=vault_path, out_file_name=output_file_name)


if __name__ == "__main__":
    file_path = 'path to file'
    output_name = 'output file name'
    first_page_number = 10
    last_page_number = 150

    main(path=file_path, output_file_name=output_name, first_page=first_page_number, last_page=last_page_number)
