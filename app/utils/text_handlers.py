from unidecode import unidecode
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_document(path, first_page, last_page):
    loader = PyPDFLoader(file_path=path)
    pages = loader.load()
    pages = pages[first_page: last_page]

    text = ""
    for page in pages:
        text += page.page_content
    text = text.replace('\t', ' ')

    return text


def text_splitter(text):
    splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=10000, chunk_overlap=3000)
    docs = splitter.create_documents([text])

    return docs


def normalize_text(text):
    normalized_text = unidecode(text)
    return normalized_text


def export_as_md(text, vault_path, out_file_name):
    with open(fr'{vault_path}\{out_file_name}.md', 'w') as file:
        file.write(text)
