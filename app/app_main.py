import openai
import streamlit as st
from utils.embedding_handlers import get_embedding_clusters
from utils.summarization_handlers import get_chunks_summaries, get_final_summary
from utils.text_handlers import load_document, text_splitter, normalize_text, export_as_md


def main():
    st.title('Text to Markdown Summarization')

    col1, col2 = st.columns([2, 1])
    openai.api_key = st.text_input('Enter your OpenAI API key: ')

    if openai.api_key:
        with col1:
            file_path = st.text_input('Enter your file path: ')
            output_file_name = st.text_input('Enter your output file name: ')
            output_path = st.text_input('Enter your output path: ')

        with col2:
            first_page = st.number_input('Enter the first page of the relevant text: ', min_value=0)
            last_page = st.number_input('Enter the last page of the relevant text: ', min_value=0)

        start_process = st.button('Start Summarization', key='start_summarization_button')
        if start_process:
            text = load_document(path=file_path, first_page=first_page, last_page=last_page)
            docs = text_splitter(text=text)
            selected_indices = get_embedding_clusters(docs=docs, num_clusters=7)
            summaries = get_chunks_summaries(selected_indices=selected_indices, docs=docs)
            final_summary = get_final_summary(summaries=summaries)
            normalized_text = normalize_text(text=final_summary)
            st.text('MD file generated successfully')
            st.download_button(label='Download Summary', key='download_summary_button',
                               data=export_as_md(text=normalized_text, vault_path=output_path, out_file_name=output_file_name))


if __name__ == '__main__':
    main()
