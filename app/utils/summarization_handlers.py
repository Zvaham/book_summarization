import time
from tqdm import tqdm
from langchain import PromptTemplate
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain


def get_chunks_summaries(selected_indices, docs):
    map_prompt = """
    You will be given a single passage of an essay. This section will be enclosed in triple backticks (```)
    Your goal is to output a summary of the section so that a reader will have a full understanding of the topics discussed 
    and what happened.
    Your response should be at least 350 words, or three paragraphs and fully encompass what was said in the passage.

    ```{text}```
    FULL SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    chat_llm = ChatOpenAI(temperature=0, max_tokens=1000, model='gpt-3.5-turbo-16k')

    map_chain = load_summarize_chain(llm=chat_llm, chain_type="stuff", prompt=map_prompt_template)
    selected_docs = [docs[doc] for doc in selected_indices]

    summary_list = []
    progress_bar = tqdm(total=len(selected_docs), dynamic_ncols=True, desc="Processing chunks")
    for i, doc in enumerate(selected_docs):
        chunk_summary = map_chain.run([doc])
        summary_list.append(chunk_summary)
        progress_bar.update(1)

    progress_bar.close()
    elapsed_time = time.time() - start_time
    print(f"Total processing time: {elapsed_time:.2f} seconds")

    summaries = "\n".join(summary_list)
    summaries = Document(page_content=summaries)

    return summaries


def get_final_summary(summaries):
    chat_llm2 = ChatOpenAI(temperature=0, max_tokens=3000, model='gpt-3.5-turbo-16k', request_timeout=120)

    # combine_prompt = """
    # You will be given a series of summaries from an essay. The summaries will be enclosed in triple backticks (```)
    # Your goal is to give a verbose summary of the topics discussed and the events that happened.
    # You must write it as of the summary is the original text. do not reference it as "the passage".
    #
    # The reader should be able to answer questions about the topics without reading the original essay.
    # The summary should be at least 400 words, formatted correctly, and with paragraph titles.
    #
    # ```{text}```
    # VERBOSE SUMMARY:
    # """

    combine_prompt = """
        You will be given a series of summaries from an essay. The summaries will be enclosed in triple backticks (```)
        Your task is to write a new essay using the given summaries, including the topics discussed and the events that happened.
        The reader should be able to understand the overall context of the essay and its topics and events.
        The new essay should be at least 350 words, formatted correctly, and with paragraph titles.

        make sure to add some tags at the bottom of the essay.

        ```{text}```
        VERBOSE SUMMARY:
        """

    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
    reduce_chain = load_summarize_chain(llm=chat_llm2, chain_type="stuff", prompt=combine_prompt_template)
    output = reduce_chain.run([summaries])
    print(output)
    return output


if __name__ == "__main__":
    pass
