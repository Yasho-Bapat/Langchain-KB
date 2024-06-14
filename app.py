import time

import streamlit as st
from split_experiment.main import SplittingTest


def response_generator(splitter: str, query: str):
    test = SplittingTest(splitter)
    response, topk = test.query_documents(query)

    for word in response.split():
        yield word + " "
        time.sleep(0.05)
    yield "\nTOP 3 DOCUMENTS RETRIEVED: \n"

    for document in topk:
        yield document + "\n" + "\t" + str(document.metadata) + "\n\n\n"
        time.sleep(0.05)

    print(topk)


st.set_page_config(layout="wide", page_title="Chunking Demo")
st.title("Chunking Strategies Demo")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        cols = st.columns(2, gap="small")

        with cols[0]:
            with st.chat_message("assistant"):
                semantic_container = st.container(border=True)
                semantic_container.subheader("Semantic")
                semantic_container.markdown(message["semantic"])
        with cols[1]:
            with st.chat_message("assistant"):
                recursive_container = st.container(border=True)
                recursive_container.subheader("Recursive")
                recursive_container.markdown(message["recursive"])

if prompt := st.chat_input("Ask a Query..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    cols = st.columns(2, gap="small")

    with cols[0]:
        with st.chat_message("assistant"):
            semantic_container = st.container(border=True)
            semantic_container.subheader("Semantic")
            semantic_response = semantic_container.write_stream(
                response_generator(splitter="semantic", query=prompt)
            )
    with cols[1]:
        with st.chat_message("assistant"):
            recursive_container = st.container(border=True)
            recursive_container.subheader("Recursive")
            recursive_response = recursive_container.write_stream(
                response_generator(splitter="recursive", query=prompt)
            )
        st.session_state.messages.append({"role": "assistant", "semantic": semantic_response, "recursive": recursive_response})

    st.rerun()
