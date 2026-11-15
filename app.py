import streamlit as st

from rag import RAG

st.set_page_config(page_title="Multi-Document RAG", page_icon="📄")
st.title("📄 Multi-Document RAG Chat")

if "rag" not in st.session_state:
    st.session_state.rag, st.session_state.files, st.session_state.messages = RAG(), set(), []
rag = st.session_state.rag

with st.sidebar:
    st.header("Documents")
    for f in st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True) or []:
        if f.name not in st.session_state.files:
            with st.spinner(f"Indexing {f.name}..."):
                n = rag.add_pdf(f, f.name)
            st.session_state.files.add(f.name)
            st.success(f"{f.name}: {n} chunks")
    st.caption(f"{len(st.session_state.files)} documents, {len(rag.chunks)} chunks indexed")
    if st.button("Clear chat history"):
        rag.reset()
        st.session_state.messages = []


def show(msg):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for i, s in enumerate(msg["sources"], 1):
                    st.markdown(f"**[{i}] {s['source']} — page {s['page']}**\n\n{s['text'][:300]}...")


for msg in st.session_state.messages:
    show(msg)

if question := st.chat_input("Ask a question about your documents"):
    if not rag.chunks:
        st.warning("Upload at least one PDF first.")
        st.stop()
    show(user := {"role": "user", "content": question})
    with st.spinner("Thinking..."):
        answer, sources = rag.ask(question)
    show(bot := {"role": "assistant", "content": answer, "sources": sources})
    st.session_state.messages += [user, bot]
