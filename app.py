import hashlib
import streamlit as st
from openai import APIConnectionError
from rag import extract, split_pages, build_index, answer

st.set_page_config(page_title='StudyPDF • Research assistant', page_icon='📚', layout='wide')
st.title('📚 StudyPDF')
st.caption('Read together. Ask better questions. Check every source.')
for key, default in [('index', None), ('messages', []), ('signature', None), ('warnings', []), ('stats', None)]:
    st.session_state.setdefault(key, default)
with st.sidebar:
    st.header('Workspace')
    provider = st.selectbox('Model provider', ['OpenAI', 'Ollama (local)'])
    local = provider.startswith('Ollama')
    api_key = st.text_input(
        'Your OpenAI API key',
        type='password',
        help='Used for this app session only; it is not saved to a file.',
    ) if not local else ''
    base_url = st.text_input('Ollama URL', value='http://localhost:11434') if local else ''
    chat_model = st.text_input('Chat model', value='qwen3:4b' if local else 'gpt-4o-mini')
    embed_model = st.text_input('Embedding model', value='nomic-embed-text' if local else 'text-embedding-3-small')
    st.caption('Local mode keeps PDF text on your machine. OpenAI mode sends PDF chunks and questions to OpenAI.')
    uploads = st.file_uploader('Upload PDFs', type=['pdf'], accept_multiple_files=True)
    with st.expander('Retrieval settings'):
        size = st.slider('Chunk size (characters)', 600, 2400, 1200, 100)
        overlap = st.slider('Chunk overlap', 0, 400, 200, 50)
        k = st.slider('Source chunks per answer', 2, 12, 6)
    process = st.button('Process documents', type='primary', use_container_width=True)
    if st.button('Clear conversation'):
        st.session_state.messages = []
    if st.button('Reset workspace'):
        for key in ['index', 'signature', 'stats']:
            st.session_state[key] = None
        st.session_state.messages = []
        st.session_state.warnings = []
        st.rerun()

files = [(f.name, f.getvalue()) for f in uploads]
signature = hashlib.sha256(repr([(n, hashlib.sha256(b).hexdigest()) for n, b in files] +
    [provider, embed_model, base_url, size, overlap, hashlib.sha256(api_key.encode()).hexdigest()]).encode()).hexdigest()

def clients():
    if local:
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        return (ChatOllama(model=chat_model, base_url=base_url, temperature=0, num_ctx=16384),
                OllamaEmbeddings(model=embed_model, base_url=base_url))
    if not api_key:
        raise ValueError('Enter your OpenAI API key first.')
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    return (ChatOpenAI(model=chat_model, api_key=api_key, temperature=0, timeout=90, max_retries=2),
            OpenAIEmbeddings(model=embed_model, api_key=api_key, timeout=90, max_retries=5))

if process:
    try:
        if not files:
            raise ValueError('Upload at least one PDF.')
        if sum(len(b) for _, b in files) > 50 * 1024 * 1024:
            raise ValueError('Please keep the total upload below 50 MB.')
        with st.spinner('Extracting pages, splitting text and embedding chunks…'):
            _, embeddings = clients()
            pages, warnings = extract(files)
            chunks = split_pages(pages, size, overlap)
            if len(chunks) > 5000:
                raise ValueError('Too many chunks (limit: 5,000). Upload fewer documents.')
            index = build_index(chunks, embeddings)
        st.session_state.update(index=index, signature=signature, messages=[], warnings=warnings,
                                stats=(len(files), len(pages), len(chunks)))
        st.success('Documents ready.')
    except Exception as exc:
        if isinstance(exc, APIConnectionError):
            cause = exc.__cause__ or exc.__context__
            cause_hint = f' Transport error: {type(cause).__name__}.' if cause else ''
            st.error(
                'OpenAI could not complete the embedding connection after retries. '
                'Check your internet connection, VPN/firewall and OpenAI service status, then retry. '
                'This connection error does not confirm whether your API key is valid. '
                'The key is not displayed here; do not paste it into chat.'
                f'{cause_hint} No new index was saved.'
            )
        else:
            st.error(f'Processing failed ({type(exc).__name__}). Check PDFs, model names, credentials and provider connection. No new index was saved.')
        if isinstance(exc, ValueError):
            st.warning(str(exc))

ready = st.session_state.index is not None and st.session_state.signature == signature
if st.session_state.index is not None and not ready:
    st.warning('Uploads or indexing settings changed. Click Process documents before asking questions.')
if ready:
    a, b, c = st.columns(3)
    for col, label, val in zip([a, b, c], ['Documents', 'Readable pages', 'Searchable chunks'], st.session_state.stats):
        col.metric(label, val)
    if st.session_state.warnings:
        with st.expander('Extraction notices'):
            for warning in st.session_state.warnings:
                st.warning(warning)
else:
    st.info('Upload lecture notes, textbooks or research papers, then process them to begin.')
mode = st.radio('Study mode', ['Ask a question', 'Explain simply', 'Summarize a topic', 'Create a quiz'], horizontal=True)
st.caption('Examples: Explain the main method • Summarize the limitations • Create 5 quiz questions about regression. Summaries cover retrieved excerpts, not necessarily every page.')

def render(message):
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        if message.get('citation_warning'):
            st.warning('Citations are missing or include an unknown source ID. Verify the answer against the passages below.')
        if message.get('sources'):
            with st.expander('Retrieved source chunks / PDF pages', expanded=True):
                for s in message['sources']:
                    st.markdown(f"**[{s['id']}] {s['file']} · PDF page {s['page']} · chunk {s['chunk_id']}**")
                    st.caption(f"Cosine similarity: {s['score']:.3f} (retrieval ranking, not answer confidence)")
                    st.text(s['text'])
for message in st.session_state.messages:
    render(message)
question = st.chat_input('Ask about your PDFs…', disabled=not ready)
if question:
    try:
        model, _ = clients()
        render({'role': 'user', 'content': question})
        with st.spinner('Retrieving evidence and preparing your answer…'):
            content, sources, warning = answer(st.session_state.index, model, question, mode, k, st.session_state.messages)
        response = {'role': 'assistant', 'content': content, 'sources': sources, 'citation_warning': warning}
        st.session_state.messages.extend([{'role': 'user', 'content': question}, response])
        render(response)
    except Exception as exc:
        st.error(f'Answer failed ({type(exc).__name__}). Check that the selected chat model is installed/available and the provider is reachable; then retry.')
if st.session_state.messages:
    export = []
    for m in st.session_state.messages:
        export.append(f"## {m['role'].title()}\n\n{m['content']}")
        for s in m.get('sources', []):
            export.append(f"### [{s['id']}] {s['file']} — PDF page {s['page']}\n\n{s['text']}")
    st.download_button('Download study conversation', '\n\n'.join(export), 'study_conversation.md', 'text/markdown')
