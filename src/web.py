from flask import Flask, render_template, request, jsonify, redirect
import os
from pathlib import Path
from werkzeug.utils import secure_filename
from threading import Thread

# Import RAG pipeline parts
from src.ingest import process_documents
from src.ingest import update_vectorstore_incremental
try:
    from src.rag_engine import load_vectorstore, create_conversational_chain, run_conversational_query
except Exception as _err:
    load_vectorstore = create_conversational_chain = run_conversational_query = None
    print("Warning: could not import rag_engine:", _err)

try:
    from src.conversation_manager import create_memory, clear_memory
except Exception as _err:
    create_memory = clear_memory = None
    print("Warning: could not import conversation_manager:", _err)

# Topic tracking (optional)
try:
    from src.topic_tracker import TopicTracker
except Exception as _err:
    TopicTracker = None
    print("Warning: could not import topic_tracker:", _err)

# Watcher for continuous updates
try:
    from src.updater import start_watcher
except Exception as _err:
    start_watcher = None
    print("Warning: could not import updater:", _err)

# Configure upload settings
UPLOAD_FOLDER = "data/raw"
ALLOWED_EXTENSIONS = {"txt", "pdf", "docx"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Track ingestion/embedding progress
processing_status = {
    "current_file": None,
    "total_files": 0,
    "processed_files": 0,
    "current_stage": None,
    "error": None,
    "new_knowledge": False,
}

def allowed_file(filename):
    """Check if a filename has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def process_uploaded_files(files):
    """Process uploaded files in a background thread."""
    global processing_status
    try:
        processing_status["current_stage"] = "ingesting"
        # process_documents will load, split and build the vectorstore (persisted to data/index)
        process_documents()
        
        processing_status["current_stage"] = "indexing"
        # index persisted by ingest.process_documents
        
        processing_status["current_stage"] = "done"
    except Exception as e:
        processing_status["error"] = str(e)
        processing_status["current_stage"] = "error"

app = Flask(__name__, template_folder="../templates", static_folder="../static")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# Conversation memory (keeps last 3 turns) — create if available
memory = create_memory(k=3) if create_memory is not None else None

# Topic tracker instance
topic_tracker = TopicTracker() if 'TopicTracker' in globals() and TopicTracker is not None else None

# Try loading existing index at startup (optional)
vectorstore = None
conv_chain = None
_watcher_started = False
try:
    if os.path.exists("data/index") and load_vectorstore is not None:
        vectorstore = load_vectorstore("data/index")
        # chain will be created lazily when first query arrives
except Exception as _err:
    print("Warning: failed to load existing index:", _err)
    vectorstore = None

# Start filesystem watcher in a background daemon thread
def _on_fs_change():
    try:
        added = update_vectorstore_incremental("data/raw", "data/index")
        if added and isinstance(added, int) and added > 0:
            processing_status["new_knowledge"] = True
    except Exception as e:
        # store error non-fatally
        processing_status["error"] = str(e)

if start_watcher is not None and not _watcher_started:
    from threading import Thread
    t = Thread(target=lambda: start_watcher("data/raw", callback=_on_fs_change), daemon=True)
    t.start()
    _watcher_started = True

@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle file uploads and start processing."""
    if "files[]" not in request.files:
        return jsonify({"error": "No files provided"}), 400
    
    files = request.files.getlist("files[]")
    if not files:
        return jsonify({"error": "No files selected"}), 400
    
    # Reset processing status
    global processing_status
    processing_status["current_file"] = None
    processing_status["total_files"] = len(files)
    processing_status["processed_files"] = 0
    processing_status["current_stage"] = "uploading"
    processing_status["error"] = None
    
    # Save uploaded files
    saved_files = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            saved_files.append(filename)
            processing_status["processed_files"] += 1
    
    if saved_files:
        # Start processing in background
        Thread(target=process_uploaded_files, args=(saved_files,), daemon=True).start()
        return jsonify({
            "message": f"Uploaded {len(saved_files)} files",
            "files": saved_files
        })
    return jsonify({"error": "No valid files uploaded"}), 400

@app.route("/status")
def get_status():
    """Return current processing status."""
    return jsonify(processing_status)

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main page handler."""
    answer = None
    results = None
    query = ""
    top_k = 3
    theme = request.cookies.get("theme", "light")
    re_rank = request.form.get("re_rank", "cosine")
    
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        # validate top_k from form
        try:
            top_k = int(request.form.get('top_k', 3))
            top_k = max(1, min(top_k, 50))  # clamp between 1 and 50
        except Exception:
            top_k = 3

        if query:
            # ensure we have a vectorstore (build if missing)
            global vectorstore, conv_chain
            if vectorstore is None:
                # attempt to build index from raw files (process_documents returns client,collection)
                try:
                    proc = process_documents()
                    if proc is not None:
                        # reload chroma-backed store
                        try:
                            vectorstore = load_vectorstore("data/index")
                        except Exception as _err:
                            processing_status["error"] = str(_err)
                            vectorstore = None
                except Exception as e:
                    processing_status["error"] = str(e)
                    vectorstore = None

            if vectorstore is not None and conv_chain is None:
                conv_chain = create_conversational_chain(vectorstore, memory=memory)

            if conv_chain is not None and vectorstore is not None:
                res = run_conversational_query(conv_chain, vectorstore, query, top_k=top_k)
                # res: {answer, source_docs, raw}
                answer = {
                    'text': res.get('answer') or "",
                    'confidence': round((sum([d['score'] for d in res.get('source_docs', [])]) / max(1, len(res.get('source_docs', [])))) * 100, 1),
                    'method': 'conversational_retrieval'
                }
                results = res.get('source_docs', [])
                # Assign topic (best-effort)
                try:
                    if topic_tracker is not None and answer.get('text'):
                        topic_name = topic_tracker.assign_topic(query, query, answer['text'])
                        answer['topic'] = topic_name
                except Exception:
                    pass
            else:
                # fallback: no index available
                answer = { 'text': "No documents available to answer the question.", 'confidence': 0.0, 'method': 'none' }
    
    # List uploaded files with sizes
    uploaded_files = []
    file_sizes = {}
    if os.path.exists(UPLOAD_FOLDER):
        for f in os.listdir(UPLOAD_FOLDER):
            if allowed_file(f):
                uploaded_files.append(f)
                size = os.path.getsize(os.path.join(UPLOAD_FOLDER, f))
                if size < 1024:
                    file_sizes[f] = f"{size} B"
                elif size < 1024 * 1024:
                    file_sizes[f] = f"{size/1024:.1f} KB"
                else:
                    file_sizes[f] = f"{size/(1024*1024):.1f} MB"
    
    # expose recent chat history from memory (if available)
    chat_history = []
    try:
        mem_vars = memory.load_memory_variables({})
        # memory may return messages when return_messages=True
        chat_history = mem_vars.get('history') or mem_vars.get('chat_history') or []
    except Exception:
        chat_history = []

    return render_template('index.html',
        query=query,
        answer=answer,
        results=results,
        top_k=top_k,
        theme=theme,
        re_rank=re_rank,
        uploaded_files=uploaded_files,
        file_sizes=file_sizes,
        processing_status=processing_status,
        chat_history=chat_history
    )


@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    """Clear the conversation memory (chat history)."""
    try:
        clear_memory(memory)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route('/streamlit')
def streamlit_redirect():
    """Redirect to the Streamlit UI if it's running locally."""
    return redirect('http://localhost:8501', code=302)


@app.route('/ack_update', methods=['POST'])
def ack_update():
    """Acknowledge the new knowledge banner and clear the flag."""
    try:
        processing_status["new_knowledge"] = False
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
