
import sys
import os
import traceback
import requests
from flask import Flask, request, jsonify, render_template, send_from_directory, Response
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

_rag_system = None
def get_rag_system():
    global _rag_system
    if _rag_system is None:
        from main import AntimicrobialRAG
        _rag_system = AntimicrobialRAG("../datasets")
    return _rag_system

# Warm-up: pre-build/load index in background thread to reduce first request latency
try:
    import threading
    threading.Thread(target=get_rag_system, daemon=True).start()
except Exception:
    pass


app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/images/<path:filename>')
def images_files(filename):
    images_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'images'))
    return send_from_directory(images_dir, filename)

@app.route('/ask_stream', methods=['POST'])
def ask_stream():
    data = request.get_json()
    question = data.get('question', '')
    messages = data.get('messages', [])
    if not question:
        return Response('event: error\ndata: No question provided\n\n', mimetype='text/event-stream')
    try:
        print(f"[ask_stream] Received question: {question}")
        rag = get_rag_system()

        def generate():
            for chunk in rag.stream_query(question):
                yield f'data: {chunk}\n\n'
            yield 'data: [DONE]\n\n'
        return Response(generate(), mimetype='text/event-stream')
    except Exception as e:
        print('---RAG STREAM ERROR---')
        traceback.print_exc()
        return Response(f'event: error\ndata: {str(e)}\n\n', mimetype='text/event-stream')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question', '')
    messages = data.get('messages', [])
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    try:
        rag = get_rag_system()
        answer = rag.query(question)
        return jsonify({'answer': answer})
    except Exception as e:
        print('---RAG ERROR---')
        traceback.print_exc()
        return jsonify({'error': f'Backend error: {str(e)}'}), 500
@app.route('/modeling-report', methods=['POST'])
def modeling_report():
    """Endpoint for generating Section 8 mathematical modeling report."""
    data = request.get_json()
    query = data.get('question', 'Generate Section 8 mathematical modeling report')

    try:
        rag = get_rag_system()
        # Use a few documents as context (mainly providing system architecture information)
        relevant_docs = rag.retriever.retrieve(query, top_k=3)

        # Generate modeling report
        answer = rag.generator.generate_modeling_report(relevant_docs, query)

        return jsonify({"answer": answer})
    except Exception as e:
        print('---MODELING REPORT ERROR---')
        traceback.print_exc()
        return jsonify({'error': f'Modeling report generation error: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(e):
    return render_template('index.html'), 200

if __name__ == '__main__':

    HOST = os.getenv('APP_HOST', '127.0.0.1')
    PORT = int(os.getenv('APP_PORT', '5000'))

    try:
        from waitress import serve
        print(' * Using waitress WSGI server')
        print(f' * Server running at http://{HOST}:{PORT} (open /)')
        print(' * SSE streaming endpoint: POST /ask_stream with JSON {"question": "..."}')
        serve(app, host=HOST, port=PORT)
    except ImportError:
        print('Please install waitress first with pip install waitress, or use gunicorn or other production servers.\nTemporarily using Flask development server:')
        print(f' * Server running at http://{HOST}:{PORT} (open /)')
        print(' * SSE streaming endpoint: POST /ask_stream with JSON {"question": "..."}')
        app.run(debug=True, host=HOST, port=PORT)
