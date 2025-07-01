from app.backend import app, initialize_rag
from vercel_wsgi import handle_request

initialize_rag()

def handler(request, context):
    return handle_request(app, request, context)
