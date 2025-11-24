from flask import Blueprint, request, jsonify
from services.chat_service import ChatService

bp = Blueprint('chat', __name__, url_prefix='/api')
chat_service = ChatService()

@bp.route('/chat', methods=['POST'])
def chat():
    """
    Chat with AI assistant
    Expected JSON: { "message": "...", "context": {...} }
    """
    try:
        data = request.json
        message = data.get('message')
        context = data.get('context')

        if not message:
            return jsonify({"error": "Message is required"}), 400

        response = chat_service.get_response(message, context)
        
        return jsonify({
            "success": True,
            "response": response
        })

    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
