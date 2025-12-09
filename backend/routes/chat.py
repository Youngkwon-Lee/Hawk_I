from flask import Blueprint, request, jsonify
from services.chat_service import ChatService

bp = Blueprint('chat', __name__, url_prefix='/api')
chat_service = ChatService()

@bp.route('/chat', methods=['POST'])
def chat():
    """
    Chat with AI assistant
    Expected JSON: { "message": "...", "context": {...} }

    If VLM analysis is triggered, response includes vlm_result
    """
    try:
        data = request.json
        message = data.get('message')
        context = data.get('context')

        if not message:
            return jsonify({"error": "Message is required"}), 400

        # Get response - now returns tuple (response_text, vlm_result or None)
        response_text, vlm_result = chat_service.get_response(message, context)

        result = {
            "success": True,
            "response": response_text
        }

        # Include VLM result if available
        if vlm_result:
            result["vlm_analysis"] = {
                "performed": True,
                "score": vlm_result.get("score"),
                "confidence": vlm_result.get("confidence"),
                "findings": vlm_result.get("findings"),
                "reasoning": vlm_result.get("reasoning")
            }
        else:
            result["vlm_analysis"] = {"performed": False}

        return jsonify(result)

    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
