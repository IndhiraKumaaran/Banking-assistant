"""Flask web interface for the AI Banking Assistant."""

import logging

from flask import Flask, jsonify, render_template, request

from rag_chain import BankingAssistantError, answer_question


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
MAX_QUESTION_LENGTH = 2000


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/ask")
def ask():
    if not request.is_json:
        return jsonify({"error": "Send the question as a JSON request."}), 400

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "The request body must be a JSON object."}), 400

    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "Please enter a banking question."}), 400
    if len(question) > MAX_QUESTION_LENGTH:
        return jsonify(
            {"error": f"Please keep your question under {MAX_QUESTION_LENGTH} characters."}
        ), 400

    try:
        result = answer_question(question.strip())
        return jsonify(result)
    except BankingAssistantError as error:
        return jsonify({"error": str(error)}), 503
    except Exception:
        app.logger.exception("The banking question could not be answered.")
        return jsonify(
            {
                "error": (
                    "The assistant could not process your question right now. "
                    "Check the application configuration and try again."
                )
            }
        ), 500


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)
