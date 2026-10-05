from flask import Blueprint, jsonify, render_template, request

from . import license_model, similarity

bp = Blueprint("main", __name__)


@bp.get("/")
def index():
    return render_template("index.html")


@bp.get("/health")
def health():
    return jsonify(status="ok")


@bp.post("/api/compare")
def api_compare():
    data = request.get_json(silent=True) or {}
    code_a, code_b = data.get("code_a", ""), data.get("code_b", "")
    if not code_a.strip() or not code_b.strip():
        return jsonify(error="Both 'code_a' and 'code_b' are required."), 400
    return jsonify(similarity.compare(code_a, code_b))


@bp.post("/api/license")
def api_license():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not text.strip():
        return jsonify(error="'text' is required."), 400
    return jsonify(license_model.predict(text))
