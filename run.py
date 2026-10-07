import os

from app import create_app

app = create_app()

if __name__ == "__main__":
    # macOS reserves port 5000 for AirPlay Receiver; set PORT=5001 to avoid it.
    app.run(debug=True, port=int(os.environ.get("PORT", "5000")))
