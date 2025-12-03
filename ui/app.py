import os
import json
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit

# Optional sample snapshot image — replace with real captured frames later
SAMPLE_IMAGE = "/mnt/data/Screenshot 2025-11-03 at 10.56.00 PM.png"

# Flask + SocketIO setup
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")


# -----------------------------------------------------
#  API ENDPOINT — receives detection events from Python
# -----------------------------------------------------
@app.route("/api/event", methods=["POST"])
# --- Temporary safe handler (write request to file, do NOT emit) ---
@app.route("/api/event", methods=["POST"])
# ----- receive_event() with Socket.IO emit -----
@app.route("/api/event", methods=["POST"])
def receive_event():
    try:
        data = request.get_json(force=True)

        name = data.get("name", "Unknown")
        confidence = float(data.get("confidence", 0))
        snapshot = data.get("snapshot", SAMPLE_IMAGE)

        payload = {
            "name": name,
            "confidence": confidence,
            "snapshot": snapshot,
            "timestamp": int(time.time()),
            "timestr": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Emit event to all connected browser clients
        try:
            socketio.emit("recognition_event", payload)
            print("Emitted to UI:", payload)
        except Exception as e:
            # If emit fails, still return success and log error
            print("SocketIO emit error:", e)

        return jsonify({"status": "ok", "received": payload}), 200

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("ERROR in /api/event:", tb)
        return jsonify({"status": "error", "error": str(e), "trace": tb}), 500




# -----------------------------------------------------
#  ROUTE: Home UI Page
# -----------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", sample_image=SAMPLE_IMAGE)


# -----------------------------------------------------
#  SOCKET.IO CONNECTION EVENTS
# -----------------------------------------------------
@socketio.on("connect")
def on_connect():
    print("UI Connected")


@socketio.on("disconnect")
def on_disconnect():
    print("UI Disconnected")


# -----------------------------------------------------
#  RUN SERVER
# -----------------------------------------------------
if __name__ == "__main__":
    print("Starting UI server: http://localhost:5000")
    socketio.run(app, host="127.0.0.1", port=5000)

