from flask import Flask, send_from_directory, render_template_string, abort # type: ignore
import os
import mimetypes
import logging

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the Playback directory exists
PLAYBACK_DIR = "Playback"
if not os.path.exists(PLAYBACK_DIR):
    os.makedirs(PLAYBACK_DIR)
    logger.info(f"Created Playback directory at {os.path.abspath(PLAYBACK_DIR)}")


@app.route("/playback/<path:filename>")
def serve_video(filename):
    """Serve video files from the Playback directory securely."""
    try:
        # Security check: Ensure the filename is safe
        if ".." in filename or filename.startswith("/"):
            logger.warning(f"Attempted unsafe file access: {filename}")
            abort(400, description="Invalid filename")

        # Check if the file exists
        if not os.path.isfile(os.path.join(PLAYBACK_DIR, filename)):
            logger.warning(f"Video not found: {filename}")
            abort(404, description="Video not found")

        # Serve the file securely
        return send_from_directory(
            PLAYBACK_DIR,
            filename,
            mimetype=mimetypes.guess_type(filename)[0] or "video/mp4",
        )

    except Exception as e:
        logger.error(f"Error serving video {filename}: {str(e)}")
        abort(500, description="Internal server error")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
