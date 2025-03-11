from flask import Flask, send_from_directory, render_template_string, abort
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


@app.route("/")
def video_list():
    """Show a list of available videos with embedded players."""
    try:
        # Get list of .mp4 files in the Playback directory
        videos = [f for f in os.listdir(PLAYBACK_DIR) if f.endswith(".mp4")]
        if not videos:
            logger.info("No videos found in the Playback directory")

        return render_template_string(
            """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Video Playback</title>
                <style>
                    body { 
                        font-family: Arial, sans-serif; 
                        margin: 20px;
                        background: #f0f0f0;
                    }
                    .video-container {
                        margin-bottom: 30px;
                        background: white;
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        max-width: 800px;
                        margin: 0 auto 30px auto;
                    }
                    video {
                        width: 100%;
                        height: auto;
                        border-radius: 4px;
                    }
                    h1 {
                        text-align: center;
                        color: #333;
                        margin-bottom: 20px;
                    }
                    h2 {
                        color: #333;
                        margin-bottom: 10px;
                    }
                </style>
            </head>
            <body>
                <h1>Available Videos</h1>
                {% if videos %}
                    {% for video in videos %}
                    <div class="video-container">
                        <h2>{{ video }}</h2>
                        <video controls>
                            <source src="{{ url_for('serve_video', filename=video) }}" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>
                    </div>
                    {% endfor %}
                {% else %}
                    <p>No videos available.</p>
                {% endif %}
            </body>
            </html>
            """,
            videos=videos,
        )

    except Exception as e:
        logger.error(f"Error rendering video list: {str(e)}")
        abort(500, description="Internal server error")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
