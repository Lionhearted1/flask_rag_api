from app import create_app
from flask_cors import CORS

app = create_app()
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})


if __name__ == '__main__':
    app.run(debug=True)
