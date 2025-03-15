import os
from flask import Flask
from routes.predict import predict_bp
from routes.personalized_pathway import pathway_bp

app = Flask(__name__)

# Register blueprints
app.register_blueprint(predict_bp)
app.register_blueprint(pathway_bp)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if no PORT is set
    app.run(debug=True, host="0.0.0.0", port=port)
