import sys
import os

# Add the directory containing app.py to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dotenv
dotenv.load_dotenv()

print("DEBUG URI:", os.environ.get("SQLALCHEMY_DATABASE_URI", "NOT FOUND"))

from SafeMumApp import create_app, db

app = create_app()

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)