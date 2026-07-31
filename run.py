import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from app import app
app.run(debug=False, host="127.0.0.1", port=5000)
