from src.app import app

@app.route("/")
def test():
    return 'Run /predict!'


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)