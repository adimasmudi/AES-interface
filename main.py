from flask import Flask, request, render_template, redirect, url_for,jsonify
from transformers import BertTokenizer, BertModel
import time
from model import BERTScoring

app = Flask(__name__)

# Load IndoBERT finetuned model
savedFinetunedModel = 'indobert-base-aes-pretrained'
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p2')
bert_model = BertModel.from_pretrained(savedFinetunedModel)
modelForScoring = BERTScoring(bert_model, tokenizer)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result')
def result():
    score = float(request.args.get('score'))
    reference_answer = request.args.get('reference_answer')
    answer = request.args.get('answer')
    time_cost = request.args.get('time_cost')

    # Determine score class
    if score >= 0.75:
        score_class = 'score-high'
    elif 0.5 <= score < 0.75:
        score_class = 'score-medium'
    else:
        score_class = 'score-low'

    score = 0 if score < 0 else score

    return render_template('result.html', score=score, reference_answer=reference_answer, answer=answer, time_cost=time_cost, score_class=score_class)

@app.route("/api/v1/predict", methods=["POST"])
def predict():
    response = {
        "message": "prediction success",
        "code": 200,
        "status": "success",
        "data": {}
    }
    if request.method == "POST":
        try:
            referenceAnswer = ''
            answer = ''
            if request.is_json:
                referenceAnswer = request.json.get("reference_answer")
                answer = request.json.get("answer")
            else:
                referenceAnswer = request.form.get("reference_answer")
                answer = request.form.get("answer")
            
            t1 = time.time()
            predictionResult = modelForScoring.predict(referenceAnswer, answer)
            time_cost = time.time() - t1

            response["data"] = {
                "score": predictionResult[0],
                "time_cost": f"{time_cost} seconds"
            }

            # Redirect to result page with arguments
            return redirect(url_for('result', 
                                    score=predictionResult[0], 
                                    reference_answer=referenceAnswer, 
                                    answer=answer, 
                                    time_cost=f"{time_cost} seconds"))

        except Exception as err:
            response["status"] = "error"
            response["message"] = str(err)
            response["code"] = 500

    return jsonify(response) if request.is_json else redirect(url_for('result'))

if __name__ == "__main__":
    app.run(port=5000)
