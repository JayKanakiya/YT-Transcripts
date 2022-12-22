from flask import Flask, render_template, request
from fetch import getTranscripts
from summarizer import transformer_summarizer, generate_summary, nltk_summarizer
import requests

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        url = request.form.get('videoURL')
        id = url.split('?')
        id = id[1].split('=')[1]
        res = getTranscripts(id)
        print(res)

        tokens = res.split()
        n_tokens = len(tokens)
        s = []
        prev = 0
        i = 0
        while i < n_tokens:

            if i == 0:
                i += 200
                continue
            if i >= n_tokens:
                i = n_tokens
            
            # print(len(tokens[prev:i]))
            temp = transformer_summarizer(" ".join(tokens[prev:i]))
            # print(tokens[prev:i])
            # print("HERE", temp)
            s.append(temp)
            prev = i
            i += 200

        # summary = " ".join(s)
        # nltk_sum = nltk_summarizer(res, 7)
        # spacy_sum = generate_summary(res, 7)

        # print('------NLTK ----------')
        # print(nltk_sum)
        # print('------Spacy-----------')
        # print(spacy_sum)
       
        return render_template('result.html', transcript=res, url="https://www.youtube.com/embed/" + id, summary=summary)

if "__main__" == __name__:
    app.run(debug=True)
