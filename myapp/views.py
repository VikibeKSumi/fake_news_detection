from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, redirect
from .forms import MyForm
from django.urls import reverse

import re
import pickle
from nltk.stem.porter import PorterStemmer


def home(request):
    return render(request, 'home.html')


def stopwords():
    stopwords_lst = ['i','me','my','myself','we', 'our', 'ours', 'ourselves', 'you', "you're", "you've","you'll", "you'd", 'your','yours', 'yourself', 'yourselves', 'he', 'him','his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it',"it's", 'its','itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these','those','am', 'is', 'are', 'was', 'were', 'be','been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a','an', 'the','and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor','not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y','ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",'wasn',"wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    return stopwords_lst

def text_cleaning(message):
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', message)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords()]
    review = ' '.join(review)
    return review


def action(request):

    if request.method == 'POST':
        # All post request will be processed here
        form = MyForm(request.POST)
        if form.is_valid():
            text = form.cleaned_data['text']
        
            # text processing
            text = text_cleaning(text)
            # text vectorization
            with open('myapp\\vectorizer_small3', 'rb') as f:
                vectorizer = pickle.load(f)
            X_vectors = vectorizer.transform([text])
            # case prediction
            with open('myapp\\classifier_news', 'rb') as f:
                model = pickle.load(f)
            prediction = model.predict(X_vectors)

            if prediction == 1:
                result = 'News is True'
            elif prediction == 0:
                result = 'News is False'

            context = {'result':result}
            return render(request, 'home.html', context)
        else:
            msg = 'Text field cannot be empty'
            return render(request, 'home.html', {'msg': msg})

    return render(request, 'home.html')
    
def success(request, result):
    return render(request, 'success.html', {'result':result})
