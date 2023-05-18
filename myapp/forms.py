from django import forms

class MyForm(forms.Form):
    text = forms.CharField(label='Text')