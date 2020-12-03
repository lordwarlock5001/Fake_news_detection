import pandas as pd
import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import joblib
import tkinter as tk
import tkinter.messagebox as tmsg
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
pa_classifier=PassiveAggressiveClassifier(max_iter=50)
root=tk.Tk()
labela=tk.Label(root,text="")
def train():
    df=pd.read_csv('dataset/fake-news/train.csv')
    print(df.shape)
    print(df.head())
    df.loc[(df['label'] == 1) , ['label']] = 'FAKE'
    df.loc[(df['label'] == 0) , ['label']] = 'REAL'
    labels = df.label
    print(labels.head())
    true_data=df
    true_data=true_data.drop(columns=["title","author","id"])
    true_data=true_data.set_index("label")
    true_data=true_data.groupby('label')
    true_data=true_data.count()
    print(true_data)
    true_data.plot(kind='bar')
    plt.title("Bar Chart")
    #plt.show()
    plt.savefig("bar.png")
    print("----Start Training------")
    x_train,x_test,y_train,y_test=train_test_split(df['text'].values.astype('str'), labels, test_size=0.02, random_state=7)


    tfidf_train=tfidf_vectorizer.fit_transform(x_train)
    tfidf_test=tfidf_vectorizer.transform(x_test)


    pa_classifier.fit(tfidf_train,y_train)
    file="model.sav"
    joblib.dump(pa_classifier,file)
    joblib.dump(tfidf_vectorizer,"v.sav")
    print("----Training Finished------")
    print("----Start Predication------")
    y_pred=pa_classifier.predict(tfidf_test)
    score=accuracy_score(y_test,y_pred)
    print("----Predication Finished------")
    print(f'Accuracy: {round(score*100,2)}%')
    labela.config(text=f'Accuracy: {round(score*100,2)}%')
    confusion_matrix1=confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
    print(confusion_matrix1)
    plt.figure(figsize=(2,2))
    sn.heatmap(confusion_matrix1,annot=True,xticklabels=['FAKE','REAL'],yticklabels=['FAKE','REAL'])
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.show()
def find(str):
    test_data=np.array([str])
    print(test_data)
    tf=tfidf_vectorizer.transform(test_data)
    pred=pa_classifier.predict(tf)
    print(pred)
    tmsg.showinfo("Result",pred[0])

btn1=tk.Button(root,text="Train Model",command=train)
btn1.pack()

labela.pack()
textbox=tk.Text(root,height=10,width=40)
textbox.pack()
btn=tk.Button(root,text="Find",command=lambda :find(textbox.get("1.0","end-1c")))
btn.pack()
root.mainloop()
