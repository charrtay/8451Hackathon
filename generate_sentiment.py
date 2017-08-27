import string
import csv
import pandas as pd

from afinn import Afinn


#Preprocessing the data -- Need to add to access only twitter
rawData = pd.read_csv('Test.csv')
rawData.to_csv('First.csv',sep = ',')
with open('First.csv') as file:
    reader = csv.reader(file)
    with open('Preprocessed.csv','w',newline = '') as output_file:
        writer = csv.writer(output_file, delimiter = ',')
        for rows in reader:
            line = rows[11]
            line = line.translate(str.maketrans('', '', string.punctuation))
            line = line.translate(str.maketrans('', '', string.digits))
            rows[11] = line
            writer.writerow(rows)





# Cleaning the data and finding the sentiment score for each tweet in the dataset.

df= pd.read_csv('Preprocessed.csv',index_col=0)
length=df.__len__()
#print length

df=df.dropna()

afinn = Afinn()
pscore = []

#This gives the sentiment score for every tweet
for text in df['Text']:
    pscore.append(afinn.score(text))
df['pscore']= pscore
#print df.__len__()
df.to_csv('Sentiment_data.csv', delimiter = ',')