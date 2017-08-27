import string
import csv
import pandas as pd
import matplotlib as plt
import numpy
from scipy.stats.stats import pearsonr

#from afinn import Afinn
#Preprocessing the data -- Need to add to access only twitter
'''
rawData = pd.read_csv('Reviews.csv')
rawData.to_csv('First.csv',sep = ',')
with open('First.csv') as file:
    reader = csv.reader(file)
    with open('Preprocessed.csv','w',newline = '') as output_file:
        writer = csv.writer(output_file, delimiter = ',')
        for rows in reader:
            line = rows[10]
            line = line.translate(str.maketrans('', '', string.punctuation))
            line = line.translate(str.maketrans('', '', string.digits))
            rows[10] = line
            writer.writerow(rows)
'''
'''
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
df['SentimentScore']= pscore
#print df.__len__()
#print ("Printing")
df.to_csv('Sentiment_data.csv')

#######################Corelation Matrix
df = pd.read_csv('Correlation bertween Sentiment versus Score.csv',names = [0,1])
print(df.corr())



#Ground truth
ratioList = []
dataframe = pd.read_csv('Ground Truth.csv')
helpfulNumerator = dataframe['HelpfulnessNumerator'].tolist()
helpfulDenominator = dataframe['HelpfulnessDenominator'].tolist()
for i in range(len(helpfulDenominator)):
    try:
        ratio = float(helpfulNumerator[i]) / float(helpfulDenominator[i])
    except ZeroDivisionError:
        ratio = 0
    if(helpfulNumerator[i] == 0):
        ratioList.append(ratio)
    elif(helpfulNumerator[i] >= 1 and helpfulNumerator[i] <= 10):
        ratioList.append((0.50 *ratio))
    elif(helpfulNumerator[i] > 10 and helpfulNumerator[i] <= 25):
        ratioList.append((0.75 * ratio))
    elif(helpfulNumerator[i] > 25 and helpfulNumerator[i] <= 75):
        ratioList.append((1 * ratio))
    elif (helpfulNumerator[i] > 75):
        ratioList.append((1.25 * ratio))
groundtruth = sum(ratioList)/ len(ratioList)
print("The ground truth is", groundtruth)
with open('ratioList.csv', 'w', newline = '') as myfile:
    wr = csv.writer(myfile)
    for value in ratioList:
        wr.writerow([value])

'''

dataframe = pd.read_csv('Correlation bertween Sentiment versus Score.csv')
sentimentScore = dataframe['SentimentScore'].tolist()
reviewScore = dataframe['Score'].tolist()
print (numpy.corrcoef2(sentimentScore,reviewScore))
