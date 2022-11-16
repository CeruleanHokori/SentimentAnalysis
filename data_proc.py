import csv

#Opening file
def open_file(filepath): #Returns an iterator on file lines
    file = open(filepath,"r",newline="",encoding="utf8")
    reader = csv.reader(file,delimiter=",")
    return reader

#Encoding/Decoding labels
def encoder(sentiment):
    return 0*(sentiment == "Positive") + 1*(sentiment == "Neutral") + 2*(sentiment == "Negative")
def decoder(sentiment):
    return "Positive"*(sentiment == 0) + "Neutral"*(sentiment == 1) + "Negative"*(sentiment == 2)


def line_analyzer(line): #Analyzes the line returned by open_file iterator to return (tweet,encoded(sentiment))
    if line != []:
        return line[3],encoder(line[2])
    else:
        print("Line seems to be empty")

def samples_labels(filepath): #Returns X,y samples and labels lists as a tuple
    reader = open_file(filepath) #We open the reader
    X,y = [],[]
    for line in reader:
        tweet,sentiment = line_analyzer(line)
        X.append(tweet)
        y.append(sentiment)
    return X,y

#Example to extract training samples and labels
#X,y = samples_labels("data/twitter_training.csv")



