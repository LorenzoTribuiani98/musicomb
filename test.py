
from feature_extractor.SVMClassifier import SVMClassifier
import matplotlib.pyplot as plt

classifier = SVMClassifier(['dataset/new_midis/commu00002.mid']) 

feat = classifier.preprocess__in(classifier.midis[0])

plt.bar(['mean chord number','mean note number', 'mean chord note distance', 'mean note distance', 'mean notes per chord', 'min octave', 'max octave', 'mean octave', 'mean chord duration', 'mean notes duration', 'instrument'],
        feat)
plt.xticks(rotation=90)
plt.tick_params(axis='x',labelsize=30)
plt.subplots_adjust()
plt.grid()
plt.show()