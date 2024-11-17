#Training the classifier model for ISL
#using pickle datasets which contains bytes of information from the extracted landmarks

import pickle
import numpy
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.preprocessing.sequence import pad_sequences  # Import pad_sequences




data_dict = pickle.load(open('./data.pickle' , 'rb'))

# print(type(data_dict['data']))
# print(len(data_dict['data']))
# for i, item in enumerate(data_dict['data']):
#     print(f"Item {i}: {type(item)}, Length: {len(item) if hasattr(item, '__len__') else 'N/A'}")

#data = np.asarray(data_dict['data'])
padded_data = pad_sequences(data_dict['data'], padding='post')
labels = np.asarray(data_dict['labels'])

x_train, x_test , y_train, y_test = train_test_split(padded_data, labels , test_size=0.2,
                                                     shuffle=True,
                                                     stratify=labels)
model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)


score = accuracy_score(y_predict , y_test)
print('{}% of samples were classified correctly!'.format(score*100))

f= open('model.p' , 'wb')
pickle.dump({'model' : model} , f)
f.close()

