import jsonpickle
import json
import pickle

with open('generations.pickle', 'wb') as f:
    pickle.dump(generations, f)

with open('generations.pickle') as f:
    loaded_obj = pickle.load(f)


jsonpickle.encode(generations, unpicklable=False)

text_file = open("generations.json", "w")
n = text_file.write(jsonpickle.encode(generations, unpicklable=False))
text_file.close()

gg = jsonpickle.encode(generations, unpicklable=False)
jj = json.loads(gg)
len(jj) #number of individuals


#Save file
text_file = open("sample.txt", "w")
n = text_file.write('Welcome to pythonexamples.org')
text_file.close()