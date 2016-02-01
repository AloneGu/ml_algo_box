from sklearn import tree
import pydot
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.cross_validation import cross_val_score, train_test_split

def dct_process(x_data,y_data,file_name):
    feature_name = ['variance','average','max','min','data_count']
    class_name = ['close','open']
    clf = tree.DecisionTreeClassifier(max_depth=3)
    clf.fit(x_data,y_data)
    print 'scores with 3 fold', cross_val_score(clf,x_data,y_data)

    with open("iris.dot", 'w') as f:
        f = tree.export_graphviz(clf, out_file=f)

    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,feature_names=feature_name,class_names=class_name)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(file_name+".pdf")


file_name = 'audio_training.txt'
content = open(file_name).read()
x_data_0,y_data_0=[],[]
x_data_1,y_data_1=[],[]

rows = content.split('\r\n')
for r in rows:
    try:
        index = r.find(':')
        tag = r[:index]
        data = r[index + 1:]
        data = map(int, data.split(','))
        if tag == '0':
            x_data_0.append(data)
            y_data_0.append(int(tag))
        else:
            x_data_1.append(data)
            y_data_1.append(int(tag))
    except:
        pass
x_data = x_data_0 + x_data_1
y_data = y_data_0 + y_data_1
dct_process(x_data,y_data,'full_train')
x_train_new_1, X_test, y_train_new_1, y_test = train_test_split(x_data_1, y_data_1, test_size=0.6, random_state=0)
x_data = x_data_0 + x_train_new_1
y_data = y_data_0 + y_train_new_1
dct_process(x_data,y_data,'pre_processed_train')