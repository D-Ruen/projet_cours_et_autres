from six import StringIO
import pandas as pd
import numpy as np
from sklearn import tree
import pydot


def getFeatures():
    features = ['Age', 
                'Sex', 
                'Chest pain', 
                'resting blood pressure',
                'serum cholestoral',
                'fasting blood sugar', 
                'resting electrocardiographic',
                'max heart rate', 
                'exercise induced angina',
                'oldpeak',
                'ST segment',
                'major vessels']
    return features
    

def generateSubmission(model, test_path='./test.txt'):
    """ Generates a CSV submission file for the Kaggle competition from a trained model and test data file.
    
    Parameters
    ----------
        - model: your decision tree
        - test_path: path to the test data file (default: './test.txt')
        
    """
    
    X_test = np.loadtxt(test_path)
    pred = model.predict(X_test)
    idx = np.array(list(range(pred.shape[0])))
    df = pd.DataFrame(data=np.concatenate((idx[:,None], pred[:,None]), axis=1),
                      index=None,
                      columns=['PatientId', 'label'])
    df.to_csv('./submission.csv', index=False)



def DT_to_PNG(model, feature_names, file_name):
    """ Exports a DT to a PNG image file for inspection.
    
    Parameters
    ----------
        - model: a decision tree (class sklearn.tree.DecisionTreeClassifier)
        - feature_names: a list of feature names
        - file_name: name of file to be produced (without '.png' extension)
    
    Notes
    -----
    This function requires the pydot Python package and the Graphviz library.
    
    For more information about tree export, see http://scikit-learn.org/stable/
    modules/generated/sklearn.tree.export_graphviz.html#sklearn.tree.export_graphviz
    """

    dot_data = StringIO()
    tree.export_graphviz(model, out_file=dot_data, feature_names=feature_names)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]
    graph.write_png("%s.png" % file_name)