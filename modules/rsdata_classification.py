import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as py
from plotly.subplots import make_subplots
py.init_notebook_mode(connected=True)
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, fbeta_score, jaccard_score, log_loss, precision_score, recall_score, roc_auc_score #ver se precisa tudo isso
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

__name__ = "rsdata_classification"

def k_means(dataset, features, k, random_state):
    X = dataset[features]
    clf = KMeans(n_clusters=k, random_state=random_state).fit(X)
    dataset['Cluster'] = clf.labels_
    
    return dataset

def multilayer_perceptron(X, y, X_real, y_real, n_epochs):   
    assessment = dict()
    confusion_matrices = dict()
    acc = dict()
    b_acc = dict()
    f1_macro = dict()
    f1_weighted = dict()
    fbeta_macro = dict() 
    fbeta_weighted = dict()
    pr_macro = dict()
    rc_macro = dict()
    pr_weighted = dict()
    rc_weighted = dict()
    
    for i in range(0, n_epochs):
        # holdout
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=i)
        
        neural_network = MLPClassifier(activation='identity', 
                               alpha=1e-05, 
                               hidden_layer_sizes=20, 
                               max_iter=100, 
                               solver='lbfgs', 
                               random_state=i).fit(X, y)#treina com 100% do dart

        y_pred = neural_network.predict(X_real)#testa com usgs

        error_analysis = pd.DataFrame([],columns=['Real', 'Prediction', 'Status'])
        error_analysis['Real'] = y_real
        error_analysis['Prediction'] =  y_pred
        error_analysis['Status'] = (error_analysis['Prediction'] == error_analysis['Real'])
        
        assessment.update({i: error_analysis})
        confusion_matrices.update({i:confusion_matrix(y_real, y_pred, labels=['Water', 'Plastic'])})
        acc.update({i:accuracy_score(y_real, y_pred)}) 
        b_acc.update({i:balanced_accuracy_score(y_real, y_pred)}) 
        f1_macro.update({i:f1_score(y_real, y_pred, average='macro')}) #F1micro é igual a acurácia geral
        f1_weighted.update({i:f1_score(y_real, y_pred, average='weighted')})
        fbeta_macro.update({i:fbeta_score(y_real, y_pred, average='macro', beta=0.5)})
        fbeta_weighted.update({i:fbeta_score(y_real, y_pred, average='weighted', beta=0.5)})
        pr_macro.update({i:precision_score(y_real, y_pred, average='macro')}) 
        pr_weighted.update({i:precision_score(y_real, y_pred, average='weighted')}) 
        rc_macro.update({i:recall_score(y_real, y_pred, average='macro')})
        rc_weighted.update({i:recall_score(y_real, y_pred, average='weighted')})
        
        errors = X_real.loc[assessment[i].query('Status == False').index]
        hits = X_real.loc[assessment[i].query('Status == True').index]
        
    return assessment, errors, hits, confusion_matrices, acc, b_acc, f1_macro, f1_weighted, fbeta_macro, fbeta_weighted, pr_macro, rc_macro, pr_weighted, rc_weighted

def stats_classification(assessment, ground_truth):   
    errors_bags = []
    errors_bottles = []
    errors_bags_bottles = []
    errors_hdpe = []
    hits_bags = []
    hits_bottles = []
    hits_bags_bottles = []
    hits_hdpe = []

    errors_2021_p = []
    errors_2019_p = []
    hits_2021_p = []
    hits_2019_p = []
    errors_2021_w = []
    errors_2019_w = []
    hits_2021_w = []
    hits_2019_w = []

    errors_cp_25 = []
    errors_cp_50 = []
    errors_cp_100 = []
    errors_cp_unknown = []
    errors_cp_unknown_100 = []
    hits_cp_25 = []
    hits_cp_50 = []
    hits_cp_100 = []
    hits_cp_unknown = []
    hits_cp_unknown_100 = []

    errors_2019_04_18 = []
    errors_2019_05_03 = []
    errors_2019_05_18 = []
    errors_2019_05_28 = []
    errors_2019_06_07 = []
    errors_2021_06_21 = []
    errors_2021_07_01 = []
    errors_2021_07_06 = []
    errors_2021_07_21 = []
    errors_2021_08_25 = [] 
    hits_2019_04_18 = []
    hits_2019_05_03 = []
    hits_2019_05_18 = []
    hits_2019_05_28 = []
    hits_2019_06_07 = []
    hits_2021_06_21 = []
    hits_2021_07_01 = []
    hits_2021_07_06 = []
    hits_2021_07_21 = []
    hits_2021_08_25 = [] 


    for i in assessment.keys():
        errors = ground_truth.loc[assessment[i].query('Status == False').index]
        hits = ground_truth.loc[assessment[i].query('Status == True').index]

        #Repetir 100 vezes e ver se a tendencia se mantem (mais acertos em 2021/malha hdpe e erros e 2019/sacolas e garrafas)
        errors_bags.append(errors.query('Label == "Plastic" and Polymer == "Bags"')['Polymer'].count()) 
        errors_bottles.append(errors.query('Label == "Plastic" and Polymer == "Bottles"')['Polymer'].count()) 
        errors_bags_bottles.append(errors.query('Label == "Plastic" and Polymer == "Bags and Bottles"')['Polymer'].count()) 
        errors_hdpe.append(errors.query('Label == "Plastic" and Polymer == "HDPE mesh"')['Polymer'].count()) 
        hits_bags.append(hits.query('Label == "Plastic" and Polymer == "Bags"')['Polymer'].count()) 
        hits_bottles.append(hits.query('Label == "Plastic" and Polymer == "Bottles"')['Polymer'].count()) 
        hits_bags_bottles.append(hits.query('Label == "Plastic" and Polymer == "Bags and Bottles"')['Polymer'].count()) 
        hits_hdpe.append(hits.query('Label == "Plastic" and Polymer == "HDPE mesh"')['Polymer'].count()) 

        errors_2021_p.append(errors.query('Label == "Plastic" and Year == "2021"')['Year'].count())
        errors_2019_p.append(errors.query('Label == "Plastic" and Year == "2019"')['Year'].count())
        hits_2021_p.append(hits.query('Label == "Plastic" and Year == "2021"')['Year'].count())
        hits_2019_p.append(hits.query('Label == "Plastic" and Year == "2019"')['Year'].count())

        errors_2021_w.append(errors.query('Label == "Water" and Year == "2021"')['Year'].count())
        errors_2019_w.append(errors.query('Label == "Water" and Year == "2019"')['Year'].count())
        hits_2021_w.append(hits.query('Label == "Water" and Year == "2021"')['Year'].count())
        hits_2019_w.append(hits.query('Label == "Water" and Year == "2019"')['Year'].count())

        errors_cp_25.append(errors.query('Label == "Plastic" and Cover_percent > 0 and Cover_percent <= 25')['Cover_percent'].count())
        errors_cp_50.append(errors.query('Label == "Plastic" and Cover_percent > 25 and Cover_percent <= 50')['Cover_percent'].count())
        errors_cp_100.append(errors.query('Label == "Plastic" and Cover_percent > 50')['Cover_percent'].count())
        errors_cp_unknown.append(errors.query('Label == "Plastic" and Cover_percent < 0')['Cover_percent'].count())
        errors_cp_unknown_100.append(errors.query('Label == "Plastic" and Cover_percent < -99')['Cover_percent'].count())
        hits_cp_25.append(hits.query('Label == "Plastic" and Cover_percent > 0 and Cover_percent <= 25')['Cover_percent'].count())
        hits_cp_50.append(hits.query('Label == "Plastic" and Cover_percent > 25 and Cover_percent <= 50')['Cover_percent'].count())
        hits_cp_100.append(hits.query('Label == "Plastic" and Cover_percent > 50')['Cover_percent'].count())
        hits_cp_unknown.append(hits.query('Label == "Plastic" and Cover_percent == -1')['Cover_percent'].count())
        hits_cp_unknown_100.append(hits.query('Label == "Plastic" and Cover_percent == -100')['Cover_percent'].count())

        errors_2019_04_18.append(errors.query('Path == "2019_04_18"')['Path'].count())
        errors_2019_05_03.append(errors.query('Path == "2019_05_03"')['Path'].count())
        errors_2019_05_18.append(errors.query('Path == "2019_05_18"')['Path'].count())
        errors_2019_05_28.append(errors.query('Path == "2019_05_28"')['Path'].count())
        errors_2019_06_07.append(errors.query('Path == "2019_06_07"')['Path'].count())
        errors_2021_06_21.append(errors.query('Path == "2021_06_21"')['Path'].count())
        errors_2021_07_01.append(errors.query('Path == "2021_07_01"')['Path'].count())
        errors_2021_07_06.append(errors.query('Path == "2021_07_06"')['Path'].count())
        errors_2021_07_21.append(errors.query('Path == "2021_07_21"')['Path'].count())
        errors_2021_08_25.append(errors.query('Path == "2021_08_25"')['Path'].count())
        hits_2019_04_18.append(hits.query('Path == "2019_04_18"')['Path'].count())
        hits_2019_05_03.append(hits.query('Path == "2019_05_03"')['Path'].count())
        hits_2019_05_18.append(hits.query('Path == "2019_05_18"')['Path'].count())
        hits_2019_05_28.append(hits.query('Path == "2019_05_28"')['Path'].count())
        hits_2019_06_07.append(hits.query('Path == "2019_06_07"')['Path'].count())
        hits_2021_06_21.append(hits.query('Path == "2021_06_21"')['Path'].count())
        hits_2021_07_01.append(hits.query('Path == "2021_07_01"')['Path'].count())
        hits_2021_07_06.append(hits.query('Path == "2021_07_06"')['Path'].count())
        hits_2021_07_21.append(hits.query('Path == "2021_07_21"')['Path'].count())
        hits_2021_08_25.append(hits.query('Path == "2021_08_25"')['Path'].count())


    errors_bags = pd.DataFrame(errors_bags, columns=['Error_bags'])
    errors_bottles = pd.DataFrame(errors_bottles, columns=['Error_bottles'])
    errors_bags_bottles = pd.DataFrame(errors_bags_bottles, columns=['Error_bags_bottles'])
    errors_hdpe = pd.DataFrame(errors_hdpe, columns=['Error_hdpe'])

    hits_bags = pd.DataFrame(hits_bags, columns=['Hits_bags'])
    hits_bottles = pd.DataFrame(hits_bottles, columns=['Hits_bottles'])
    hits_bags_bottles = pd.DataFrame(hits_bags_bottles, columns=['Hits_bags_bottles'])
    hits_hdpe = pd.DataFrame(hits_hdpe, columns=['Hits_hdpe'])

    errors_2021_p = pd.DataFrame(errors_2021_p, columns=['Errors_2021_Plastic'])
    errors_2019_p = pd.DataFrame(errors_2019_p, columns=['Errors_2021_Plastic'])
    hits_2021_p = pd.DataFrame(hits_2021_p, columns=['Hits_2021_Plastic'])
    hits_2019_p = pd.DataFrame(hits_2019_p, columns=['Hits_2019_Plastic'])
    errors_2021_w = pd.DataFrame(errors_2021_w, columns=['Errors_2021_Water'])
    errors_2019_w = pd.DataFrame(errors_2019_w, columns=['Errors_2019_Water'])
    hits_2021_w = pd.DataFrame(hits_2021_w, columns=['Hits_2021_Water'])
    hits_2019_w = pd.DataFrame(hits_2019_w, columns=['Hits_2019_Water'])

    errors_cp_25 = pd.DataFrame(errors_cp_25, columns=['Errors_cp_25'])
    errors_cp_50 = pd.DataFrame(errors_cp_50, columns=['Errors_cp_50'])
    errors_cp_100 = pd.DataFrame(errors_cp_100, columns=['Errors_cp_100'])
    errors_cp_unknown = pd.DataFrame(errors_cp_unknown, columns=['Errors_cp_unknown'])
    errors_cp_unknown_100 = pd.DataFrame(errors_cp_unknown_100, columns=['Errors_cp_unknown_100'])
    hits_cp_25 = pd.DataFrame(hits_cp_25, columns=['Hits_cp_25'])
    hits_cp_50 = pd.DataFrame(hits_cp_50, columns=['Hits_cp_50'])
    hits_cp_100 = pd.DataFrame(hits_cp_100, columns=['Hits_cp_100'])
    hits_cp_unknown = pd.DataFrame(hits_cp_unknown, columns=['Hits_cp_unknown'])
    hits_cp_unknown_100 = pd.DataFrame(hits_cp_unknown_100, columns=['Hits_cp_unknown_100'])

    errors_2019_04_18 = pd.DataFrame(errors_2019_04_18, columns = ['errors_2019_04_18'])
    errors_2019_05_03 = pd.DataFrame(errors_2019_05_03, columns = ['errors_2019_05_03'])
    errors_2019_05_18 = pd.DataFrame(errors_2019_05_18, columns = ['errors_2019_05_18'])
    errors_2019_05_28 = pd.DataFrame(errors_2019_05_28, columns = ['errors_2019_05_28'])
    errors_2019_06_07 = pd.DataFrame(errors_2019_06_07, columns = ['errors_2019_06_07'])
    errors_2021_06_21 = pd.DataFrame(errors_2021_06_21, columns = ['errors_2021_06_21'])
    errors_2021_07_01 = pd.DataFrame(errors_2021_07_01, columns = ['errors_2021_07_01'])
    errors_2021_07_06 = pd.DataFrame(errors_2021_07_06, columns = ['errors_2021_07_06'])
    errors_2021_07_21 = pd.DataFrame(errors_2021_07_21, columns = ['errors_2021_07_21'])
    errors_2021_08_25 = pd.DataFrame(errors_2021_08_25, columns = ['errors_2021_08_25'])
    hits_2019_04_18 = pd.DataFrame(hits_2019_04_18, columns = ['hits_2019_04_18'])
    hits_2019_05_03 = pd.DataFrame(hits_2019_05_03, columns = ['hits_2019_05_03'])
    hits_2019_05_18 = pd.DataFrame(hits_2019_05_18, columns = ['hits_2019_05_18'])
    hits_2019_05_28 = pd.DataFrame(hits_2019_05_28, columns = ['hits_2019_05_28'])
    hits_2019_06_07 = pd.DataFrame(hits_2019_06_07, columns = ['hits_2019_06_07'])
    hits_2021_06_21 = pd.DataFrame(hits_2021_06_21, columns = ['hits_2021_06_21'])
    hits_2021_07_01 = pd.DataFrame(hits_2021_07_01, columns = ['hits_2021_07_01'])
    hits_2021_07_06 = pd.DataFrame(hits_2021_07_06, columns = ['hits_2021_07_06'])
    hits_2021_07_21 = pd.DataFrame(hits_2021_07_21, columns = ['hits_2021_07_21'])
    hits_2021_08_25 = pd.DataFrame(hits_2021_08_25, columns = ['hits_2021_08_25'])
    
    stats_by_polymer = pd.DataFrame([
                                        [
                                             'Bags', 
                                             ground_truth.query('Polymer == "Bags"')['Label'].count(), 
                                             str(hits_bags['Hits_bags'].mean()) + ' (' + 
                                                str(round(hits_bags['Hits_bags'].mean() / 
                                                ground_truth.query('Polymer == "Bags"')['Label'].count() * 100 ,1)) + ' %)',
                                             str(hits_bags['Hits_bags'].min()) + ' (' + 
                                                str(round(hits_bags['Hits_bags'].min() / 
                                                ground_truth.query('Polymer == "Bags"')['Label'].count() * 100 ,1)) + ' %)', 
                                             str(hits_bags['Hits_bags'].max()) + ' (' + 
                                                str(round(hits_bags['Hits_bags'].max() / 
                                                ground_truth.query('Polymer == "Bags"')['Label'].count() * 100 ,1)) + ' %)',
                                             str(ground_truth.query('Polymer == "Bags"')['Cover_percent'].mean()) + ' %'
                                        ],
                                        [
                                            'Bottles', 
                                             ground_truth.query('Polymer == "Bottles"')['Label'].count(), 
                                                 str(hits_bottles['Hits_bottles'].mean()) + ' (' + 
                                                 str(round(hits_bottles['Hits_bottles'].mean() / 
                                                 ground_truth.query('Polymer == "Bottles"')['Label'].count() * 100, 1)) + ' %)',
                                             str(hits_bottles['Hits_bottles'].min()) + ' (' + 
                                                str(round(hits_bottles['Hits_bottles'].min() / 
                                                ground_truth.query('Polymer == "Bottles"')['Label'].count() * 100 ,1)) + ' %)', 
                                             str(hits_bottles['Hits_bottles'].max()) + ' (' + 
                                                str(round(hits_bottles['Hits_bottles'].max() / 
                                                ground_truth.query('Polymer == "Bottles"')['Label'].count() * 100 ,1)) + ' %)',
                                             str(ground_truth.query('Polymer == "Bottles"')['Cover_percent'].mean()) + ' %'
                                        ],
                                        [
                                            'Bags and Bottles', 
                                             ground_truth.query('Polymer == "Bags and Bottles"')['Label'].count(), 
                                                 str(hits_bags_bottles['Hits_bags_bottles'].mean()) + ' (' + 
                                                 str(round(hits_bags_bottles['Hits_bags_bottles'].mean() / 
                                                 ground_truth.query('Polymer == "Bags and Bottles"')['Label'].count() * 100, 1)) + ' %)',
                                             str(hits_bags_bottles['Hits_bags_bottles'].min()) + ' (' + 
                                                str(round(hits_bags_bottles['Hits_bags_bottles'].min() / 
                                                ground_truth.query('Polymer == "Bags and Bottles"')['Label'].count() * 100 ,1)) + ' %)', 
                                             str(hits_bags_bottles['Hits_bags_bottles'].max()) + ' (' + 
                                                str(round(hits_bags_bottles['Hits_bags_bottles'].max() / 
                                                ground_truth.query('Polymer == "Bags and Bottles"')['Label'].count() * 100 ,1)) + ' %)',
                                             str(ground_truth.query('Polymer == "Bags and Bottles"')['Cover_percent'].mean()) + ' %'
                                        ],
                                        [
                                            'HDPE mesh', 
                                             ground_truth.query('Polymer == "HDPE mesh"')['Label'].count(), 
                                                 str(hits_hdpe['Hits_hdpe'].mean()) + ' (' + 
                                                 str(round(hits_hdpe['Hits_hdpe'].mean() / 
                                                 ground_truth.query('Polymer == "HDPE mesh"')['Label'].count() * 100, 1)) + ' %)',
                                             str(hits_hdpe['Hits_hdpe'].min()) + ' (' + 
                                                str(round(hits_hdpe['Hits_hdpe'].min() / 
                                                ground_truth.query('Polymer == "HDPE mesh"')['Label'].count() * 100 ,1)) + ' %)', 
                                             str(hits_hdpe['Hits_hdpe'].max()) + ' (' + 
                                                str(round(hits_hdpe['Hits_hdpe'].max() / 
                                                ground_truth.query('Polymer == "HDPE mesh"')['Label'].count() * 100 ,1)) + ' %)',
                                             '100% (15 samples) and < 100% (39 samples)'
                                        ]
                                    ],
                                    columns = ['Polymer', 'Total', 'Mean hits', 'Min hits', 'Max hits', 'Mean cover percent'])
    
    stats_by_label_year = pd.DataFrame([
                                        [
                                            '2019', 
                                            'Water', 
                                             ground_truth.query('Label == "Water" and Year == "2019"')['Label'].count(), 
                                             str(hits_2019_w['Hits_2019_Water'].mean()) + ' (' + 
                                                str(round(hits_2019_w['Hits_2019_Water'].mean() / 
                                                ground_truth.query('Label == "Water" and Year == "2019"')['Label'].count() * 100 ,1)) + ' %)',
                                             str(hits_2019_w['Hits_2019_Water'].min()) + ' (' + 
                                                str(round(hits_2019_w['Hits_2019_Water'].min() / 
                                                ground_truth.query('Label == "Water" and Year == "2019"')['Label'].count() * 100 ,1)) + ' %)', 
                                             str(hits_2019_w['Hits_2019_Water'].max()) + ' (' + 
                                                str(round(hits_2019_w['Hits_2019_Water'].max() / 
                                                ground_truth.query('Label == "Water" and Year == "2019"')['Label'].count() * 100 ,1)) + ' %)',
                                             str(round(ground_truth.query('Label == "Water" and Year == "2019"')['Cover_percent'].mean(),2)) + ' %'
                                        ],
                                        [
                                            '2019', 
                                            'Plastic',
                                             ground_truth.query('Label == "Plastic" and Year == "2019"')['Label'].count(), 
                                                 str(hits_2019_p['Hits_2019_Plastic'].mean()) + ' (' + 
                                                 str(round(hits_2019_p['Hits_2019_Plastic'].mean() / 
                                                 ground_truth.query('Label == "Plastic" and Year == "2019"')['Label'].count() * 100, 1)) + ' %)',
                                             str(hits_2019_p['Hits_2019_Plastic'].min()) + ' (' + 
                                                str(round(hits_2019_p['Hits_2019_Plastic'].min() / 
                                                ground_truth.query('Label == "Plastic" and Year == "2019"')['Label'].count() * 100 ,1)) + ' %)', 
                                             str(hits_2019_p['Hits_2019_Plastic'].max()) + ' (' + 
                                                str(round(hits_2019_p['Hits_2019_Plastic'].max() / 
                                                ground_truth.query('Label == "Plastic" and Year == "2019"')['Label'].count() * 100 ,1)) + ' %)',
                                             str(round(ground_truth.query('Label == "Plastic" and Year == "2019"')['Cover_percent'].mean(),2)) + ' %'
                                        ],
                                        [
                                            '2021', 
                                            'Water', 
                                             ground_truth.query('Label == "Water" and Year == "2021"')['Label'].count(), 
                                                 str(hits_2021_w['Hits_2021_Water'].mean()) + ' (' + 
                                                 str(round(hits_2021_w['Hits_2021_Water'].mean() / 
                                                 ground_truth.query('Label == "Water" and Year == "2021"')['Label'].count() * 100, 1)) + ' %)',
                                             str(hits_2021_w['Hits_2021_Water'].min()) + ' (' + 
                                                str(round(hits_2021_w['Hits_2021_Water'].min() / 
                                                ground_truth.query('Label == "Water" and Year == "2021"')['Label'].count() * 100 ,1)) + ' %)', 
                                             str(hits_2021_w['Hits_2021_Water'].max()) + ' (' + 
                                                str(round(hits_2021_w['Hits_2021_Water'].max() / 
                                                ground_truth.query('Label == "Water" and Year == "2021"')['Label'].count() * 100 ,1)) + ' %)',
                                             str(round(ground_truth.query('Label == "Water" and Year == "2021"')['Cover_percent'].mean(),2)) + ' %'
                                        ],
                                        [
                                            '2021', 
                                            'Plastic', 
                                             ground_truth.query('Label == "Plastic" and Year == "2021"')['Label'].count(), 
                                                 str(hits_2021_p['Hits_2021_Plastic'].mean()) + ' (' + 
                                                 str(round(hits_2021_p['Hits_2021_Plastic'].mean() / 
                                                 ground_truth.query('Label == "Plastic" and Year == "2021"')['Label'].count() * 100, 1)) + ' %)',
                                             str(hits_2021_p['Hits_2021_Plastic'].min()) + ' (' + 
                                                str(round(hits_2021_p['Hits_2021_Plastic'].min() / 
                                                ground_truth.query('Label == "Plastic" and Year == "2021"')['Label'].count() * 100 ,1)) + ' %)', 
                                             str(hits_2021_p['Hits_2021_Plastic'].max()) + ' (' + 
                                                str(round(hits_2021_p['Hits_2021_Plastic'].max() / 
                                                ground_truth.query('Label == "Plastic" and Year == "2021"')['Label'].count() * 100 ,1)) + ' %)',
                                             '100% (15 samples) and < 100% (39 samples)'
                                        ]
                                    ],
                                    columns = ['Year', 'Label', 'Total', 'Mean hits', 'Min hits', 'Max hits', 'Mean cover percent'])
    
    
    stats_by_plastic_cover_percent = pd.DataFrame([
                                            [
                                                '<=25%', 
                                                 ground_truth.query('Label == "Plastic" and Cover_percent > 0 and Cover_percent <= 25')['Label'].count(),
                                                 str(hits_cp_25['Hits_cp_25'].mean()) + ' (' + 
                                                    str(round(hits_cp_25['Hits_cp_25'].mean() / 
                                                    ground_truth.query('Label == "Plastic" and Cover_percent > 0 and Cover_percent <= 25')['Label'].count() * 100, 1)) + ' %)',
                                                 str(hits_cp_25['Hits_cp_25'].min()) + ' (' + 
                                                    str(round(hits_cp_25['Hits_cp_25'].min() / 
                                                    ground_truth.query('Label == "Plastic" and Cover_percent > 0 and Cover_percent <= 25')['Label'].count() * 100 ,1)) + ' %)', 
                                                 str(hits_cp_25['Hits_cp_25'].max()) + ' (' + 
                                                    str(round(hits_cp_25['Hits_cp_25'].max() / 
                                                    ground_truth.query('Label == "Plastic" and Cover_percent > 0 and Cover_percent <= 25')['Label'].count() * 100 ,1)) + ' %)',
                                       
                                                 str(round(ground_truth.query('Label == "Plastic" and Cover_percent > 0 and Cover_percent <= 25')['Cover_percent'].mean(),2)) + ' %'
                                            ],
                                        [
                                            '26% - 50%',
                                             ground_truth.query('Label == "Plastic" and Cover_percent > 25 and Cover_percent <= 50')['Label'].count(), 
                                                 str(hits_cp_50['Hits_cp_50'].mean()) + ' (' + 
                                                 str(round(hits_cp_50['Hits_cp_50'].mean() / 
                                                 ground_truth.query('Label == "Plastic" and Cover_percent > 25 and Cover_percent <= 50')['Label'].count() * 100, 1)) + ' %)',
                                             str(hits_cp_50['Hits_cp_50'].min()) + ' (' + 
                                                str(round(hits_cp_50['Hits_cp_50'].min() / 
                                                ground_truth.query('Label == "Plastic" and Cover_percent > 25 and Cover_percent <= 50')['Label'].count() * 100 ,1)) + ' %)', 
                                             str(hits_cp_50['Hits_cp_50'].max()) + ' (' + 
                                                str(round(hits_cp_50['Hits_cp_50'].max() / 
                                                ground_truth.query('Label == "Plastic" and Cover_percent > 25 and Cover_percent <= 50')['Label'].count() * 100 ,1)) + ' %)',
                                             str(round(ground_truth.query('Label == "Plastic" and Cover_percent > 25 and Cover_percent <= 50')['Cover_percent'].mean(),2)) + ' %'
                                        ],
                                        [
                                            '51% - 99%', 
                                             ground_truth.query('Label == "Plastic" and Cover_percent > 50')['Label'].count(), 
                                                 str(hits_cp_100['Hits_cp_100'].mean()) + ' (' + 
                                                 str(round(hits_cp_100['Hits_cp_100'].mean() / 
                                                 ground_truth.query('Label == "Plastic" and Cover_percent > 50')['Label'].count() * 100, 1)) + ' %)',
                                             str(hits_cp_100['Hits_cp_100'].min()) + ' (' + 
                                                str(round(hits_cp_100['Hits_cp_100'].min() / 
                                                ground_truth.query('Label == "Plastic" and Cover_percent > 50')['Label'].count() * 100 ,1)) + ' %)', 
                                             str(hits_cp_100['Hits_cp_100'].max()) + ' (' + 
                                                str(round(hits_cp_100['Hits_cp_100'].max() / 
                                                ground_truth.query('Label == "Plastic" and Cover_percent > 50')['Label'].count() * 100 ,1)) + ' %)',
                                             str(round(ground_truth.query('Label == "Plastic" and Cover_percent > 50')['Cover_percent'].mean(),2)) + ' %'
                                        ],
                                        [
                                            '100%', 
                                             ground_truth.query('Label == "Plastic" and Cover_percent == -100')['Label'].count(), 
                                                 str(hits_cp_unknown_100['Hits_cp_unknown_100'].mean()) + ' (' + 
                                                 str(round(hits_cp_unknown_100['Hits_cp_unknown_100'].mean() / 
                                                 ground_truth.query('Label == "Plastic" and Cover_percent == -100')['Label'].count() * 100, 1)) + ' %)',
                                             str(hits_cp_unknown_100['Hits_cp_unknown_100'].min()) + ' (' + 
                                                str(round(hits_cp_unknown_100['Hits_cp_unknown_100'].min() / 
                                                ground_truth.query('Label == "Plastic" and Cover_percent == -100')['Label'].count() * 100 ,1)) + ' %)', 
                                             str(hits_cp_unknown_100['Hits_cp_unknown_100'].max()) + ' (' + 
                                                str(round(hits_cp_unknown_100['Hits_cp_unknown_100'].max() / 
                                                ground_truth.query('Label == "Plastic" and Cover_percent == -100')['Label'].count() * 100 ,1)) + ' %)',#
                                             '100%'
                                        ],
                                        [
                                            'Unknown', 
                                             ground_truth.query('Label == "Plastic" and Cover_percent == -1')['Label'].count(), 
                                                 str(hits_cp_unknown['Hits_cp_unknown'].mean()) + ' (' + 
                                                 str(round(hits_cp_unknown['Hits_cp_unknown'].mean() / 
                                                 ground_truth.query('Label == "Plastic" and Cover_percent == -1')['Label'].count() * 100, 1)) + ' %)',
                                             str(hits_cp_unknown['Hits_cp_unknown'].min()) + ' (' + 
                                                str(round(hits_cp_unknown['Hits_cp_unknown'].min() / 
                                                ground_truth.query('Label == "Plastic" and Cover_percent == -1')['Label'].count() * 100 ,1)) + ' %)', 
                                             str(hits_cp_unknown['Hits_cp_unknown'].max()) + ' (' + 
                                                str(round(hits_cp_unknown['Hits_cp_unknown'].max() / 
                                                ground_truth.query('Label == "Plastic" and Cover_percent == -1')['Label'].count() * 100 ,1)) + ' %)',#
                                             'Unknown'
                                        ]
                                    ],
                                    columns = ['Cover_percent', 'Total', 'Mean hits', 'Min hits', 'Max hits', 'Mean cover percent'])


    stats_by_date = pd.DataFrame([
                                            [
                                                '18/04/2019', 
                                                 ground_truth.query('Path == "2019_04_18" and Label =="Water"')['Path'].count(),
                                                ground_truth.query('Path == "2019_04_18" and Label == "Plastic"')['Path'].count(),
                                                 str(hits_2019_04_18['hits_2019_04_18'].mean()) + ' (' + 
                                                    str(round(hits_2019_04_18['hits_2019_04_18'].mean() / 
                                                    ground_truth.query('Path == "2019_04_18"')['Path'].count() * 100, 1)) + ' %)',
                                                 str(hits_2019_04_18['hits_2019_04_18'].min()) + ' (' + 
                                                    str(round(hits_2019_04_18['hits_2019_04_18'].min() / 
                                                    ground_truth.query('Path == "2019_04_18"')['Path'].count() * 100 ,1)) + ' %)', 
                                                 str(hits_2019_04_18['hits_2019_04_18'].max()) + ' (' + 
                                                    str(round(hits_2019_04_18['hits_2019_04_18'].max() / 
                                                    ground_truth.query('Path == "2019_04_18"')['Path'].count() * 100 ,1)) + ' %)',
                                       
                                                 str(round(ground_truth.query('Label == "Plastic" and Path == "2019_04_18"')['Cover_percent'].mean(),2)) + ' %'
                                            ],
                                            [
                                                '03/05/2019', 
                                                 ground_truth.query('Path == "2019_05_03" and Label =="Water"')['Path'].count(),
                                                 ground_truth.query('Path == "2019_05_03" and Label == "Plastic"')['Path'].count(),
                                                 str(hits_2019_05_03['hits_2019_05_03'].mean()) + ' (' + 
                                                    str(round(hits_2019_05_03['hits_2019_05_03'].mean() / 
                                                    ground_truth.query('Path == "2019_05_03"')['Path'].count() * 100, 1)) + ' %)',
                                                 str(hits_2019_05_03['hits_2019_05_03'].min()) + ' (' + 
                                                    str(round(hits_2019_05_03['hits_2019_05_03'].min() / 
                                                    ground_truth.query('Path == "2019_05_03"')['Path'].count() * 100, 1)) + ' %)', 
                                                 str(hits_2019_05_03['hits_2019_05_03'].max()) + ' (' + 
                                                    str(round(hits_2019_05_03['hits_2019_05_03'].max() / 
                                                    ground_truth.query('Path == "2019_05_03"')['Path'].count() * 100, 1)) + ' %)',
                                       
                                                 str(round(ground_truth.query('Label == "Plastic" and Path == "2019_05_03"')['Cover_percent'].mean(),2)) + ' %'
                                            ],
                                            [
                                                '18/05/2019', 
                                                 ground_truth.query('Path == "2019_05_18" and Label == "Water"')['Path'].count(),
                                                 ground_truth.query('Path == "2019_05_18" and Label == "Plastic"')['Path'].count(),
                                                 str(hits_2019_05_18['hits_2019_05_18'].mean()) + ' (' + 
                                                    str(round(hits_2019_05_18['hits_2019_05_18'].mean() / 
                                                    ground_truth.query('Path == "2019_05_18"')['Path'].count() * 100, 1)) + ' %)',
                                                 str(hits_2019_05_18['hits_2019_05_18'].min()) + ' (' + 
                                                    str(round(hits_2019_05_18['hits_2019_05_18'].min() / 
                                                    ground_truth.query('Path == "2019_05_18"')['Path'].count() * 100, 1)) + ' %)', 
                                                 str(hits_2019_05_18['hits_2019_05_18'].max()) + ' (' + 
                                                    str(round(hits_2019_05_18['hits_2019_05_18'].max() / 
                                                    ground_truth.query('Path == "2019_05_18"')['Path'].count() * 100, 1)) + ' %)',
                                       
                                                 str(round(ground_truth.query('Label == "Plastic" and Path == "2019_05_18"')['Cover_percent'].mean(),2)) + ' %'
                                            ],
                                            [
                                                '28/05/2019', 
                                                 ground_truth.query('Path == "2019_05_28" and Label == "Water"')['Path'].count(),
                                                 ground_truth.query('Path == "2019_05_28" and Label == "Plastic"')['Path'].count(),
                                                 str(hits_2019_05_28['hits_2019_05_28'].mean()) + ' (' + 
                                                    str(round(hits_2019_05_28['hits_2019_05_28'].mean() / 
                                                    ground_truth.query('Path == "2019_05_28"')['Path'].count() * 100, 1)) + ' %)',
                                                 str(hits_2019_05_28['hits_2019_05_28'].min()) + ' (' + 
                                                    str(round(hits_2019_05_28['hits_2019_05_28'].min() / 
                                                    ground_truth.query('Path == "2019_05_28"')['Path'].count() * 100, 1)) + ' %)', 
                                                 str(hits_2019_05_28['hits_2019_05_28'].max()) + ' (' + 
                                                    str(round(hits_2019_05_28['hits_2019_05_28'].max() / 
                                                    ground_truth.query('Path == "2019_05_28"')['Path'].count() * 100, 1)) + ' %)',
                                       
                                                 str(round(ground_truth.query('Label == "Plastic" and Path == "2019_05_28"')['Cover_percent'].mean(),2)) + ' %'
                                            ],
                                            [
                                                '07/06/2019', 
                                                 ground_truth.query('Path == "2019_06_07" and Label == "Water"')['Path'].count(),
                                                 ground_truth.query('Path == "2019_06_07" and Label == "Plastic"')['Path'].count(),
                                                 str(hits_2019_06_07['hits_2019_06_07'].mean()) + ' (' + 
                                                    str(round(hits_2019_06_07['hits_2019_06_07'].mean() / 
                                                    ground_truth.query('Path == "2019_06_07"')['Path'].count() * 100, 1)) + ' %)',
                                                 str(hits_2019_06_07['hits_2019_06_07'].min()) + ' (' + 
                                                    str(round(hits_2019_06_07['hits_2019_06_07'].min() / 
                                                    ground_truth.query('Path == "2019_06_07"')['Path'].count() * 100, 1)) + ' %)', 
                                                 str(hits_2019_06_07['hits_2019_06_07'].max()) + ' (' + 
                                                    str(round(hits_2019_06_07['hits_2019_06_07'].max() / 
                                                    ground_truth.query('Path == "2019_06_07"')['Path'].count() * 100, 1)) + ' %)',
                                       
                                                 str(round(ground_truth.query('Label == "Plastic" and Path == "2019_06_07"')['Cover_percent'].mean(),2)) + ' %'
                                            ],
                                            [
                                                '21/06/2021', 
                                                 ground_truth.query('Path == "2021_06_21" and Label == "Water"')['Path'].count(),
                                                 ground_truth.query('Path == "2021_06_21" and Label == "Plastic"')['Path'].count(),
                                                 str(hits_2021_06_21['hits_2021_06_21'].mean()) + ' (' + 
                                                    str(round(hits_2021_06_21['hits_2021_06_21'].mean() / 
                                                    ground_truth.query('Path == "2021_06_21"')['Path'].count() * 100, 1)) + ' %)',
                                                 str(hits_2021_06_21['hits_2021_06_21'].min()) + ' (' + 
                                                    str(round(hits_2021_06_21['hits_2021_06_21'].min() / 
                                                    ground_truth.query('Path == "2021_06_21"')['Path'].count() * 100, 1)) + ' %)', 
                                                 str(hits_2021_06_21['hits_2021_06_21'].max()) + ' (' + 
                                                    str(round(hits_2021_06_21['hits_2021_06_21'].max() / 
                                                    ground_truth.query('Path == "2021_06_21"')['Path'].count() * 100, 1)) + ' %)',
                                       
                                                 'Unknown'
                                            ],
                                            [
                                                '01/07/2021', 
                                                 ground_truth.query('Path == "2021_07_01" and Label == "Water"')['Path'].count(),
                                                 ground_truth.query('Path == "2021_07_01" and Label == "Plastic"')['Path'].count(),
                                                 str(hits_2021_07_01['hits_2021_07_01'].mean()) + ' (' + 
                                                    str(round(hits_2021_07_01['hits_2021_07_01'].mean() / 
                                                    ground_truth.query('Path == "2021_07_01"')['Path'].count() * 100, 1)) + ' %)',
                                                 str(hits_2021_07_01['hits_2021_07_01'].min()) + ' (' + 
                                                    str(round(hits_2021_07_01['hits_2021_07_01'].min() / 
                                                    ground_truth.query('Path == "2021_07_01"')['Path'].count() * 100, 1)) + ' %)', 
                                                 str(hits_2021_07_01['hits_2021_07_01'].max()) + ' (' + 
                                                    str(round(hits_2021_07_01['hits_2021_07_01'].max() / 
                                                    ground_truth.query('Path == "2021_07_01"')['Path'].count() * 100, 1)) + ' %)',
                                       
                                                 'Unknown'
                                            ],
                                            [
                                                '06/07/2021', 
                                                 ground_truth.query('Path == "2021_07_06" and Label == "Water"')['Path'].count(),
                                                 ground_truth.query('Path == "2021_07_06" and Label == "Plastic"')['Path'].count(),
                                                 str(hits_2021_07_06['hits_2021_07_06'].mean()) + ' (' + 
                                                    str(round(hits_2021_07_06['hits_2021_07_06'].mean() / 
                                                    ground_truth.query('Path == "2021_07_06"')['Path'].count() * 100, 1)) + ' %)',
                                                 str(hits_2021_07_06['hits_2021_07_06'].min()) + ' (' + 
                                                    str(round(hits_2021_07_06['hits_2021_07_06'].min() / 
                                                    ground_truth.query('Path == "2021_07_06"')['Path'].count() * 100, 1)) + ' %)', 
                                                 str(hits_2021_07_06['hits_2021_07_06'].max()) + ' (' + 
                                                    str(round(hits_2021_07_06['hits_2021_07_06'].max() / 
                                                    ground_truth.query('Path == "2021_07_06"')['Path'].count() * 100, 1)) + ' %)',
                                       
                                                 'Unknown'
                                            ],
                                            [
                                                '21/07/2021', 
                                                 ground_truth.query('Path == "2021_07_21" and Label == "Water"')['Path'].count(),
                                                 ground_truth.query('Path == "2021_07_21" and Label == "Plastic"')['Path'].count(),
                                                 str(hits_2021_07_21['hits_2021_07_21'].mean()) + ' (' + 
                                                    str(round(hits_2021_07_21['hits_2021_07_21'].mean() / 
                                                    ground_truth.query('Path == "2021_07_21"')['Path'].count() * 100, 1)) + ' %)',
                                                 str(hits_2021_07_21['hits_2021_07_21'].min()) + ' (' + 
                                                    str(round(hits_2021_07_21['hits_2021_07_21'].min() / 
                                                    ground_truth.query('Path == "2021_07_21"')['Path'].count() * 100, 1)) + ' %)', 
                                                 str(hits_2021_07_21['hits_2021_07_21'].max()) + ' (' + 
                                                    str(round(hits_2021_07_21['hits_2021_07_21'].max() / 
                                                    ground_truth.query('Path == "2021_07_21"')['Path'].count() * 100, 1)) + ' %)',
                                       
                                                 'Unknown'
                                            ], 
                                            [
                                                '25/08/2021', 
                                                 ground_truth.query('Path == "2021_08_25" and Label == "Water"')['Path'].count(),
                                                 ground_truth.query('Path == "2021_08_25" and Label == "Plastic"')['Path'].count(),
                                                 str(hits_2021_08_25['hits_2021_08_25'].mean()) + ' (' + 
                                                    str(round(hits_2021_08_25['hits_2021_08_25'].mean() / 
                                                    ground_truth.query('Path == "2021_08_25"')['Path'].count() * 100, 1)) + ' %)',
                                                 str(hits_2021_08_25['hits_2021_08_25'].min()) + ' (' + 
                                                    str(round(hits_2021_08_25['hits_2021_08_25'].min() / 
                                                    ground_truth.query('Path == "2021_08_25"')['Path'].count() * 100, 1)) + ' %)', 
                                                 str(hits_2021_08_25['hits_2021_08_25'].max()) + ' (' + 
                                                    str(round(hits_2021_08_25['hits_2021_08_25'].max() / 
                                                    ground_truth.query('Path == "2021_08_25"')['Path'].count() * 100, 1)) + ' %)',
                                       
                                                 'Unknown'
                                            ]
                                            
                                    ],
                                    columns = ['Cover_percent', 'Total water', 'Total plastic', 'Mean hits', 'Min hits', 'Max hits', 'Plastic mean cover percent'])
        
    return stats_by_polymer, stats_by_label_year, stats_by_plastic_cover_percent, stats_by_date
