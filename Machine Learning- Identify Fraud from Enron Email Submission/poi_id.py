#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import numpy as np

### Task 1: Select what features you'll use.

# all units are in US dollars
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 
                     'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 
                     'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 
                     'restricted_stock', 'director_fees']

# units are generally number of emails messages; notable exception is ‘email_address’, which is a text string
email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 
                 'from_this_person_to_poi', 'shared_receipt_with_poi'] 

# boolean, represented as integer    
poi_label = ['poi']

# Full feature list
features_list =  poi_label + financial_features + email_features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# In[2]:


# Total number of data points
total_data_points = len(data_dict)
print "Total number of data points:", total_data_points


# In[3]:


# Allocation across classes (POI/non-POI)
poi_total = 0
non_poi_total = 0

for person_name in data_dict:
    if data_dict[person_name]["poi"]==1:
        poi_total += 1
    else:
        non_poi_total += 1

print "Allocation across classes (POI):", poi_total
print "Allocation across classes (non-POI):", non_poi_total


# In[4]:


total_features = len(features_list)
print "Number of Features:", total_features 


# In[5]:


# Determine amount of missing values per feature:
missing_value = {}

for feature in features_list:
    missing_value[feature] = 0
for person_name in data_dict:
    for feature in data_dict[person_name]:
        if data_dict[person_name][feature]=='NaN':
            missing_value[feature] += 1

print "Amount of features with missing values:", len(missing_value)
print "Amount of missing values from each feature:"
for feature in missing_value:
    print feature, "=", missing_value[feature]


# In[6]:


# Determine features with many missing values
# Defined "many" as more than 50% of total data points
print "Features with many missing values (More than 50% of data points are missing):"

# Dictionary of features with many missing values
many_missing_values = {}
for feature in missing_value:
    if missing_value[feature] > (total_data_points * 0.5):
        many_missing_values[feature] = missing_value[feature]
        
for feature in many_missing_values:
    print feature, "=", many_missing_values[feature]


# In[7]:


### Task 2: Remove outliers
# Out of the remaing financial features, I decided to examine "salary" and "bonus" for outliers. 
import matplotlib.pyplot as plt
features = ["salary", "bonus"]

data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

print data.max()


# In[8]:


# Remove outlier that correlates to outlier found in previous plot
data_dict.pop('TOTAL',0)

#Additionaly removals based on analysis of pdf

##Remove feature that were unique and therefore not needed for analysis
features_list.remove('email_address')

##Footnote stated that "THE TRAVEL AGENCY IN THE PARK" is an entity and not a real employee or possible poi
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)

print len(features_list)


# In[9]:


### Task 3: Create new feature(s)

my_dataset = data_dict

#New financial features
for name in my_dataset:
    bonus = my_dataset[name]['bonus']
    salary = my_dataset[name]['salary']
    total_payments = my_dataset[name]['total_payments']
    
    # Create bonus_to_salary feature
    if bonus == 'NaN' or salary  == 'NaN':
        my_dataset[name]['bonus_to_salary'] = 0.0
    else:
        my_dataset[name]['bonus_to_salary'] = bonus / float(salary)
    
    # Create bonus_to_total feature
    if bonus == 'NaN' or total_payments  == 'NaN':
        my_dataset[name]['bonus_to_total'] = 0.0
    else:    
        my_dataset[name]['bonus_to_total'] = bonus / float(total_payments)


#New e-mail features
for name in my_dataset:
    to_poi = my_dataset[name]['from_this_person_to_poi']
    from_poi = my_dataset[name]['from_poi_to_this_person']
    to_messages = my_dataset[name]['to_messages']
    from_messages = my_dataset[name]['from_messages']
    shared_receipt_with_poi = my_dataset[name]['shared_receipt_with_poi']
    
    # Create from_poi_ratio feature
    if from_poi =='NaN' or from_messages == 'NaN':
        my_dataset[name]['from_poi_ratio'] = 0.0
    else:
        my_dataset[name]['from_poi_ratio'] = from_poi / float(to_messages)
    
    # Create percentage_to_poi feature
    if to_poi == 'NaN' or to_messages  == 'NaN':
        my_dataset[name]['to_poi_ratio'] = 0.0
    else: 
        my_dataset[name]['to_poi_ratio'] = to_poi / float(from_messages)
     
    # Create shared_poi_ratio feature
    if shared_receipt_with_poi == 'NaN' or to_messages == 'NaN':
        my_dataset[name]['shared_poi_ratio'] = 0.0        
    else:
        my_dataset[name]['shared_poi_ratio'] = shared_receipt_with_poi / float(to_messages)
        
# Add new features to features list
features_list = features_list + ['bonus_to_salary', 'bonus_to_total', 'from_poi_ratio', 
                                'to_poi_ratio', 'shared_poi_ratio']


# In[10]:


# Selecting Features
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.feature_selection import SelectKBest, f_classif 
from sklearn.preprocessing import MinMaxScaler

def best_features(k):
    selector = SelectKBest(f_classif, k)
    selector.fit_transform(features, labels)
    scores = sorted(zip(features_list[1:], selector.scores_), key = lambda x: x[1], reverse=True)
    best_features = list(map(lambda x: x[0], scores))[0:k]
    
    best_features_list = poi_label + best_features
    return best_features_list

best_features_list = best_features(15)


# In[11]:


###Task 4: Try a varity of classifiers
data = featureFormat(my_dataset, best_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


from tester import test_classifier, dump_classifier_and_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Naive Bayes Classifer
naive_bayes_clf = GaussianNB()
naive_bayes_clf.fit(features, labels)
print 'Naive Bayes: Before Tuning', test_classifier(naive_bayes_clf, my_dataset, best_features_list)

#Decision Tree Classifer
decision_tree_clf = DecisionTreeClassifier()
decision_tree_clf.fit(features, labels)
print 'Decision Tree: Before Tuning', test_classifier(decision_tree_clf, my_dataset, best_features_list)


# In[16]:


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB

kbest = SelectKBest(f_classif)
cv = StratifiedShuffleSplit(n_splits = 1000, random_state=42)
from tester import test_classifier

# Naive Bayes Tuning
# Pipeline Creation
nb_pipeline = Pipeline([('scaler', MinMaxScaler()), ('kbest', kbest), ('nb', GaussianNB())]) 
nb_parameters = {'kbest__k': [6, 8, 10]}
nb_grid_search = GridSearchCV(nb_pipeline, nb_parameters, cv = cv, scoring = 'f1') 
nb_grid_search.fit(features, labels)

print 'Naive Bayes Classifer: After Tuning', test_classifier(nb_grid_search.best_estimator_, 
                                                                         my_dataset, best_features_list)


# In[15]:


# Decision Tree Tuning
dt_clf = DecisionTreeClassifier()
dt_parameters = [{'criterion': ['gini', 'entropy'], 'max_depth':[4, 5, 6]}]
dt_grid_search = GridSearchCV(dt_clf, dt_parameters, cv = cv, scoring = 'f1')
dt_grid_search.fit(features, labels)

print 'Decision Tree Classifer: After Tuning',test_classifier(dt_grid_search.best_estimator_, 
                                                                         my_dataset, best_features_list)


# In[17]:


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

clf = nb_grid_search.best_estimator_
features_list = best_features_list

dump_classifier_and_data(clf, my_dataset, features_list)

