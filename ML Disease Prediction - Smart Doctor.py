#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


filename='Training.csv'
data=read_csv(filename)
data.head()


# In[4]:


df_x=data[['itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain','stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition','spotting_ urination','fatigue','weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy','patches_in_throat','irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness','sweating','dehydration','indigestion','headache','yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine','yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach','swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation','redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs','fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails','swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints','movement_stiffness','spinning_movements','loss_of_balance','unsteadiness','weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine','continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)','depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling','silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze'
]]
df_y=data[['prognosis']]


# In[5]:


df_x.head()


# In[6]:


df_y.head()


# In[7]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.2,random_state=0)


# In[8]:


x_test.shape


# In[9]:


from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb = gnb.fit(df_x, np.ravel(df_y))


# In[10]:


from sklearn.metrics import accuracy_score
y_pred=gnb.predict(x_test)
print(accuracy_score(y_test,y_pred))
print(accuracy_score(y_test,y_pred,normalize=False))


# In[11]:


x_test.head(10)


# In[12]:


y_test.head(10)


# In[13]:


prediction=gnb.predict(x_test)
print(prediction[0:10])


# In[14]:


import joblib as joblib
joblib.dump(gnb,'model/naive_bayes.pkl')


# In[15]:


nb=joblib.load('model/naive_bayes.pkl')


# In[16]:


prediction=nb.predict(x_test)
print(prediction[0:10])


# In[17]:


list_a=['itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain','stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition','spotting_ urination','fatigue','weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy','patches_in_throat','irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness','sweating','dehydration','indigestion','headache','yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine','yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach','swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation','redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs','fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails','swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints','movement_stiffness','spinning_movements','loss_of_balance','unsteadiness','weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine','continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)','depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling','silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze']
print(len(list_a))


# In[18]:


list_c=[]
for x in range(0,len(list_a)):
    list_c.append(0)


# In[19]:


print(list_c)


# In[20]:


list_b=['shivering','headache','vomiting','muscle_pain','diarrhoea']


# In[21]:


for z in range(0,len(list_a)):
    for k in list_b:
        if(k==list_a[z]):
            list_c[z]=1


# In[22]:


print(list_c)


# In[23]:


test=np.array(list_c)
test=np.array(test).reshape(1,-1)
print(test.shape)


# In[24]:


prediction=nb.predict(test)
print(prediction)


# In[25]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf=rf.fit(df_x,np.ravel(df_y))


# In[26]:


y_pred=rf.predict(x_test)
print(accuracy_score(y_test,y_pred))
print(accuracy_score(y_test,y_pred,normalize=False))


# In[27]:


rf.score(x_test,y_test)


# In[28]:


prediction=rf.predict(x_test)
print(prediction[0:10])


# In[29]:


joblib.dump(rf,'model/random_forest.pkl')


# In[30]:


rand_forest=joblib.load('model/random_forest.pkl')


# In[31]:


prediction=rand_forest.predict(x_test)
print(prediction[0:10])


# In[32]:


prediction=rand_forest.predict(test)
print(prediction[0])


# In[33]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(df_x, df_y)


# In[35]:


logreg.score(x_test, y_test)


# In[36]:


import joblib as joblib
joblib.dump(logreg, 'model/lopistic_regression.pkl')


# In[37]:


dt = joblib.load('model/lopistic_regression.pkl')


# In[38]:


prediction = dt.predict(test)
print(prediction[0])


# In[39]:


from sklearn import tree

clf3 = tree.DecisionTreeClassifier()   # empty model of the decision tree
clf3 = clf3.fit(df_x,df_y)


# In[41]:


from sklearn.metrics import accuracy_score
y_pred=clf3.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred,normalize=False))


# In[42]:


import joblib as joblib
joblib.dump(clf3, 'model/decision_tree.pkl')


# In[43]:


dt = joblib.load('model/decision_tree.pkl')


# In[44]:


prediction = dt.predict(test)
print(prediction[0])


# In[ ]:




