#!/usr/bin/env python
# coding: utf-8

# # Surviving the Titanic: An EDA of Passenger Data to Uncover Key Survival Factors

# **Goal :**  
# Our objective for this project is to clean the data first and foremost, address issues such as missing values, inconsistencies, and incorrect data types to ensure the dataset is ready for analysis, examine how different variables (e.g., age, sex, class, and fare) relate to one another and how they might influence other variables and discover patterns and trends in the data that can provide insights into factors affecting survival rates.

# **About Dataset** :  
#     The dataset was provided by Prodigy Info tech from kaggle. The Titanic dataset contains information about the passengers aboard the RMS Titanic, which sank on its maiden voyage in 1912. It was intended to be used for a machine learning competition so it includes 3 tables consisting of training data, testing data and what the predicted values should look like. We'll only be working with our training data since it's the only one with complete features.  
# Link to the dataset : [Titanic](https://www.kaggle.com/c/titanic/data)

# ## Importing Libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ## Importing Dataset

# In[2]:


df=pd.read_csv('C:/Users/obalabi adepoju/Downloads/titanic/train.csv')


# ## Data Cleaning & Inspection

# We'll first of all look at a general overview of our data followed by short descriptions of a few of its columns.

# In[3]:


df.info()


# In[4]:


print(f"The dataset consists of {df.shape[0]} rows and {df.shape[1]} columns")


# #### Brief Desciption of Our Columns  
# 
# 1. **PassengerId**: A unique identifier for each passenger.
# 2. **Survived**: A binary indicator of survival (0 = No, 1 = Yes).
# 3. **pclass**: A proxy for socio-economic status (SES), it represents the class of the tickets passengers bought 
#     * 1st = Upper  
#     * 2nd = Middle  
#     * 3rd = Lower
# 4. **Name**: The name of the passenger.
# 5. **Sex**: The gender of the passenger.
# 6. **Age**: The age of the passenger in years.
# 7. **sibsp**: The number of siblings or spouses the passenger had aboard, the dataset defines family relations in this way...  
#     * Sibling = brother, sister, stepbrother, stepsister  
#     * Spouse = husband, wife (mistresses and fiancés were ignored)
# 8. **parch**: The number of parents or children the passenger had aboard, the dataset defines family relations in this way...  
#     * Parent = mother, father  
#     * Child = daughter, son, stepdaughter, stepson
# 9. **Ticket**: The ticket number.
# 10. **Fare**: The amount of money the passenger paid for the ticket.
# 11. **Cabin**: The cabin number.
# 12. **Embarked**: The port where the passenger boarded (C = Cherbourg, Q = Queenstown, S = Southampton).

# In[5]:


# Let's look at the a few of the records we have hear to gain an understanding of what we are dealing with.
df.head(10)


# We'll start with our passenger id column to ensure there are no duplicates.  
# We'll rename our column "ID" for ease.

# In[6]:


df.rename(columns={"PassengerId":'ID'},inplace =True)

#Checking for duplicates
print(f"Duplicates : {df['ID'].duplicated().any()}")


# In[7]:


# Next we'll examine our survived column to make sure it only consists of the values (0,1)
df['Survived'].unique()


# ### Class

# Next we'll be looking at our pclass column.  
# We'll rename it "class" for clarity and we'll view the distribution of status in our data.

# In[8]:


df.rename(columns={'Pclass':'class'},inplace=True)


# In[9]:


plt = px.histogram(df, x='class',title='Social Status Distribution',color='class')
plt.show()


# In[10]:


df['class'].value_counts()


# * First Class: Tickets for first-class accommodations on the Titanic were very expensive. Prices ranged from about 150 to 4,350 dollars (equivalent to approximately 4,000 to 115,000 today, adjusting for inflation). This high cost was due to the luxurious amenities and services offered, including spacious cabins, fine dining, and exclusive access to various facilities.
# 
# * Second Class: Second-class tickets were less expensive but still a significant expense, ranging from about 60 to 150 dollars (about 1,600 to 4,000 dollars today). Second-class passengers enjoyed a high level of comfort and service, though not as lavish as in first class.
# 
# * Third Class: Third-class tickets were much more affordable, typically costing between 15 and 40 dollars (approximately 400 to 1,100 today). These accommodations were more basic but still provided reasonable comfort for the time. 
#   
#   
#   
# > Recall our data was grouped into 2 so the information present here might not 100 % reflect our overall data. Regardless, the information above likely explains why we have the lower class tickets occupying more than half of our data due to it being the cheapest option, although it doesn't factor in why we have first class as being more than the average class probably due to the incomplete data or that may just be the way our it is. 

# ### Gender

# In[11]:


#Let's check out our gender column
plt = px.pie(df, names='Sex',title='Sex Distribution',hole=0.5)

plt.show()


# ### Age

# Next we'll be looking at our age column.  
# As we previously saw, our age is a floating point variable, this is wrong as ages can only be integers, we'll first examine our values to check why this is the case.

# In[12]:


data = df[(df['Age'] % 1 != 0) & (~df['Age'].isnull()) ].sort_values('Age',ascending=False)
data


# After cross referencing these names with multiple sources online, we've been able to confirm the ages of all of them and we'll now clean our data accordingly. We found that all ages that end in ".5" must have been a data entry error as the numbers before the decimal point are all correct.  So we'll first of all deal with that.

# In[13]:


data['Age']=data['Age'].astype(int)


# After researching the names of the passengers starting with 0, we discovered they were either 3 or below.

# In[14]:


# Now we'll manually enter the ages of the ones with 0
data.iloc[18:26,5] = pd.Series([1,1,1,3,3,1,1])


# In[15]:


data.sort_values('ID',inplace=True)
data


# In[16]:


#Now we fix our original dataframe
df.update(data)


# In[17]:


# We check to see if there are any other abnormal ages after correction
df[(df['Age'] % 1 != 0) & (~df['Age'].isnull()) ]


# We see we have successfully cleaned our data, let's now correct null values.

# In[18]:


df['Age'].info()


# As we can see, there are 177 null values, that's a whole lot of missing values which can't be inputed manually due to lack of information and neither can it be replaced because doing so would skew the distribution of ages in our dataset leading to inaccurate insights drawn when conducting our analyses.  
# So we'll leave these null values as they are and simply work with the ages present.

# Let's now check out our distribution.

# In[19]:


plt=px.violin(df,x='Age',title='Age Distribution',color_discrete_sequence=['mediumseagreen'])
plt.show()


# * Our violin plot gives us a very detailed description of our Age distribution, part of which includes the range of our data with the youngest passenger being a year old and the oldest, 80 years, that gives us the range (1 - 80) years.  
# * Another thing to note is the shape of our plot between the ages of 20 and 40, this tells us where the bulk of our data is situated with about half of all our ages being either 28 or below.
# * One last thing it shows us is the the upper fence of our distribution which tells us that all ages above it are extreme values which lies outside the main range of our data and those outliers include just 5 values (66, 70, 71, 74 and 80).

# ### Family

# Now, we want to check how many family relations every passenger had on the boat. We'll do this by creating a new column.

# In[20]:


df['family'] = df['SibSp'] + df['Parch']
df.head()


# In[21]:


#Let's now check out it's distribution.
plt=px.violin(df,x='family',title='Family Distribution',color_discrete_sequence=['royalblue'])
plt.show()


# * Our plot of the distribution shows us the range of our data with the lowest and also the most common number being zero and the oldest, 10 members, that gives us the range (0 - 10) people.  
# * Another thing to note is the shape of our plot between the numbers of 0 and 2, this tells us where the bulk of our data is situated with about half of all our data being either 1 or below. This tells us almost every passenger had either 1 family member or none.  
# * One last thing it shows us is the the upper fence of our distribution which shows us our extreme values which lies outside the main range of our data and those outliers include just 6 values (3,4,5,6,7 and 10).

# ### Fare

# In[22]:


plt=px.histogram(df,x='Fare',title='Fare Distribution',color_discrete_sequence=['royalblue'])
plt.show()


# Our fare prices are fairly concentrated between the 5 and 100 with the highest distribution being between 5 - 15 dollars, we do have a few outliers which (100 - 165), (200 - 265) and (505 - 515). Let's check out these values in our data.  
# > Note that fare prices are directly affected by the total number of family associated with the passenger so each fare price for a passenger might or might not also include fare for family members or 

# In[23]:


df[df['Fare'] > 100].sort_values('Ticket')


#     Worthy to note all significantly higher prices are all from first class tickets with multiple people sharing the same tickets and they either embarked from southampton or cherbourg. This would explain the outliers as first class tickets are generally more expensive and the higher numbers also contribute to the higher fare prices and note that this isn't oue entire dataset so it's very likely there are more people associated with the tickets than shown.

# ### Embarked

# In[24]:


#Our last column to inspect is the embarked column, we'll first check to see if there are null values
df['Embarked'].info()


# In[25]:


#Just 2, let's check it out
df[df.Embarked.isnull()]


# We went ahead to search them and discovered they embarked at southampton.

# In[26]:


df['Embarked'].fillna('S',inplace=True)


# In[27]:


#Let's check out it's distribution
plt=px.pie(df,names='Embarked',title='Embarkation Distribution',color='Embarked',
           color_discrete_map = {'C':'deepskyblue','S':'dodgerblue','Q':'lightblue'}, hole = 0.5)
plt.show()


# ## Exploratory Analysis

# We'll go through this next step by answering some questions about our data relative to the survival column and they include:

# 1. How does gender impact survival rates?
# 2. What is the connection between class and survival outcomes?
# 3. How does age affect survival chances?
# 4. Do gender, class, and age together influence survival probabilities?

# ### Gender Vs Survival

# Our initial hypothesis for this phase is:
# * Females have a higher chance of survival than men. So we'll now check that

# In[28]:


#We want to check the distribution of gender relative to survival
plt = px.histogram(df, x='Sex',title='Gender Distribution by Survival',color='Survived',barmode='group')
plt.show()


# Upon calculating the numbers in our data, we discovered that only about 18.9 % of male passengers survived and on the contrary, 74.2 % of female passengers survived. Not to make assumptions so early on but ***'Women and Children First'***  is likely to be a clear cause of the disparity in numbers.  
# With this difference in numbers we can conclude that there's a higher probability of women surviving rather than the men thanks to the sacrifices of those brave men.  
# > This information will be very useful during the data modelling phase.

#     Note that each phase of our analysis will build on the previous. With that, let's check out our next column.|

# ### Class vs Survival

# Our next hypothesis for this phase is that:
# * The higher your class level, the higher your chances of survival, based on the assumption that individuals such as royalty or wealthy families would be given a higher priority during the evacuation.

# In[29]:


# As before we'll check the distibution of class relative to survival
plt = px.histogram(df, x='class',title='Class Distribution by Survival',color='Survived',
                   barmode='group',color_discrete_map={1: 'royalblue', 0: '#EF553B'})
plt.show()


#     We can see from this distribution that the number of survivors when compared to the number that didn't survive rises as it moves from the 3rd class to the 1st, this observation clearly affirms our hypothesis but let's dive in further into these factors.

# In[30]:


plt = px.histogram(df, x='Sex',title='Gender Distribution by Class',color='class'
                   ,barmode='group',color_discrete_map={3:'royalblue',1:'#EF553B',2:'mediumseagreen'})
plt.show()


#      The differences in distributions for ticket classes is fairly equal in both genders, but only for first and average, It would seem as tho the third class males make up the most of our population.  
#      Let's Check!

# In[31]:


print(f"Third Class ticket makes up about {round((347/891)*100)} % of our entire population")


#     With respect to our grouping, that's quite the number.

# Next we want to visualize the connection between our first and second hypotheses to determine how these two factors would interact with respect to our survival.

# In[32]:


# Create subplots
fig = make_subplots(rows=1, cols=2, subplot_titles=("Male Distribution", "Female Distribution"))

# Filter data for males and females
df_male = df[df['Sex'] == 'male']
df_female = df[df['Sex'] == 'female']

# Create histograms for each gender
hist_male = px.histogram(df_male, x='class', color='Survived', barmode='group',
                         color_discrete_map={1: 'royalblue', 0: '#EF553B'})
hist_female = px.histogram(df_female, x='class', color='Survived', barmode='group',
                           color_discrete_map={1: 'royalblue', 0: '#EF553B'})

# Add histograms to subplots
for trace in hist_male.data:
    fig.add_trace(trace, row=1, col=1)

for trace in hist_female.data:
    fig.add_trace(trace, row=1, col=2)

# Update layout
fig.update_layout(title_text='Gender Distribution by Survival', showlegend=True)

# Show plot
fig.show()


# * Starting with our male population, we see survival is still ridiculously low in all classes still proving our initial hypothesis, most notably in the 3rd class which we already know houses most of our population with 86 % of 3rd class males not surviving, looking at the plot alone, we are not able to conclude the chances of survival increase as the class increases in our male population but we'll conduct further calculations to thoroughly test this.
# * Alas, the female distribution repersents most clearly shows how our two theories interact with each other, there's a higher number of survivors overall among the female populace and the number of survivors exponentially increases as we go further in class. So we see our two theories gives birth to another conclusion entirely
# > Females have a higher chances of survival when compared with males and those chances increases as the class level increases.  
# 
#     But don't take my word for it, we'll do further calculations next to affirm or disprove this.

# In[33]:


# We want to check the chances of survival for male and female population for each class
def prob(df1, df2):
    
    prob_male = df1.groupby('class').agg(total=('Survived','count'),survived=('Survived','sum')).reset_index()
    prob_female = df2.groupby('class').agg(total=('Survived','count'),survived=('Survived','sum')).reset_index()
    
    prob_male['probability(%)'] = round((prob_male['survived']/prob_male['total'] * 100),1)
    prob_female['probability(%)'] =round((prob_female['survived']/prob_female['total'] * 100),1)
    
    return prob_male,prob_female


# In[34]:


prob_male,prob_female = prob(df_male,df_female)


# In[35]:


# Create subplots
fig = make_subplots(
    rows=1, cols=2, 
    subplot_titles=("Male Probability", "Female Probability"),
    horizontal_spacing=0.1  # Adjust space between plots
)


bar_male = go.Bar(
    x=prob_male['class'],
    y=prob_male['probability(%)'],
    marker_color='mediumseagreen',
   
)

bar_female = go.Bar(
    x=prob_female['class'],
    y=prob_female['probability(%)'],
    marker_color='mediumseagreen',
  
)

# Add bar plots to subplots
fig.add_trace(bar_male, row=1, col=1)
fig.add_trace(bar_female, row=1, col=2)

# Update layout
fig.update_layout(
    title_text='Probability Distribution',
    xaxis_title='Class',
    yaxis_title='Probability (%)',
    xaxis2_title='Class',
    yaxis2_title='Probability (%)',
    yaxis=dict(range=[0, 100]),
    showlegend=False
)

# Show plot
fig.show()


#     Well this certainly affirms our hypotheses, males generally have lower chances of survival with first class males being the highest and having only a 37 % chance of survival whilst on the contrary, our data shows first class females have a 97 % chance of surviving with third class having an even 50. All in all, this shows the interaction of our two hypotheses and how they complement each other.

# ### Age vs Survival

# After a quick overview of the evacuation procedure online, our hypothesis for this next bout is that:
# * Children have the highest chances of survival among all the age groups due to them being given priority during evacuation.

# In[36]:


''' We want to separate our data into different categories based on age groups
Child : 12 and below
Teenager : (13 - 19)
Adult : (20 - 50)
Elderly : 51 and below '''

df['category'] = df['Age'].apply(lambda x:'Child' if x <= 12 else 'Teenager' if x <= 19 else 'Adult' if x <= 50 else 'Elderly' if x>50 else 'Nan' )


# In[37]:


# What our new column looks like
df.head()


# In[38]:


# Filtering our data to not include null values
d = df[df.category != 'Nan']

# Viewing at our distribution
plt = px.histogram(d, x='category',title='Age Distribution by Survival',color='Survived',
                   barmode='group',color_discrete_map={1: 'royalblue', 0: '#EF553B'})
plt.show()


# Among children, there were more survivors than non-survivors, which stands out as the exception. In every other age category, including teenagers, the number of survivors is lower than those who didn't make it. This outcome aligns with our expectations and reinforces our initial hypothesis. The next step is to examine this trend in relation to gender.

# In[39]:


# We want to check the chances of survival for male and female population for each class
def prob1(df1, df2):
    
    prob_male = df1.groupby('category').agg(total=('Survived','count'),survived=('Survived','sum')).reset_index()
    prob_female = df2.groupby('category').agg(total=('Survived','count'),survived=('Survived','sum')).reset_index()
    
    prob_male['probability(%)'] = round((prob_male['survived']/prob_male['total'] * 100),1)
    prob_female['probability(%)'] =round((prob_female['survived']/prob_female['total'] * 100),1)
    
    return prob_male,prob_female


# In[40]:


d_male = d[d['Sex'] == 'male']
d_female = d[d['Sex'] == 'female']


# In[41]:


prob1_male,prob1_female = prob1(d_male,d_female)


# In[42]:


# Create subplots
fig = make_subplots(
    rows=1, cols=2, 
    subplot_titles=("Male Probability", "Female Probability"),
    horizontal_spacing=0.1  # Adjust space between plots
)

# Create bar plots for each gender with opacity based on probability
bar_male = go.Bar(
    x=prob1_male['category'],
    y=prob1_male['probability(%)'],
    marker_color='royalblue',
   
)

bar_female = go.Bar(
    x=prob1_female['category'],
    y=prob1_female['probability(%)'],
    marker_color='royalblue',
  
)

# Add bar plots to subplots
fig.add_trace(bar_male, row=1, col=1)
fig.add_trace(bar_female, row=1, col=2)

# Update layout
fig.update_layout(
    title_text='Probability Distribution',
    xaxis_title='Category',
    yaxis_title='Probability (%)',
    xaxis2_title='Category',
    yaxis2_title='Probability (%)',
    yaxis=dict(range=[0, 100]),
    showlegend=False
)

# Show plot
fig.show()


# For male children, chances of survival is 57 % while for females is 59 %, while there does exist a difference, there's nothing considerable about it which shows no gender preference. This is not the case with the remainder of our categories, as the disparity between the populations is still ridiculously significant most notably in the female elderly group which has a 94 % chance of survival compared to its counterpart which has a 13 % chance.  
# We've now seen the interaction between gender and age, let's look at the connection between age and class.

# In[43]:


# We want to check the chances of survival for male and female population for each class
def prob2(df1, df2, df3):
    
    prob_1 = df1.groupby('category').agg(total=('Survived','count'),survived=('Survived','sum')).reset_index()
    prob_2 = df2.groupby('category').agg(total=('Survived','count'),survived=('Survived','sum')).reset_index()
    prob_3 = df3.groupby('category').agg(total=('Survived','count'),survived=('Survived','sum')).reset_index()
    
    prob_1['probability(%)'] = round((prob_1['survived']/prob_1['total'] * 100),1)
    prob_2['probability(%)'] = round((prob_2['survived']/prob_2['total'] * 100),1)
    prob_3['probability(%)'] = round((prob_3['survived']/prob_3['total'] * 100),1)
    
    return prob_1,prob_2,prob_3

d_1 = d[d['class'] == 1]
d_2 = d[d['class'] == 2]
d_3 = d[d['class'] == 3]

prob_1,prob_2,prob_3 = prob2(d_1,d_2,d_3)


# In[44]:


# Create subplots
fig = make_subplots(
    rows=1, cols=3, 
    subplot_titles=("First Class Probability", "Second Class Probability",'Third Class Probability'),
    horizontal_spacing=0.1  
)


bar_f = go.Bar(
    x=prob_1['category'],
    y=prob_1['probability(%)'],
    marker_color='royalblue',width=0.5
   
)

bar_s = go.Bar(
    x=prob_2['category'],
    y=prob_2['probability(%)'],
    marker_color='#EF553B',width =0.5
  
)

bar_t = go.Bar(
    x=prob_3['category'],
    y=prob_3['probability(%)'],
    marker_color='mediumseagreen',width =0.5
  
)

# Add bar plots to subplots
fig.add_trace(bar_f, row=1, col=1)
fig.add_trace(bar_s, row=1, col=2)
fig.add_trace(bar_t, row=1, col=3)

# Update layout
fig.update_layout(
    title_text='Probability Distribution',
    xaxis_title='Category',
    yaxis_title='Probability (%)',
    xaxis2_title='Category',
    yaxis2_title='Probability (%)',
    xaxis3_title='Category',
    yaxis3_title='Probability (%)',
    yaxis=dict(range=[0, 100]),
    yaxis3=dict(range=[0, 100]),
    showlegend=False
)

# Show plot
fig.show()


# * We observe that for all age groups other than children, the chances of survival decrease significantly as you move from higher to lower classes. The plot shows that not a single category in the third class has a survival rate up to 50%. This indicates that class was a very significant factor in survival.
# 
# *  For the children age group, we see they have the highest chances in all classes other than the 1st which is held by teenagers, overall it shows that age was also a very significant factor in survival.  
# > Despite previous observations suggesting that population differences did not affect survival in cases of children, this plot reveals a different story. We still see a significant disparity in survival rates: the third class has a survival rate of 41%, meaning that 4 out of every 10 children in the third class survived. In contrast, the survival rate for the second class is 100%, indicating that 10 out of 10 children survived, and for the first class, it is 75%, meaning that 3 out of every 4 children survived. This difference might not solely be attributed to class, as children are generally prioritized during evacuation. However, the data insists on a different narrative.

# ### All 3 factors

# To examine all 3 factors and the role they play, we'll group our data into six groups to analyze their distribution.

# In[45]:


d1_male = d_1[d_1.Sex == 'male']
d1_female = d_1[d_1.Sex == 'female']

d2_male = d_2[d_2.Sex == 'male']
d2_female = d_2[d_2.Sex == 'female']

d3_male = d_3[d_3.Sex == 'male']
d3_female = d_3[d_3.Sex == 'female']


# In[46]:


def prob3(df1, df2, df3, df4, df5, df6):
    
    prob_1 = df1.groupby('category').agg(total=('Survived','count'),survived=('Survived','sum')).reset_index()
    prob_2 = df2.groupby('category').agg(total=('Survived','count'),survived=('Survived','sum')).reset_index()
    prob_3 = df3.groupby('category').agg(total=('Survived','count'),survived=('Survived','sum')).reset_index()
    prob_4 = df4.groupby('category').agg(total=('Survived','count'),survived=('Survived','sum')).reset_index()
    prob_5 = df5.groupby('category').agg(total=('Survived','count'),survived=('Survived','sum')).reset_index()
    prob_6 = df6.groupby('category').agg(total=('Survived','count'),survived=('Survived','sum')).reset_index()
    
    prob_1['probability(%)'] = round((prob_1['survived']/prob_1['total'] * 100),1)
    prob_2['probability(%)'] = round((prob_2['survived']/prob_2['total'] * 100),1)
    prob_3['probability(%)'] = round((prob_3['survived']/prob_3['total'] * 100),1)
    prob_4['probability(%)'] = round((prob_4['survived']/prob_4['total'] * 100),1)
    prob_5['probability(%)'] = round((prob_5['survived']/prob_5['total'] * 100),1)
    prob_6['probability(%)'] = round((prob_6['survived']/prob_6['total'] * 100),1)
    
    return prob_1,prob_2,prob_3,prob_4,prob_5,prob_6

pro_1,pro_2,pro_3,pro_4,pro_5,pro_6 = prob3(d1_male,d1_female,d2_male,d2_female,d3_male,d3_female)


# In[47]:


# Create subplots
fig = make_subplots(
    rows=3, cols=2, 
    subplot_titles = ('Male','Female'),
    horizontal_spacing=0.03,  # Adjust space between plots
     vertical_spacing=0.03,
)


bar_m1 = go.Bar(
    x=pro_1['category'],
    y=pro_1['probability(%)'],
    marker_color='royalblue',width=0.3
   
)

bar_f1 = go.Bar(
    x=pro_2['category'],
    y=pro_2['probability(%)'],
    marker_color='#EF553B',width =0.3
  
)

bar_m2 = go.Bar(
    x=pro_3['category'],
    y=pro_3['probability(%)'],
    marker_color='mediumseagreen',width =0.3
  
)

bar_f2 = go.Bar(
    x=pro_4['category'],
    y=pro_4['probability(%)'],
    marker_color='royalblue',width =0.3
  
)

bar_m3 = go.Bar(
    x=pro_5['category'],
    y=pro_5['probability(%)'],
    marker_color='#EF553B',width =0.3
  
)

bar_f3 = go.Bar(
    x=pro_6['category'],
    y=pro_6['probability(%)'],
    marker_color='mediumseagreen',width =0.3
  
)

# Add bar plots to subplots
fig.add_trace(bar_m1, row=1, col=1)
fig.add_trace(bar_f1, row=1, col=2)
fig.add_trace(bar_m2, row=2, col=1)
fig.add_trace(bar_f2, row=2, col=2)
fig.add_trace(bar_m3, row=3, col=1)
fig.add_trace(bar_f3, row=3, col=2)

# Update layout
fig.update_layout(
    title_text='Probability Distribution',
    yaxis_title='Class 1',
    yaxis3_title='Class 2',
    yaxis5_title='Class 3 ',
    yaxis5=dict(range=[0, 100]),
    height=1000,  
    width=900,   
    margin=dict(l=50, r=50, t=100, b=50),
    showlegend=False
)

# Show plot
fig.show()


# * By now, it's news to no one to see the difference between male and female chances of survival so we'll first of all focus on the children category, we see they generally have the highest chances among all male classes, although the same cannot be said in the female population but we already know that's due to other circumstances other than gender disparity, putting both genders together, it's clear class differences do play a role in their chances despite the priority given to this group during the disaster.
# * Another thing worthy of noting is the higher chances for the elderly group and teenagers for the females as these 2 categories  other than children have the highest chances out of all our passengers in terms of survival and we can also see how these 2 chances vary and increase as we go up the classes. This suggests the crucial role all 3 factors played in the survival of the passengers.

# ## Insights & Conclusion

# 
# 1. **Gender and Survival**: Women had a significantly higher survival rate (74.2%) compared to men (18.9%), reflecting the "Women and Children First" evacuation policy.
# 
# 2. **Class and Survival**: Survival chances increased with higher ticket classes, with first-class passengers having the best survival rates, particularly among women.
# 
# 3. **Age and Survival**: Children had the highest survival rates across all classes, with minimal gender disparity (57% for boys, 59% for girls). However, survival rates plummeted as we moved down the classes, especially in third class. Among the elderly, women had a 94% survival rate, while men had only 13%. Overall, survival rates were highest in younger age groups and higher classes.
# 
# 4. **Interaction of Factors**: Gender, age, and class together played a crucial role in determining survival, with women and children in higher classes having the best chances of survival.
# 
# 5. **Exceptions**: While children generally had high survival rates, third-class children had notably lower chances, highlighting that class remained a significant factor even among the prioritized groups.

# In[48]:


import nbformat
from nbconvert import PythonExporter

# Load the notebook
with open('Titanic.ipynb") as f:
    notebook = nbformat.read(f, as_version=4)

# Export only code cells
python_exporter = PythonExporter()
python_code, _ = python_exporter.from_notebook_node(notebook)

# Save the code cells to a .py file
with open("your_notebook_code_cells.py", "w") as f:
    f.write(python_code)



# In[ ]:




