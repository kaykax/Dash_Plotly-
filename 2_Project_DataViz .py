
from urllib.request import urlopen
import json
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.express as px
import pandas as pd
import numpy as np


import dash
from dash import dcc
from dash import html
import dash_cytoscape as cyto
from dash.dependencies import Input, Output
from collections import defaultdict

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score,cross_val_predict



def gen_null_bar_traces(df):
    '''Plot histogram of null values'''
    fig = go.Figure()
    for reg in list(df['loc'].unique()):
        dfx = df[df['loc'] == reg].drop(['loc','target'],axis=1)
        fig.add_trace(go.Bar(x=dfx.isna().sum().index, y=dfx.isna().sum().values,name=reg))
    fig.update_layout(
        title={
        'text': f'Total Count of Null Values/Features',
        #'y':0.9,
        #'x':0.5,
        #'xanchor': 'center',
        #'yanchor': 'top'
        },
        
        )
    return fig

def filt_corrmat(c_mat,thresh):
    '''Drops rows/cols from correlation matrix for which all values below threshold'''
    c_mat[c_mat < thresh] = None
    c_mat[c_mat==1] = None
    c_mat.dropna(axis=0,how='all',inplace=True)
    c_mat.dropna(axis=1,how='all',inplace=True)
    return df[c_mat.keys().to_list()].corr()


# Function to plot correlation matrix
def plot_corr(dfx,region,thresh=0.3):
    '''Computes and filters correlation matrix for dataframe and plots as image'''
    df  = dfx[dfx['loc'] == region]
    corr_mat = filt_corrmat(df.corr(),thresh)
    fig = px.imshow(corr_mat,title=f'Correlated Features > {thresh} for "{region}"', labels={'color':'Correlation'})

    return fig,corr_mat

def feature_plot(dfx,reg,f,type='bar'):
    '''Plots bar or pie graph of an individual feature. Bar graph groups by target, pie graph filters by target'''
    colors = ['#56B4E9','#E69F00']

    df  = dfx[dfx['loc'] == reg]
    print(f'reg ={reg},f = {f}')

    if f not in df.keys():
        print('Unrecognized feature name')
        return
        
    if type=='bar':
        # Plot histogram of feature grouped by target
        fig = px.histogram(df,x=f,color='target',title = f.title() + f' of Heart Diseased Patients for "{reg}"',
                          color_discrete_sequence=colors,barmode='group',category_orders={'target':['No Heart Disease','Heart Disease']})
        
    
    elif type=='pie':
        # Filter by target and plot pie chart of feature
        fig = px.pie(df.loc[df['target']=='Heart Disease'], names=f, title= f.title() + ' of Heart Diseased Patients')
        
    else:
        print("Unrecognized type. Please choose 'bar' or 'pie'")
    
    return fig

def feature_gridPlot(dfx,reg,f):
    df  = dfx[dfx['loc'] == reg]
    fig = ff.create_facet_grid(df, x =f, color_name='sex', trace_type='histogram',color_is_cat=True,facet_row ='cp')
    fig.update_layout(
        title={
        'text': f'Histogram Plot of {f} for {reg}',
        #'y':0.9,
        #'x':0.5,
        #'xanchor': 'center',
        #'yanchor': 'top'
        },
        
        )
    return fig

def filter_corrCols(cor_mat):
    '''Filter columns in a correlation matrix'''
    max_dict = {i:v for i,v in zip(cor_mat[cor_mat != 1].max().index,cor_mat[cor_mat != 1].max().values)}
    dropCols = set()
    for c in max_dict.keys():
        if c not in dropCols:
            tmp = cor_mat[cor_mat != 1][c] == max_dict[c]
            dropCols.add(list(tmp.index[np.where(tmp.values == True)])[0])
    return dropCols


def prepClassifierData(df,location,dropCols = []):
    '''Train SGD classifier and run on dataset for given site'''
    imputer = SimpleImputer(strategy="median")
    if len(list(dropCols)) :
        print(f'[INFO] dropping columns : {dropCols}')
        df_numeric = df.drop(list(dropCols),axis=1).loc[:,df.dtypes != object]
    else :
        df_numeric = df.loc[:,df.dtypes != object]
    df_cat = df[['sex']]
    df_labels = df[['target']]
    
    ordinal_encoder = OrdinalEncoder()
    df_label_encoded = 1- ordinal_encoder.fit_transform(df_labels)

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])
    
    num_attribs = df_numeric.columns.to_list()
    cat_attribs = ['sex']
    num_attribs + cat_attribs

    
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])
    
    
    loc_mask = df['loc'] == location
    X_train,X_test,Y_train,Y_test = train_test_split(df[loc_mask][num_attribs + cat_attribs],df_label_encoded[loc_mask],test_size=0.3, random_state=42,stratify=df_label_encoded[loc_mask])
    X_train_prepared = full_pipeline.fit_transform(X_train)
    X_test_prepared = full_pipeline.transform(X_test)
    print(X_test_prepared)
    
    sgd_clf = SGDClassifier(max_iter=100, tol=1e-3, random_state=42)
    y_train_pred = cross_val_predict(sgd_clf,X_train_prepared, Y_train.ravel(), cv=10)
    Y = cross_val_score(sgd_clf, X_train_prepared, Y_train.ravel(), cv=10, scoring="accuracy")
    fig_cv  =px.bar(x=[f'cross_fold_score_{i}' for i in range (1,11)], y = Y, title =f' {location}: 10-Fold Cross-Validation ML Classifier Accuracy',labels={'x':'','y':'Accuracy'})
    
    sgd_clf.fit(X_train_prepared,Y_train.ravel())
    y_test_pred = sgd_clf.predict(X_test_prepared)
    Y = sgd_clf.coef_[0]
    fig_fimp = px.bar(x=X_train.columns.to_list()[:-1] + ['sex_M', 'sex_F'], y=np.abs(Y), color=np.abs(Y), title =f' {location} : Feature Importance',range_color=[0,np.max(np.abs(Y))],labels={'color':'Feature Importance','y':'Feature Importance','x':''})
    return [fig_cv,fig_fimp]

# Set display theme
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Read data
df = pd.read_csv('./alldata.csv')


app.layout = html.Div([ 
     html.Div([
        # Bar plot of heart disease by site
        dcc.Graph(
            id='startBar-plot',
            hoverData={'points': [{'x': 'Cleveland Clinic Foundation'}]},
            figure = px.histogram(df.rename(columns= {'target' : 'condition'}), x ='loc' , color = 'condition' , title= 'Total Count of Heart Disease Across Different Locations',color_discrete_sequence=['#56B4E9','#E69F00'],category_orders={'condition':['No Heart Disease','Heart Disease']},labels={'loc':'Clinical Site'})
        ),
        # Bar plot of null values
        dcc.Graph(id='null-barPlot', figure=gen_null_bar_traces(df)),
        # Bar plot of feature importances
        dcc.Graph(id='feature-impPlot'),
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
    html.Div([
        # Dropdown for correlation thresholds
        dcc.Dropdown(
                id='corr-column',
                options=[{'label': round(i*0.1,1), 'value': round(i*0.1,1)}
                         for i in range(0,10)],
                value= 0.3
            ),
        # Correlation heatmap
        dcc.Graph(id='county-corrPlot',hoverData={'points': [{'x': 'Cleveland Clinic Foundation'}]}, figure=plot_corr(df,'Cleveland Clinic Foundation')[0]),
        # Dropdown for features to plot
        dcc.Dropdown(
                id='feature-column',
                options=[{'label': i, 'value': i}
                         for i in df.drop(['loc','target'],axis=1).columns.to_list()],
                value='age'
            ),
        # Feature distribution histogram
        dcc.Graph(id='age-barPlot'),
        # Bar graph of training scores
        dcc.Graph(id='Mscore-Plot'),
        ], style={'display': 'inline-block', 'width': '49%'}),
])


@app.callback(
    dash.dependencies.Output('county-corrPlot', 'figure'),
    [
        dash.dependencies.Input('startBar-plot', 'hoverData'),
        dash.dependencies.Input('corr-column', 'value')
    ]
    )
def update_corrPlot(hoverData,cor_thres):
    '''Update correlation heatmap when user hovers over location bar graph'''
   # print(hoverData['points'][0])
    location  = str(hoverData['points'][0]['x'])
   # print(f'location = {location} ')
    fig,cor_mat = plot_corr(df,location,cor_thres)
    dropColumns = filter_corrCols(cor_mat)
    return fig

@app.callback(
    Output('feature-impPlot', 'figure'),
    Output('Mscore-Plot', 'figure'),
    Input('startBar-plot', 'hoverData'),
    Input('corr-column', 'value')   
)
def update_featureImpPlot(hoverData,cor_val):
    '''Update feature importance bar graph when user hovers over location bar graph'''
    location  = str(hoverData['points'][0]['x'])
    print(f'location = {location} ')
    cor_mat = filt_corrmat(df.corr(),cor_val)
    fig_cv,fig_imp = prepClassifierData(df,location,filter_corrCols(cor_mat))
    
    return fig_imp,fig_cv
   
@app.callback(
    Output('age-barPlot', 'figure'),
    Input('startBar-plot', 'hoverData'),
    Input('feature-column', 'value'),
)
def update_featurePlot(hoverData,feature_val):
    '''Update feature histogram when user hovers over location bar graph'''
   # print(hoverData['points'][0])
    location  = str(hoverData['points'][0]['x'])
   # print(f'location = {location} , feature_val : {feature_val}')
    fig = feature_plot(df,location,feature_val)
    fig.update_xaxes(title=feature_val)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)