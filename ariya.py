import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv('Mall_Customers.csv')
X = df.iloc[: , [3,4]]
print(f"X Shape {X.shape}")
X.head()

#show data
st.header("Isi Dataset")
st.write(df)

#elbow
n_clusters = range(2 , 13)
inertia_errors = []
silhouette_scores = []
# Tambahkan loop `for` untuk melatih model dan menghitung inersia, skor siluet
for k in n_clusters:
    model = KMeans(n_clusters= k , random_state= 42)
    #TRAIN MODEL
    model.fit(X)
    #CALCULATE INERTIA
    inertia_errors.append(model.inertia_)
    #CALCULATE SILHOUETTE SCORE
    silhouette_scores.append(silhouette_score(X , model.labels_))
print("Inertia:", inertia_errors[:3])
print()
print("Silhouette Scores:", silhouette_scores[:3])

# Buat plot garis `inertia_errors` vs `n_clusters`
fig = px.line(x= range(2 , 13) , y= inertia_errors , title="K-Means Model: Inertia vs Number of Clusters")
fig.update_layout(xaxis_title="Number of Clusters" , yaxis_title="Inertia")
fig.show()

st.header("Elbow Point") 
st.plotly_chart(fig)

st.sidebar.header("Mall Custumers")
clust = st.sidebar.slider("Pilih Jumlah Cluster :", 2,10,3,1)

df = pd.read_csv('Mall_Customers.csv')

def k_means(clust):

    final_model = KMeans(n_clusters=5 , random_state= 42)
    final_model.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])

    labels = final_model.labels_
    centroids = final_model.cluster_centers_
    print(labels[:5])

    #plot "Annual Income" vs "Spending Score" with final_model labels
    sns.scatterplot(x=df['Annual Income (k$)'] , y= df['Spending Score (1-100)'] ,
               hue=labels,
               palette='deep')
    sns.scatterplot(
                    x= centroids[:,0],
                    y= centroids[: ,1],
                    color= 'gray',
                    marker= '*',
                    s= 500
                  )

    fig = px.scatter(df, x='Annual Income (k$)', y='Spending Score (1-100)', color=labels,
                     title="Annual Income vs. Spending Score with K-Means Labels",
                     color_continuous_scale='deep')
    fig.add_scatter(x=centroids[:, 0], y=centroids[:, 1], mode='markers', marker=dict(color='gray', size=10, symbol='star'))

    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.title("Annual Income vs. Spending Score")
    st.plotly_chart(fig)
    st.write(df)
 
k_means(clust)
 




