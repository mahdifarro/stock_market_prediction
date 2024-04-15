import altair as alt
import pandas as pd
from sklearn.decomposition import PCA


def visualize_loss(training_losses):
    df_loss = pd.DataFrame(enumerate(training_losses), columns=["epoch", "training_loss"])
    chart = alt.Chart(df_loss).mark_bar().encode(alt.X("epoch"), alt.Y("training_loss", scale=alt.Scale(type="log")))
    chart.save('training_loss_for_doc2vec.html')
def pca_2d(paragraph_matrix):
    pca = PCA(n_components=2)
    reduced_dims = pca.fit_transform(paragraph_matrix)
    print(f"2-component PCA, explains {sum(pca.explained_variance_):.2f}% of variance")
    df = pd.DataFrame(reduced_dims, columns=["x", "y"])
    return df
