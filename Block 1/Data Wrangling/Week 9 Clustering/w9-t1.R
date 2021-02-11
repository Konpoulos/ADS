library(MASS)
library(ggdendro)
library(ISLR)
library(factoextra)
library(dendextend)
library(patchwork)
library(cluster)


# Hierarchical clustering --> Agglomerative clustering, 
#                         --> and Divisive hierarchical clustering
#                             --> Distance metric (similarity and dissimilarity)
#                             --> Linkage method
#                                  --> Maximum or complete linkage clustering
#                                  --> Minimum or single linkage clustering
#                                  --> Mean or average linkage clustering
#                                  --> Centroid linkage clustering
#                                  --> Ward’s minimum variance method


# Partitional clustering  --> k-means
# Model-based clustering  --> Assumptions about the clusters are explicit, not implicit 

# Example from the lecture
distances <- dist(faithful, method = "euclidean")

result    <- hclust(distances, method = "average")
ggdendrogram(result, labels = FALSE)

result1   <- hclust(distances, method = "complete")

ggdendrogram(result, labels = FALSE) + ggtitle("average") + ylim(0, 10) +
  ggdendrogram(result1, labels = FALSE) + ggtitle("complete") + ylim(0, 10)

cutree(result, h = 20)
cutree(result, k = 5)

# Compute 2 hierarchical clusterings
df  <- USArrests
d   <- dist(df, method = "euclidean")
hc1 <- hclust(d, method = "average")
hc2 <- hclust(d, method = "centroid")
# Create two dendrograms
dend1 <- as.dendrogram (hc1)
dend2 <- as.dendrogram (hc2)
tanglegram(dend1, dend2)

# Divisive hierarchical clustering
res.diana <- diana(df, stand = TRUE)
# Plot the dendrogram
fviz_dend(res.diana, cex = 0.5, # size of labels
          k = 4, # Cut in four groups
          palette = "jco" # Color palette
)

# Partitional clustering
pclust <- kmeans(df, centers = 3)
str(pclust)
pclust
pclust$centers
pclust$cluster
fviz_cluster(pclust, data = df)

df %>%
  as_tibble() %>%
  mutate(cluster = pclust$cluster,
         state = row.names(USArrests)) %>%
  ggplot(aes(UrbanPop, Murder, color = factor(cluster), label = state)) +
  geom_text()


# Dertermining And Visualizing The Optimal Number Of Clusters
# https://uc-r.github.io/hc_clustering
fviz_nbclust(df, FUN = hcut, method = "wss") # Elbow Method
fviz_nbclust(df, FUN = hcut, method = "silhouette") # Average Silhouette Method
gap_stat <- clusGap(df, FUN = hcut, nstart = 25, K.max = 10, B = 50)
fviz_gap_stat(gap_stat) # Gap Statistic Method
