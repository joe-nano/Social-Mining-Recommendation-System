# Social-Mining-Recommendation-System
Datamining for Big Data
 
### About Project
Content based Recommedation system based on Graph mining and Unsupervised Machine learning method. The application is visualizes the similarity between the content of different publications and how this publications are similar in terms of content. 

Using advance unsupervised machine learning models, we recommend research papers that are worth reading/consulting in terms of researching using the content of a prior paper. We use cosine measure as a metric to build the similarity matrix between different books.

We also use graph mining modelling in finding different communities which a book can be classified with highly related papers. Highly related research papers are grouped in the same communities using the graph modeling approach.

### Project Workflow

<ul>
  <li>Import and and preprocess all 1269 French books</li>
    <ul>
    <li>Convert all english research papers to french</li>
    <li>Update/replace translated papers to french papers in original document</li>
    </ul>
  <li>Stemming & Lemmatization of extracted tokens from each research paper</li>
  <li>Visualize most frequent words on hover and return output in barplot on Web App</li>
  <li>TF-IDF Model for vectorizing document into float</li>
  <li>Document Similarity using Cosine distance of paper content</li>
    <li>Clustering based Recommender System</li>
    <ul>
      <li>Kernel Principal component analysis</li>
      <li>Kernel KMeans clustering</li>
     </ul>
    <li>Graph based Recommender System</li>
     <ul>
      <li>Greedy approach</li>
      <li>Louvain Algorithm</li>
      <li>Clique Based Approach</li>
    </ul>
  <li>Web GUI Viusalization</li>
    <ul>
    <li>TSNE 2D Visualizaion</li>
    <li>Hover over data points to see <Title, year, and author of paper></li>
    <li>Hover over data to see top recommendation based</li>
    <li>Bar chat for 15 most frequent words in research paper</li>
    </ul>
</ul>


### How to use

```python
git clone https://github.com/beteko/Social-Mining-Recommendation-System
```
```
Change the directory of the project to system directory format
```

Open the script folder in your terminal and run the following command

```python
python mainapp.py
```

```python
Navigate http://127.0.0.1:8050/ 
```

