Module 10 Application
=====================

Challenge: Crypto Clustering
----------------------------

In this Challenge, you’ll combine your financial Python programming
skills with the new unsupervised learning skills that you acquired in
this module.

The CSV file provided for this challenge contains price change data of
cryptocurrencies in different periods.

The steps for this challenge are broken out into the following sections:

-  Import the Data (provided in the starter code)
-  Prepare the Data (provided in the starter code)
-  Find the Best Value for ``k`` Using the Original Data
-  Cluster Cryptocurrencies with K-means Using the Original Data
-  Optimize Clusters with Principal Component Analysis
-  Find the Best Value for ``k`` Using the PCA Data
-  Cluster the Cryptocurrencies with K-means Using the PCA Data
-  Visualize and Compare the Results

Import the Data
~~~~~~~~~~~~~~~

This section imports the data into a new DataFrame. It follows these
steps:

1. Read the “crypto_market_data.csv” file from the Resources folder into
   a DataFrame, and use ``index_col="coin_id"`` to set the
   cryptocurrency name as the index. Review the DataFrame.

2. Generate the summary statistics, and use HvPlot to visualize your
   data to observe what your DataFrame contains.

..

   **Rewind:** The
   `Pandas\ ``describe()``\ function <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html>`__
   generates summary statistics for a DataFrame.

.. code:: ipython3

    # Import required libraries and dependencies
    import pandas as pd
    import hvplot.pandas
    from pathlib import Path
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

.. code:: ipython3

    # Load the data into a Pandas DataFrame
    df_market_data = pd.read_csv(
        Path("Resources/crypto_market_data.csv"),
        index_col="coin_id")
    
    # Display sample data
    df_market_data.head(10)

.. code:: ipython3

    # Generate summary statistics
    df_market_data.describe()

.. code:: ipython3

    # Plot your data to see what's in your DataFrame
    df_market_data.hvplot.line(
        width=800,
        height=400,
        rot=90
    )

--------------

Prepare the Data
~~~~~~~~~~~~~~~~

This section prepares the data before running the K-Means algorithm. It
follows these steps:

1. Use the ``StandardScaler`` module from scikit-learn to normalize the
   CSV file data. This will require you to utilize the ``fit_transform``
   function.

2. Create a DataFrame that contains the scaled data. Be sure to set the
   ``coin_id`` index from the original DataFrame as the index for the
   new DataFrame. Review the resulting DataFrame.

.. code:: ipython3

    # Use the `StandardScaler()` module from scikit-learn to normalize the data from the CSV file
    scaled_data = StandardScaler().fit_transform(df_market_data)

.. code:: ipython3

    # Create a DataFrame with the scaled data
    df_market_data_scaled = pd.DataFrame(
        scaled_data,
        columns=df_market_data.columns
    )
    
    # Copy the crypto names from the original data
    df_market_data_scaled["coin_id"] = df_market_data.index
    
    # Set the coinid column as index
    df_market_data_scaled = df_market_data_scaled.set_index("coin_id")
    
    # Display sample data
    df_market_data_scaled.head()

--------------

Find the Best Value for k Using the Original Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this section, you will use the elbow method to find the best value
for ``k``.

1. Code the elbow method algorithm to find the best value for ``k``. Use
   a range from 1 to 11.

2. Plot a line chart with all the inertia values computed with the
   different values of ``k`` to visually identify the optimal value for
   ``k``.

3. Answer the following question: What is the best value for ``k``?

.. code:: ipython3

    # Create a list with the number of k-values to try
    # Use a range from 1 to 11
    # YOUR CODE HERE!

.. code:: ipython3

    # Create an empy list to store the inertia values
    # YOUR CODE HERE!

.. code:: ipython3

    # Create a for loop to compute the inertia with each possible value of k
    # Inside the loop:
    # 1. Create a KMeans model using the loop counter for the n_clusters
    # 2. Fit the model to the data using `df_market_data_scaled`
    # 3. Append the model.inertia_ to the inertia list
    # YOUR CODE HERE!

.. code:: ipython3

    # Create a dictionary with the data to plot the Elbow curve
    # YOUR CODE HERE!
    
    # Create a DataFrame with the data to plot the Elbow curve
    # YOUR CODE HERE!

.. code:: ipython3

    # Plot a line chart with all the inertia values computed with 
    # the different values of k to visually identify the optimal value for k.
    # YOUR CODE HERE!

Answer the following question: What is the best value for k?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Question:** What is the best value for ``k``?

**Answer:** # YOUR ANSWER HERE!

--------------

Cluster Cryptocurrencies with K-means Using the Original Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this section, you will use the K-Means algorithm with the best value
for ``k`` found in the previous section to cluster the cryptocurrencies
according to the price changes of cryptocurrencies provided.

1. Initialize the K-Means model with four clusters using the best value
   for ``k``.

2. Fit the K-Means model using the original data.

3. Predict the clusters to group the cryptocurrencies using the original
   data. View the resulting array of cluster values.

4. Create a copy of the original data and add a new column with the
   predicted clusters.

5. Create a scatter plot using hvPlot by setting
   ``x="price_change_percentage_24h"`` and
   ``y="price_change_percentage_7d"``. Color the graph points with the
   labels found using K-Means and add the crypto name in the
   ``hover_cols`` parameter to identify the cryptocurrency represented
   by each data point.

.. code:: ipython3

    # Initialize the K-Means model using the best value for k
    # YOUR CODE HERE!

.. code:: ipython3

    # Fit the K-Means model using the scaled data
    # YOUR CODE HERE!

.. code:: ipython3

    # Predict the clusters to group the cryptocurrencies using the scaled data
    # YOUR CODE HERE!
    
    # View the resulting array of cluster values.
    # YOUR CODE HERE!

.. code:: ipython3

    # Create a copy of the DataFrame
    # YOUR CODE HERE!

.. code:: ipython3

    # Add a new column to the DataFrame with the predicted clusters
    # YOUR CODE HERE!
    
    # Display sample data
    # YOUR CODE HERE!

.. code:: ipython3

    # Create a scatter plot using hvPlot by setting 
    # `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. 
    # Color the graph points with the labels found using K-Means and 
    # add the crypto name in the `hover_cols` parameter to identify 
    # the cryptocurrency represented by each data point.
    # YOUR CODE HERE!

--------------

Optimize Clusters with Principal Component Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this section, you will perform a principal component analysis (PCA)
and reduce the features to three principal components.

1. Create a PCA model instance and set ``n_components=3``.

2. Use the PCA model to reduce to three principal components. View the
   first five rows of the DataFrame.

3. Retrieve the explained variance to determine how much information can
   be attributed to each principal component.

4. Answer the following question: What is the total explained variance
   of the three principal components?

5. Create a new DataFrame with the PCA data. Be sure to set the
   ``coin_id`` index from the original DataFrame as the index for the
   new DataFrame. Review the resulting DataFrame.

.. code:: ipython3

    # Create a PCA model instance and set `n_components=3`.
    # YOUR CODE HERE!

.. code:: ipython3

    # Use the PCA model with `fit_transform` to reduce to 
    # three principal components.
    # YOUR CODE HERE!
    
    # View the first five rows of the DataFrame. 
    # YOUR CODE HERE!

.. code:: ipython3

    # Retrieve the explained variance to determine how much information 
    # can be attributed to each principal component.
    # YOUR CODE HERE!

Answer the following question: What is the total explained variance of the three principal components?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Question:** What is the total explained variance of the three
principal components?

**Answer:** # YOUR ANSWER HERE!

.. code:: ipython3

    # Create a new DataFrame with the PCA data.
    # Note: The code for this step is provided for you
    
    # Creating a DataFrame with the PCA data
    # YOUR CODE HERE!
    
    # Copy the crypto names from the original data
    # YOUR CODE HERE!
    
    # Set the coinid column as index
    # YOUR CODE HERE!
    
    # Display sample data
    # YOUR CODE HERE!

--------------

Find the Best Value for k Using the PCA Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this section, you will use the elbow method to find the best value
for ``k`` using the PCA data.

1. Code the elbow method algorithm and use the PCA data to find the best
   value for ``k``. Use a range from 1 to 11.

2. Plot a line chart with all the inertia values computed with the
   different values of ``k`` to visually identify the optimal value for
   ``k``.

3. Answer the following questions: What is the best value for k when
   using the PCA data? Does it differ from the best k value found using
   the original data?

.. code:: ipython3

    # Create a list with the number of k-values to try
    # Use a range from 1 to 11
    # YOUR CODE HERE!

.. code:: ipython3

    # Create an empy list to store the inertia values
    # YOUR CODE HERE!

.. code:: ipython3

    # Create a for loop to compute the inertia with each possible value of k
    # Inside the loop:
    # 1. Create a KMeans model using the loop counter for the n_clusters
    # 2. Fit the model to the data using `df_market_data_pca`
    # 3. Append the model.inertia_ to the inertia list
    # YOUR CODE HERE!

.. code:: ipython3

    # Create a dictionary with the data to plot the Elbow curve
    # YOUR CODE HERE!
    
    # Create a DataFrame with the data to plot the Elbow curve
    # YOUR CODE HERE!

.. code:: ipython3

    # Plot a line chart with all the inertia values computed with 
    # the different values of k to visually identify the optimal value for k.
    # YOUR CODE HERE!

Answer the following questions: What is the best value for k when using the PCA data? Does it differ from the best k value found using the original data?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  **Question:** What is the best value for ``k`` when using the PCA
   data?

   -  **Answer:** # YOUR ANSWER HERE!

-  **Question:** Does it differ from the best k value found using the
   original data?

   -  **Answer:** # YOUR ANSWER HERE!

--------------

Cluster Cryptocurrencies with K-means Using the PCA Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this section, you will use the PCA data and the K-Means algorithm
with the best value for ``k`` found in the previous section to cluster
the cryptocurrencies according to the principal components.

1. Initialize the K-Means model with four clusters using the best value
   for ``k``.

2. Fit the K-Means model using the PCA data.

3. Predict the clusters to group the cryptocurrencies using the PCA
   data. View the resulting array of cluster values.

4. Add a new column to the DataFrame with the PCA data to store the
   predicted clusters.

5. Create a scatter plot using hvPlot by setting ``x="PC1"`` and
   ``y="PC2"``. Color the graph points with the labels found using
   K-Means and add the crypto name in the ``hover_cols`` parameter to
   identify the cryptocurrency represented by each data point.

.. code:: ipython3

    # Initialize the K-Means model using the best value for k
    # YOUR CODE HERE!

.. code:: ipython3

    # Fit the K-Means model using the PCA data
    # YOUR CODE HERE!

.. code:: ipython3

    # Predict the clusters to group the cryptocurrencies using the PCA data
    # YOUR CODE HERE!
    
    # View the resulting array of cluster values.
    # YOUR CODE HERE!

.. code:: ipython3

    # Create a copy of the DataFrame with the PCA data
    # YOUR CODE HERE!
    
    # Add a new column to the DataFrame with the predicted clusters
    # YOUR CODE HERE!
    
    # Display sample data
    # YOUR CODE HERE!

.. code:: ipython3

    # Create a scatter plot using hvPlot by setting 
    # `x="PC1"` and `y="PC2"`. 
    # Color the graph points with the labels found using K-Means and 
    # add the crypto name in the `hover_cols` parameter to identify 
    # the cryptocurrency represented by each data point.
    # YOUR CODE HERE!

--------------

Visualize and Compare the Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this section, you will visually analyze the cluster analysis results
by contrasting the outcome with and without using the optimization
techniques.

1. Create a composite plot using hvPlot and the plus (``+``) operator to
   contrast the Elbow Curve that you created to find the best value for
   ``k`` with the original and the PCA data.

2. Create a composite plot using hvPlot and the plus (``+``) operator to
   contrast the cryptocurrencies clusters using the original and the PCA
   data.

3. Answer the following question: After visually analyzing the cluster
   analysis results, what is the impact of using fewer features to
   cluster the data using K-Means?

..

   **Rewind:** Back in Lesson 3 of Module 6, you learned how to create
   composite plots. You can look at that lesson to review how to make
   these plots; also, you can check `the hvPlot
   documentation <https://holoviz.org/tutorial/Composing_Plots.html>`__.

.. code:: ipython3

    # Composite plot to contrast the Elbow curves
    # YOUR CODE HERE!

.. code:: ipython3

    # Compoosite plot to contrast the clusters
    # YOUR CODE HERE!

Answer the following question: After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  **Question:** After visually analyzing the cluster analysis results,
   what is the impact of using fewer features to cluster the data using
   K-Means?

-  **Answer:** # YOUR ANSWER HERE!
