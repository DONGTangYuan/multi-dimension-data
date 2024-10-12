# Additional Files For Experiment Section
The code includes two methods: Exponential Smoothing and LightGBM. The steps can be divided as follows:

1. **Data Acquisition and Preprocessing**: Retrieve historical trading data for stocks from eight different industries. Perform necessary preprocessing, including handling missing values and industry classification, and store the processed data in the corresponding folders.

2. **Normalization**: Apply min-max normalization to the data.

3. **Data Preparation and Model Training**: For each industry, load the stock data, process the training and testing sets using a sliding window approach, and input the data into the models for training.

4. **Evaluation**: Calculate the results of evaluation metrics.
