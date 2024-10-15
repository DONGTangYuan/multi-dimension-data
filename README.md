# Additional Files For Experiment Section
The code includes two methods: Exponential Smoothing and LightGBM for stock prediction as addditional files for paper *A Stock Prediction Method Based on Multidimensional and Multilevel Feature Dynamic Fusion*. The steps can be divided as follows:

1. **Data Acquisition and Preprocessing**: Retrieve historical trading data for stocks from eight different industries. Perform necessary preprocessing, including handling missing values and industry classification, and store the processed data in the corresponding folders. All stock information and the corresponding industry information are obtained from https://www.eastmoney.com/.

2. **Normalization**: Apply min-max normalization to the data.

3. **Data Preparation and Model Training**: For each industry, load the stock data, process the training and testing sets using a sliding window approach, and input the data into the models for training. For the Exponential Smoothing model, we use the Holt-Winters model, considering that the fluctuations in the A-share market are limited and cannot follow an exponential growth pattern. Therefore, we set the `trend` parameter to `add`. For the LightGBM model, we set the random seed to a random value, so multiple experiments are needed to take the average as the final result.

4. **Evaluation**: Calculate the results of evaluation metrics.

5. run main_linux.sh for Linux environment.\\
6. run main_win.bat for Win enviroment.
