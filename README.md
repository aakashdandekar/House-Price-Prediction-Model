# House Price Prediction

A linear regression model that predicts house prices from four property features: square footage, number of rooms, age of the house, and distance to the city centre. The project covers data loading, cleaning, model training, evaluation, and saving the trained model for reuse.

## Dataset

The file `house_prices_dataset.csv` contains 10,000 rows and 5 columns with no missing values.

| Column | Type | Description | Range (raw data) |
|---|---|---|---|
| `square_feet` | float | Living area of the house | 500 to 3,963 |
| `num_rooms` | integer | Number of rooms | 2 to 7 |
| `age` | integer | Age of the house in years | 0 to 99 |
| `distance_to_city(km)` | float | Distance from the city in kilometres | 1 to 30 |
| `price` | float | Sale price (target variable) | -95,613 to 660,168 |

The raw `price` column contains negative values and some very large values, which are not realistic sale prices. The notebook removes them during cleaning, as described below.

## Project Structure

| File | Purpose |
|---|---|
| `house_prices_dataset.csv` | Raw dataset |
| `house_price_prediction.ipynb` | Notebook with the full workflow: loading, cleaning, training, evaluation, and saving |
| `model.pkl` | Trained model saved with joblib (generated when the notebook runs) |

## Workflow

1. **Load the data.** The notebook asks for the dataset filename at runtime and reads it with pandas.
2. **Explore.** It inspects the structure with `info()` and `head()`/`tail()`, and plots price against number of rooms with a box plot.
3. **Clean.** Rows with a price above 600,000 are removed (4 rows), and rows with a price below 10,000 are removed (55 rows). This drops the implausible values. The notebook leaves 9,941 rows.
4. **Prepare features.** The target column `price` is separated from the features. All features are already numeric, so the one-hot encoding step does not change the data.
5. **Split.** The data is split into 85% training and 15% test sets with `random_state=42` for reproducibility. The test set has 1,492 rows.
6. **Train.** A scikit-learn `LinearRegression` model is fitted on the training set.
7. **Evaluate.** The model is scored on the test set using R² (coefficient of determination).
8. **Save.** The trained model is saved as `model.pkl` with joblib, so it can be loaded later for predictions without retraining.

## Results

On the held-out test set, the model achieves:

| Metric | Value |
|---|---|
| R² score | 0.955 |
| Root mean squared error (RMSE) | about 20,200 |

An R² of 0.955 means the model explains about 95.5% of the variance in house prices on unseen data. The fitted coefficients are roughly:

- Each extra square foot adds about 150 to the price.
- Each extra room adds about 20,100.
- Each extra year of age reduces the price by about 1,000.
- Each extra kilometre from the city reduces the price by about 5,000.

These results are unusually clean for real estate data. The relationships in this dataset appear close to linear, so this model may not perform as well on real-world housing data, which usually has more variables (location, condition, amenities) and more complex patterns.

## Requirements

- Python 3.8 or newer
- pandas
- scikit-learn
- seaborn
- matplotlib
- joblib
- Jupyter Notebook or JupyterLab (or Google Colab)

Install the libraries with pip, or open the notebook in Google Colab, where most of them are already available.

## How to Run

1. Clone this repository or download the files.
2. Open `house_price_prediction.ipynb` in Jupyter or Google Colab.
3. Run the cells in order. When prompted with "Enter Dataset:", enter `house_prices_dataset.csv` (or the path to the file on your machine).
4. The trained model is saved as `model.pkl` in the working directory.

To use the saved model later, load it with joblib and call `predict` with new data that has the same four feature columns, in the same order as the training data.

## Notes and Limitations

- The notebook imports `accuracy_score`, which is a classification metric and is not used here. It can be removed.
- The one-hot encoding step has no effect on this dataset because all columns are numeric. It is kept so the notebook still works if categorical columns are added later.
- The price filter thresholds (10,000 and 600,000) are chosen by inspection of this dataset and are not based on market data.
- Pickle and joblib files can run arbitrary code when loaded. Only load `model.pkl` files you created or trust.
- The model was saved with the scikit-learn and joblib versions used during training. Loading it with very different versions may cause errors.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.


