# Predicting Flight Delays

This project aims to predict flight delays using historical flight data. The project is structured to facilitate data processing, model development, and performance evaluation.

## Project Structure

- **data/**: Contains datasets used in the project.
  - **interim/**: Intermediate data that has been transformed during processing.
  - **processed/**: Final datasets that are ready for modeling.
  - **raw/**: Original, immutable data, including `flights_sample_3m.csv`.

- **models/**: Stores various machine learning models.
  - **deep_learning/**: Saved deep learning model files.
  - **machine_learning/**: Saved traditional machine learning model files.
  - **time_series/**: Saved time series model files.

- **notebooks/**: Jupyter notebooks for exploration and analysis.
  - **exploratory/**: Notebooks for initial data exploration and analysis.
    - `01_initial_data_exploration.ipynb`: Initial exploration of the dataset.
    - `02_delay_patterns_analysis.ipynb`: Analysis of patterns in flight delays.
    - `03_feature_importance.ipynb`: Assessment of feature importance in predicting delays.
  - **feature_engineering/**: Notebooks focused on feature creation.
  - **modeling/**: Notebooks for model development.
  - **preprocessing/**: Notebooks for data cleaning.

- **reports/**: Contains reports and visualizations.
  - **figures/**: Generated visualizations related to the analysis.
  - **performance/**: Reports on model performance.

- **src/**: Source code for the project.
  - **data/**: Utilities for data handling.
    - `cleaner.py`: Functions for data cleaning, including handling missing values and outliers.
    - `loader.py`: Functions for loading data from various sources.
  - **features/**: Code related to feature engineering.
  - **models/**: Implementations of various models.
    - **deep_learning/**: Deep learning model implementations.
    - **machine_learning/**: Traditional machine learning model implementations.
    - **time_series/**: Time series model implementations.
  - **pipelines/**: End-to-end workflows for data processing and model training.
    - `data_pipeline.py`: Workflow for data processing and preparation.
    - `model_pipeline.py`: Workflow for model training and evaluation.
  - **visualization/**: Utilities for creating visualizations.
    - `visualize.py`: Functions for visualizing data and model results.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   ```

2. Navigate to the project directory:
   ```
   cd predicting-flight-delays
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Run the Jupyter notebooks in the `notebooks/` directory for exploration and analysis.

## Usage

- Use the notebooks in the `notebooks/` directory to explore the data, engineer features, and develop models.
- The `src/` directory contains the code for data processing, model training, and visualization.
- Generated reports and figures can be found in the `reports/` directory.

## License

This project is licensed under the MIT License. See the LICENSE file for details.