import pandas as pd

from tsfresh import extract_features
from tsfresh.feature_extraction.settings import from_columns


class FeatureExtractor:
    """Extract selected TSFresh features from raw sensor data."""

    SENSOR_COLUMNS = [
        "id",
        "Sample",
        "ADXL_X",
        "ADXL_Y",
        "ADXL_Z",
        "GYRO_X",
        "GYRO_Y",
        "GYRO_Z",
        "MMA_X",
        "MMA_Y",
        "MMA_Z",
    ]

    @staticmethod
    def extract_selected_features(
        raw_test_file,
        selected_features,
        target,
    ):
        """
        Extract only the requested TSFresh features.

        Parameters
        ----------
        raw_test_file : str
            Path to test.csv

        selected_features : list[str]
            Features selected by the ranking method.

        target : str
            "Subject_ID" or "Task_ID"

        Returns
        -------
        X_test : pandas.DataFrame
        y_test : pandas.Series
        """

        # Load raw test data
        df = pd.read_csv(raw_test_file)
        df.insert( 0, "id", 
                   df["Subject_ID"].astype(str)
                   + "_"
                   + df["Task_ID"].astype(str)
                   + "_"
                   + df["Trial"].astype(str),
                 )


        # Save labels before removing them
        y_test = (
            df.groupby("id")[target]
                .first()
                .sort_index()
        )

        # Keep only sensor signals
        sensor_df = df[FeatureExtractor.SENSOR_COLUMNS]

        # Build TSFresh settings from selected feature names
        fc_parameters = from_columns(selected_features)

        # Extract ONLY the requested features
        X_test = extract_features(
            sensor_df,
            column_id="id",
            column_sort="Sample",
            kind_to_fc_parameters=fc_parameters,
        )

        # Keep the same feature order as the training data
        X_test = X_test[selected_features]
        y_test = y_test.loc[X_test.index]

        return X_test, y_test