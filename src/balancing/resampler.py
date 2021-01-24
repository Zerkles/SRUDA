from src.balancing import oversampling, undersampling, multiclass_resampling

from src.balancing.data_controller import DataController
import numpy as np


class Resampler:
    def __init__(self, method_name: str, filepath_source: str):
        self.unbalanced_dataset = DataController.read_categorized_criteo(filepath_source)
        self.X_unbalanced, self.y_unbalanced = DataController.split_data_on_x_y(self.unbalanced_dataset)
        self.resampling_method_object = self.__method_selector(method_name)

    def __method_selector(self, balancing_type: str):
        if balancing_type == 'ros':
            return oversampling.random_over_sampler_optimized()
        elif balancing_type == 'smotenc':
            if self.unbalanced_dataset.shape[0] > 10000:  # it just cant handle more than 10k samples because of ram
                self.X_unbalanced = self.X_unbalanced.head(10000)
                self.y_unbalanced = self.y_unbalanced.head(10000)
            return oversampling.smotenc_optimized(self.X_unbalanced)
        elif balancing_type == 'rus':
            return undersampling.random_under_sampler_optimized()
        elif balancing_type == 'nearmiss':
            return undersampling.nearmiss_optimized()
        elif balancing_type == 'enn':
            return undersampling.edited_nearest_neighbours_optimized()
        elif balancing_type == 'renn':
            return undersampling.repeated_edited_nearest_neighbours_optimized()
        elif balancing_type == 'allknn':
            return undersampling.allknn_optimized()
        elif balancing_type == 'onesided':
            return undersampling.one_sided_selection_optimized()
        elif balancing_type == 'ncr':
            return undersampling.neighbourhood_cleaning_rule_optimized()
        elif balancing_type == 'iht':
            return undersampling.instance_hardness_threshold_optimized()
        elif balancing_type == 'globalcs':
            return multiclass_resampling.global_cs_optimized()
        elif balancing_type == 'soup':
            return multiclass_resampling.soup_optimized()
        else:
            raise ValueError("Incorrect resampler type: " + balancing_type)

    def get_name(self) -> str:
        return self.resampling_method_object.__str__().title().split("(")[0]

    def set_params(self, **params):
        self.resampling_method_object.set_params(**params)

    def get_params(self) -> str:
        print(self.resampling_method_object.get_params())
        return str(self.resampling_method_object.get_params())

    def resample_to_ndarray(self) -> (np.ndarray, np.ndarray):
        X_resampled, y_resampled = self.resampling_method_object.fit_resample(self.X_unbalanced, self.y_unbalanced)
        return X_resampled.values, y_resampled.values.ravel()

    def resample_and_write_to_csv(self, filepath_destination: str, name: str = None) -> str:
        if name is None:
            name = self.resampling_method_object.__str__().split("(")[0]

        X_resampled, y_resampled = self.resampling_method_object.fit_resample(self.X_unbalanced, self.y_unbalanced)

        balanced_df = X_resampled
        balanced_df["Sales"] = y_resampled

        filepath_balanced = f"{filepath_destination}/{name}.csv"

        balanced_df.to_csv(filepath_balanced, index=False)
        print("Balanced:", name, DataController.count_classes_size(y_resampled))

        return filepath_balanced
