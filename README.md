# Beyond the Tax Haven: A Graph Analysis of Business Attraction in Swiss Municipalities

Arthur Capozzi* and Damian Dailisan*

*Computational Social Science, ETH Zürich, Stampfenbachstr., Zurich, 8092, Switzerland.

## Abstract
Switzerland's decentralized fiscal structure has long been anecdotally credited with creating intense tax competition among its municipalities, famously attracting businesses to cantons like Zug. This research proposes a data-driven analysis of the factors that influence the business landscape in 226 Swiss municipalities from 2011 to 2022. By leveraging a rich collection of spatio-temporal datasets, we build a predictive model of business creation and use explainable AI techniques to uncover the key socioeconomic drivers of municipal attractiveness.

Our core methodology uses machine learning models, particularly graph neural networks (GNNs), to learn and capture the complex interdependencies between municipalities. Here, a GNN using attention mechanisms performs the best with a median $R^2 = 0.832$ when using business sector demographics, population, municipal expenditure, and tax rate feature sets. Combining the trained models with explainable AI, we find that the most important features are coming from the business statistics datasets, rather than the tax data. However, a more granular analysis of municipalities grouped by primary language shows a different set of important features, highlighting the importance of a contextual, localized approach rather than a one-size-fits-all analysis. This study will provide a nuanced understanding of the interaction between tax policies, demographics, infrastructure, and other factors in shaping Switzerland's economic geography.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Data Placement**:
    Place the following **6 files** in the `Data` directory:
    - `expenditure_reduced.csv` (Expenditure data)
    - `normalized_statent2_EE.csv` (STATENT data)
    - `data_reduced_residents.csv` (Population data)
    - `TAX_without_child.csv` (Tax data)
    - `car_table_gmaps_renamed.csv` (Distance matrix)
    - `data_swiss.csv` (National level data)

## Running Experiments

You can run experiments directly using the python script:

```bash
# Run a single experiment from the repository root
python scripts/run_experiment.py -c config.yaml
```

Results will be saved in the `results` directory.


## 📜 Citation
If you use this code or data, please cite:
> Beyond the tax haven: a graph analysis of business attraction in Swiss municipalities
> Capozzi, A., & Dailisan, D. (2026)
> EPJ Data Science, 15(1), 15.

BibTeX:
> @article{Capozzi2026,
>   author = {Capozzi, Arthur and Dailisan, Damian},
>   title = {Beyond the tax haven: a graph analysis of business attraction in Swiss municipalities},
>   journal = {EPJ Data Science},
>   year = {2026},
>   volume = {15},
>   number = {1},
>   pages = {15},
>   doi = {10.1140/epjds/s13688-026-00619-4},
>   url = {https://doi.org/10.1140/epjds/s13688-026-00619-4}
> }