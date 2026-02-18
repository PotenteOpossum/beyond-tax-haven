import pandas as pd
import os
from typing import List, Dict, Tuple, Any
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
import torch
from torch_geometric.data import Data
import numpy as np
from sklearn.preprocessing import QuantileTransformer, LabelEncoder, StandardScaler, MinMaxScaler, PowerTransformer, FunctionTransformer
import networkx as nx
from sklearn.model_selection import train_test_split

class TemporalDatasetLoader:
    def __init__(self, config: Dict[str, Any], annotate: bool = False):
        self.config = config
        self.base_path = config['data']['base_path']
        self.distance_path = config['data']['distance_path']
        
        # Columns to exclude from feature renaming or usage
        self.exception_list = ['name', 'year', 'Number of workplaces', 'Number of employees', 
                          'growth of workplaces - swiss', 'growth of employees - swiss']
        
        self._load_data(annotate)

    def _rename_columns(self, df: pd.DataFrame, suffix: str, except_on: List[str] = None) -> pd.DataFrame:
        if except_on is None:
            except_on = self.exception_list
        df = df.rename(columns={col: f"{col} [{suffix}]" for col in df.columns if col not in except_on})
        return df

    def _load_data(self, annotate: bool):
        # Load datasets
        expenditure = pd.read_csv(os.path.join(self.base_path, self.config['data']['expenditure_file']))
        statent = pd.read_csv(os.path.join(self.base_path, self.config['data']['statent_file']))
        population = pd.read_csv(os.path.join(self.base_path, self.config['data']['population_file']))
        taxes = pd.read_csv(os.path.join(self.base_path, self.config['data']['tax_file']))

        # Remove redundant column
        if 'Permanent resident population' in population.columns:
            del population['Permanent resident population']

        # Rename columns to avoid conflicts if requested
        if annotate:
            expenditure = self._rename_columns(expenditure, 'E')
            statent = self._rename_columns(statent, 'S')
            population = self._rename_columns(population, 'P')
            taxes = self._rename_columns(taxes, 'T')

        # Identify feature columns
        self.taxes_cols = [c for c in taxes.columns if c not in ['name', 'year']]
        self.expenditure_cols = [c for c in expenditure.columns if c not in ['name', 'year']]
        self.statent_cols = [c for c in statent.columns if c not in ['name', 'year', 'Number of workplaces', 'Number of employees']]
        self.population_cols = [c for c in population.columns if c not in ['name', 'year']]

        # Load Distance/OD Matrix
        od_path = os.path.join(self.distance_path, self.config['data']['od_matrix_file'])
        self.od_df = pd.read_csv(od_path, index_col=0).fillna(0).astype(int)

        # Merge DataFrames
        self.df = expenditure.merge(statent, on=['name', 'year'])\
                                .merge(population, on=['name', 'year'])\
                                .merge(taxes, on=['name', 'year'])\
                                .sort_values('year').reset_index(drop=True)
        
        # Clean up Unnamed columns
        for colname in self.df.columns:
            if 'Unnamed: 0' in colname:
                del self.df[colname]

    def _filter_mun_full_coverage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only municipalities that have data for all years."""
        years = sorted(df['year'].unique())
        muni_year_counts = df.groupby("name")["year"].nunique()
        valid_munis = muni_year_counts[muni_year_counts == len(years)].index
        return df[df["name"].isin(valid_munis)].reset_index(drop=True)

    def _get_growth(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Calculate percentage growth from previous year."""
        df['Previous_Year_Workplaces'] = df.groupby('name')['Number of '+col].shift(1)
        df['growth of '+col] = df.apply(
            lambda row: ((row['Number of '+col] - row['Previous_Year_Workplaces']) /
                        row['Previous_Year_Workplaces']) * 100
            if pd.notna(row['Previous_Year_Workplaces']) and row['Previous_Year_Workplaces'] != 0
            else np.nan, axis=1
        )
        del df['Previous_Year_Workplaces']
        del df['Number of '+col]
        return df

    def _compute_net_feature(self, od_df: pd.DataFrame, knn_k: int = None) -> pd.DataFrame:
        """
        Computes network features (centrality, pagerank, clustering) from a distance matrix.
        """
        G = nx.Graph()
        G.add_nodes_from(od_df.index)

        # K-NN Graph or Fully Connected
        if knn_k is not None and knn_k > 0:
            for node in od_df.index:
                neighbors = od_df.loc[node].nsmallest(knn_k + 1).index[1:]
                for neighbor in neighbors:
                    distance = od_df.loc[node, neighbor]
                    if distance > 0:
                        G.add_edge(node, neighbor, weight=1.0/distance)
        else:
            for i, municipality1 in enumerate(od_df.index):
                for j, municipality2 in enumerate(od_df.index):
                    if j > i:
                        distance = od_df.loc[municipality1, municipality2]
                        if distance > 0:
                            G.add_edge(municipality1, municipality2, weight=1.0/distance)

        N = G.number_of_nodes()
        strength = dict(G.degree(weight='weight'))
        features = {
            'degree_centrality': {n: (s / (N - 1)) if N > 1 else 0 for n, s in strength.items()},
            'pagerank': nx.pagerank(G, weight='weight') if N > 0 else {n:0 for n in G.nodes()},
            'clustering_coefficient': nx.clustering(G, weight='weight')
        }
        return pd.DataFrame(features)

    def get_dataset(self, feature_combination: List[str], model: str, use_real_weights: bool = True, lags: int = 3,
                    as_temporal: bool = True, filter_method: str = 'None', iteration: int = 0):
        
        temp_df = self.df.copy()
        temp_df = temp_df.dropna()
        temp_df = self._filter_mun_full_coverage(temp_df)
        
        predictor = self.config['experiment']['predictor']

        # Feature selection
        cols_to_drop = []
        if 'expenditure' in feature_combination: cols_to_drop.extend(self.expenditure_cols)
        if 'population' in feature_combination: cols_to_drop.extend(self.population_cols)
        if 'statent' in feature_combination: cols_to_drop.extend(self.statent_cols)
        if 'taxes' in feature_combination: cols_to_drop.extend(self.taxes_cols)
        temp_df = temp_df.drop(columns=cols_to_drop, errors='ignore')

        # Filter municipalities to match OD matrix
        df_municipalities = sorted(temp_df['name'].unique())
        od_municipalities = self.od_df.index.tolist()
        common_municipalities = sorted(set(df_municipalities) & set(od_municipalities))

        temp_df = temp_df[temp_df['name'].isin(common_municipalities)].reset_index(drop=True)
        od_df_filtered = self.od_df.loc[common_municipalities, common_municipalities]
        self.municipality_to_idx = {name: i for i, name in enumerate(common_municipalities)}
        num_nodes = len(common_municipalities)

        # Merge Swiss data if enabled
        if self.config['experiment']['swiss']:
            swiss_df = pd.read_csv(os.path.join(self.base_path, self.config['data']['swiss_file']))
            self.swiss_df = swiss_df[['year', 'Number of workplaces - swiss']]
            temp_df = temp_df.merge(self.swiss_df, on='year')

        # Handle predictor column selection
        col = 'workplaces' if 'workplaces' in predictor else 'employees'
        
        # Remove target artifacts from features
        if self.config['experiment']['cross_features']:
            drop_pattern = '- Number of employees' if col == 'employees' else '- Number of workplaces'
            temp_df = temp_df.drop(columns=[c for c in temp_df.columns if drop_pattern in c])
            del temp_df['Number of workplaces' if col == 'employees' else 'Number of employees']
        else:
             drop_pattern = '- Number of workplaces' if col == 'employees' else '- Number of employees'
             temp_df = temp_df.drop(columns=[c for c in temp_df.columns if drop_pattern in c])
             del temp_df['Number of workplaces' if col == 'employees' else 'Number of employees']

        # Calculate growth if needed
        if "growth" in predictor:
            temp_df = self._get_growth(temp_df.copy(), col)
            if self.config['experiment']['swiss']:
                temp_df = self._get_growth(temp_df.copy(), col+' - swiss')

        # Network features
        if self.config['experiment']['use_net_features'] and model in ['RandomForestRegressor', 'XGBRegressor'] and use_real_weights:
            net_df = self._compute_net_feature(od_df_filtered, self.config['experiment'].get('knn_k', None)).reset_index(drop=False)
            temp_df = temp_df.merge(net_df, left_on='name', right_on='index')
            del temp_df['index']

        # --- Graph Construction ---
        od_matrix = od_df_filtered.values
        if use_real_weights and filter_method == 'quartile':
             values_to_consider = od_matrix.flatten()
             values_to_consider = values_to_consider[values_to_consider != 0]
             percentile_threshold = np.percentile(values_to_consider, self.config['experiment']['quartile'])
             src, dst = np.nonzero(od_matrix)
             weights = od_matrix[src, dst]
             mask = weights <= percentile_threshold
             edge_index = torch.tensor(np.array([src[mask], dst[mask]]), dtype=torch.long)
        elif use_real_weights and filter_method == 'knn':
             k = self.config['experiment'].get('knn_k', 10)
             new_src, new_dst = [], []
             for i in range(num_nodes):
                 distances = od_matrix[i, :]
                 neighbor_indices = np.where(distances > 0)[0]
                 neighbor_distances = distances[neighbor_indices]
                 knn_indices = neighbor_indices[np.argsort(neighbor_distances)][:k]
                 for neighbor_idx in knn_indices:
                     new_src.append(i)
                     new_dst.append(neighbor_idx)
             edge_index = torch.tensor(np.array([new_src, new_dst]), dtype=torch.long)
        else:
             src, dst = np.nonzero(od_matrix)
             edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)

        self.edge_index = edge_index

        # Edge weights
        if use_real_weights:
            final_src, final_dst = self.edge_index.numpy()
            non_zero_weights = od_matrix[final_src, final_dst]
            if self.config['experiment']['gaussian_normalization']:
                sigma = non_zero_weights.std()
                edge_weight = torch.tensor(np.exp(-(non_zero_weights**2) / (2 * sigma**2)), dtype=torch.float)
            else:
                edge_weight = torch.tensor(np.divide(3600, non_zero_weights, where=non_zero_weights!=0), dtype=torch.float)
        else:
            edge_weight = torch.ones(self.edge_index.size(1), dtype=torch.float)

        if self.config['experiment']['normalization'] and use_real_weights:
            row, col = self.edge_index
            deg = torch.zeros(num_nodes, dtype=torch.float, device=row.device).scatter_add_(0, row, edge_weight)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
            self.edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        else:
            self.edge_weight = edge_weight

        # Center predictor per municipality
        mean_per_municipality = temp_df.groupby('name')[predictor].transform('mean')
        temp_df[predictor] = temp_df[predictor] - mean_per_municipality
        
        feature_cols = temp_df.drop(columns=['name', predictor, 'year'], errors='ignore').columns
        years = sorted(temp_df['year'].unique())
        
        # Prepare arrays
        all_features_np = np.zeros((len(years), num_nodes, len(feature_cols)))
        all_targets_np = np.zeros((len(years), num_nodes, 1))
        all_targets_names_np = np.full((len(years), num_nodes, 1), fill_value='', dtype=object)

        self.temp_df = temp_df.copy()

        for i, year in enumerate(years):
            df_year = temp_df[temp_df['year'] == year]
            for _, row in df_year.iterrows():
                if row['name'] in self.municipality_to_idx:
                    idx = self.municipality_to_idx[row['name']]
                    all_features_np[i, idx, :] = row[feature_cols].values
                    all_targets_np[i, idx, :] = row[predictor]
                    all_targets_names_np[i, idx, :] = row['name']

        self.num_nodes = num_nodes
        self.num_features = len(feature_cols)
        self.feature_cols = feature_cols

        # Splitting and Scaling
        if as_temporal:
            train_years_count = self.config['experiment']['train_years']

            train_features_np = all_features_np[:train_years_count]
            train_targets_np = all_targets_np[:train_years_count]
            test_features_np = all_features_np[train_years_count:]
            test_targets_np = all_targets_np[train_years_count:]

            train_features_flat = train_features_np.reshape(-1, self.num_features)
            train_targets_flat = train_targets_np.reshape(-1, 1)

            scaler_x = StandardScaler().fit(train_features_flat)
            scaler_y = StandardScaler().fit(train_targets_flat)

            self.scaler_x = scaler_x
            self.scaler_y = scaler_y

            train_features_scaled = scaler_x.transform(train_features_flat).reshape(train_features_np.shape)
            train_targets_scaled = scaler_y.transform(train_targets_flat).reshape(train_targets_np.shape)

            test_features_flat = test_features_np.reshape(-1, self.num_features)
            test_features_scaled = scaler_x.transform(test_features_flat).reshape(test_features_np.shape)
            test_targets_scaled = test_targets_np # Test targets usually kept raw for evaluation, or handled inside model wrapper

            train_dataset = StaticGraphTemporalSignal(self.edge_index, self.edge_weight, train_features_scaled, train_targets_scaled)
            test_dataset = StaticGraphTemporalSignal(self.edge_index, self.edge_weight, test_features_scaled, test_targets_scaled)
            
            return train_dataset, None, test_dataset
        
        else: # Random Split
             # (Logic for random split remains similar but can be cleaned if needed. 
             #  For now I'm simplifying by omitting the copious repeated comments)
             # ... implementation of random split ...
             # For brevity in this refactor I will implement the random split logic as well since it was in the original file
            
            all_nodes_map = []
            for i in range(len(years)):
                for j in range(num_nodes):
                    all_nodes_map.append((i, j))

            indices = np.arange(len(all_nodes_map))
            train_val_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42 + iteration)
            
            val_ratio = self.config['experiment']['val_ratio']
            if val_ratio > 0:
                train_indices, val_indices = train_test_split(train_val_indices, test_size=val_ratio, random_state=42 + iteration)
            else:
                train_indices = train_val_indices
                val_indices = np.array([], dtype=int)

            train_features_to_fit = all_features_np.reshape(-1, self.num_features)[train_indices]
            train_targets_to_fit = all_targets_np.reshape(-1, 1)[train_indices]

            scaling_method = self.config['experiment'].get('scaling_method', 'yeo-johnson')
            
            if scaling_method == 'None':
                scaler_x = FunctionTransformer(lambda x: x, validate=False)
                scaler_y = FunctionTransformer(lambda x: x, validate=False)
            elif scaling_method == "yeo-johnson":
                scaler_x = PowerTransformer(method="yeo-johnson", standardize=True).fit(train_features_to_fit)
                scaler_y = PowerTransformer(method="yeo-johnson", standardize=True).fit(train_targets_to_fit)
            elif scaling_method == "log":
                scaler_x = FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=True).fit(train_features_to_fit)
                scaler_y = FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=True).fit(train_targets_to_fit)
            elif scaling_method == 'standard':
                scaler_x = StandardScaler().fit(train_features_to_fit)
                scaler_y = StandardScaler().fit(train_targets_to_fit)
            elif scaling_method == 'quantile':
                scaler_x = QuantileTransformer(output_distribution='normal').fit(train_features_to_fit)
                scaler_y = QuantileTransformer(output_distribution='normal').fit(train_targets_to_fit)
            
            self.scaler_x = scaler_x
            self.scaler_y = scaler_y

            all_features_scaled = scaler_x.transform(all_features_np.reshape(-1, self.num_features)).reshape(all_features_np.shape)
            all_targets_scaled = scaler_y.transform(all_targets_np.reshape(-1, 1)).reshape(all_targets_np.shape)

            all_graphs = []
            for i in range(len(years)):
                data = Data(
                            features=torch.from_numpy(all_features_scaled[i]).float(),
                            edge_index=self.edge_index,
                            edge_weight=self.edge_weight,
                            targets=torch.from_numpy(all_targets_scaled[i]).float(),
                            targets_names=all_targets_names_np[i]
                            )
                data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
                data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
                data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
                data.year = years[i]
                all_graphs.append(data)

            for idx in train_indices:
                graph_idx, node_idx = all_nodes_map[idx]
                all_graphs[graph_idx].train_mask[node_idx] = True
            for idx in val_indices:
                graph_idx, node_idx = all_nodes_map[idx]
                all_graphs[graph_idx].val_mask[node_idx] = True
            for idx in test_indices:
                graph_idx, node_idx = all_nodes_map[idx]
                all_graphs[graph_idx].test_mask[node_idx] = True

            from torch_geometric.loader import DataLoader
            loader = DataLoader(all_graphs, batch_size=1, shuffle=False)

            return loader, None, None
