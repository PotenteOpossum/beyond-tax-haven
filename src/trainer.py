import torch
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.optim.lr_scheduler import ExponentialLR
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from torch.utils.data import DataLoader

from .models import RecurrentGCN, GNNGAT, SimpleGCN, GraphTransformer

class Experiment:
    def __init__(self, config: Dict[str, Any], data_loader: DataLoader, as_temporal: bool = True) -> None:
        self.config = config
        self.data_loader = data_loader
        self.as_temporal = config['experiment'].get('as_temporal', True)
        self.results: List[Dict[str, Any]] = []
        self.trained_models: Dict[str, Any] = {}
        self.timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = self.config['experiment']['output_path']
        
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            
        self.data_loader_data = []

    def _convert_numpy_to_list(self, obj):
        if isinstance(obj, dict):
            return {k: self._convert_numpy_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_list(elem) for elem in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def _compute_metrics(self, y_test_denorm: np.ndarray, y_pred_denorm: np.ndarray, y_pred_norm: np.ndarray, y_true_norm: np.ndarray) -> Dict[str, float]:
        mse = mean_squared_error(y_test_denorm, y_pred_denorm)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_denorm, y_pred_denorm)
        r2 = r2_score(y_test_denorm, y_pred_denorm)

        mse_norm = mean_squared_error(y_true_norm, y_pred_norm)
        rmse_norm = np.sqrt(mse_norm)
        mae_norm = mean_absolute_error(y_true_norm, y_pred_norm)
        r2_norm = r2_score(y_true_norm, y_pred_norm)

        return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, 
                "mse_norm": mse_norm, "rmse_norm": rmse_norm, "mae_norm": mae_norm, "r2_norm": r2_norm}
                
    def _save_data_loader_data(self):
        results_json = os.path.join(self.output_path, "data_loader_check.json")
        processed_results = self._convert_numpy_to_list(self.data_loader_data)
        new_entry = {self.timestamp: processed_results}

        all_results = []
        if os.path.exists(results_json):
            with open(results_json, 'r') as f:
                all_results = json.load(f)
        all_results.append(new_entry)
        with open(results_json, 'w') as f:
            json.dump(all_results, f, indent=4)

    def _store_data_loader_data(self, iteration, model_name, weight_desc, feat_key):
        self.data_loader_data.append({
            'iteration': iteration,
            'model_name': model_name,
            'weight_desc': weight_desc,
            'feat_key': feat_key,
            'num_nodes': self.data_loader.num_nodes,
            'num_features': self.data_loader.num_features,
            'feature_cols': list(self.data_loader.feature_cols)
        })

    def run_experiments(self) -> None:
        models = self.config['experiment']['models']
        weight_options = self.config['experiment']['weight_options']
        feature_combinations = self.config['experiment']['feature_combinations']
        n_runs = self.config['experiment']['n_runs']
        filter_method = self.config['experiment']['filter_method']
        as_temporal = self.config['experiment']['as_temporal']

        for i in range(n_runs):
            run_pbar = tqdm(models, desc=f"Run {i+1}/{n_runs}")
            for model_name in run_pbar:
                run_pbar.set_description(f"Run {i+1}/{n_runs} - Model: {model_name}")
                for weight_desc, use_real_weights in weight_options.items():
                    for filter_type in filter_method:
                        for feat_key, feat_combination in feature_combinations.items():
                            
                            train_dataset, val_dataset, test_dataset = self.data_loader.get_dataset(
                                feat_combination,
                                model_name,
                                use_real_weights=use_real_weights,
                                as_temporal=as_temporal,
                                filter_method=filter_type,
                                iteration=i
                            )
                            self._store_data_loader_data(i+1, model_name, weight_desc, feat_key)
                            
                            scaler_y = self.data_loader.scaler_y
                            edge_index = self.data_loader.edge_index
                            edge_weight = self.data_loader.edge_weight
                            num_features = self.data_loader.num_features

                            info = {
                                'iteration': i + 1,
                                'model_name': model_name,
                                'as_temporal': as_temporal,
                                'weight_desc': weight_desc,
                                'filter': filter_type,
                                'feat_key': feat_key,
                                'num_features': num_features,
                                'cross_features': self.config['experiment']['cross_features'],
                                'use_net_features': self.config['experiment']['use_net_features'],
                                'gaussian_normalization': self.config['experiment']['gaussian_normalization']
                            }
                            
                            if model_name in ["RandomForestRegressor", 'XGBRegressor']:
                                metrics, model = self._run_rf_experiment(info, train_dataset, test_dataset, scaler_y)
                            else:
                                metrics, model = self._run_gcn_experiment(info, train_dataset, val_dataset, test_dataset, scaler_y, edge_index, edge_weight)
                            
                            if self.trained_models.get(model_name) is None:
                                self.trained_models.setdefault(model_name, model)
                            
                            result_entry = info.copy()
                            result_entry.update(metrics)
                            self.results.append(result_entry)

        self._save_results()
        self._save_data_loader_data()

    def _save_results(self) -> None:
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer,)): return int(obj)
                elif isinstance(obj, (np.floating,)): return float(obj)
                elif isinstance(obj, (np.ndarray,)): return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        
        results_json = os.path.join(self.output_path, "results.json")
        processed_results = self._convert_numpy_to_list(self.results)
        new_entry = {self.timestamp: processed_results}
        
        all_results = []
        if os.path.exists(results_json):
            with open(results_json, 'r') as f:
                all_results = json.load(f)
        all_results.append(new_entry)
        
        with open(results_json, 'w') as f:
            json.dump(all_results, f, indent=4, cls=NumpyEncoder)
        print(f"Results saved to {results_json}")

    def _run_gcn_experiment(self, info: Dict, train_dataset, val_dataset, test_dataset, scaler_y, edge_index, edge_weight):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        lags = self.config['experiment']['lags'] if 'lags' in self.config['experiment'] else 1
        prediction_window = self.config['experiment']['prediction_window'] if 'prediction_window' in self.config['experiment'] else 1
        
        # Instantiate model
        hidden_channels = self.config['gcn_params'][info['model_name']]['hidden_channels']
        
        if info['model_name'] == "RecurrentGCN":
            model = RecurrentGCN(info['num_features'] * lags, hidden_channels, prediction_window).to(device)
        elif info['model_name'] == "GNNGAT":
            model = GNNGAT(info['num_features'] * lags, hidden_channels, prediction_window).to(device)
        elif info['model_name'] == "SimpleGCN":
            model = SimpleGCN(info['num_features'] * lags, hidden_channels, prediction_window).to(device)
        elif info['model_name'] == "GraphTransformer":
            model = GraphTransformer(info['num_features'] * lags, hidden_channels, prediction_window).to(device)
        else:
            raise ValueError(f"Unknown GCN model: {info['model_name']}")

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['gcn_params'][info['model_name']]['learning_rate'])
        loss_fn = torch.nn.MSELoss()
        
        if self.config['gcn_params'][info['model_name']]['adaptive_learning_rate']:
             scheduler = ExponentialLR(optimizer, gamma=0.98)

        # Training Loop
        for _ in tqdm(range(self.config['gcn_params'][info['model_name']]['epochs']), desc="Epochs", leave=False):
            model.train()
            total_train_loss = 0
            
            if info['as_temporal'] and info['model_name'] == 'RecurrentGCN':
                hidden_state = None
                for t, seq in enumerate(train_dataset):
                    optimizer.zero_grad()
                    seq.to(device)
                    out, hidden_state = model(seq.x, seq.edge_index, seq.edge_weight, hidden_state)
                    hidden_state = hidden_state.detach()
                    loss = loss_fn(out, seq.y.view_as(out))
                    loss.backward()
                    optimizer.step()
                    total_train_loss += loss.item()
                if t > 0: total_train_loss /= (t + 1)
            else:
                 for seq in train_dataset:
                    optimizer.zero_grad()
                    seq.to(device)
                    # For non-recurrent/static dataset, x is features
                    x = seq.x if getattr(seq, 'x', None) is not None else seq.features
                    y = seq.y if getattr(seq, 'y', None) is not None else seq.targets
                    edge_w = seq.edge_weight if seq.edge_weight is not None else edge_weight
                    
                    if info['model_name'] == 'GraphTransformer':
                         out = model(x, seq.edge_index, edge_attr=edge_w)
                    else:
                         out = model(x, seq.edge_index, edge_w)
                    
                    if hasattr(seq, 'train_mask'):
                         mask = seq.train_mask
                         loss = loss_fn(out[mask], y.view_as(out)[mask])
                    else:
                         loss = loss_fn(out, y.view_as(out))
                    
                    loss.backward()
                    optimizer.step()
                    total_train_loss += loss.item()
                 if len(train_dataset) > 0: total_train_loss /= len(train_dataset)

            if self.config['gcn_params'][info['model_name']]['adaptive_learning_rate']:
                scheduler.step()

        if test_dataset is None: test_dataset = train_dataset
        prediction_dict = self._predict(model, test_dataset, scaler_y, device)
        
        # Save model state
        iteration_folder = os.path.join(self.output_path, f"iteration_{info['iteration']}")
        os.makedirs(iteration_folder, exist_ok=True)
        model_filename = f"{model.__name__}_it_{info['iteration']}_{info['weight_desc']}_{info['feat_key']}.pth"
        torch.save(model.state_dict(), os.path.join(iteration_folder, model_filename))
        
        return self._compute_metrics(**prediction_dict), model

    def _predict(self, model, dataset, scaler_y, device):
        model.eval()
        all_preds, all_truths, all_preds_norm, all_truths_norm = [], [], [], []
        
        with torch.no_grad():
            hidden_state = None
            is_recurrent = model.__name__ == "RecurrentGCN"
            
            for seq in dataset:
                seq.to(device) # Keep on device for inference
                
                if is_recurrent:
                    y_pred, hidden_state = model(seq.x, seq.edge_index, seq.edge_weight, hidden_state)
                    y_true = seq.y
                else:
                    x = seq.x if getattr(seq, 'x', None) is not None else seq.features
                    y_true = seq.y if getattr(seq, 'y', None) is not None else seq.targets
                    # y_true = seq.y if hasattr(seq, 'y') else seq.targets
                    edge_w = seq.edge_weight
                    if model.__name__ == 'GraphTransformer':
                        y_pred = model(x, seq.edge_index, edge_attr=edge_w)
                    else:
                        y_pred = model(x, seq.edge_index, edge_w)

                y_pred_np = y_pred.cpu().numpy()
                y_true_np = y_true.cpu().numpy()

                # Handle masking for static graph dataset
                if not is_recurrent and hasattr(seq, 'test_mask'):
                     y_pred_np = y_pred_np[seq.test_mask.cpu().numpy()]
                     y_true_np = y_true_np[seq.test_mask.cpu().numpy()]

                y_true_denorm = scaler_y.inverse_transform(y_true_np)
                y_pred_denorm = scaler_y.inverse_transform(y_pred_np)

                all_preds.append(y_pred_denorm)
                all_truths.append(y_true_denorm)
                all_preds_norm.append(y_pred_np)
                all_truths_norm.append(y_true_np)

        return {
            'y_pred_denorm': np.concatenate(all_preds).flatten(),
            'y_test_denorm': np.concatenate(all_truths).flatten(),
            'y_pred_norm': np.concatenate(all_preds_norm).flatten(),
            'y_true_norm': np.concatenate(all_truths_norm).flatten()
        }

    def _run_rf_experiment(self, info: dict, train_dataset, test_dataset, scaler_y):
        # Extract data from loader for sklearn models
        first_batch = next(iter(train_dataset))
        is_random_split = hasattr(first_batch, 'train_mask')
        
        if is_random_split:
            X_train, y_train, X_test, y_test = [], [], [], []
            for batch in train_dataset: # Only one batch usually
                X_train.append(batch.features[batch.train_mask])
                y_train.append(batch.targets[batch.train_mask])
                X_test.append(batch.features[batch.test_mask])
                y_test.append(batch.targets[batch.test_mask])
            X_train = np.concatenate(X_train)
            y_train = np.concatenate(y_train)
            X_test = np.concatenate(X_test)
            y_test = np.concatenate(y_test)
        else:
             # Temporal: Train set is separate from test set
             X_train = np.concatenate([b.features for b in train_dataset])
             y_train = np.concatenate([b.targets for b in train_dataset])
             X_test = np.concatenate([b.features for b in test_dataset])
             y_test = np.concatenate([b.targets for b in test_dataset])

        if info['model_name'] == "RandomForestRegressor":
             model = RandomForestRegressor(**self.config['random_forest_params'], random_state=42+info['iteration'])
        elif info['model_name'] == "XGBRegressor":
             model = XGBRegressor(**self.config['xgb_params'], random_state=42+info['iteration'])
        
        model.fit(X_train, y_train.squeeze())
        y_pred = model.predict(X_test).reshape(-1, 1)
        
        y_test_denorm = scaler_y.inverse_transform(y_test)
        y_pred_denorm = scaler_y.inverse_transform(y_pred)
        
        return self._compute_metrics(y_test_denorm, y_pred_denorm, y_pred, y_test), model
