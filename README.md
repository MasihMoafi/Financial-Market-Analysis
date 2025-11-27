# Financial Market Analysis

A collection of machine learning projects for financial market analysis, covering time-series forecasting, clustering, reinforcement learning, and dynamic pricing strategies.

---

## Projects

### 1. BTC Time Series Analysis (ResNet-LSTM)

Hybrid deep learning model combining LSTM with Residual Network architecture for Bitcoin price prediction.

**Architecture:**
- Two LSTM layers with dropout regularization
- TimeDistributed Dense layer applied to each time step
- ResNet-style residual connections via custom ResidualWrapper

**Results:**
| Metric | Value |
|--------|-------|
| MAE | 0.015 |
| MSE | 0.0005 |
| RMSE | 0.023 |

**Key Features:**
- Predicts price *changes* rather than absolute prices for robustness
- 6-day lookback window predicting 6-day forward changes
- Includes swing trading simulation with SMA-30 indicator
- Compared against CNN-Transformers, vanilla LSTM, regression, and ConvNet variants

---

### 2. Clustering 100 Crypto Coins

Unsupervised clustering of cryptocurrencies using K-Means based on market data attributes.

**Pipeline:**
- Data retrieval via CoinGecko API
- Preprocessing and feature engineering
- K-Means clustering with PCA for visualization

**Results:**
- Silhouette Score: **0.914**

---

### 3. Transformers for Time-Series Analysis

Transformer-based architecture for financial time-series forecasting.

**Branches:**
- `focal-loss-gpu-opts` - Focal loss implementation with GPU optimizations
- `gtn-transformer-variant` - Gated Transformer Network variant

---

### 4. RL Trader Agent

Reinforcement learning agent for automated trading strategies.

*Work in progress*

---

### 5. Dynamic Pricing Strategy

Random Forest Regression model for predicting optimal pricing based on market conditions.

**Features:**
- Number of riders/drivers
- Vehicle type
- Expected ride duration

**Pipeline:**
- Categorical encoding and outlier detection
- EDA with Plotly visualizations
- Random Forest training and evaluation

---

## Tech Stack

- **Deep Learning:** PyTorch, TensorFlow/Keras
- **ML:** scikit-learn, XGBoost
- **Data:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **APIs:** CoinGecko

---

## Contact

For detailed findings and proprietary implementations, contact: masihmoafi12@gmail.com
