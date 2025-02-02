# AI for Coastal Resilience: Surrogate Modeling of High-Fidelity Hydrodynamic Simulators of Extreme Water Levels

## Background

Coastal regions worldwide face increasing threats from rising sea levels and extreme weather events due to climate change. Accurate prediction of water levels is crucial for effective flood risk assessment and coastal management. Traditional high-fidelity hydrodynamic models, such as the Delft3D Flexible Mesh (D3D FM) Suite, provide accurate simulations but are computationally intensive, limiting their use in real-time forecasting and long-term planning scenarios.

Recent research has focused on developing surrogate models to address this challenge. Hybrid statistical-dynamical frameworks combining climate emulators with machine learning techniques have shown promise in efficiently predicting water levels and flood risks. These advancements aim to balance computational efficiency with prediction accuracy, enabling more responsive coastal flood risk assessments and adaptation strategies.

## Vision Statement

This project aims to develop a highly efficient and accurate surrogate model for predicting coastal water levels in San Francisco Bay. By leveraging machine learning techniques including a feed-forward neural network, we will create a model that significantly reduces computational time for water level predictions compared to traditional hydrodynamic models.

The model will maintain a high level of accuracy comparable to existing high-fidelity models across diverse environmental conditions. Additionally, it will demonstrate the potential for scalable application to other coastal regions, paving the way for broader implementation of machine learning techniques in coastal hydrodynamic modeling.

## Minimum Viable Product

The Minimum Viable Product (MVP) will be a surrogate model using neural network architectures to predict water levels in San Francisco Bay. The model will:

### Features
- Take environmental parameters as inputs, including:
  - Tides
  - Mean monthly sea level anomalies
  - Waves
  - Sea level pressure
  - Winds
  - River flows
- Output steady-state water levels at 179,294 nodes across San Francisco Bay
- Process environmental input parameters using neural networks for water level prediction
- Predict steady-state water levels for individual time steps based on input conditions

### Success Criteria
- **Accuracy**: Evaluated using Root Mean Square Error by comparing:
  - Predicted water levels against high-fidelity hydrodynamic model simulations
  - Predictions against NOAA observed water levels at 15 gauges within the area
- **Computational Efficiency**: 
  - Generate predictions for a full year of hourly data in a fraction of the time required by current methods
  - Maintain performance when scaled to all 179,294 nodes of the San Francisco Bay grid

## Project Partners
- Peter Ruggiero & Zhenqiang Wang - College of Earth, Ocean, and Atmospheric Sciences, OSU
