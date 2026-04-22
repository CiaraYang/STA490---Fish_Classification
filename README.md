# Fish_Classification

## Overview

The Fish Species Classification project investigates whether wideband acoustic signals can be used to reliably distinguish fish species, focusing on classifying alewife and rainbow smelt. The analysis uses wideband acoustic recordings collected from [data source to be specified].

The project integrates data preprocessing, feature representation, and neural network modeling to capture the spectral and structural characteristics of acoustic signals. Four neural network architectures were implemented: deep neural networks (DNN), convolutional neural networks (CNN), residual networks (ResNet), and recurrent neural networks (RNN). These models allow us to capture both local patterns and sequential dependencies in the acoustic data.

Model performance was evaluated using validation-based metrics, and the final model was selected based on predictive accuracy and generalization performance on a held-out test set. This project demonstrates the potential of non-invasive acoustic monitoring for ecological research and fisheries management.

## File Structure

| Folder / File | Description |
|--------------|-------------|
| `Data/` | Processed and raw hydroacoustic datasets |
| `EDA/` | Exploratory data analysis |
| `Models/` | All neural network models (DNN, CNN, ResNet, RNN) and training scripts |
| `Resources/` | External repositories and additional resources |
| `README.md` | Project documentation |
