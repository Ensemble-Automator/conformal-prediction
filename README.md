# conformal-prediction
Quantify uncertainty in any ML model

- Conformal prediction is a framework that quantifies uncertainity by estimating the confidence and credibility of test point predictions.
- Currently, our model is able to accurately predict (100%) on new predictions when it has a confidence ≥ 75%. The model used is a simple ANN for loan risk
calculation. 
- Conformal prediction works using a nearest centroid classifier, along with computing non-conformal and p-value score.
- So far, our framework has been testing on ANN's only. We're currently working on testing it on RNN and CNN's.
* This is part of our project on ML interability via Actor-Critic networks.
