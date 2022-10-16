# splines paper

we present a color enhancement method based on the estimation of parameters of a spline that approximates a 3dlut

first we show that there is room for improvement:
  [x] 3dlut image oracle is better than per channel image oracle
  [x] (failed) add residual to oracle
  [x] (failed) code their spline as oracle
  [] perchannel free xs spline oracle performs better than perchannel fixed xs spline oracles
  [] free 3dlut (vector) oracles perform better than free perchannel spline oracles
  [] we can speed up the computation by predicting the alphas instead of the ys

then we show some other interesting results:
  [] vector spline performs better than free perchannel spline with same number of params (6xN vs. 6xN)
  [] 3dlut spline in lab performs better than in rgb (compare curves for different N)
  and we choose parameters accordingly

then we train the models and evaluate the performance (over validation!)
  [] implement collate function to deal with different shapes
  [] fixed xs rgb per channel (baseline)
  [] free xs rgb per channel
  [] free xs lab per channel
  [] free 3dlut lab/rgb per channel
  [] free 3dlut lab/rgb per channel different architecture
  [] free 3dlut lab/rgb per channel different architecture + hsv hist
  [] use color balance before hand


now retrain the best model and put a table comparing against other methods

def pseudocode():
    if usual_training:
        params = network(raw)  # params is xs, alphas
        out = predict_training(raw, params)  # in their code
    elif naive_training:
        params = network(raw)  # params is xs, ys
        out = predict_training(raw, params)  # in their code
    else:
        params = fit(raw, enh)  # oracle
        out = predict_oracle(raw, params)



## Oracles

- show how splines work 

## performance of oracles
- for different colorspaces: (RGB, CIELab, HSV)
    - 3d lut oracle
    - per channel oracle
    - for different N knots: (5, 10, 20, 50, 100, 200, 500, 1000)
        for different spline functions: (gaussian)
            per channel spline oracle

- add extra information (deep features + histogram)
  - train with batch size 1
  - train with rotated images

# experiments

## free the xs

## xs in 3D

## different spline functions
- exp, thin, poly1, poly2, poly3

## different colorspaces

## using augmentations

# list of
0.1 - oracle 3dlut with knots from images
0.2 - oracle per channel with knots from images
for each spline type:
0.3 - oracle 3dlut with N knots via gradient descent
0.4 - oracle per channel with N knots via gradient descent
1.1 - free the xs for per channel transformations (oracle)

2 - different functions for per channel transformation (in RGB)
3 - different functions for 3dlut transformation (in RGB)
4 - different colorspaces per channel using the best function
5 - augmentations using the best result

