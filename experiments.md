
def fit_K_Gaussian(xs, ys, sigma=1):
    n = len(xs)
    K = np.zeros((n,n))
    X1 = np.repeat(xs, (n,1))
    X2 = np.tile(xs, (1,n))
    D = np.linalg.norm(X1-X2)
    E = np.exp(-D/(2*sigma**2))
    K = E.reshape((n,n))
    return np.linalg.inv(K)@ys



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

