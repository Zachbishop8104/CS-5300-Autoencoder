from numpy.random import RandomState
from sklearn.datasets import fetch_olivetti_faces
import gallary as gal

from layers.linearLayer import linearLayer
from layers.reluLayer import reluLayer

def main():
    n_row, n_col = 3, 4
    n_components = n_row * n_col
    image_shape = (64, 64)
    
    rng = RandomState(0)
    faces, _ = fetch_olivetti_faces(return_X_y=True, shuffle=True, random_state=rng)
    n_samples, n_features = faces.shape
    
    #global centering
    faces_centered = faces - faces.mean(axis=0)
    
    #local centering
    faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)
    
    print("Dataset consists of %d faces" % n_samples)
    gal.plot_gallery("Faces from dataset", faces_centered[:n_components], n_col=n_col, n_row=n_row)
    
    #create test data
    h1 = linearLayer(n_features, 20)
    pass1 = h1.forward(faces_centered[:100])

if __name__ == "__main__":
    main()