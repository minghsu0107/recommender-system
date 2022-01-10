# see more here https://scikit-learn.org/stable/modules/metrics.html#sigmoid-kernel
from sklearn.svm import SVC
# ch-square kernel kis commonly used in histograms (bags) of visual words (computer vision)
from sklearn.metrics.pairwise import chi2_kernel
X = [[0, 1], [1, 0], [.2, .8], [.7, .3]]
y = [0, 1, 0, 1]
K = chi2_kernel(X, gamma=.5)
'''
array([[1.        , 0.36787944, 0.89483932, 0.58364548],
       [0.36787944, 1.        , 0.51341712, 0.83822343],
       [0.89483932, 0.51341712, 1.        , 0.7768366 ],
       [0.58364548, 0.83822343, 0.7768366 , 1.        ]])
'''
svm = SVC(kernel='precomputed').fit(K, y)
print(svm.predict(K))  # array([0, 1, 0, 1])
