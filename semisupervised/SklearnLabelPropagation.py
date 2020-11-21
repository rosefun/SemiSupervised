from sklearn.semi_supervised import LabelPropagation
import warnings
warnings.simplefilter('error')
warnings.filterwarnings('default', category=PendingDeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)
# filter warning of numpy
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ImportWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class LabelPropagation(LabelPropagation):
	def __init__(self, kernel='rbf', *, gamma=20, n_neighbors=7,
				 max_iter=1000, tol=1e-3, n_jobs=None):
		super().__init__(kernel=kernel, gamma=gamma,
						 n_neighbors=n_neighbors, max_iter=max_iter,
						 tol=tol, n_jobs=n_jobs, alpha=None)
	