from sklearn.semi_supervised import LabelPropagation

class LabelPropagation(LabelPropagation):
	def __init__(self, kernel='rbf', *, gamma=20, n_neighbors=7,
				 max_iter=1000, tol=1e-3, n_jobs=None):
		super().__init__(kernel=kernel, gamma=gamma,
						 n_neighbors=n_neighbors, max_iter=max_iter,
						 tol=tol, n_jobs=n_jobs, alpha=None)
	