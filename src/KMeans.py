class Node:
	"""docstring for Node"""
	def __init__(self, key):
		self.key = key
		self.left = None
		self.right = None


class Tree:
	"""docstring for Tree"""
	def insert(self, root, key):
		if root == None:
			return Node(key)
		elif key <= root.key:
			root.left = self.insert(root.left, key)
		elif key > root.key:
			root.right = self.insert(root.right, key)

	def search(self, root, key):
		








class KNN:
	"""docstring for KNN"""
	def __init__(self, neighbour=5):
		self.N = neighbour
		self.train_set = None		

	def add(self):
		pass

