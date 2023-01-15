from indexer import Indexer
import calculation


indexer = Indexer()
indexer.main()

print(calculation.make_vector_from_query(indexer))
