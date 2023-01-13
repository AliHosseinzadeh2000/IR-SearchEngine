import indexer
import Calculation

indexer = indexer.Indexer()
indexer.main()

print(Calculation.make_vector_from_query(indexer))
