import pstats
from pstats import SortKey
p = pstats.Stats('restats')
p.strip_dirs().sort_stats(2).print_stats()
