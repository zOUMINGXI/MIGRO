"""Exact native nearest-movie edit distance with length lower-bound pruning."""
from functools import lru_cache
import numpy as np
from editdistance import eval as edit_distance

@lru_cache(maxsize=2)
def candidates(movie_tuple):
    names=list(set(movie_tuple))
    lower=[x.lower() for x in names]
    buckets={}
    for i,text in enumerate(lower):buckets.setdefault(len(text),[]).append(i)
    return names,lower,buckets

def find_nearest_movie(target_name,candidate_movies):
    names,lower,buckets=candidates(tuple(candidate_movies))
    query=target_name.lower()
    best_index=0;best_distance=edit_distance(query,lower[0])
    for length in sorted(buckets,key=lambda n:abs(n-len(query))):
        # Unit-cost Levenshtein distance cannot be smaller than length gap.
        if abs(length-len(query))>best_distance:break
        for i in buckets[length]:
            if i==0:continue
            distance=edit_distance(query,lower[i])
            if distance<best_distance or distance==best_distance and i<best_index:
                best_index,best_distance=i,distance
    return dict(movie_name=target_name,min_edit_distance=best_distance,nearest_movie=names[best_index])
