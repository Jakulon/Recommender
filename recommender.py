from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k
import numpy as np
import kaggle


data = fetch_movielens(min_rating=4.0)

print(repr(data['train']))
print(repr(data['test']))


model = LightFM(loss='warp')

model.fit(data['train'], epochs=30, num_threads=1)

print("Train precision: %.2f" % precision_at_k(model, data['train'], k=5).mean())
print("Test precision: %.2f" % precision_at_k(model, data['test'], k=5).mean())

def sample_recommendation(model, data, user_ids):
    n_users, m_movies = data['train'].shape

    for user_id in user_ids:
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        
        scores = model.predict(user_id, np.arange(m_movies))
        top_items = data['item_labels'][np.argsort(-scores)]

        print("UserID %s" % user_id)
        print("     Known positives:")

        for i in known_positives[:3]:
            print("         %s" % i)

        print("     Recommended:")

        for i in top_items[:3]:
            print("         %s" % i)



sample_recommendation(model, data, [3, 25, 450])


