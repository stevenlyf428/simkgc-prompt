def plot_entity_embeddings(path='./vectors.json'):
    import json
    import pandas as pd

    import seaborn as sns
    import matplotlib.pyplot as plt

    from sklearn.manifold import TSNE

    tail_id_to_str = {'Q5': 'Human',
                      'Q16521': 'Taxon',
                      'Q482994': 'Album',
                      'Q11424': 'Movie',
                      'Q486972': 'Community',
                      'Q4830453': 'Company',
                      'Q571': 'Book',
                      'Q532': 'Village'}

    examples = json.load(open(path, 'r', encoding='utf-8'))
    x, y = [], []
    for ex in examples:
        x.append([float(n) for n in ex['head_vector'].split(',')])
        y.append(tail_id_to_str[ex['tail_id']])

    tsne = TSNE(n_components=2, verbose=1, random_state=12)
    z = tsne.fit_transform(x)
    df = pd.DataFrame()
    df["entity type"] = y
    df["x"] = z[:, 0]
    df["y"] = z[:, 1]

    sns.scatterplot(x="x", y="y", hue='entity type',
                    palette=sns.color_palette("hls", len(tail_id_to_str)),
                    data=df).set(title="Entity embedding visualization")


    plt.show()