"""Import required libraries"""
import pandas as pd
import datetime as dt
import seaborn as sns
import plotly.graph_objects as go
import plotly.figure_factory as ff

"""Import data"""
posts = pd.read_csv('../Task_1/posts.csv', index_col=0)

"""Data preprocessing"""
target = {
    'family': 0,
    'Parenting': 1,
    'depression': 2,
    'askwomenadvice': 3,
    'SuicideWatch': 4,
    'Marriage': 5,
    'TwoXChromosomes': 6,
    'love': 7,
    'relationships': 8,
    'DecidingToBeBetter': 9
}
posts['target'] = posts['subreddit'].apply(lambda x: target[x])
posts['date'] = posts['created_utc'].apply(dt.datetime.fromtimestamp)
posts['cleared_text'] = posts['cleared_text'].apply(lambda x: x.strip())
posts['title_cleared'] = posts['title_cleared'].apply(lambda x: str(x).strip())


def text_len(text):
    return len(text.split(" "))


posts['text_len'] = posts['cleared_text'].apply(text_len)
posts['title_len'] = posts['title_cleared'].apply(text_len)

"""Delete invalid documents"""
posts.drop(posts[posts['text_len'] < 30].index, inplace=True)  # документ с кол-вом слов меньше 30

"""Exploratory data analysis"""
df_stat = posts[['target', 'score', 'num_comments', 'title_len', 'text_len']]
df_stat.drop('target', axis=1).describe()
df_stat.groupby('target').describe()

"""Visualization in eda.ipynb"""
def plot_distribution(metric, start, end, size):
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=df_stat[metric],
            name=metric,
            xbins=dict(
                start=start,
                end=end,
                size=size
            )
        )
    )
    fig.update_layout(
        barmode='overlay',
        title_text=metric + ' distribution',
        xaxis_title_text='Value',  # xaxis label
        yaxis_title_text='Count',  # yaxis label
    )
    return fig


img = plot_distribution('score', 0, 50, 2)
img.write_image("figures/score.png")
img = plot_distribution('num_comments', 0, 100, 2)
img.write_image("figures/num_comments.png")
img = plot_distribution('text_len', 0, 1000, 25)
img.write_image("figures/text_len.png")
img = plot_distribution('title_len', 0, 40, 1)
img.write_image("figures/title_len.png")

sns.set(
    style="white",
    palette="muted",
    color_codes=True
)
sns_plot = sns.pairplot(df_stat, hue='target')
sns_plot.savefig("figures/pair_grid.png")

corrs = df_stat.drop('target', axis=1).corr()
fig = ff.create_annotated_heatmap(
    z=corrs.values,
    x=list(corrs.columns),
    y=list(corrs.index),
    annotation_text=corrs.round(2).values,
    showscale=True)

fig.write_image("figures/matrix_corr.png")

"""Export data"""
posts.to_csv('train.csv')
