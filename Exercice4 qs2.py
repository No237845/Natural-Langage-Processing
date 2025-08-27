#2. Visualiser les similarités entre différents mots à l'aide de PCA 
import numpy as np
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
import plotly.io as pio

def display_pca_scatterplot_3D(model, user_input=None, words=None, label=None, color_map=None, topn=5, sample=10):
    if words is None:
        if sample > 0:
            # Use key_to_index instead of vocab
            words = np.random.choice(list(model.key_to_index.keys()), sample, replace=False)
        else:
            words = list(model.key_to_index.keys())
    
    # If user_input is provided, get similar words for each input word
    if user_input is not None:
        word_vectors = []
        words = []
        for word in user_input:
            try:
                similar_words = model.most_similar(word, topn=topn)
                words.extend([w for w, _ in similar_words])
                word_vectors.extend([model[w] for w, _ in similar_words])
            except KeyError:
                print(f"Word '{word}' not in vocabulary. Skipping.")
                continue
        
        # Add the input words at the end
        words.extend(user_input)
        word_vectors.extend([model[word] for word in user_input if word in model])
    else:
        # Get vectors for all words
        word_vectors = [model[word] for word in words if word in model]

    # Convert to numpy array
    word_vectors = np.array(word_vectors)
    
    # Perform PCA
    three_dim = PCA(random_state=0, n_components=3).fit_transform(word_vectors)

    data = []
    count = 0
    
    if user_input is not None:
        for i in range(len(user_input)):
            if count + topn > len(words):
                break
                
            trace = go.Scatter3d(
                x=three_dim[count:count+topn,0], 
                y=three_dim[count:count+topn,1],  
                z=three_dim[count:count+topn,2],
                text=words[count:count+topn],
                name=user_input[i],
                textposition="top center",
                textfont_size=20,
                mode='markers+text',
                marker={
                    'size': 10,
                    'opacity': 0.8,
                    'color': i  # Use different colors for each group
                }
            )
            data.append(trace)
            count += topn

        # Add input words trace
        if count < len(words):
            trace_input = go.Scatter3d(
                x=three_dim[count:,0], 
                y=three_dim[count:,1],  
                z=three_dim[count:,2],
                text=words[count:],
                name='input words',
                textposition="top center",
                textfont_size=20,
                mode='markers+text',
                marker={
                    'size': 10,
                    'opacity': 1,
                    'color': 'black'
                }
            )
            data.append(trace_input)
    else:
        # Single trace for all words when no user_input
        trace = go.Scatter3d(
            x=three_dim[:,0], 
            y=three_dim[:,1],  
            z=three_dim[:,2],
            text=words,
            textposition="top center",
            textfont_size=20,
            mode='markers+text',
            marker={
                'size': 10,
                'opacity': 0.8,
            }
        )
        data.append(trace)

    # Configure the layout
    layout = go.Layout(
        title=label,
        margin={'l': 0, 'r': 0, 'b': 0, 't': 30},
        showlegend=True,
        legend=dict(
            x=1,
            y=0.5,
            font=dict(
                family="Courier New",
                size=25,
                color="black"
            )),
        font=dict(
            family="Courier New",
            size=15),
        autosize=False,
        width=1000,
        height=1000
    )

    plot_figure = go.Figure(data=data, layout=layout)
    
    # Save the plot as an HTML file
    pio.write_html(plot_figure, file='word_vectors_3d.html', auto_open=True)
    
    # Also try to show it directly
    plot_figure.show()

# Load the model
try:
    model = KeyedVectors.load_word2vec_format(
        "GoogleNews-vectors-negative300.bin.gz", binary=True
    )
    print("Model loaded successfully!")
    
    # Display the scatterplot
    display_pca_scatterplot_3D(
        model,
        user_input=["coffee", "tea", "sugar", "drink", "computer"],
        words=None,
        label="3D PCA Projection of Words",
        color_map=None,
        topn=5,
        sample=10
    )
except FileNotFoundError:
    print("Model file not found. Please make sure 'GoogleNews-vectors-negative300.bin.gz' is in the current directory.")
except Exception as e:
    print(f"An error occurred: {e}")