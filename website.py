import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Visualisasi Asosiasi Modul dengan Algoritma Apriori")

uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'nama_modul' not in df.columns:
        st.error('Kolom "nama_modul" tidak ditemukan pada file yang diunggah.')
    else:
        support = st.selectbox("Pilih nilai minimum support", [0.1, 0.05, 0.02, 0.01, 0.005])
        confidence = st.selectbox("Pilih nilai minimum confidence", [0.9, 0.7, 0.5, 0.3, 0.1])

        df["nama_modul"] = df["nama_modul"].astype(str)
        transactions = df["nama_modul"].apply(lambda x: [mod.strip() for mod in x.split(',')]).tolist()

        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_array, columns=te.columns_)

        try:
            freq_items = apriori(df_encoded, min_support=support, use_colnames=True)
            rules = association_rules(freq_items, metric="confidence", min_threshold=confidence)
        except MemoryError:
            st.error("Kesalahan: Naikkan nilai minimum support untuk mengurangi ukuran dataset.")
            st.stop()

        # Filter aturan dengan 1 item di antecedent dan consequent
        rules = rules[
            (rules['antecedents'].apply(lambda x: len(x) == 1)) &
            (rules['consequents'].apply(lambda x: len(x) == 1))
        ]

        if rules.empty:
            st.warning("Tidak ditemukan aturan asosiasi untuk parameter yang dipilih.")
        else:
            # Ubah frozenset menjadi string
            rules['antecedents'] = rules['antecedents'].apply(lambda x: next(iter(x)))
            rules['consequents'] = rules['consequents'].apply(lambda x: next(iter(x)))

            st.subheader("Aturan Asosiasi")
            rules_display = rules.rename(columns={
                "antecedents": "Modul A",
                "consequents": "Modul B",
                "support": "Support",
                "confidence": "Confidence"
            })
            st.dataframe(rules_display[["Modul A", "Modul B", "Support", "Confidence"]])

            # Bangun grafik berarah
            G = nx.DiGraph()

            for _, row in rules.iterrows():
                G.add_edge(row['antecedents'], row['consequents'],
                           confidence=row['confidence'])

            pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

            # Koordinat node
            node_x = []
            node_y = []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)

            # Koordinat garis antar node
            edge_x = []
            edge_y = []
            edge_hover_x = []
            edge_hover_y = []
            edge_hover_text = []

            for edge in G.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

                # Titik tengah untuk hover
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2

                conf = edge[2]['confidence']
                edge_hover_x.append(mx)
                edge_hover_y.append(my)

                hovertext = (
                    f"{edge[0]} → {edge[1]}<br>"
                    f"Kepercayaan: {conf:.2f}"
                )
                edge_hover_text.append(hovertext)

            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=1, color='#888'),
                hoverinfo='none',
                mode='lines'
            )

            edge_hover_trace = go.Scatter(
                x=edge_hover_x,
                y=edge_hover_y,
                mode='markers',
                hoverinfo='text',
                marker=dict(size=10, color='rgba(0,0,0,0)'),
                hovertext=edge_hover_text,
                showlegend=False
            )

            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                text=[str(node) for node in G.nodes()],
                textposition="bottom center",
                hoverinfo='text',
                marker=dict(
                    showscale=True,
                    colorscale='YlGnBu',
                    color=[rules.loc[rules['antecedents'] == node, 'confidence'].mean() if node in rules['antecedents'].values else 0 for node in G.nodes()],
                    size=20,
                    colorbar=dict(
                        title=dict(text='Rata-rata Kepercayaan', side='right')
                    ),
                    line_width=2)
            )

            node_hover_text = []
            for node in G.nodes():
                avg_conf = rules.loc[rules['antecedents'] == node, 'confidence'].mean()
                if pd.isna(avg_conf):
                    avg_conf = 0
                node_hover_text.append(f"Modul: {node}<br>Rata-rata Kepercayaan: {avg_conf:.2f}")

            node_trace.hovertext = node_hover_text

            fig = go.Figure(data=[edge_trace, edge_hover_trace, node_trace],
                            layout=go.Layout(
                                title=dict(
                                    text="Graf Jaringan Asosiasi Modul",
                                    x=0.5,
                                    xanchor='center',
                                    font=dict(size=24)
                                ),
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20, l=20, r=20, t=60),
                                annotations=[dict(
                                    text="",
                                    showarrow=False,
                                    xref="paper",
                                    yref="paper")],
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                            ))

            st.plotly_chart(fig, use_container_width=True)
