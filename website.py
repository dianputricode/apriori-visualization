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
        st.error("Kolom 'nama_modul' tidak ditemukan pada file yang diunggah.")
    else:
        support = st.selectbox("Pilih minimum support", [0.1, 0.05, 0.02, 0.01, 0.005])
        confidence = st.selectbox("Pilih minimum confidence", [0.9, 0.7, 0.5, 0.3, 0.1])

        df["nama_modul"] = df["nama_modul"].astype(str)
        transactions = df["nama_modul"].apply(lambda x: [mod.strip() for mod in x.split(',')]).tolist()

        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_array, columns=te.columns_)

        try:
            freq_items = apriori(df_encoded, min_support=support, use_colnames=True)
            rules = association_rules(freq_items, metric="confidence", min_threshold=confidence)
        except MemoryError:
            st.error("Kesalahan memori: Naikkan nilai minimum support atau kurangi ukuran dataset.")
            st.stop()

        # Filter hanya aturan dengan 1 modul di setiap sisi
        rules = rules[
            (rules['antecedents'].apply(lambda x: len(x) == 1)) &
            (rules['consequents'].apply(lambda x: len(x) == 1))
        ]

        if rules.empty:
            st.warning("Tidak ditemukan aturan asosiasi dengan parameter yang dipilih.")
        else:
            # Ubah frozenset ke string
            rules['Modul A'] = rules['antecedents'].apply(lambda x: next(iter(x)))
            rules['Modul B'] = rules['consequents'].apply(lambda x: next(iter(x)))

            st.subheader("Aturan Asosiasi")
            st.dataframe(rules[["Modul A", "Modul B", "support", "confidence"]])

            # Bangun grafik
            G = nx.DiGraph()

            for _, row in rules.iterrows():
                G.add_edge(row['Modul A'], row['Modul B'],
                           confidence=row['confidence'])

            pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

            node_x = []
            node_y = []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)

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

                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                conf = edge[2]['confidence']

                edge_hover_x.append(mx)
                edge_hover_y.append(my)
                hovertext = (
                    f"{edge[0]} → {edge[1]}<br>"
                    f"Confidence: {conf:.2f}"
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

            # Pewarnaan berdasarkan jumlah koneksi
            node_colors = [G.degree(node) for node in G.nodes()]
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
                    color=node_colors,
                    size=20,
                    colorbar=dict(
                        title=dict(text='Jumlah Koneksi', side='right')
                    ),
                    line_width=2)
            )

            node_hover_text = []
            for node in G.nodes():
                degree = G.degree(node)
                node_hover_text.append(f"Modul: {node}<br>Jumlah Koneksi: {degree}")

            node_trace.hovertext = node_hover_text

            fig = go.Figure(data=[edge_trace, edge_hover_trace, node_trace],
                            layout=go.Layout(
                                title="Jaringan Asosiasi Modul",
                                title_x=0.5,
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20, l=5, r=5, t=40),
                                annotations=[dict(
                                    text="",
                                    showarrow=False,
                                    xref="paper",
                                    yref="paper")],
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                            ))

            st.plotly_chart(fig, use_container_width=True)
