import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Visualisasi Aturan Asosiasi Modul")

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
        te_array = te.fit(transactions).transform(transactions, sparse=True)
        df_encoded = pd.DataFrame.sparse.from_spmatrix(te_array, columns=te.columns_)

        try:
            freq_items = apriori(df_encoded, min_support=support, use_colnames=True)
        except MemoryError:
            st.error("Memory error: Coba naikkan minimum support atau kecilkan ukuran data.")
            st.stop()

        if freq_items.empty:
            st.warning("Tidak ada itemset yang memenuhi minimum support. Coba turunkan nilai support.")
            st.stop()
        
        rules = association_rules(freq_items, metric="confidence", min_threshold=confidence)

        # Filter aturan dengan 1 item pada antecedent dan consequent
        rules = rules[
            (rules['antecedents'].apply(lambda x: len(x) == 1)) &
            (rules['consequents'].apply(lambda x: len(x) == 1))
        ]

        if rules.empty:
            st.warning("Tidak ada aturan asosiasi yang ditemukan.")
        else:
            # Konversi frozenset ke string
            rules['antecedents'] = rules['antecedents'].apply(lambda x: next(iter(x)))
            rules['consequents'] = rules['consequents'].apply(lambda x: next(iter(x)))

            rules.rename(columns={
                'antecedents': 'Modul A',
                'consequents': 'Modul B',
                'support': 'Support',
                'confidence': 'Confidence',
                'lift': 'Lift'
            }, inplace=True)

            rules_display = rules.copy()
            rules_display["Support"] = rules_display["Support"].apply(lambda x: float(f"{x:.3f}"))
            rules_display["Confidence"] = rules_display["Confidence"].apply(lambda x: float(f"{x:.3f}"))
            rules_display["Lift"] = rules_display["Lift"].apply(lambda x: float(f"{x:.3f}"))

            # Urutkan berdasarkan Support, lalu Confidence
            rules_display_sorted = rules_display.sort_values(by=["Support", "Confidence"], ascending=[False, False])

            st.subheader("Association Rules")
            st.dataframe(rules_display_sorted[["Modul A", "Modul B", "Support", "Confidence", "Lift"]])

            # Bangun grafik asosiasi
            G = nx.DiGraph()
            for _, row in rules.iterrows():
                G.add_edge(row['Modul A'], row['Modul B'],
                           confidence=row['Confidence'],
                           lift=row['Lift'])

            if G.number_of_edges() == 0:
                st.info("Tidak ada hasil untuk ditampilkan dalam visualisasi.")
            else:
                pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

                node_x, node_y = [], []
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)

                edge_x, edge_y = [], []
                edge_hover_x, edge_hover_y, edge_hover_text = [], [], []

                for edge in G.edges(data=True):
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x += [x0, x1, None]
                    edge_y += [y0, y1, None]

                    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                    edge_hover_x.append(mx)
                    edge_hover_y.append(my)

                    conf = edge[2]['confidence']
                    lift = edge[2]['lift']
                    hovertext = (
                        f"{edge[0]} → {edge[1]}<br>"
                        f"Confidence: {conf:.3f}<br>"
                        f"Lift: {lift:.3f}"
                    )

                    reverse = rules[
                        (rules['Modul A'] == edge[1]) & (rules['Modul B'] == edge[0])
                    ]
                    if not reverse.empty:
                        rc = reverse.iloc[0]['Confidence']
                        rl = reverse.iloc[0]['Lift']
                        hovertext += (
                            f"<br><br>{edge[1]} → {edge[0]}<br>"
                            f"Confidence: {rc:.3f}<br>"
                            f"Lift: {rl:.3f}"
                        )

                    edge_hover_text.append(hovertext)

                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y, line=dict(width=1, color='#888'),
                    hoverinfo='none', mode='lines'
                )

                edge_hover_trace = go.Scatter(
                    x=edge_hover_x, y=edge_hover_y,
                    mode='markers', hoverinfo='text',
                    marker=dict(size=10, color='rgba(0,0,0,0)'),
                    hovertext=edge_hover_text,
                    showlegend=False
                )

                # Pewarnaan berdasarkan jumlah hubungan (degree)
                node_color = [G.degree(node) for node in G.nodes()]
                node_hover_text = [f"Modul: {node}<br>Jumlah Hubungan: {G.degree(node)}" for node in G.nodes()]

                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    text=[str(n) for n in G.nodes()],
                    textposition="bottom center",
                    hoverinfo='text',
                    marker=dict(
                        showscale=True,
                        colorscale='YlGnBu',
                        color=node_color,
                        size=20,
                        colorbar=dict(title=dict(text='Jumlah Hubungan', side='right')),
                        line_width=2
                    ),
                    hovertext=node_hover_text
                )

                fig = go.Figure(
                    data=[edge_trace, edge_hover_trace, node_trace],
                    layout=go.Layout(
                        title="Visualisasi Aturan Asosiasi Modul",
                        title_x=0.4,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    )
                )

                st.plotly_chart(fig, use_container_width=True)
