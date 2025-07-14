import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
import matplotlib.pyplot as plt

st.title("Visualisasi Asosiasi Modul dengan Apriori")

# Upload file
uploaded_file = st.file_uploader("Unggah file CSV berisi transaksi (list modul)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, header=None)
    transactions = df.values.tolist()

    # Validasi isi transaksi
    if len(transactions) == 0 or not all(isinstance(t, list) for t in transactions):
        st.error("Data tidak valid. Pastikan file berisi daftar transaksi dalam format list of list.")
        st.stop()

    try:
        # Encode data
        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions, sparse=True)
        df_encoded = pd.DataFrame.sparse.from_spmatrix(te_array, columns=te.columns_)

        # Pilihan parameter
        min_support = st.slider("Minimum Support", 0.001, 0.1, 0.005, 0.001)
        min_confidence = st.slider("Minimum Confidence", 0.1, 1.0, 0.3, 0.05)

        # Apriori
        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

        if rules.empty:
            st.warning("Tidak ditemukan aturan asosiasi dengan parameter yang dipilih.")
        else:
            st.subheader("Aturan Asosiasi")
            st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence']])

            # Visualisasi graf
            st.subheader("Visualisasi Grafik Asosiasi")
            G = nx.DiGraph()

            for _, row in rules.iterrows():
                a = ', '.join(list(row['antecedents']))
                b = ', '.join(list(row['consequents']))
                G.add_edge(a, b, weight=row['support'], confidence=row['confidence'])

            pos = nx.spring_layout(G, k=0.5, iterations=50)

            edge_weights = [G[u][v]['weight'] * 10 for u, v in G.edges()]
            edge_colors = [G[u][v]['confidence'] for u, v in G.edges()]
            node_sizes = [500 + 100 * G.degree(n) for n in G.nodes()]

            plt.figure(figsize=(12, 8))
            nx.draw(G, pos,
                    with_labels=True,
                    node_color='skyblue',
                    edge_color=edge_colors,
                    width=edge_weights,
                    node_size=node_sizes,
                    edge_cmap=plt.cm.viridis,
                    font_size=8,
                    arrowsize=10)
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                       norm=plt.Normalize(vmin=min(edge_colors), vmax=max(edge_colors)))
            sm.set_array([])
            cbar = plt.colorbar(sm)
            cbar.set_label('Confidence')

            st.pyplot(plt)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")
